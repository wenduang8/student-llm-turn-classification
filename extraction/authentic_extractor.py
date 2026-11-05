#!/usr/bin/env python3
"""
authentic_extractor.py

Normalize the authentic COMP2017 student-LLM reference submissions (de-identified,
one flat folder of P###_<assignment>_<origname> files, mixed formats) into the
consistent CSV schema the rest of the pipeline already consumes.

Everything runs from this one entrypoint (no manual steps). Outputs land in
data/processed/. See docs/pipeline_reports/extraction_dev_log.md and the auto-generated
data/processed/EXTRACTION_REPORT.md for the reproducible run summary.

Target schema (unchanged from simulation_conversations.csv / huggingface_integration.py):
    Student ID, Student Prompt, LLM Response, Delegation Label, Cognitive Label, Coded, Source
We preserve those 7 canonical columns exactly and add additive provenance columns
(Conversation ID, Segment Order, Source File) which the loader ignores.
"""

from __future__ import annotations
import os
import re
import sys
import csv
import json
import time
import hashlib
import logging
import datetime
from pathlib import Path

import pandas as pd

# Reuse existing PDF extraction
try:
    from pdf_extractor import extract_pdf_text
except Exception:
    extract_pdf_text = None

# ---------------------------------------------------------------------------
# Config (committed; deterministic)
# ---------------------------------------------------------------------------
SEED = 42
REPO = Path(__file__).resolve().parent.parent   # repo root (this file lives in extraction/)
INPUT_DIR = REPO / "data" / "Raw data from COMP2017" / "AI transcripts (de-identified)"
OUT_DIR = REPO / "data" / "processed"
TEXT_DIR = OUT_DIR / "normalized_text"
LOG_DIR = OUT_DIR / "logs"
CACHE_DIR = OUT_DIR / ".cache"          # link/ocr caches -> idempotent, resumable
DEVLOG = REPO / "docs" / "pipeline_reports" / "extraction_dev_log.md"

SOURCE_TAG = "authentic"

TEXT_EXTS = {".txt", ".md", ""}
RICH_EXTS = {".rtf", ".pdf", ".docx"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
# everything else (.h .json .yaml .xml .save .mdc .rar ...) -> non-transcript, recorded & skipped

PREVIEW_CHARS = 500
# Coverage sanity: flag if extracted text is implausibly small vs source bytes.
COVERAGE_MIN_CHARS = 20
COVERAGE_RATIO = 0.02   # extracted_chars < ratio * source_bytes -> suspicious

URL_RE = re.compile(r'https?://[^\s<>"\')]+', re.I)
SHARE_RE = re.compile(
    r'https?://(?:chatgpt\.com|chat\.openai\.com)/share/[A-Za-z0-9\-]+'
    r'|https?://claude\.ai/share/[A-Za-z0-9\-]+'
    r'|https?://g\.co/gemini/share/[A-Za-z0-9\-]+'
    r'|https?://(?:www\.)?sharegpt\.com/[A-Za-z0-9\-]+', re.I)

# ---------------------------------------------------------------------------
# Turn markers.  Each pattern is line-anchored at match time.  A "family" name
# groups a student/llm pair so marker_style can be reported.
# strong markers are trusted; weak markers (Q:/A:) are a fallback only.
# ---------------------------------------------------------------------------
STRONG_STUDENT = [
    ("you_said",       r'You\s+said:'),
    ("prompt_response",r'Prompt(?:\s+\d+)?(?:\s+BEGIN)?:'),
    ("user_assistant", r'User:'),
    ("me_ai",          r'Me:'),
    ("bold_you",       r'\*\*\s*You\s*\*\*:?'),
    ("user_input",     r'User\s+input\s*[:：]'),          # "User input:"
    ("cn_said",        r'(?:您|你)\s*说\s*[:：]'),          # 您说： / 你说：
]
STRONG_LLM = [
    ("you_said",       r'ChatGPT\s+said:'),
    ("prompt_response",r'(?:Response(?:\s+Given)?|Output)(?:\s+\d+)?(?:\s+BEGIN)?\s*[:：]'),
    ("user_assistant", r'Assistant:'),
    ("me_ai",          r'AI:'),
    ("bold_you",       r'\*\*\s*ChatGPT\s*\*\*:?'),
    ("chatgpt_plain",  r'ChatGPT:'),
    ("gpt",            r'GPT(?:[-\s]?4o?|[-\s]?5|[-\s]?4)?\s*[:：]'),  # GPT-4o: / GPT: / GPT 4o:
    ("gemini",         r'Gemini:'),
    ("claude",         r'Claude:'),
    ("cn_said",        r'(?:ChatGPT|Claude|Gemini|GPT[-\s]?4o?)\s*说\s*[:：]'),  # ChatGPT 说：
]
WEAK_STUDENT = [("qa", r'Q:')]
WEAK_LLM = [("qa", r'A:')]

# Lines that are export-metadata headers (Save ChatGPT / similar) -> not turns.
HEADER_LINE_RE = re.compile(
    r'^\s*(Company|Model|Date|Public URL|Prompt History|Conversation with|Title)\s*:', re.I)

log = logging.getLogger("authentic_extractor")


# ---------------------------------------------------------------------------
# Optional dependency probes (graceful degradation)
# ---------------------------------------------------------------------------
def probe_deps():
    caps = {}
    try:
        from striprtf.striprtf import rtf_to_text  # noqa
        caps["rtf"] = True
    except Exception:
        caps["rtf"] = False
    try:
        import docx  # noqa
        caps["docx"] = True
    except Exception:
        caps["docx"] = False
    try:
        import pytesseract
        from PIL import Image  # noqa
        # tesseract binary?
        try:
            pytesseract.get_tesseract_version()
            caps["ocr"] = True
        except Exception:
            # try known install locations (PATH miss)
            cands = [
                r"D:\Tesseract-OCR\tesseract.exe",
                os.path.expandvars(r"%ProgramFiles%\Tesseract-OCR\tesseract.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            ]
            caps["ocr"] = False
            for cand in cands:
                if os.path.isfile(cand):
                    pytesseract.pytesseract.tesseract_cmd = cand
                    try:
                        pytesseract.get_tesseract_version()
                        caps["ocr"] = True
                        break
                    except Exception:
                        continue
    except Exception:
        caps["ocr"] = False
    try:
        from playwright.sync_api import sync_playwright  # noqa
        caps["playwright"] = True
    except Exception:
        caps["playwright"] = False
    caps["pdf"] = extract_pdf_text is not None
    return caps


# ---------------------------------------------------------------------------
# Text extraction per format
# ---------------------------------------------------------------------------
def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_rtf(path: Path) -> str:
    try:
        from striprtf.striprtf import rtf_to_text
        return rtf_to_text(read_text_file(path))
    except Exception as e:
        log.warning("rtf extract failed %s: %s", path.name, e)
        # crude fallback: strip control words
        txt = read_text_file(path)
        txt = re.sub(r'\\[a-zA-Z]+-?\d* ?', ' ', txt)
        txt = re.sub(r'[{}]', '', txt)
        return txt


def extract_docx(path: Path) -> str:
    """Capture paragraphs AND table cells AND text boxes, in reading order."""
    try:
        import docx
        doc = docx.Document(str(path))
        W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        parts, buf = [], []
        # Walk the body in document order; w:t = text run (incl. tables & textboxes),
        # w:p / w:tr boundaries -> newlines.
        for el in doc.element.body.iter():
            tag = el.tag
            if tag == W + "t":
                if el.text:
                    buf.append(el.text)
            elif tag in (W + "p", W + "tr"):
                if buf:
                    parts.append("".join(buf)); buf = []
                parts.append("\n")
        if buf:
            parts.append("".join(buf))
        text = re.sub(r'\n{3,}', '\n\n', "".join(parts)).strip()
        if len(text) < COVERAGE_MIN_CHARS:
            # cross-check with docx2txt (also grabs tables)
            try:
                import docx2txt
                alt = docx2txt.process(str(path)) or ""
                if len(alt) > len(text):
                    text = alt.strip()
            except Exception:
                pass
        return text
    except Exception as e:
        log.warning("docx extract failed %s: %s", path.name, e)
        try:
            import docx2txt
            return (docx2txt.process(str(path)) or "").strip()
        except Exception:
            return ""


def extract_pdf(path: Path) -> str:
    if extract_pdf_text is None:
        return ""
    try:
        return extract_pdf_text(str(path)) or ""
    except Exception as e:
        log.warning("pdf extract failed %s: %s", path.name, e)
        return ""


def ocr_image(path: Path, caps, cache) -> str:
    if not caps.get("ocr"):
        return ""
    key = f"ocr::{path.name}"
    if key in cache:
        return cache[key]
    try:
        import pytesseract
        from PIL import Image
        txt = pytesseract.image_to_string(Image.open(str(path)))
        txt = txt.strip()
    except Exception as e:
        log.warning("ocr failed %s: %s", path.name, e)
        txt = ""
    cache[key] = txt
    return txt


# ---------------------------------------------------------------------------
# Link recovery (3 tiers)
# ---------------------------------------------------------------------------
def _turns_from_chatgpt_json(data) -> list:
    """Parse ChatGPT share JSON (mapping or linear_conversation) into turns."""
    turns = []
    nodes = None
    if isinstance(data, dict):
        if isinstance(data.get("linear_conversation"), list):
            nodes = data["linear_conversation"]
        elif isinstance(data.get("mapping"), dict):
            nodes = list(data["mapping"].values())
        elif isinstance(data.get("continue_conversation_url"), str) and "mapping" in data:
            nodes = list(data["mapping"].values())
    if nodes is None:
        return turns
    for node in nodes:
        msg = node.get("message") if isinstance(node, dict) else None
        if not msg:
            continue
        role = (msg.get("author") or {}).get("role")
        content = msg.get("content") or {}
        parts = content.get("parts") or []
        text = "\n".join(p for p in parts if isinstance(p, str)).strip()
        if role in ("user", "assistant") and text:
            turns.append({"role": "student" if role == "user" else "llm", "text": text})
    return turns


def recover_link_tier1(url: str, session) -> list:
    """HTTP backend-API JSON (ChatGPT). Returns list of turns or []."""
    m = re.search(r'/share/([A-Za-z0-9\-]+)', url)
    if not m or ("chatgpt.com" not in url and "openai.com" not in url):
        return []
    sid = m.group(1)
    for api in (f"https://chatgpt.com/backend-api/share/{sid}",
                f"https://chat.openai.com/backend-api/share/{sid}"):
        try:
            r = session.get(api, timeout=20, headers={
                "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                turns = _turns_from_chatgpt_json(r.json())
                if turns:
                    return turns
        except Exception as e:
            log.debug("tier1 %s: %s", api, e)
    return []


def recover_link_tier2(url: str, browser) -> list:
    """Playwright headless render + DOM scrape. Returns list of turns or []."""
    if browser is None:
        return []
    try:
        page = browser.new_page(user_agent="Mozilla/5.0")
        page.goto(url, wait_until="domcontentloaded", timeout=25000)
        page.wait_for_timeout(2500)
        # ChatGPT share: turns carry data-message-author-role
        turns = []
        nodes = page.query_selector_all("[data-message-author-role]")
        for n in nodes:
            role = n.get_attribute("data-message-author-role")
            text = (n.inner_text() or "").strip()
            if role in ("user", "assistant") and text:
                turns.append({"role": "student" if role == "user" else "llm", "text": text})
        if not turns:
            # generic fallback: main article text, then try marker segmentation later
            body = page.query_selector("main") or page.query_selector("body")
            raw = (body.inner_text() if body else "") or ""
            page.close()
            return [{"role": "raw", "text": raw.strip()}] if raw.strip() else []
        page.close()
        return turns
    except Exception as e:
        log.debug("tier2 %s: %s", url, e)
        return []


# ---------------------------------------------------------------------------
# Turn segmentation from pasted / extracted text
# ---------------------------------------------------------------------------
def _find_markers(text, student_markers, llm_markers):
    """Return sorted list of (offset, role, family) for line-anchored markers."""
    hits = []
    for role, markers in (("student", student_markers), ("llm", llm_markers)):
        for family, pat in markers:
            for m in re.finditer(r'(?im)^[ \t>*#\-]*' + pat + r'[ \t]*', text):
                hits.append((m.start(), m.end(), role, family))
    hits.sort(key=lambda h: h[0])
    return hits


def segment_turns(text: str):
    """
    Segment pasted transcript text into ordered turns.
    Returns (turns, marker_families:set, alternation_ok:bool).
    turns: list of {"role","text"}.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # drop export-metadata header lines and long separator lines (<<<<, >>>>, ====)
    text = re.sub(r'(?m)^[<>=~_\-\*#\s]{6,}$', '', text)
    text = "\n".join(ln for ln in text.split("\n") if not HEADER_LINE_RE.match(ln))

    hits = _find_markers(text, STRONG_STUDENT, STRONG_LLM)
    families = {f for _, _, _, f in hits}
    roles = {r for _, _, r, _ in hits}

    segs = []
    if len(roles) >= 2:
        # Both roles marked: segment between consecutive markers.
        for i, (s, e, role, family) in enumerate(hits):
            nxt = hits[i + 1][0] if i + 1 < len(hits) else len(text)
            segs.append({"role": role, "text": text[e:nxt].strip()})
    elif roles == {"llm"} and hits:
        # Only the LLM side is marked (e.g. only "ChatGPT said:"): the text before
        # each llm marker is the student turn; the text after it is the response.
        prev = 0
        for i, (s, e, role, family) in enumerate(hits):
            stu = text[prev:s].strip()
            nxt = hits[i + 1][0] if i + 1 < len(hits) else len(text)
            llm = text[e:nxt].strip()
            if stu:
                segs.append({"role": "student", "text": stu})
            if llm:
                segs.append({"role": "llm", "text": llm})
            prev = nxt
    else:
        # fall back to weak markers (Q:/A:), needs both roles
        weak = _find_markers(text, WEAK_STUDENT, WEAK_LLM)
        if len({r for _, _, r, _ in weak}) >= 2:
            for i, (s, e, role, family) in enumerate(weak):
                nxt = weak[i + 1][0] if i + 1 < len(weak) else len(text)
                segs.append({"role": role, "text": text[e:nxt].strip()})
                families.add(family)
        else:
            return [], families, True  # no reliable structure -> caller decides

    # Merge consecutive same-role segments (handles S,S / L,L runs from stray markers)
    merged = []
    for seg in segs:
        if merged and merged[-1]["role"] == seg["role"]:
            if seg["text"]:
                merged[-1]["text"] = (merged[-1]["text"] + "\n" + seg["text"]).strip()
        else:
            merged.append(dict(seg))
    merged = [m for m in merged if m["text"]]

    # alternation ok if, after merging, roles strictly alternate
    roles = [m["role"] for m in merged]
    alt_ok = all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
    return merged, families, alt_ok


def turns_to_pairs(turns):
    """Pair each student turn with following llm turn(s) -> (prompt, response)."""
    # normalize 'raw' role (from generic fallback) by trying marker segmentation
    if len(turns) == 1 and turns[0]["role"] == "raw":
        seg, _, _ = segment_turns(turns[0]["text"])
        turns = seg if seg else []
    pairs = []
    i = 0
    n = len(turns)
    while i < n:
        if turns[i]["role"] == "student":
            prompt = turns[i]["text"]
            resp, j = [], i + 1
            while j < n and turns[j]["role"] == "llm":
                resp.append(turns[j]["text"]); j += 1
            pairs.append((prompt, "\n".join(resp).strip()))
            i = j
        else:
            i += 1
    return pairs


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
def is_link_only(text: str):
    urls = URL_RE.findall(text or "")
    if not urls:
        return False, None
    residual = URL_RE.sub("", text or "")
    residual = re.sub(r'\s+', '', residual)
    share = SHARE_RE.search(text or "")
    url = share.group(0) if share else urls[0]
    return (len(residual) < 30), url


def looks_like_summary(text, filename):
    if not text:
        return False
    has_markers, _, _ = segment_turns(text)
    if has_markers:
        return False
    fn = filename.lower()
    prose = len(text) < 2500
    named = "summary" in fn or "summar" in fn
    return named or prose


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_prefix(fname: str):
    m = re.match(r'^(P\d{3})_(P[12])_(.+)$', fname)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return "UNKNOWN", "NA", fname


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOG_DIR / f"run_{ts}.log"
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(logfile, encoding="utf-8"); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(ch)
    return logfile, ts


def load_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / "recovery_cache.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache):
    (CACHE_DIR / "recovery_cache.json").write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8")


_SHARE_ID_RE = re.compile(r'/(?:share|chat)/([A-Za-z0-9\-]+)', re.I)


def _share_id_for(cid, rec):
    """Share/conversation id for a conversation: from link_url, else from its text file."""
    u = rec.get("link_url") or ""
    m = _SHARE_ID_RE.search(u)
    if m:
        return m.group(1)
    tp = TEXT_DIR / f"{cid}.txt"
    if tp.exists():
        m2 = SHARE_RE.search(tp.read_text(encoding="utf-8"))
        if m2:
            m3 = _SHARE_ID_RE.search(m2.group(0))
            return m3.group(1) if m3 else None
    return None


def deduplicate(manifest_rows, master_rows, segmented_rows):
    """
    Same conversation submitted across multiple files by one student.
    - Exact content-duplicate conversations (identical turn-pair sequence, same student):
      keep the earliest conversation_id, mark the rest status=duplicate + duplicate_of,
      and drop their rows from the segmented output.
    - Same share/conversation id but DIFFERENT content (partial captures): flag needs_review,
      do not merge.
    Returns (new_segmented_rows, n_dup_convos, n_dropped_rows, n_partial_flagged).
    """
    from collections import defaultdict
    pairs_by_cid = defaultdict(list)
    for r in segmented_rows:
        pairs_by_cid[r["Conversation ID"]].append(
            (r["Student Prompt"], r["LLM Response"]))
    man_by_cid = {r["conversation_id"]: r for r in manifest_rows if r["conversation_id"]}

    # 1) exact content-duplicate conversations, grouped within a student
    groups = defaultdict(list)
    for cid, pairs in pairs_by_cid.items():
        rec = man_by_cid.get(cid)
        if rec:
            groups[(rec["pid"], tuple(pairs))].append(cid)
    dup_of = {}
    for (_pid, _sig), cids in groups.items():
        cids_sorted = sorted(cids)
        for c in cids_sorted[1:]:
            dup_of[c] = cids_sorted[0]

    # 2) partial captures (not full dups, so keep both but flag needs_review):
    #    (a) same share-id but differing content; (b) different conversations of the
    #    same student that share one or more identical turn pairs (overlap).
    live_cids = [c for c in pairs_by_cid if c not in dup_of]
    share_groups = defaultdict(set)
    for cid in live_cids:
        rec = man_by_cid.get(cid)
        if rec and (sid := _share_id_for(cid, rec)):
            share_groups[(rec["pid"], sid)].add(cid)
    share_partial = set()
    for (_pid, _sid), cids in share_groups.items():
        if len(cids) > 1 and len({tuple(pairs_by_cid[c]) for c in cids}) > 1:
            share_partial.update(cids)

    pair_owners = defaultdict(set)
    for c in live_cids:
        rec = man_by_cid.get(c)
        if not rec:
            continue
        for p in set(pairs_by_cid[c]):
            pair_owners[(rec["pid"], p)].add(c)
    overlap_partial = set()
    for _k, cids in pair_owners.items():
        if len(cids) > 1:
            overlap_partial.update(cids)

    partial_flag = share_partial | overlap_partial
    dup_cids = set(dup_of)
    new_seg = [r for r in segmented_rows if r["Conversation ID"] not in dup_cids]

    for r in manifest_rows:
        cid = r["conversation_id"]
        if cid in dup_cids:
            r["status"] = "duplicate"; r["content_type"] = "duplicate"
            r["needs_review"] = True
            r["notes"] = (r.get("notes", "") + f"; duplicate_of {dup_of[cid]}").strip("; ")
        elif cid in partial_flag:
            r["needs_review"] = True
            reason = ("same share-id, differing content" if cid in share_partial
                      else "overlaps another conversation (partial capture)")
            r["notes"] = (r.get("notes", "") + f"; {reason}").strip("; ")

    for m in master_rows:
        cid = m["conversation_id"]
        m["duplicate_of"] = dup_of.get(cid, "")
        if cid in dup_cids:
            m["status"] = "duplicate"; m["needs_review"] = True

    n_dropped = len(segmented_rows) - len(new_seg)
    return new_seg, len(dup_cids), n_dropped, len(partial_flag)


def main():
    logfile, ts = setup_logging()
    caps = probe_deps()
    log.info("Capabilities: %s", caps)
    if not INPUT_DIR.exists():
        log.error("Input dir not found: %s", INPUT_DIR); sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()

    files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file()],
                   key=lambda p: p.name.lower())
    log.info("Found %d input files", len(files))

    # Lazy browser / http session for link recovery
    import requests
    session = requests.Session()
    bstate = {"browser": None, "pw": None, "need": caps.get("playwright")}

    def get_browser():
        if bstate["browser"] is None and bstate["need"]:
            try:
                from playwright.sync_api import sync_playwright
                bstate["pw"] = sync_playwright().start()
                bstate["browser"] = bstate["pw"].chromium.launch(headless=True)
            except Exception as e:
                log.warning("playwright launch failed: %s", e); bstate["need"] = False
        return bstate["browser"]

    def recover(url):
        """Tier1 (HTTP JSON) -> Tier2 (Playwright). Cached. Returns (turns, tier)."""
        ck = f"link::{url}"
        if ck in cache:
            c = cache[ck]; return c["turns"], c["tier"]
        turns = recover_link_tier1(url, session); tier = 1 if turns else 0
        if not turns:
            turns = recover_link_tier2(url, get_browser()); tier = 2 if turns else 0
        cache[ck] = {"turns": turns, "tier": tier}
        save_cache(cache)   # persist after each live fetch -> resumable
        return turns, tier

    def turns_to_transcript_text(turns):
        return "\n\n".join(
            f"{'Student' if t['role']=='student' else 'LLM'}: {t['text']}"
            for t in turns if t["role"] in ("student", "llm"))

    manifest_rows = []
    master_rows = []
    segmented_rows = []
    manual_links = []
    tier_counts = {1: 0, 2: 0, 3: 0}      # link-only files
    embed_counts = {2: 0, 3: 0}           # no-marker files with an embedded share URL
    conv_seq = {}

    def next_conv_id(pid, assign):
        k = (pid, assign)
        conv_seq[k] = conv_seq.get(k, 0) + 1
        return f"{pid}_{assign}_{conv_seq[k]:02d}"

    for path in files:
        fname = path.name
        pid, assign, orig = parse_prefix(fname)
        ext = path.suffix.lower()
        size = path.stat().st_size
        rec = {"pid": pid, "assignment": assign, "source_file": fname,
               "format": ext or "(noext)", "size_bytes": size,
               "extract_method": "", "status": "", "recovery_tier": "",
               "content_type": "", "is_link_only": False, "link_url": "",
               "n_turns": 0, "char_len": 0, "needs_review": False,
               "conversation_id": "", "marker_style": "", "notes": ""}

        # ---- category ----
        if ext in IMAGE_EXTS:
            text = ocr_image(path, caps, cache)
            rec["extract_method"] = "ocr"
            if not text:
                rec["status"] = "image_only" if not caps.get("ocr") else "ocr_empty"
                rec["needs_review"] = True
                manifest_rows.append(rec); continue
        elif ext in TEXT_EXTS:
            text = read_text_file(path); rec["extract_method"] = "read"
        elif ext == ".rtf":
            text = extract_rtf(path); rec["extract_method"] = "rtf"
        elif ext == ".pdf":
            text = extract_pdf(path); rec["extract_method"] = "pdf"
        elif ext == ".docx":
            text = extract_docx(path); rec["extract_method"] = "docx"
        else:
            rec["status"] = "non_transcript"; rec["content_type"] = "none"
            manifest_rows.append(rec); continue

        text = (text or "").strip()

        # ---- link-only? attempt recovery ----
        lo, url = is_link_only(text)
        if lo and url:
            rec["is_link_only"] = True; rec["link_url"] = url
            turns, tier = recover(url)
            pairs = turns_to_pairs(turns) if turns else []
            if pairs:
                tier_counts[tier] += 1
                rec["recovery_tier"] = tier; rec["extract_method"] = f"link_tier{tier}"
                _emit_transcript(rec, pid, assign, next_conv_id, pairs,
                                 turns_to_transcript_text(turns),
                                 "link_recovered", master_rows, segmented_rows)
                manifest_rows.append(rec); continue
            elif turns:   # page rendered but no clean pairs (raw / one-sided) -> keep raw text for LLM
                rec["recovery_tier"] = tier; rec["extract_method"] = f"link_tier{tier}_raw"
                rec["status"] = "link_recovered"; rec["needs_review"] = True
                raw = "\n\n".join(t.get("text", "") for t in turns).strip()
                cid = next_conv_id(pid, assign); rec["conversation_id"] = cid
                _write_text(cid, raw); rec["char_len"] = len(raw)
                _add_master(master_rows, rec, raw)
                manifest_rows.append(rec); continue
            else:
                tier_counts[3] += 1
                rec["recovery_tier"] = 3; rec["status"] = "link_unrecovered"
                rec["needs_review"] = True
                manual_links.append({"pid": pid, "source_file": fname,
                                     "url": url, "status": "unrecovered"})
                manifest_rows.append(rec); continue

        if not text:
            rec["status"] = "empty"; rec["content_type"] = "none"
            rec["needs_review"] = True
            manifest_rows.append(rec); continue

        # coverage sanity
        if len(text) < COVERAGE_MIN_CHARS or (size > 2000 and len(text) < COVERAGE_RATIO * size):
            rec["needs_review"] = True
            rec["notes"] = "low-coverage extraction"

        # ---- segment ----
        turns, families, alt_ok = segment_turns(text)
        rec["marker_style"] = "+".join(sorted(families)) if families else ""
        if turns:
            pairs = turns_to_pairs(turns)
            if not alt_ok:
                rec["needs_review"] = True
                rec["notes"] = (rec["notes"] + "; non-alternating markers").strip("; ")
            _emit_transcript(rec, pid, assign, next_conv_id, pairs, text,
                             "transcript", master_rows, segmented_rows)
            manifest_rows.append(rec); continue

        # No turn markers. For any non-image text/rich file, an embedded share URL means the
        # real conversation lives on an external page -> recover it BEFORE classifying as
        # summary/no_markers. This covers 'summary'-named files that only reference the link.
        share = None if rec["format"] in IMAGE_EXTS else SHARE_RE.search(text)
        if share:
            url = share.group(0); rec["link_url"] = url
            rturns, tier = recover(url)
            pairs = turns_to_pairs(rturns) if rturns else []
            if pairs:
                embed_counts[2] += 1
                rec["recovery_tier"] = tier
                rec["extract_method"] = f"embed_link_tier{tier}"
                rec["status"] = "link_recovered_embedded"
                _emit_transcript(rec, pid, assign, next_conv_id, pairs,
                                 turns_to_transcript_text(rturns),
                                 "link_recovered_embedded", master_rows, segmented_rows)
                manifest_rows.append(rec); continue
            elif rturns:   # rendered but no clean pairs -> keep raw text (retain pasted too) for LLM
                raw = "\n\n".join(t.get("text", "") for t in rturns).strip()
                if len(raw) > len(text):
                    text = raw   # prefer the richer recovered page text; falls through to residue
                rec["extract_method"] = f"embed_link_tier{tier}_raw"
                rec["notes"] = (rec["notes"] + "; embedded link raw-recovered").strip("; ")
            else:
                embed_counts[3] += 1
                rec["notes"] = (rec["notes"] + "; embedded share URL not recovered").strip("; ")
                manual_links.append({"pid": pid, "source_file": fname,
                                     "url": url, "status": "embedded_unrecovered"})

        # classify the residue (full text always retained)
        if rec["format"] in IMAGE_EXTS:
            rec["content_type"] = "image_ocr"; rec["status"] = "image_ocr"
            rec["needs_review"] = True
        elif looks_like_summary(text, fname):
            rec["content_type"] = "summary"; rec["status"] = "summary"
        else:
            rec["content_type"] = "unknown"; rec["status"] = "no_markers"
            rec["needs_review"] = True
        cid = next_conv_id(pid, assign); rec["conversation_id"] = cid
        _write_text(cid, text); rec["char_len"] = len(text)
        _add_master(master_rows, rec, text)

        manifest_rows.append(rec)

    # cleanup browser
    if bstate["browser"] is not None:
        try: bstate["browser"].close()
        except Exception: pass
    if bstate["pw"] is not None:
        try: bstate["pw"].stop()
        except Exception: pass
    save_cache(cache)

    # ---- deduplicate conversations resubmitted across multiple files ----
    segmented_rows, n_dup, n_dropped, n_partial = deduplicate(
        manifest_rows, master_rows, segmented_rows)
    dedup_stats = {"dup_convos": n_dup, "dropped_pairs": n_dropped,
                   "partial_flagged": n_partial}
    log.info("Dedup: %d duplicate conversations removed (%d pairs), %d partial-capture flagged",
             n_dup, n_dropped, n_partial)

    # ---- write outputs ----
    _write_csv(OUT_DIR / "extraction_manifest.csv", manifest_rows, [
        "pid", "assignment", "source_file", "format", "size_bytes", "extract_method",
        "status", "recovery_tier", "content_type", "is_link_only", "link_url",
        "marker_style", "n_turns", "char_len", "needs_review", "conversation_id", "notes"])

    _write_csv(OUT_DIR / "conversations_master.csv", master_rows, [
        "pid", "assignment", "conversation_id", "source_file", "format", "status",
        "marker_style", "is_summary", "is_link_only", "link_url", "recovery_tier",
        "n_turns", "char_len", "needs_review", "duplicate_of", "text_path", "text_preview"])

    seg_df = pd.DataFrame(segmented_rows, columns=[
        "Student ID", "Conversation ID", "Segment Order", "Student Prompt",
        "LLM Response", "Delegation Label", "Cognitive Label", "Coded", "Source",
        "Source File"])
    seg_df.to_csv(OUT_DIR / "transcripts_segmented.csv", index=False, encoding="utf-8")

    _write_csv(OUT_DIR / "links_to_capture_manually.csv", manual_links,
               ["pid", "source_file", "url", "status"])

    write_report(ts, caps, files, manifest_rows, master_rows, segmented_rows,
                 manual_links, tier_counts, embed_counts, dedup_stats, logfile)
    log.info("DONE. files=%d  conversations=%d  turn_pairs=%d  manual_links=%d",
             len(files), len(master_rows), len(segmented_rows), len(manual_links))


_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def _clean(s: str) -> str:
    """Drop lone surrogates / undecodable artifacts so UTF-8 output never fails."""
    if not s:
        return s
    return _SURROGATE_RE.sub("", s)


def _write_text(cid, text):
    (TEXT_DIR / f"{cid}.txt").write_text(_clean(text), encoding="utf-8")


def _add_master(master_rows, rec, text):
    master_rows.append({
        "pid": rec["pid"], "assignment": rec["assignment"],
        "conversation_id": rec["conversation_id"], "source_file": rec["source_file"],
        "format": rec["format"], "status": rec["status"],
        "marker_style": rec["marker_style"],
        "is_summary": rec["content_type"] == "summary",
        "is_link_only": rec["is_link_only"], "link_url": rec["link_url"],
        "recovery_tier": rec["recovery_tier"], "n_turns": rec["n_turns"],
        "char_len": rec["char_len"], "needs_review": rec["needs_review"],
        "text_path": str((TEXT_DIR / f"{rec['conversation_id']}.txt").relative_to(REPO)),
        "text_preview": _clean(text[:PREVIEW_CHARS]).replace("\n", " ")})


def _emit_transcript(rec, pid, assign, next_conv_id, pairs, full_text, status,
                     master_rows, segmented_rows):
    cid = next_conv_id(pid, assign)
    rec["conversation_id"] = cid
    rec["content_type"] = "transcript"
    rec["status"] = rec["status"] or status
    rec["n_turns"] = len(pairs)
    rec["char_len"] = len(full_text)
    _write_text(cid, full_text)
    _add_master(master_rows, rec, full_text)
    for order, (prompt, resp) in enumerate(pairs, 1):
        if not prompt.strip():
            continue
        segmented_rows.append({
            "Student ID": pid, "Conversation ID": cid, "Segment Order": order,
            "Student Prompt": _clean(prompt), "LLM Response": _clean(resp),
            "Delegation Label": "", "Cognitive Label": "", "Coded": False,
            "Source": SOURCE_TAG, "Source File": rec["source_file"]})


def _write_csv(path, rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False, encoding="utf-8")


def write_report(ts, caps, files, manifest, master, segmented, manual, tier_counts,
                 embed_counts, dedup_stats, logfile):
    from collections import Counter
    status_counts = Counter(r["status"] for r in manifest)
    content_counts = Counter(r["content_type"] for r in manifest if r["content_type"])
    fmt_counts = Counter(r["format"] for r in manifest)
    needs_review = sum(1 for r in manifest if r["needs_review"])
    n_summary = sum(1 for r in master if r.get("is_summary"))
    n_transcript = sum(1 for r in master if not r.get("is_summary"))
    link_only = sum(1 for r in manifest if r["is_link_only"])
    students = len({r["Student ID"] for r in segmented})
    total_pairs = len(segmented)

    lines = []
    A = lines.append
    A(f"# Extraction Report — run {ts}\n")
    A(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}  ")
    A(f"Seed: {SEED}  ")
    A(f"Input dir: `{INPUT_DIR.relative_to(REPO)}`  ")
    A(f"Run log: `{logfile.relative_to(REPO)}`\n")

    A("## Capabilities (dependency probe)\n")
    for k, v in caps.items():
        A(f"- {k}: {'available' if v else 'MISSING (degraded)'}")
    A("")

    A("## Inputs\n")
    A(f"- Total input files: **{len(files)}**")
    for k, v in sorted(fmt_counts.items(), key=lambda x: -x[1]):
        A(f"  - `{k}`: {v}")
    A("")

    A("## Outputs\n")
    A(f"- Conversations (master rows): **{len(master)}**  "
      f"(transcripts {n_transcript}, summaries {n_summary})")
    A(f"- Turn pairs (segmented rows): **{total_pairs}**")
    A(f"- Distinct students with ≥1 turn pair: **{students}**")
    A(f"- Files flagged needs_review: **{needs_review}**")
    A("")

    A("## Cross-file deduplication\n")
    A(f"- Duplicate conversations removed (same student resubmitted the same conversation): "
      f"**{dedup_stats['dup_convos']}**  (**{dedup_stats['dropped_pairs']}** turn pairs dropped)")
    A(f"- Same share-id but differing content (partial captures) flagged `needs_review`: "
      f"**{dedup_stats['partial_flagged']}**")
    A("  Removed conversations are kept in the master with `status=duplicate` + `duplicate_of`.")
    A("")

    A("## Link recovery (3 tiers, per-tier success)\n")
    A(f"- Link-only files detected: **{link_only}**")
    A(f"- **Tier 1 — HTTP backend-API JSON: {tier_counts[1]} recovered**")
    A(f"- **Tier 2 — Playwright headless render: {tier_counts[2]} recovered**")
    A(f"- **Tier 3 — unrecovered → manual list: {tier_counts[3]}**")
    A(f"  (Tier1+Tier2+Tier3 = {tier_counts[1]+tier_counts[2]+tier_counts[3]}; "
      f"see `links_to_capture_manually.csv`)")
    A("")

    A("## Embedded share-URL recovery (no-marker files)\n")
    embed_total = embed_counts[2] + embed_counts[3]
    A(f"- No-marker files carrying a share URL, fetched via Playwright: **{embed_total}**")
    A(f"- **Recovered (Tier 2): {embed_counts[2]}**  ·  **Unrecovered: {embed_counts[3]}**")
    A("")

    A("## Status tally (per input file)\n")
    for k, v in status_counts.most_common():
        A(f"- `{k or '(blank)'}`: {v}")
    A("")

    A("## Content type\n")
    for k, v in content_counts.most_common():
        A(f"- `{k}`: {v}")
    A("")

    A("## Caveats\n")
    A("- Link recovery deviates from the CLAUDE.md rule (links captured during de-identification, "
      "not in-pipeline); enabled by explicit decision — **supervisor flag**.")
    if not caps.get("ocr"):
        A("- OCR unavailable (tesseract binary missing): screenshots flagged `image_only`, not recovered.")
    A("- Segmentation is heuristic on uncoded inference data; `needs_review` marks ambiguous files. "
      "Full text is always retained in `normalized_text/`.")
    (OUT_DIR / "EXTRACTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
