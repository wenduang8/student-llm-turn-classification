#!/usr/bin/env python3
"""
recover_manual_rules.py

Per-participant deterministic recovery for a few students whose transcripts use a fixed,
mechanical convention the general parser can't safely generalise (identified by manual review),
plus P450's Claude share pages captured as Chrome MHTML snapshots (Cloudflare blocks server-side
fetching, so the user saved the rendered pages). Every emitted turn is verbatim to its source.
Output kept in its own CSV (transcripts_segmented_manual.csv); statuses set to `manual_segmented`.

Run LAST (after authentic_extractor.py, segment_needs_review.py --apply). Idempotent.

Rules:
- P329: `<` ... [line 8 = student] [lines 9.. = LLM] ... `>`  (bracket wraps the whole exchange).
- P393: after "Prompt History:", the first `<`..`>` block is the student turn; text after `>` is the LLM.
- P450: parse saved Claude share-page MHTML; human turns carry data-testid="user-message".
"""
import re
import email
from pathlib import Path
import pandas as pd

from authentic_extractor import _clean, SOURCE_TAG, TEXT_DIR, PREVIEW_CHARS

NL = chr(10)
REPO = Path(__file__).resolve().parent.parent   # repo root (this file lives in extraction/)
SRC = REPO / "data" / "Raw data from COMP2017" / "AI transcripts (de-identified)"
OUT = REPO / "data" / "processed"
MANIFEST = OUT / "extraction_manifest.csv"
MASTER = OUT / "conversations_master.csv"
MANUAL_CSV = OUT / "transcripts_segmented_manual.csv"
REVIEW1 = REPO / "data" / "review1"   # scanned for P###* folders holding saved .mhtml page snapshots
#   (lives under data/, which is gitignored - the folder is a data source, not scratch)
MANUAL_MARK = REVIEW1 / "manual_mark.txt"   # user's hand-marked splits (see format below)

# manual_mark.txt format:
#   Pnnn            -> student section header
#   student input:[ ... ]  /  LLM:[ ... ]   -> a turn (brackets may nest in content)
#   N.  (standalone, or inline "N.student input:[") -> starts a separate file/conversation N;
#   turns without a number continue the current file.
_MM_ANCHOR = re.compile(
    r'(?m)^(?P<pid>P\d{3})[ \t]*$'
    r'|^[ \t]*(?P<numonly>\d+)\.[ \t]*$'
    r'|^[ \t]*(?:(?P<num>\d+)\.[ \t]*)?(?P<role>student input|LLM)[ \t]*:\[')


def rule_p329(text):
    lines = text.replace("\r\n", NL).split(NL)
    if len(lines) < 9:
        return None
    student = lines[7].strip()
    llm = re.sub(r'\s*>\s*$', '', NL.join(lines[8:]).strip()).strip()
    if len(student) < 5 or len(llm) < 20:
        return None
    return [(student, llm)]


def rule_p393(text):
    t = text.replace("\r\n", NL)
    i = t.find("Prompt History:")
    seg = t[i + len("Prompt History:"):] if i >= 0 else t
    lt = seg.find("<"); gt = seg.find(">", lt + 1)
    if lt < 0 or gt < 0:
        return None
    student = seg[lt + 1:gt].strip(); llm = seg[gt + 1:].strip()
    if len(student) < 5 or len(llm) < 20:
        return None
    return [(student, llm)]


def _mhtml_html(path):
    for part in email.message_from_bytes(Path(path).read_bytes()).walk():
        if part.get_content_type() == "text/html":
            return part.get_payload(decode=True).decode("utf-8", "replace")
    return ""


def _clean_claude(t):
    t = re.sub('[-]', '', t)     # icon-font private-use glyphs (Copy/Retry icons)
    t = re.sub(r'^\s*Claude responded:\s*', '', t)
    t = re.sub(r'\n\s*[A-Z][a-z]{2,8} \d{1,2}, \d{4}\s*$', '', t)   # trailing date stamp
    return t.strip()


def pairs_from_mhtml(path):
    """Extract (student, llm) pairs from a saved ChatGPT or Claude share-page MHTML snapshot."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(_mhtml_html(path), "html.parser")
    turns = []
    cg = soup.select('[data-message-author-role]')          # ChatGPT
    if cg:
        src = "chatgpt"
        for n in cg:
            role = n.get("data-message-author-role")
            if role not in ("user", "assistant"):
                continue
            rr = "student" if role == "user" else "llm"
            txt = _clean_claude(n.get_text(NL, strip=True))
            if not txt:
                continue
            if turns and turns[-1][0] == rr:
                turns[-1] = (rr, turns[-1][1] + NL + txt)
            else:
                turns.append((rr, txt))
    else:                                                   # Claude
        src = "claude"
        for b in soup.select('[data-test-render-count]'):
            um = b.select_one('[data-testid="user-message"]')
            rr = "student" if um else "llm"
            txt = _clean_claude((um or b).get_text(NL, strip=True))
            if not txt:
                continue
            if turns and turns[-1][0] == rr:
                turns[-1] = (rr, turns[-1][1] + NL + txt)
            else:
                turns.append((rr, txt))
    pairs, i = [], 0
    while i < len(turns):
        if turns[i][0] == "student":
            r = turns[i + 1][1] if i + 1 < len(turns) and turns[i + 1][0] == "llm" else ""
            pairs.append((turns[i][1], r)); i += 2 if r else 1
        else:
            i += 1
    return [(p, r) for p, r in pairs if len(p.strip()) >= 3 and len(r.strip()) >= 20], src


def process_mhtml_folders(rows):
    """Scan review1/ for P###* folders of saved .mhtml pages; extract each as a new conversation."""
    new_man, new_mst = [], []
    if not REVIEW1.exists():
        return [], []
    for folder in sorted(REVIEW1.iterdir()):
        if not folder.is_dir():
            continue
        m = re.match(r'(P\d{3})', folder.name)
        files = sorted(folder.glob("*.mhtml"))
        if not m or not files:
            continue
        pid = m.group(1)
        for i, f in enumerate(files):
            pairs, src = pairs_from_mhtml(f)
            if not pairs:
                continue
            cid = "%s_P2_CL%d" % (pid, i)
            mm = re.search(r'Snapshot-Content-Location:\s*(\S+)', f.read_text("utf-8", "replace")[:500])
            url = mm.group(1) if mm else ""
            full = _clean((NL + NL).join("Student: " + p + NL + NL + "LLM: " + r for p, r in pairs))
            (TEXT_DIR / (cid + ".txt")).write_text(full, encoding="utf-8")
            for order, (p, r) in enumerate(pairs, 1):
                rows.append({"Student ID": pid, "Conversation ID": cid, "Segment Order": order,
                             "Student Prompt": _clean(p), "LLM Response": _clean(r),
                             "Delegation Label": "", "Cognitive Label": "", "Coded": False,
                             "Source": SOURCE_TAG, "Source File": f.name,
                             "Segmentation Method": "mhtml_" + src, "Coverage": 1.0})
            base = {"pid": pid, "assignment": "P2", "conversation_id": cid,
                    "source_file": f.name, "format": ".mhtml", "status": "manual_segmented",
                    "marker_style": "", "is_link_only": False, "link_url": url,
                    "recovery_tier": "", "n_turns": len(pairs), "char_len": len(full),
                    "needs_review": False, "notes": "%s MHTML snapshot" % src}
            new_man.append({**base, "size_bytes": f.stat().st_size,
                            "extract_method": "mhtml_" + src, "content_type": "transcript"})
            new_mst.append({**base, "is_summary": False, "duplicate_of": "",
                            "text_path": str((TEXT_DIR / (cid + ".txt")).relative_to(REPO)),
                            "text_preview": _clean(full[:PREVIEW_CHARS]).replace(NL, " ")})
    return new_man, new_mst


def process_manual_mark(rows):
    """Parse review1/manual_mark.txt (user's hand-marked student/LLM splits) -> conversations."""
    if not MANUAL_MARK.exists():
        return [], []
    from collections import defaultdict, OrderedDict
    text = MANUAL_MARK.read_text(encoding="utf-8", errors="replace").replace("\r\n", NL)
    ms = list(_MM_ANCHOR.finditer(text))
    convos = OrderedDict()      # (pid, file) -> [(role, content), ...]
    pid, cur = None, 1
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        if m.group("pid"):
            pid, cur = m.group("pid"), 1; continue
        if m.group("numonly"):
            cur = int(m.group("numonly")); continue
        if m.group("num"):
            cur = int(m.group("num"))
        role = "student" if m.group("role") == "student input" else "llm"
        content = re.sub(r'\]\s*$', '', text[m.end():end]).strip()
        if pid and content:
            convos.setdefault((pid, cur), []).append((role, content))

    new_man, new_mst = [], []
    for (pid, fn), turns in convos.items():
        pairs, i = [], 0
        while i < len(turns):
            if turns[i][0] == "student":
                r = turns[i + 1][1] if i + 1 < len(turns) and turns[i + 1][0] == "llm" else ""
                if len(r.strip()) >= 15:
                    pairs.append((turns[i][1], r))
                i += 2 if r else 1
            else:
                i += 1
        if not pairs:
            continue
        cid = "%s_P2_MM%d" % (pid, fn)
        full = _clean((NL + NL).join("Student: " + p + NL + NL + "LLM: " + r for p, r in pairs))
        (TEXT_DIR / (cid + ".txt")).write_text(full, encoding="utf-8")
        for order, (p, r) in enumerate(pairs, 1):
            rows.append({"Student ID": pid, "Conversation ID": cid, "Segment Order": order,
                         "Student Prompt": _clean(p), "LLM Response": _clean(r),
                         "Delegation Label": "", "Cognitive Label": "", "Coded": False,
                         "Source": SOURCE_TAG, "Source File": "manual_mark.txt",
                         "Segmentation Method": "manual_mark", "Coverage": 1.0})
        base = {"pid": pid, "assignment": "P2", "conversation_id": cid,
                "source_file": "manual_mark.txt", "format": ".txt", "status": "manual_segmented",
                "marker_style": "", "is_link_only": False, "link_url": "",
                "recovery_tier": "", "n_turns": len(pairs), "char_len": len(full),
                "needs_review": False, "notes": "hand-marked split (manual_mark.txt)"}
        new_man.append({**base, "size_bytes": len(full), "extract_method": "manual_mark",
                        "content_type": "transcript"})
        new_mst.append({**base, "is_summary": False, "duplicate_of": "",
                        "text_path": str((TEXT_DIR / (cid + ".txt")).relative_to(REPO)),
                        "text_preview": _clean(full[:PREVIEW_CHARS]).replace(NL, " ")})
    return new_man, new_mst


def rule_p224(text):
    """`### Ref NN` blocks with inline `[User Prompt]...[AI Response]...` markers (verbatim slices)."""
    out = []
    for m in re.finditer(
            r'\[User Prompt\](.*?)\[AI Response\](.*?)'
            r'(?=\*\*Use case|\*\*Company|\n#{1,3}\s|\[User Prompt\]|$)', text, re.S):
        s = m.group(1).strip().lstrip("<").strip()
        l = m.group(2).strip().rstrip(">").strip()
        if len(s) >= 5 and len(l) >= 15:
            out.append((s, l))
    return out or None


RULES = {"P329": rule_p329, "P393": rule_p393, "P224": rule_p224}
COLS = ["Student ID", "Conversation ID", "Segment Order", "Student Prompt", "LLM Response",
        "Delegation Label", "Cognitive Label", "Coded", "Source", "Source File",
        "Segmentation Method", "Coverage"]


def main():
    man = pd.read_csv(MANIFEST)
    mst = pd.read_csv(MASTER)
    cid_of = {(r["pid"], r["source_file"]): r["conversation_id"] for _, r in man.iterrows()}

    rows = []
    marked = []
    for pid, rule in RULES.items():
        for f in sorted(SRC.glob(pid + "_*")):
            if f.suffix.lower() not in (".txt", ".md", ""):
                continue
            src = f.read_text(encoding="utf-8", errors="replace")
            pairs = rule(src)
            if not pairs:
                continue
            cid = cid_of.get((pid, f.name))
            if cid is None:
                continue
            pairs = [(p, r) for (p, r) in pairs if p in src and r in src]   # verbatim guard
            if not pairs:
                continue
            for order, (p, r) in enumerate(pairs, 1):
                rows.append({"Student ID": pid, "Conversation ID": cid, "Segment Order": order,
                             "Student Prompt": _clean(p), "LLM Response": _clean(r),
                             "Delegation Label": "", "Cognitive Label": "", "Coded": False,
                             "Source": SOURCE_TAG, "Source File": f.name,
                             "Segmentation Method": "rule_" + pid, "Coverage": 1.0})
            marked.append((cid, len(pairs)))

    # Saved MHTML page snapshots (ChatGPT/Claude) in review1/P###* folders -> new conversations
    new_man, new_mst = process_mhtml_folders(rows)
    # Hand-marked splits in review1/manual_mark.txt -> new conversations
    mm_man, mm_mst = process_manual_mark(rows)
    new_man += mm_man; new_mst += mm_mst

    pd.DataFrame(rows, columns=COLS).to_csv(MANUAL_CSV, index=False, encoding="utf-8")

    # update statuses for rule-recovered files
    nturn = dict(marked)
    cids = set(nturn)
    for df in (man, mst):
        for i, row in df.iterrows():
            cid = row.get("conversation_id")
            if cid in cids:
                df.at[i, "status"] = "manual_segmented"
                if "needs_review" in df.columns:
                    df.at[i, "needs_review"] = False
                if "is_summary" in df.columns:
                    df.at[i, "is_summary"] = False
                if "n_turns" in df.columns:
                    df.at[i, "n_turns"] = nturn[cid]

    # append P450 new rows (skip any already present -> idempotent)
    if new_man:
        exist = set(man["conversation_id"])
        man = pd.concat([man, pd.DataFrame([r for r in new_man if r["conversation_id"] not in exist])],
                        ignore_index=True)
        existm = set(mst["conversation_id"])
        mst = pd.concat([mst, pd.DataFrame([r for r in new_mst if r["conversation_id"] not in existm])],
                        ignore_index=True)

    man.to_csv(MANIFEST, index=False, encoding="utf-8")
    mst.to_csv(MASTER, index=False, encoding="utf-8")

    npair = len(rows)
    students = len({r["Student ID"] for r in rows})
    print("Manual recovery: %d pairs across %d students (MHTML snapshots: %d convos)"
          % (npair, students, len(new_man)))
    print("  -> " + MANUAL_CSV.name + "; statuses set to manual_segmented")


if __name__ == "__main__":
    main()
