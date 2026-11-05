#!/usr/bin/env python3
"""
segment_needs_review.py

Recover turn pairs from the `needs_review` leftovers of authentic_extractor.py that the
deterministic marker parser could not split: (1) `no_markers` conversations (implicit
student/LLM boundaries) and (2) `image_ocr` screenshots (stitched per conversation).

Uses the Claude API to identify turn BOUNDARIES only — the model returns, for each turn,
a short verbatim ANCHOR (the first ~50 chars where that turn begins). We then slice the
ORIGINAL text at those anchors locally, so every emitted turn is a substring of the source
(no paraphrase / hallucination possible). When an anchor is not found verbatim, a fallback
position is used and the boundary can fall a few characters early. Output is validated (anchors must be locatable,
coverage threshold) and written to a SEPARATE, flagged CSV — kept apart from the
deterministic training data until reviewed.

Runs from one entrypoint; idempotent (LLM responses cached by text hash), except that the
image-group work ids use Python's hash() and differ between runs.

    python segment_needs_review.py            # dry-run on 10 items (sanity/cost check)
    python segment_needs_review.py --all       # process the full work list
    python segment_needs_review.py --all --apply   # also update manifest/master in place
    python segment_needs_review.py --all --merge   # append rows into transcripts_segmented.csv

Auth: needs a Claude credential (ANTHROPIC_API_KEY, or `ant auth login`). See the claude-api
skill; the SDK's zero-arg client resolves it from the environment.
"""
from __future__ import annotations
import re
import sys
import json
import time
import hashlib
import logging
import argparse
import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd

import authentic_extractor as AX
from authentic_extractor import turns_to_pairs, _clean, TEXT_DIR, OUT_DIR, SOURCE_TAG

MODEL = "claude-sonnet-5"          # segmentation is mechanical; user-approved in the plan
SUMMARY_MIN_CHARS = 2500           # long 'summary' files may be markerless transcripts mislabeled
COVERAGE_MIN = 0.60
BLANK_RATE_MIN = 0.30           # share-link captures whose pairs are mostly empty on the LLM side
CONFIDENCE_MIN = 0.70              # fraction of returned anchors that were locatable
CACHE_FILE = OUT_DIR / ".cache" / "llm_segmentation_cache.json"
LLM_CSV = OUT_DIR / "transcripts_segmented_llm.csv"
REPORT = OUT_DIR / "LLM_SEGMENTATION_REPORT.md"
MANIFEST = OUT_DIR / "extraction_manifest.csv"
MASTER = OUT_DIR / "conversations_master.csv"

log = logging.getLogger("segment_needs_review")

SYSTEM_PROMPT = (
    "You segment ONE student–LLM programming conversation into turns.\n"
    "You are given the raw text of the conversation (it may include an export-metadata "
    "header such as Company/Model/Date/Public URL/Prompt History — ignore that header).\n"
    "Identify the boundaries between the student's messages and the assistant/LLM's replies.\n\n"
    "Return ONLY a JSON array (no prose, no code fence). Each element is:\n"
    '  {\"role\": \"student\" | \"llm\", \"anchor\": \"<verbatim first ~50 characters where this turn begins>\"}\n\n'
    "Rules:\n"
    "- The anchor MUST be copied character-for-character from the text (including punctuation, "
    "code, casing) so it can be located exactly. Do not paraphrase or normalise it.\n"
    "- List turns in document order. The first turn is usually the student.\n"
    "- One element per turn. Use 'student' for the human, 'llm' for the AI assistant.\n"
    "- If you cannot identify a genuine student–LLM exchange, return []."
)


# ---------------------------------------------------------------------------
# LLM boundary detection (isolated so it can be mocked in tests)
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def llm_anchors(text: str, cache: dict) -> list:
    """Return list of {'role','anchor'} from the model; cached by text hash."""
    key = "seg::" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    if key in cache:
        return cache[key]
    client = _get_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    anchors = _parse_anchor_json(raw)
    cache[key] = anchors
    _save_cache(cache)
    return anchors


def _parse_anchor_json(raw: str) -> list:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", raw).strip()
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and d.get("role") in ("student", "llm") and d.get("anchor"):
            out.append({"role": d["role"], "anchor": str(d["anchor"])})
    return out


# ---------------------------------------------------------------------------
# Slice original text at anchors -> verbatim turns
# ---------------------------------------------------------------------------
def turns_from_anchors(text: str, anchors: list):
    """
    Locate each anchor in `text` in document order and cut the text into turns.
    Returns (turns, confidence, coverage).
    turns: [{'role','text'}] with verbatim text sliced from the source.
    """
    bounds = []          # (offset, role)
    cursor = 0
    found = 0
    for a in anchors:
        anc = a["anchor"].strip()
        if not anc:
            continue
        idx = text.find(anc, cursor)
        if idx == -1:                       # try a shorter, whitespace-tolerant probe
            probe = re.sub(r"\s+", " ", anc[:25]).strip()
            idx = _fuzzy_find(text, probe, cursor)
        if idx == -1:
            continue
        bounds.append((idx, a["role"]))
        cursor = idx + 1
        found += 1
    confidence = found / len(anchors) if anchors else 0.0
    if not bounds:
        return [], confidence, 0.0
    # Collapse consecutive same-role boundaries BEFORE slicing, so each turn is a
    # single contiguous slice of the source -> verbatim by construction.
    collapsed = []
    for off, role in bounds:
        if collapsed and collapsed[-1][1] == role:
            continue
        collapsed.append((off, role))
    turns = []
    for i, (off, role) in enumerate(collapsed):
        end = collapsed[i + 1][0] if i + 1 < len(collapsed) else len(text)
        chunk = text[off:end].strip()
        if chunk:
            turns.append({"role": role, "text": chunk})
    covered = sum(len(t["text"]) for t in turns)
    coverage = covered / max(len(text.strip()), 1)
    return turns, confidence, coverage


def _fuzzy_find(text, probe, cursor):
    if not probe:
        return -1
    norm = re.sub(r"\s+", " ", text[cursor:])
    j = norm.find(probe)
    if j == -1:
        return -1
    # map normalised offset back approximately to raw offset
    return cursor + j


# ---------------------------------------------------------------------------
# Work list
# ---------------------------------------------------------------------------
def _natural_key(fn):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", fn)]


def build_work_items(manifest: pd.DataFrame):
    """Return list of dicts: {kind, work_id, pid, assignment, source_files[], text, member_cids[]}."""
    items = []
    # (1) no_markers text conversations (include already-llm_segmented for idempotent re-runs)
    tset = manifest[manifest["status"].isin(["no_markers", "llm_segmented"])]
    for _, r in tset.iterrows():
        cid = r["conversation_id"]
        tp = TEXT_DIR / f"{cid}.txt"
        if not tp.exists():
            continue
        items.append({"kind": "text", "work_id": cid, "pid": r["pid"],
                      "assignment": r["assignment"], "source_files": [r["source_file"]],
                      "text": tp.read_text(encoding="utf-8"), "member_cids": [cid]})
    # (1b) long 'summary' files — candidates for markerless transcripts mislabeled as summaries.
    # The LLM returns [] for genuine summaries (no student/LLM exchange), so this is low-risk.
    sset = manifest[(manifest["status"] == "summary") &
                    (manifest["char_len"].fillna(0) >= SUMMARY_MIN_CHARS)]
    for _, r in sset.iterrows():
        cid = r["conversation_id"]
        tp = TEXT_DIR / f"{cid}.txt"
        if not tp.exists():
            continue
        items.append({"kind": "summary_probe", "work_id": cid, "pid": r["pid"],
                      "assignment": r["assignment"], "source_files": [r["source_file"]],
                      "text": tp.read_text(encoding="utf-8"), "member_cids": [cid]})

    # (1c) residue: conversations of students who STILL have no turn pairs, with pasted text the
    # deterministic markers didn't catch (odd/positional/conflicting conventions). Targeted + cheap.
    try:
        sp_seg = pd.read_csv(OUT_DIR / "transcripts_segmented.csv")
        sp_llm = pd.read_csv(LLM_CSV) if LLM_CSV.exists() else pd.DataFrame(columns=["Student ID"])
        with_pairs = set(sp_seg["Student ID"]) | set(sp_llm["Student ID"])
    except Exception:
        with_pairs = set()
    seen = {i["work_id"] for i in items}
    rset = manifest[(~manifest["pid"].isin(with_pairs)) &
                    (manifest["status"].isin(["summary", "no_markers",
                                              "link_recovered_embedded", "link_recovered"])) &
                    (manifest["char_len"].fillna(0) >= 200)]
    for _, r in rset.iterrows():
        cid = r["conversation_id"]
        if cid in seen:
            continue
        tp = TEXT_DIR / f"{cid}.txt"
        if not tp.exists():
            continue
        txt = tp.read_text(encoding="utf-8")
        if len(txt.strip()) < 200:
            continue
        items.append({"kind": "residue", "work_id": cid, "pid": r["pid"],
                      "assignment": r["assignment"], "source_files": [r["source_file"]],
                      "text": txt, "member_cids": [cid]})

    # (1d) mis-segmented share-link captures: the page text was recovered but had no speaker
    # markers, so the deterministic parser emitted pairs whose LLM half is empty and whose
    # student half absorbs the assistant text. These pass block (1c) because the student DOES
    # have pairs -- they are just wrong. Select on the blank-LLM-response rate instead.
    try:
        det = pd.read_csv(OUT_DIR / "transcripts_segmented.csv")
        det["_blank"] = det["LLM Response"].fillna("").astype(str).str.strip() == ""
        rate = det.groupby("Conversation ID")["_blank"].mean()
        bad = set(rate[rate >= BLANK_RATE_MIN].index)
    except Exception:
        bad = set()
    lset = manifest[manifest["status"].isin(["link_recovered_embedded", "link_recovered"]) &
                    manifest["conversation_id"].isin(bad)]
    for _, r in lset.iterrows():
        cid = r["conversation_id"]
        if cid in seen:
            continue
        tp = TEXT_DIR / f"{cid}.txt"
        if not tp.exists():
            continue
        txt = tp.read_text(encoding="utf-8")
        if len(txt.strip()) < 200:
            continue
        seen.add(cid)
        items.append({"kind": "link_missegmented", "work_id": cid, "pid": r["pid"],
                      "assignment": r["assignment"], "source_files": [r["source_file"]],
                      "text": txt, "member_cids": [cid]})

    # (2) image_ocr -> stitch per (pid, base name) group
    iset = manifest[manifest["status"].isin(["image_ocr", "llm_segmented_ocr"])].copy()
    def base(fn):
        m = re.match(r"(P\d{3})_(P[12])_(.+?)[\s_\-]*\(?\d*\)?\.(png|jpg|jpeg)$", fn, re.I)
        return m.group(3) if m else fn
    iset["grp"] = iset.apply(lambda r: (r["pid"], r["assignment"], base(r["source_file"])), axis=1)
    for (pid, assign, _b), grp in iset.groupby("grp"):
        grp = grp.iloc[grp["source_file"].map(_natural_key).argsort()]
        texts, cids, files = [], [], []
        for _, r in grp.iterrows():
            tp = TEXT_DIR / f"{r['conversation_id']}.txt"
            if tp.exists():
                texts.append(tp.read_text(encoding="utf-8"))
                cids.append(r["conversation_id"]); files.append(r["source_file"])
        if not texts:
            continue
        # hash() of a str differs between processes, so these ids cannot be regenerated; the corpus
        # keeps the ones from the run that built it.
        wid = f"{pid}_{assign}_IMG{abs(hash(_b)) % 10000:04d}"
        items.append({"kind": "image", "work_id": wid, "pid": pid, "assignment": assign,
                      "source_files": files, "text": "\n\n".join(texts), "member_cids": cids})
    return items


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(items, cache, seg_fn):
    """seg_fn(text, cache) -> anchors. Returns (llm_rows, results)."""
    llm_rows, results = [], []
    for it in items:
        text = it["text"]
        try:
            anchors = seg_fn(text, cache)
        except Exception as e:
            log.warning("LLM call failed for %s: %s", it["work_id"], e)
            results.append({**_meta(it), "accepted": False, "reason": f"llm_error:{e}",
                            "n_turns": 0, "confidence": 0, "coverage": 0}); continue
        turns, conf, cov = turns_from_anchors(text, anchors)
        pairs = turns_to_pairs(turns)
        # keep only genuine exchanges: prompt AND a substantive LLM response
        pairs = [(p, r) for (p, r) in pairs if p.strip() and len(r.strip()) >= 20]
        reason = "ok"
        if not pairs:
            reason = "no_student_llm_pair"
        elif cov < COVERAGE_MIN:
            reason = "low_coverage"
        elif conf < CONFIDENCE_MIN:
            reason = "low_confidence"
        accepted = (reason == "ok")
        method = "llm+ocr" if it["kind"] == "image" else "llm"
        results.append({**_meta(it), "accepted": accepted,
                        "reason": reason,
                        "n_turns": len(pairs), "confidence": round(conf, 3),
                        "coverage": round(cov, 3)})
        if not accepted:
            continue
        # write stitched text for image groups so it has a normalized_text file
        if it["kind"] == "image":
            (TEXT_DIR / f"{it['work_id']}.txt").write_text(_clean(text), encoding="utf-8")
        for order, (p, resp) in enumerate(pairs, 1):
            if not p.strip():
                continue
            llm_rows.append({
                "Student ID": it["pid"], "Conversation ID": it["work_id"],
                "Segment Order": order, "Student Prompt": _clean(p),
                "LLM Response": _clean(resp),
                "Student delegation": "", "Student cognition": "",
                "AI delegation": "", "AI response type": "",
                "Coded": False, "Source": SOURCE_TAG,
                "Source File": "; ".join(it["source_files"]),
                "Segmentation Method": method, "Coverage": round(cov, 3)})
    return llm_rows, results


def _meta(it):
    return {"work_id": it["work_id"], "kind": it["kind"], "pid": it["pid"],
            "source_files": "; ".join(it["source_files"]),
            "member_cids": "; ".join(it["member_cids"])}


def apply_to_manifest(results):
    """Mark successfully segmented source rows in manifest/master (opt-in --apply)."""
    man = pd.read_csv(MANIFEST)
    mst = pd.read_csv(MASTER)
    accepted = {r["work_id"]: r for r in results if r["accepted"]}
    member_to_work = {}
    for r in results:
        if r["accepted"]:
            for cid in r["member_cids"].split("; "):
                member_to_work[cid] = r["work_id"]
    def mark(df):
        for i, row in df.iterrows():
            cid = row.get("conversation_id")
            if cid in member_to_work:
                w = member_to_work[cid]
                st = "llm_segmented_ocr" if accepted[w]["kind"] == "image" else "llm_segmented"
                df.at[i, "status"] = st
                df.at[i, "needs_review"] = False
                if "is_summary" in df.columns:      # reclassified out of 'summary'
                    df.at[i, "is_summary"] = False
                if "notes" in df.columns:
                    df.at[i, "notes"] = (str(row.get("notes", "") or "") +
                                         f"; llm_segmented -> {w} (cov {accepted[w]['coverage']})").strip("; ")
        return df
    man = mark(man); mst = mark(mst)

    # stitched IMAGE groups get a NEW conversation_id (work_id) that isn't an existing file row —
    # register it in manifest/master so it isn't an orphan.
    existing = set(man["conversation_id"].dropna())
    new_man, new_mst = [], []
    from authentic_extractor import PREVIEW_CHARS
    for w, r in accepted.items():
        if r["kind"] != "image" or w in existing:
            continue
        tp = TEXT_DIR / f"{w}.txt"
        text = tp.read_text(encoding="utf-8") if tp.exists() else ""
        base = {"pid": r["pid"], "assignment": w.split("_")[1] if "_" in w else "P2",
                "conversation_id": w, "source_file": r["source_files"], "format": ".png",
                "status": "llm_segmented_ocr", "marker_style": "", "is_link_only": False,
                "link_url": "", "recovery_tier": "", "n_turns": r["n_turns"],
                "char_len": len(text), "needs_review": False,
                "notes": "stitched OCR group (segment_needs_review)"}
        new_man.append({**base, "size_bytes": len(text), "extract_method": "ocr+llm",
                        "content_type": "transcript"})
        new_mst.append({**base, "is_summary": False, "duplicate_of": "",
                        "text_path": f"data/processed/normalized_text/{w}.txt",
                        "text_preview": text[:PREVIEW_CHARS].replace("\n", " ")})
    if new_man:
        man = pd.concat([man, pd.DataFrame(new_man)], ignore_index=True)
        mst = pd.concat([mst, pd.DataFrame(new_mst)], ignore_index=True)
    man.to_csv(MANIFEST, index=False, encoding="utf-8")
    mst.to_csv(MASTER, index=False, encoding="utf-8")


def write_report(results, dry):
    acc = [r for r in results if r["accepted"]]
    txt = [r for r in results if r["kind"] == "text"]
    img = [r for r in results if r["kind"] == "image"]
    lines = [f"# LLM-assisted segmentation report{' (DRY RUN)' if dry else ''}\n",
             f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}  ",
             f"Model: {MODEL}  |  coverage>={COVERAGE_MIN}, confidence>={CONFIDENCE_MIN}\n",
             f"- Work items: **{len(results)}** ({len(txt)} no_markers texts, {len(img)} stitched image groups)",
             f"- Accepted (segmented): **{len(acc)}**  |  low-confidence (still needs_review): **{len(results)-len(acc)}**",
             f"- Turn pairs produced: **{sum(r['n_turns'] for r in acc)}**",
             f"- Mean coverage (accepted): {round(sum(r['coverage'] for r in acc)/max(len(acc),1),3)}",
             "", "## Low-confidence (left as needs_review)"]
    for r in results:
        if not r["accepted"]:
            lines.append(f"- `{r['work_id']}` ({r['kind']}): {r['reason']} "
                         f"conf={r['confidence']} cov={r['coverage']}")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="process full work list (default: 10-item dry run)")
    ap.add_argument("--apply", action="store_true", help="update manifest/master statuses in place")
    ap.add_argument("--merge", action="store_true", help="append rows into transcripts_segmented.csv")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--kind", default=None,
                    help="only process this work-item kind "
                         "(text | summary_probe | residue | link_missegmented | image)")
    args = ap.parse_args()

    manifest = pd.read_csv(MANIFEST)
    items = build_work_items(manifest)
    if args.kind:
        items = [i for i in items if i["kind"] == args.kind]
        log.info("Filtered to kind=%s: %d items", args.kind, len(items))
    log.info("Work list: %d items (%d text, %d image groups)",
             len(items), sum(i["kind"] == "text" for i in items),
             sum(i["kind"] == "image" for i in items))
    if not args.all:
        items = items[: args.limit]
        log.info("DRY RUN on %d items (use --all for everything)", len(items))

    cache = _load_cache()
    llm_rows, results = process(items, cache, llm_anchors)

    cols = ["Student ID", "Conversation ID", "Segment Order", "Student Prompt",
            "LLM Response", "Student delegation", "Student cognition",
            "AI delegation", "AI response type", "Coded", "Source",
            "Source File", "Segmentation Method", "Coverage"]
    pd.DataFrame(llm_rows, columns=cols).to_csv(LLM_CSV, index=False, encoding="utf-8")
    pd.DataFrame(results).to_csv(OUT_DIR / "llm_segmentation_manifest.csv",
                                 index=False, encoding="utf-8")
    write_report(results, dry=not args.all)

    if args.apply and args.all:
        apply_to_manifest(results)
        log.info("Applied statuses to manifest/master.")
    if args.merge and args.all:
        base = pd.read_csv(OUT_DIR / "transcripts_segmented.csv")
        base["Segmentation Method"] = "deterministic"; base["Coverage"] = ""
        pd.concat([base, pd.DataFrame(llm_rows, columns=cols)], ignore_index=True) \
          .to_csv(OUT_DIR / "transcripts_segmented.csv", index=False, encoding="utf-8")
        log.info("Merged %d LLM rows into transcripts_segmented.csv", len(llm_rows))

    acc = sum(r["accepted"] for r in results)
    log.info("DONE. items=%d accepted=%d pairs=%d  -> %s",
             len(results), acc, len(llm_rows), LLM_CSV.name)


if __name__ == "__main__":
    main()
