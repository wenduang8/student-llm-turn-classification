#!/usr/bin/env python3
"""
review_report.py

Final reporting step. Splits the remaining `needs_review` files into two categories and
locks the ones that are intentional bookkeeping rather than work to do:

- flagged_for_provenance  (LOCKED)  — duplicate conversations + partial/overlapping captures.
    These already have data (or are intentional exclusions); they are NOT processed further.
- genuinely_unrecovered   (ACTIONABLE) — unmarked / low-quality residue the parsers and the
    LLM pass could not segment (no_markers, image_ocr, empty, low-coverage, dead links).

Adds `review_category` + `locked` columns to extraction_manifest.csv and conversations_master.csv,
and writes data/processed/NEEDS_REVIEW_REPORT.md. Re-runnable; run it LAST (after
authentic_extractor.py and segment_needs_review.py --apply).
"""
from pathlib import Path
import pandas as pd

P = Path(__file__).resolve().parent.parent / "data" / "processed"   # repo root / data (this file lives in extraction/)
MANIFEST = P / "extraction_manifest.csv"
MASTER = P / "conversations_master.csv"
REPORT = P / "NEEDS_REVIEW_REPORT.md"

PROVENANCE_NOTE_MARKERS = ("partial capture", "differing content", "overlaps another")

# One-line explanation of each genuinely_unrecovered status (for the report).
STATUS_EXPLAIN = {
    "no_markers":
        "Genuine conversations with no detectable turn delimiters — the boundary between the "
        "student's message and the LLM reply is implicit, or only one side is marked (e.g. a "
        "student code dump with no reply). Both the deterministic parser and the LLM pass could "
        "not confidently split them. Full pasted text is retained in normalized_text/.",
    "image_ocr":
        "Screenshots of conversations. OCR captured the text, but screenshots don't preserve turn "
        "markers, and the LLM pass on the stitched OCR text couldn't confidently segment them "
        "(OCR noise / UI chrome / no clear student-vs-LLM structure). OCR text is retained.",
    "empty":
        "Files that yielded no usable text after extraction — a blank file, an extraction failure, "
        "or a container with no transcript content. Nothing to segment.",
    "transcript":
        "Files that DID segment into turn pairs but were flagged for quality (non-alternating "
        "markers or low extraction coverage). The pairs are already in transcripts_segmented.csv; "
        "this is a segmentation-quality spot-check, not missing data.",
    "summary":
        "Content-classified as a conversation-level summary (a description, not a turn-by-turn "
        "transcript) and flagged low-coverage/uncertain. Metadata — not student–LLM training turns.",
    "link_unrecovered":
        "A link-only file whose external share page could not be fetched (dead or blocked). The "
        "actual conversation lives on the external page and needs manual capture upstream.",
    "link_recovered":
        "A share link that was recovered but the reconstructed text was flagged low-coverage — "
        "review the recovered transcript before use.",
    "link_recovered_embedded":
        "An embedded share URL recovered from a no-marker file, flagged low-coverage — "
        "review the recovered transcript before use.",
}


def closed_files():
    """Source-file names the user triaged as unrecoverable (manual_split_resolutions.csv)."""
    from pathlib import Path as _P
    p = P / "manual_split_resolutions.csv"
    if not p.exists():
        return set()
    r = pd.read_csv(p)
    closed = r[r["resolution_category"].isin(["not_useful", "no_llm_response", "too_messy", "chinese"])]
    return {_P(str(f)).name for f in closed["file"].dropna()}


def categorize(status, needs_review, notes, source_file, closed):
    if not bool(needs_review):
        return "", False
    notes = str(notes or "")
    if status == "duplicate" or any(m in notes for m in PROVENANCE_NOTE_MARKERS):
        return "flagged_for_provenance", True          # locked
    if source_file in closed:
        return "reviewed_closed", True                  # human-triaged as unrecoverable -> locked
    return "genuinely_unrecovered", False               # actionable


def add_columns(df, closed):
    cats, locks = [], []
    for _, r in df.iterrows():
        c, lk = categorize(r.get("status", ""), r.get("needs_review", False),
                           r.get("notes", ""), r.get("source_file", ""), closed)
        cats.append(c); locks.append(lk)
    df["review_category"] = cats
    df["locked"] = locks
    return df


def main():
    closed = closed_files()
    man = add_columns(pd.read_csv(MANIFEST), closed)
    man.to_csv(MANIFEST, index=False, encoding="utf-8")

    if MASTER.exists():
        mst = pd.read_csv(MASTER)
        if "notes" not in mst.columns:
            mst["notes"] = ""
        mst = add_columns(mst, closed)
        mst.to_csv(MASTER, index=False, encoding="utf-8")

    nr = man[man["needs_review"] == True]
    prov = nr[nr["review_category"] == "flagged_for_provenance"]
    reviewed = nr[nr["review_category"] == "reviewed_closed"]
    genu = nr[nr["review_category"] == "genuinely_unrecovered"]

    dup = int((man["status"] == "duplicate").sum())
    partial = len(prov) - dup

    lines = []
    A = lines.append
    A("# Needs-review report — split & lock\n")
    A(f"Total `needs_review` files: **{len(nr)}**\n")
    A("## LOCKED — flagged_for_provenance (not processed further)\n")
    A(f"Count: **{len(prov)}**  — intentional bookkeeping, already handled.")
    A(f"- Duplicate conversations (resubmitted; excluded from training, kept with `duplicate_of`): **{dup}**")
    A(f"- Partial/overlapping captures (kept, not merged): **{partial}**")
    A("")
    A("## REVIEWED-CLOSED — human-triaged as unrecoverable\n")
    A(f"Count: **{len(reviewed)}**  — you reviewed these and judged them not-useful / no LLM reply / "
      "too messy / Chinese (see `manual_split_resolutions.csv`). Full text retained; not processed further.")
    A("")
    A("## ACTIONABLE — genuinely_unrecovered (the remaining to-do)\n")
    A(f"Count: **{len(genu)}**  — unmarked / low-quality residue not yet human-reviewed. What each type means:\n")
    for status, n in genu["status"].value_counts().items():
        A(f"- **`{status}` ({n})** — {STATUS_EXPLAIN.get(status, 'Flagged for review.')}")
    A("")
    A("_`flagged_for_provenance` and `reviewed_closed` are locked; only `genuinely_unrecovered` "
      "items remain candidates for further processing._")
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"needs_review={len(nr)}  locked(provenance)={len(prov)}  reviewed_closed={len(reviewed)}  actionable(unrecovered)={len(genu)}")
    print("genuinely_unrecovered by status:")
    print(genu["status"].value_counts().to_string())
    print(f"\nWrote {REPORT.name}; added review_category + locked to manifest/master.")


if __name__ == "__main__":
    main()
