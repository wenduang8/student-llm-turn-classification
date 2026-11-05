#!/usr/bin/env python3
"""
combine_dedup.py

Merge the three turn-pair CSVs (deterministic / LLM / manual) into ONE deduplicated training set,
`data/processed/transcripts_all.csv`. Removes exact-duplicate pairs (same Student ID + Student Prompt
+ LLM Response) that arise when the same conversation was captured by more than one route
(e.g. a student's auto-extracted pairs and their hand-marked pairs). Keeps one copy, preferring the
deterministic parse, then the human-marked/manual recovery, then the LLM split.

Run LAST (after recover_manual_rules.py, review_report.py). Idempotent.
"""
import re
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "processed"   # repo root / data (this file lives in extraction/)
# Label columns: the combined training set uses the symmetric 4-column coding scheme
# (student side + AI side), replacing the earlier 2-column Delegation/Cognitive labels.
# All blank here — this is the uncoded inference corpus for human coders to fill.
LABELS = ["Student delegation", "Student cognition", "AI delegation", "AI response type"]
CANON = (["Student ID", "Conversation ID", "Segment Order", "Student Prompt", "LLM Response"]
         + LABELS + ["Coded", "Source", "Source File", "Segmentation Method", "Coverage"])
PRIORITY = {"deterministic": 0, "manual": 1, "llm": 2}   # which copy to keep on an exact tie


def _load(name, default_method):
    df = pd.read_csv(OUT / name)
    if "Segmentation Method" not in df.columns:
        df["Segmentation Method"] = default_method
    if "Coverage" not in df.columns:
        df["Coverage"] = ""
    for c in CANON:
        if c not in df.columns:
            df[c] = ""
    return df[CANON]


def main():
    det = _load("transcripts_segmented.csv", "deterministic")
    llm = _load("transcripts_segmented_llm.csv", "llm")
    man = _load("transcripts_segmented_manual.csv", "manual")
    allp = pd.concat([det, llm, man], ignore_index=True)
    n0 = len(allp)

    allp["_sp"] = allp["Student Prompt"].fillna("")
    allp["_lr"] = allp["LLM Response"].fillna("")
    # bucket the many method labels into the 3 priority families
    def fam(m):
        m = str(m)
        if m.startswith("mhtml") or m.startswith("rule_") or m == "manual_mark" or m == "manual":
            return "manual"
        if m == "llm" or m.startswith("llm"):
            return "llm"
        return "deterministic"
    allp["_fam"] = allp["Segmentation Method"].map(fam)

    # Pass 1 — cross-source content dedup: when an auto (deterministic/llm) pair is content-identical
    # (ignoring formatting) to a hand-marked/manual pair for the same student, drop the auto copy and
    # keep the manual one. Pairs whose content differs are kept (different data inputs).
    def _norm(t):
        return re.sub(r'[^a-z0-9]', '', re.sub(r'\s+', ' ', str(t).lower()))
    allp["_nk"] = (allp["Student ID"].astype(str) + "|" +
                   allp["_sp"].map(_norm) + "|" + allp["_lr"].map(_norm))
    manual_nk = set(allp.loc[allp["_fam"] == "manual", "_nk"])
    cross_drop = (allp["_fam"] != "manual") & (allp["_nk"].isin(manual_nk))
    n_cross = int(cross_drop.sum())
    allp = allp[~cross_drop]

    # Pass 2 — exact-string dedup (same student + prompt + response), keep by priority.
    allp["_prio"] = allp["_fam"].map(PRIORITY)
    allp = allp.sort_values("_prio", kind="stable")
    key = ["Student ID", "_sp", "_lr"]
    dupmask = allp.duplicated(subset=key, keep="first")
    dropped = int(dupmask.sum())
    kept = allp[~dupmask].drop(columns=["_sp", "_lr", "_prio", "_fam", "_nk"])
    kept = kept.sort_values(["Student ID", "Conversation ID", "Segment Order"], kind="stable")
    # renumber Segment Order 1..n per conversation (dropping pairs can leave gaps); a pair removed
    # here therefore leaves no gap, unlike the unpaired-row exclusion below
    kept["Segment Order"] = kept.groupby("Conversation ID").cumcount() + 1

    # Exclude unpaired rows. A row with no LLM response is not a student->LLM exchange and so
    # fails the turn-level-sequence inclusion criterion (methodology 2.5(4)). Most come from
    # AI-usage reference documents that list the student's prompts with an embedded share link:
    # the document body is unpaired and is dropped, while the scraped exchange at the tail is
    # paired and is kept. Segment Order is deliberately NOT renumbered afterwards, so a gap
    # stays visible and the loader can treat it as a context boundary rather than letting
    # non-adjacent turns appear adjacent in the reading window.
    n_dedup = len(kept)
    unpaired = kept["LLM Response"].fillna("").astype(str).str.strip() == ""
    excluded = kept[unpaired]
    kept = kept[~unpaired]
    excluded.to_csv(OUT / "transcripts_excluded_unpaired.csv", index=False, encoding="utf-8")

    kept.to_csv(OUT / "transcripts_all.csv", index=False, encoding="utf-8")

    from collections import Counter
    by_src = Counter(kept["Segmentation Method"].map(fam))
    print(f"combined pairs: {n0}  ->  deduplicated: {n_dedup}  "
          f"(removed {n_cross} auto copies identical to a manual pair + {dropped} exact-duplicate rows)")
    print("retained by source:", dict(by_src))
    print(f"unpaired rows excluded: {len(excluded)} "
          f"(-> data/processed/transcripts_excluded_unpaired.csv)")
    print(f"analysis corpus: {len(kept)} paired rows")
    print("distinct students:", kept["Student ID"].nunique(),
          "| conversations:", kept["Conversation ID"].nunique())
    print("-> data/processed/transcripts_all.csv")


if __name__ == "__main__":
    main()
