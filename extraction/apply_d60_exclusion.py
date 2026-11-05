#!/usr/bin/env python3
"""
apply_d60_exclusion.py  --  execute D-60, option (a), reversibly.

THE DECISION (DW, 20 September 2026; see docs/decisions.md D-60 and
docs/corpus/corpus-number-chain-20260920.md section 5)

    Two conversations are near-copies of another conversation BY THE SAME
    STUDENT. Exclude the lesser member of each pair FROM THE ANALYSIS DATA ONLY,
    keep the raw material, and record the correspondence so the exclusion can be
    undone.

        P054_P2_02  ->  kept: P054_P2_01
        P492_P2_03  ->  kept: P492_P2_04

    DW's reason, in his words: this preserves conversation structure, where
    dropping only the overlapping rows would leave orphaned tail rows that are
    not conversations.

WHAT THIS SCRIPT DOES AND DOES NOT DO

    It does NOT modify data/processed/transcripts_all.csv. That file stays as
    the PRE-EXCLUSION BASELINE, and every figure computed before this exclusion
    remains reproducible from it. Nothing is deleted and nothing is renumbered:
    Segment Order is carried through untouched, so a row's key
    (Conversation ID, Segment Order) means the same thing before and after.

    It writes two new files:

        data/processed/exclusions_d60.csv       one row per excluded record,
            with its original key, the kept record it corresponds to, how the
            two differ, and the basis for excluding it
        data/processed/transcripts_analysis.csv the analysis data with those
            records removed

    Reversing it means deleting transcripts_analysis.csv, or re-adding the rows
    named in exclusions_d60.csv from the baseline. No information is lost.

    Raw material is untouched: data/processed/normalized_text/ keeps the files
    for both excluded conversations, and so do the original transcript folders.

    It does NOT repoint anything that reads transcripts_all.csv. Several
    scripts under analysis/ and coding/ read the baseline directly and will keep
    reporting pre-exclusion figures until someone changes them deliberately.
    That is a separate decision and this script does not take it.

CORRESPONDENCE IS RE-DERIVED, NOT COPIED

    The row-level mapping and the differences are recomputed here from the data
    on every run, by normalising whitespace and case on Student Prompt and
    LLM Response. Nothing is read from a document. The guard below then checks
    the result against what D-60 was decided on, and refuses to write if the
    baseline has moved.

USAGE
    python extraction/apply_d60_exclusion.py            verify, then write
    python extraction/apply_d60_exclusion.py --check    verify only, write nothing
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASELINE = os.path.join(ROOT, "data", "processed", "transcripts_all.csv")
OUT_LIST = os.path.join(ROOT, "data", "processed", "exclusions_d60.csv")
OUT_DATA = os.path.join(ROOT, "data", "processed", "transcripts_analysis.csv")

DECISION = "D-60"
DECIDED = "DW, 20 September 2026, option (a)"
BASIS = ("near-copy of another conversation by the same student; the kept member "
         "contains every exchange this one contains")

# excluded -> kept
PAIRS = {"P054_P2_02": "P054_P2_01", "P492_P2_03": "P492_P2_04"}

# What D-60 was decided on. If the baseline no longer matches this, the decision
# was taken against different data and must be re-examined, not re-applied.
EXPECTED_BASELINE = dict(rows=6595, conversations=1204, students=361)
EXPECTED_PAIR_ROWS = {"P054_P2_02": 23, "P054_P2_01": 23, "P492_P2_03": 5, "P492_P2_04": 7}

# The CONTENT D-60 was adjudicated on, not merely its shape. Counting records is
# not enough: a baseline could keep 6,595 rows while one response changed, and a
# guard that only counts would let the new difference be swept into D-60 as
# though DW had ruled on it. He ruled on exactly these two differences.
#
#   key: (excluded conversation, its Segment Order)
#   value: the kept record's Segment Order, and the response lengths that were
#          put in front of DW when he chose option (a)
ADJUDICATED_DIFFERENCES = {
    ("P054_P2_02", 23): dict(kept_segment_order=23, excluded_response_chars=258, kept_response_chars=1063),
    ("P492_P2_03", 5): dict(kept_segment_order=5, excluded_response_chars=1312, kept_response_chars=1250),
}
# sha256 over the normalised (Student Prompt, LLM Response) of all four
# conversations in Segment Order. Any edit to the adjudicated material changes
# it. Changes elsewhere in the baseline do not, so unrelated corpus work is not
# blocked by this guard.
ADJUDICATED_CONTENT_SHA256 = "1c025e1c0cd6ce155d9c701fa244a132b72d79784e20f1fd351d7d2cd2048dbe"

# Post-exclusion counts, from the arithmetic in the decision. Verified, not assumed.
EXPECTED_AFTER = dict(rows=6567, rows_with_student_text=6565, conversations=1202, students=361)


def norm(s):
    return re.sub(r"\s+", " ", "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)).strip().lower()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_fingerprint(df):
    """sha256 over the normalised content of the four conversations D-60 ruled on.

    Restricted to those four on purpose: the guard has to fire when the
    adjudicated material changes and stay quiet when unrelated corpus work
    changes something else."""
    h = hashlib.sha256()
    for cid in sorted(set(PAIRS) | set(PAIRS.values())):
        g = df[df["Conversation ID"] == cid].sort_values("Segment Order")
        for _, r in g.iterrows():
            h.update(f"{cid}\x1f{int(r['Segment Order'])}\x1f"
                     f"{norm(r['Student Prompt'])}\x1f{norm(r['LLM Response'])}\x1e".encode())
    return h.hexdigest()


def fail(msg):
    print(f"ABORTED: {msg}")
    print("Nothing was written. The baseline is unchanged.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    df = pd.read_csv(BASELINE, low_memory=False)
    print("BASELINE, pre-exclusion")
    print(f"  file      data/processed/transcripts_all.csv")
    print(f"  sha256    {sha256(BASELINE)}")
    base = dict(rows=len(df),
                conversations=df["Conversation ID"].nunique(),
                students=df["Student ID"].nunique())
    print(f"  records {base['rows']}, conversations {base['conversations']}, students {base['students']}")
    for k, v in EXPECTED_BASELINE.items():
        if base[k] != v:
            fail(f"baseline {k} is {base[k]}, but D-60 was decided on {v}")
    for cid, n in EXPECTED_PAIR_ROWS.items():
        got = int((df["Conversation ID"] == cid).sum())
        if got != n:
            fail(f"{cid} has {got} records, expected {n}")
    fp = content_fingerprint(df)
    print(f"  adjudicated-content sha256  {fp}")
    if ADJUDICATED_CONTENT_SHA256 == "PLACEHOLDER":
        print("  NOTE: the adjudicated-content fingerprint is not pinned yet. Set")
        print("        ADJUDICATED_CONTENT_SHA256 to the value above once it is verified.")
    elif fp != ADJUDICATED_CONTENT_SHA256:
        fail("the content of the four conversations D-60 adjudicated has changed since the "
             "decision was taken. Re-examine D-60 against the new material; do not re-apply it.")
    print("  guard passed: the baseline is the data D-60 was decided on")
    print()

    # ---- re-derive the row-level correspondence -----------------------------
    out = []
    for drop, keep in PAIRS.items():
        D = df[df["Conversation ID"] == drop].sort_values("Segment Order")
        K = df[df["Conversation ID"] == keep].sort_values("Segment Order")
        kept_rows = [(int(r["Segment Order"]), norm(r["Student Prompt"]), norm(r["LLM Response"]),
                      len(str(r["LLM Response"])) if pd.notna(r["LLM Response"]) else 0)
                     for _, r in K.iterrows()]
        matched_keep = set()
        for _, r in D.iterrows():
            so = int(r["Segment Order"])
            p, resp = norm(r["Student Prompt"]), norm(r["LLM Response"])
            rlen = len(str(r["LLM Response"])) if pd.notna(r["LLM Response"]) else 0
            same = [k for k in kept_rows if k[1] == p and k[2] == resp]
            if same:
                k = same[0]
                rel, diff = "identical", "prompt and response identical after normalising whitespace and case"
            else:
                sp = [k for k in kept_rows if k[1] == p]
                if sp:
                    k = sp[0]
                    rel = "same prompt, response differs"
                    diff = (f"same prompt; response {rlen} chars here against {k[3]} chars in the kept record "
                            f"({'shorter' if rlen < k[3] else 'longer'} by {abs(rlen - k[3])})")
                else:
                    k = None
                    rel, diff = "NO MATCH", "no record in the kept conversation has this prompt"
            if k:
                matched_keep.add(k[0])
            out.append(dict(
                decision=DECISION, decided_by=DECIDED,
                excluded_conversation_id=drop, excluded_segment_order=so,
                excluded_student_id=str(r["Student ID"]),
                excluded_source_file=str(r["Source File"]),
                kept_conversation_id=keep,
                kept_segment_order=(k[0] if k else ""),
                relation=rel, difference=diff, basis=BASIS,
                student_prompt_chars=len(str(r["Student Prompt"])) if pd.notna(r["Student Prompt"]) else 0,
                llm_response_chars=rlen,
                reversible_by="re-add these (Conversation ID, Segment Order) rows from transcripts_all.csv",
            ))
        only_keep = [k[0] for k in kept_rows if k[0] not in matched_keep]
        print(f"{drop} -> {keep}")
        rel_counts = {}
        for o in out:
            if o["excluded_conversation_id"] == drop:
                rel_counts[o["relation"]] = rel_counts.get(o["relation"], 0) + 1
        for rel, n in sorted(rel_counts.items()):
            print(f"  {n:3d}  {rel}")
        print(f"  records the kept conversation has and this one does not: "
              f"{len(only_keep)}" + (f", Segment Order {only_keep}" if only_keep else ""))
        if "NO MATCH" in rel_counts:
            fail(f"{drop} has records with no counterpart in {keep}; option (a) assumes the kept "
                 f"member contains everything this one contains")
        # Every record must be either identical to its counterpart or one of the
        # two differences DW actually ruled on. A NEW difference is not covered by
        # D-60 and must not be applied under its name.
        for o in out:
            if o["excluded_conversation_id"] != drop or o["relation"] == "identical":
                continue
            key = (drop, o["excluded_segment_order"])
            spec = ADJUDICATED_DIFFERENCES.get(key)
            if spec is None:
                fail(f"{drop} Segment Order {o['excluded_segment_order']} differs from its "
                     f"counterpart, and that difference is not one D-60 adjudicated. "
                     f"D-60 covers only {sorted(ADJUDICATED_DIFFERENCES)}.")
            if (o["kept_segment_order"] != spec["kept_segment_order"]
                    or o["llm_response_chars"] != spec["excluded_response_chars"]):
                fail(f"{drop} Segment Order {o['excluded_segment_order']} no longer matches the "
                     f"difference D-60 adjudicated (expected kept SO {spec['kept_segment_order']}, "
                     f"{spec['excluded_response_chars']} chars; found kept SO {o['kept_segment_order']}, "
                     f"{o['llm_response_chars']} chars).")
        for key in ADJUDICATED_DIFFERENCES:
            if key[0] != drop:
                continue
            if not any(o["excluded_conversation_id"] == drop
                       and o["excluded_segment_order"] == key[1]
                       and o["relation"] != "identical" for o in out):
                fail(f"{drop} Segment Order {key[1]} was adjudicated as differing from its "
                     f"counterpart but no longer does. The material has changed.")
    print()

    # ---- manual labels: a GUARD, not a note ---------------------------------
    # D-60 excludes records from the analysis data. If a hand-coded label exists
    # on an excluded record, or a coding worksheet refers to an excluded
    # conversation, then executing D-60 silently drops human work. This aborts.
    # It reads spreadsheets as spreadsheets: the worksheets under data/coding/
    # are .xlsx, and reading them as text finds nothing whatever they contain.
    print("MANUAL-LABEL GUARD")
    coded_vals = sorted(set(df[df["Conversation ID"].isin(PAIRS)]["Coded"].astype(str)))
    print(f"  Coded values on the records to exclude: {coded_vals}")
    if any(v.strip().lower() in ("true", "1", "yes") for v in coded_vals):
        fail("a record marked for exclusion carries Coded=True. Executing D-60 would drop a "
             "hand-coded record. Resolve the label first.")

    coding_dir = os.path.join(ROOT, "data", "coding")
    hits, unreadable = [], []
    skip_ext = (".bak", ".png", ".jpg", ".pdf", ".zip")
    for dirpath, _, files in os.walk(coding_dir):
        for f in sorted(files):
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, ROOT)
            low = f.lower()
            if low.endswith(skip_ext) or ".bak-" in low or ".bak" in low:
                continue  # dated backups are history, not live coding
            found = False
            if low.endswith((".xlsx", ".xlsm")):
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(full, read_only=True, data_only=True)
                    for ws in wb.worksheets:
                        for row in ws.iter_rows(values_only=True):
                            for cell in row:
                                if cell is not None and any(cid in str(cell) for cid in PAIRS):
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    wb.close()
                except Exception as e:                      # noqa: BLE001
                    unreadable.append(f"{rel} ({type(e).__name__})")
                    continue
            else:
                try:
                    t = open(full, encoding="utf-8", errors="replace").read()
                except OSError as e:
                    unreadable.append(f"{rel} ({type(e).__name__})")
                    continue
                found = any(cid in t for cid in PAIRS)
            if found:
                hits.append(rel)
    print(f"  files under data/coding/ inspected for either excluded conversation id, "
          f"spreadsheets cell by cell")
    if unreadable:
        fail("could not read " + ", ".join(unreadable) + ". The guard cannot confirm that no "
             "hand coding refers to an excluded conversation, so it does not pass.")
    if hits:
        fail("these files under data/coding/ refer to an excluded conversation: "
             + ", ".join(hits) + ". Executing D-60 would orphan them.")
    print("  passed: no live file under data/coding/ refers to either excluded conversation, "
          "and no record to exclude is marked Coded")
    print()

    # ---- apply --------------------------------------------------------------
    keep_mask = ~df["Conversation ID"].isin(PAIRS)
    after = df[keep_mask]
    res = dict(rows=len(after),
               rows_with_student_text=int(after["Student Prompt"].notna().sum()),
               conversations=after["Conversation ID"].nunique(),
               students=after["Student ID"].nunique())
    print("AFTER EXCLUSION, each figure with its own denominator")
    print(f"  records in the analysis corpus                     {res['rows']}   (of {base['rows']} in the baseline)")
    print(f"  of those, records carrying student text            {res['rows_with_student_text']}   (of {res['rows']} analysis records)")
    print(f"  conversations                                      {res['conversations']}   (of {base['conversations']})")
    print(f"  students                                           {res['students']}   (of {base['students']})")
    print("  These are not universal denominators. Sample exclusions and per-field")
    print("  applicability still apply, so every reported figure states its own.")
    bad = [k for k, v in EXPECTED_AFTER.items() if res[k] != v]
    if bad:
        fail("post-exclusion counts do not match the decision: "
             + ", ".join(f"{k} is {res[k]}, expected {EXPECTED_AFTER[k]}" for k in bad))
    print("  verified against the counts recorded with the decision")
    print()

    if args.check:
        print("--check: verified, nothing written.")
        return

    fields = list(out[0].keys())
    with open(OUT_LIST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for o in out:
            w.writerow(o)
    after.to_csv(OUT_DATA, index=False)
    print(f"wrote data/processed/exclusions_d60.csv        ({len(out)} excluded records)")
    print(f"wrote data/processed/transcripts_analysis.csv  ({len(after)} records)")
    print(f"     sha256 {sha256(OUT_DATA)}")
    print("     transcripts_all.csv was not modified and remains the pre-exclusion baseline")


if __name__ == "__main__":
    main()
