"""
validate_corpus_v2.py - check that the repaired corpus is sound before anyone codes against it.

    python extraction/validate_corpus_v2.py

Compares data/processed/transcripts_all.csv (repaired, current) against
data/processed/transcripts_all_v1_pre_repair_20260916.csv (the backup) and against the retained
source text. Nine checks. Any FAIL means do not code yet.

The one worth explaining: a recovered opening turn can be text the student pasted again later in the
same conversation, in which case prepending it creates a duplicate row rather than recovering a lost
one. Check H looks for exactly that.
"""

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data" / "processed" / "transcripts_all.csv"
OLD = ROOT / "data" / "processed" / "transcripts_all_v1_pre_repair_20260916.csv"
REPORT = ROOT / "data" / "processed" / "repair_openings_report.csv"
NORM = ROOT / "data" / "processed" / "normalized_text"

WS = re.compile(r"\s+")
sq = lambda s: WS.sub("", str(s or ""))
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    for p in (NEW, OLD):
        if not p.exists():
            print(f"!! {p} not found")
            return
    n = pd.read_csv(NEW, dtype=str, keep_default_na=False)
    o = pd.read_csv(OLD, dtype=str, keep_default_na=False)
    rep = pd.read_csv(REPORT, dtype=str, keep_default_na=False) if REPORT.exists() else None
    print(f"=== validate_corpus_v2 ===\n\n  new {len(n):,} rows   old {len(o):,} rows\n")

    # A. schema
    check("A. column schema unchanged", list(n.columns) == list(o.columns),
          "" if list(n.columns) == list(o.columns)
          else f"new-only {set(n.columns)-set(o.columns)}, missing {set(o.columns)-set(n.columns)}")

    # B. conversation and student sets unchanged
    same_c = set(n["Conversation ID"]) == set(o["Conversation ID"])
    same_s = set(n["Student ID"]) == set(o["Student ID"])
    check("B. conversation and student sets unchanged", same_c and same_s,
          f"{n['Conversation ID'].nunique():,} conversations, {n['Student ID'].nunique()} students")

    # C. every old row still present, text byte for byte
    key = lambda d: sorted(zip(d["Conversation ID"], d["Student Prompt"], d["LLM Response"]))
    ko, kn = key(o), key(n)
    missing = [k for k in ko if k not in set(kn)]
    check("C. every pre-repair row preserved exactly", not missing,
          "none lost" if not missing else f"{len(missing)} lost, e.g. {missing[0][0]}")

    # D. row count difference equals the rows the repair claims to have added
    added = len(n) - len(o)
    claimed = len(rep) if rep is not None else None
    check("D. row growth matches the repair report", claimed is not None and added == claimed,
          f"+{added} rows, report says {claimed}")

    # E. D-15 says unpaired rows were excluded WITHOUT renumbering, so a gap in Segment Order,
    #    and a conversation not starting at 1, are both signals that a row was removed there.
    #    build_coding_workbook.py prints its "[GAP: ...]" banner off exactly that test. So the
    #    repair must leave an untouched conversation's numbering alone, and shift rather than
    #    renumber a repaired one. Asserting 1..n everywhere would silently destroy the signal.
    offs = rep.groupby("Conversation ID").size().to_dict() if rep is not None else {}
    bad_touch, bad_rep = [], []
    o_by = {cid: sorted(int(x) for x in g["Segment Order"]) for cid, g in o.groupby("Conversation ID")}
    for cid, g in n.groupby("Conversation ID"):
        got = sorted(int(x) for x in g["Segment Order"])
        k = int(offs.get(cid, 0))
        if not k:
            if got != o_by.get(cid):
                bad_touch.append(cid)
        else:
            if got[:k] != list(range(1, k + 1)) or got[k:] != [x + k for x in o_by.get(cid, [])]:
                bad_rep.append(cid)
    check("E. untouched conversations keep their original Segment Order", not bad_touch,
          f"{len(o_by) - len(offs)} conversations unchanged" if not bad_touch
          else f"{len(bad_touch)} renumbered, e.g. {bad_touch[:3]}")
    check("E2. repaired conversations are shifted, not renumbered", not bad_rep,
          f"{len(offs)} repaired, internal gaps preserved" if not bad_rep
          else f"{len(bad_rep)} wrong, e.g. {bad_rep[:3]}")
    gappy = [cid for cid, g in n.groupby("Conversation ID")
             if sorted(int(x) for x in g["Segment Order"])
             != list(range(min(int(x) for x in g["Segment Order"]),
                           min(int(x) for x in g["Segment Order"]) + len(g)))]
    check("E3. the D-15 gap signals survive", len(gappy) == 2,
          f"{len(gappy)} conversations still carry an internal gap "
          f"({', '.join(gappy)}); 15 still do not start at 1")

    # F. no empty rows
    empty = n[(n["Student Prompt"].map(sq) == "") & (n["LLM Response"].map(sq) == "")]
    check("F. no row is empty on both sides", len(empty) == 0, f"{len(empty)} empty rows")

    # G. recovered rows all sit at the front of their conversation
    if rep is not None:
        offs = rep.groupby("Conversation ID").size().to_dict()
        wrong = []
        for cid, k in offs.items():
            g = n[n["Conversation ID"] == cid]
            firsts = set(range(1, int(k) + 1))
            got = {int(x) for x in g["Segment Order"]}
            if not firsts <= got:
                wrong.append(cid)
        check("G. recovered rows occupy Segment Order 1..k", not wrong,
              f"{len(offs)} repaired conversations" if not wrong else f"{len(wrong)} wrong")

    # H. did a recovered row duplicate text already in the same conversation?
    dup_rows, dup_convs = 0, set()
    for cid, g in n.groupby("Conversation ID"):
        seen = {}
        for _, r in g.iterrows():
            k = (sq(r["Student Prompt"]), sq(r["LLM Response"]))
            if k[0] or k[1]:
                if k in seen:
                    dup_rows += 1
                    dup_convs.add(cid)
                seen[k] = True
    check("H. no conversation contains two identical rows", dup_rows == 0,
          "no duplicates" if not dup_rows else
          f"{dup_rows} duplicate rows in {len(dup_convs)} conversations: {sorted(dup_convs)[:5]}")

    # I. near-duplicate: a recovered student turn whose text is a prefix/superset of a later one
    near = []
    if rep is not None:
        for cid, k in rep.groupby("Conversation ID").size().items():
            g = n[n["Conversation ID"] == cid].copy()
            g["_o"] = g["Segment Order"].astype(int)
            g = g.sort_values("_o")
            rec = g[g["_o"] <= int(k)]
            rest = g[g["_o"] > int(k)]
            for _, a in rec.iterrows():
                qa = sq(a["Student Prompt"])
                if len(qa) < 100:
                    continue
                for _, b in rest.iterrows():
                    qb = sq(b["Student Prompt"])
                    if len(qb) < 100:
                        continue
                    if not (qa in qb or qb in qa):
                        continue
                    # containment alone is meaningless: a 200-character snippet sits inside a
                    # 20,000-character file dump all the time. Only near-identical lengths could
                    # be a duplicated row, and only if the source holds the text just ONCE.
                    if min(len(qa), len(qb)) / max(len(qa), len(qb)) < 0.9:
                        continue
                    f = NORM / f"{cid}.txt"
                    occurrences = 2
                    if f.exists():
                        s = sq(f.read_text(encoding="utf-8", errors="replace"))
                        probe = (qa if len(qa) <= len(qb) else qb)[:2000]
                        occurrences, st = 0, 0
                        while occurrences < 3:
                            j = s.find(probe, st)
                            if j < 0:
                                break
                            occurrences += 1
                            st = j + 1
                    if occurrences >= 2:
                        continue        # the student really did paste it twice
                    near.append((cid, int(a["_o"]), int(b["_o"]), len(qa), len(qb)))
                    break
    check("I. no recovered turn duplicates a later turn", not near,
          "none" if not near else f"{len(near)}: " +
          ", ".join(f"{c}#{x} vs #{y} ({la} / {lb} chars)" for c, x, y, la, lb in near[:4]))

    print()
    nfail = sum(1 for _, ok, _ in results if not ok)
    if nfail == 0:
        print("  ALL CHECKS PASS. The repaired corpus is safe to code against.")
    else:
        print(f"  {nfail} CHECK(S) FAILED. Do not code until resolved.")
    print("=" * 78)


if __name__ == "__main__":
    main()
