"""
repair_openings.py - recover the conversation openings the extractor dropped, into a NEW corpus file.

    python extraction/repair_openings.py            # dry run, report only
    python extraction/repair_openings.py --write    # also write transcripts_all_v2.csv

Nothing existing is overwritten. transcripts_all.csv, the workbooks and the markdown are untouched.

WHAT IT DOES. For each conversation it anchors the first transcripts_all.csv row inside
data/processed/normalized_text/<cid>.txt, takes the raw text before that anchor, and re-segments only
that leading region with the two bugs fixed:

  * text before the first marker is captured instead of discarded (it is treated as a student turn
    only when the first marker in the region is an LLM marker, i.e. something answered it; otherwise
    it is an export header and stays dropped);
  * a leading assistant turn is kept instead of skipped, as a row with an empty Student Prompt.

Recovered rows are prepended to that conversation and Segment Order is renumbered 1..n. Every
original row keeps its text byte for byte; only its Segment Order moves.

WHAT IT REPORTS. Rows recovered, conversations affected, the shape of what came back, and the effect
on the exclusion funnel, because the floor at "more than three interactions" is the figure most
likely to move.
"""

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extraction"))

NORM = ROOT / "data" / "processed" / "normalized_text"
ALL = ROOT / "data" / "processed" / "transcripts_all.csv"
OUT = ROOT / "data" / "processed" / "transcripts_all_v2.csv"
SRC_OVERRIDE = None   # --source lets this be re-run from a backup
REPORT = ROOT / "data" / "processed" / "repair_openings_report.csv"

WS = re.compile(r"\s+")
MIN_SOLID = 40
FLOOR, CAP = 3, 100          # D-25 / D-26: more than 3 interactions, at most 100


def sq(s):
    return WS.sub("", str(s or ""))


def raw_index_of_solid(raw, k):
    if k <= 0:
        return 0
    c = 0
    for i, ch in enumerate(raw):
        if not ch.isspace():
            if c == k:
                return i
            c += 1
    return len(raw)


def anchor(s, g):
    """Solid-char position of the first corpus row, chosen so the rest still runs in order."""
    per_order = []
    for _, r in g.iterrows():
        pos = []
        for col in ("Student Prompt", "LLM Response"):
            q = sq(r[col])
            if len(q) < 20:
                continue
            probe = q if len(q) <= 4000 else q[:4000]
            k, start = [], 0
            while len(k) < 8:
                j = s.find(probe, start)
                if j < 0:
                    break
                k.append(j)
                start = j + 1
            if not k and len(probe) > 200:
                j = s.find(probe[:200])
                if j >= 0:
                    k = [j]
            pos.extend(k)
        per_order.append(sorted(set(pos)))
    per_order = [p for p in per_order if p]
    if not per_order:
        return None

    def chain(start):
        prev, c = start, 1
        for pl in per_order[1:]:
            nxt = [x for x in pl if x > prev]
            if nxt:
                prev = min(nxt)
                c += 1
        return c

    return min(per_order[0], key=lambda p: (-chain(p), p))


def segment_head(head_raw, _find_markers, STRONG_STUDENT, STRONG_LLM, HEADER_LINE_RE):
    """segment_turns(), with the pre-first-marker region no longer discarded."""
    text = head_raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'(?m)^[<>=~_\-\*#\s]{6,}$', '', text)
    text = "\n".join(ln for ln in text.split("\n") if not HEADER_LINE_RE.match(ln))
    hits = _find_markers(text, STRONG_STUDENT, STRONG_LLM)
    if not hits:
        return []

    segs = []
    lead = text[:hits[0][0]].strip()
    # only a turn if something answered it; a student marker first means this was a header
    if lead and hits[0][2] == "llm":
        segs.append({"role": "student", "text": lead})
    for i, (st, en, role, fam) in enumerate(hits):
        nxt = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        segs.append({"role": role, "text": text[en:nxt].strip()})

    merged = []
    for s in segs:
        if merged and merged[-1]["role"] == s["role"]:
            if s["text"]:
                merged[-1]["text"] = (merged[-1]["text"] + "\n" + s["text"]).strip()
        else:
            merged.append(dict(s))
    return [m for m in merged if m["text"]]


def pair_head(turns):
    """turns_to_pairs(), keeping a leading assistant turn instead of skipping it."""
    pairs, i, n = [], 0, len(turns)
    while i < n:
        if turns[i]["role"] == "student":
            prompt = turns[i]["text"]
            resp, j = [], i + 1
            while j < n and turns[j]["role"] == "llm":
                resp.append(turns[j]["text"])
                j += 1
            pairs.append((prompt, "\n".join(resp).strip()))
            i = j
        else:
            resp, j = [], i
            while j < n and turns[j]["role"] == "llm":
                resp.append(turns[j]["text"])
                j += 1
            pairs.append(("", "\n".join(resp).strip()))     # opening assistant turn, kept
            i = j
    return pairs


def funnel(df):
    g = df.groupby("Conversation ID").size()
    keep = g[(g > FLOOR) & (g <= CAP)]
    return len(g), int(g.sum()), len(keep), int(keep.sum())


def main():
    write = "--write" in sys.argv
    src = ALL
    if "--source" in sys.argv:
        try:
            src = ROOT / "data" / "processed" / sys.argv[sys.argv.index("--source") + 1]
        except IndexError:
            pass
    try:
        from authentic_extractor import _find_markers, STRONG_STUDENT, STRONG_LLM, HEADER_LINE_RE
    except Exception as e:
        print(f"!! could not import authentic_extractor: {type(e).__name__}: {e}")
        return

    d = pd.read_csv(src, dtype=str, keep_default_na=False)
    print(f"  source: {src.name}")
    d["_o"] = pd.to_numeric(d["Segment Order"], errors="coerce").fillna(-1).astype(int)
    cols = [c for c in d.columns if c != "_o"]

    out, rec = [], []
    n_full = n_llm_only = n_stu_only = 0
    for cid, g in d.groupby("Conversation ID", sort=False):
        g = g.sort_values("_o")
        f = NORM / f"{cid}.txt"
        new_rows = []
        if f.exists():
            try:
                raw = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                raw = ""
            if raw:
                s = sq(raw)
                a = anchor(s, g)
                if a is not None and a >= MIN_SOLID:
                    head = raw[:raw_index_of_solid(raw, a)]
                    turns = segment_head(head, _find_markers, STRONG_STUDENT,
                                         STRONG_LLM, HEADER_LINE_RE)
                    for stu, llm in pair_head(turns):
                        if not sq(stu) and not sq(llm):
                            continue
                        if sq(stu) and sq(llm):
                            n_full += 1
                            kind = "pair"
                        elif sq(llm):
                            n_llm_only += 1
                            kind = "assistant-only"
                        else:
                            n_stu_only += 1
                            continue          # student with no reply: D-15 excludes these
                        new_rows.append((stu, llm, kind))

        base = g.iloc[0]
        # D-15: unpaired rows were excluded WITHOUT renumbering, so a gap in Segment Order and a
        # conversation that does not start at 1 are both signals that a row was removed there.
        # build_coding_workbook.py prints its "[GAP: a row was removed...]" banner off exactly that
        # test. So only a conversation that actually gains rows is touched, and it is SHIFTED by k
        # rather than renumbered, which preserves any internal gap. A conversation that gains
        # nothing keeps its original numbering untouched.
        k = len(new_rows)
        order = 1
        for stu, llm, kind in new_rows:
            r = {c: "" for c in cols}
            r["Student ID"] = base["Student ID"]
            r["Conversation ID"] = cid
            r["Segment Order"] = order
            r["Student Prompt"] = stu
            r["LLM Response"] = llm
            for c in ("Coded", "Source", "Source File", "Segmentation Method", "Coverage"):
                if c in r:
                    r[c] = base[c] if c != "Segmentation Method" else "recovered_opening"
            out.append(r)
            rec.append({"Conversation ID": cid, "Segment Order": order, "kind": kind,
                        "student_chars": len(sq(stu)), "llm_chars": len(sq(llm))})
            order += 1
        for _, r in g.iterrows():
            rr = {c: r[c] for c in cols}
            rr["Segment Order"] = (int(r["_o"]) + k) if k else r["Segment Order"]
            out.append(rr)

    nd = pd.DataFrame(out, columns=cols)

    print("=== repair_openings ===\n")
    print(f"  rows recovered                      {len(rec):>7}")
    print(f"    full (student + assistant)        {n_full:>7}")
    print(f"    assistant-only (empty prompt)     {n_llm_only:>7}")
    print(f"    student-only, NOT added (D-15)    {n_stu_only:>7}")
    print(f"  conversations gaining rows          {len({r['Conversation ID'] for r in rec}):>7}"
          f"  of {d['Conversation ID'].nunique()}")
    print(f"  corpus rows  {len(d):,}  ->  {len(nd):,}")

    oc, oi, okc, oki = funnel(d)
    nc, ni, nkc, nki = funnel(nd)
    print("\n  exclusion funnel (floor > 3 interactions, cap 100)\n")
    print(f"                        before            after")
    print(f"    conversations   {oc:>8,}        {nc:>8,}")
    print(f"    interactions    {oi:>8,}        {ni:>8,}")
    print(f"    codable convs   {okc:>8,}        {nkc:>8,}   ({nkc - okc:+,})")
    print(f"    codable ints    {oki:>8,}        {nki:>8,}   ({nki - oki:+,})")

    if write:
        nd.to_csv(OUT, index=False, encoding="utf-8")
        pd.DataFrame(rec).to_csv(REPORT, index=False, encoding="utf-8")
        print(f"\n  wrote {OUT.relative_to(ROOT)}  and  {REPORT.relative_to(ROOT)}")
        print("  transcripts_all.csv is UNCHANGED.")
    else:
        print("\n  dry run. Pass --write to produce transcripts_all_v2.csv.")


if __name__ == "__main__":
    main()
