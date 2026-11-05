#!/usr/bin/env python3
"""
finetuning/data.py - the parts of the training data path that do not depend on the label set.

    python finetuning/data.py            # report the corpus, the context windows and a split

WHAT THIS MODULE DELIBERATELY DOES NOT DO

It does not read, validate or encode labels, and it does not build a model, a loss, a training
loop or a checkpointing scheme. Which axes the classifier predicts, and over what value sets, is
now recorded in D-70, but the experimental data/evaluation protocol and training alignment
are deferred by DW (D-79). `finetuning/config.json` remains historical. The default below still
reads the pre-exclusion baseline; running this old demonstration does not create an adopted
current experimental split or establish readiness to train.
Checkpointing and resumption belong to whichever training arm is chosen (multi-head classifier or
structured generation, methodology section 4.2, not committed in advance), and building a second
one here would have to be thrown away.

What is fixed, and therefore buildable now, is everything about rows, students and context
windows. That is the data contract in CLAUDE.md, and it does not move when the label set does.

THE TWO INVARIANTS THIS MODULE EXISTS TO ENFORCE

1. **Splits are grouped by student.** One student contributes over five hundred rows. A random
   row-level split leaks that student's turns across train, validation and test, and the reported
   accuracy then measures memorisation of one person's phrasing. Grouping by student also groups
   by conversation, since a conversation has one student.

2. **A gap in `Segment Order` is a context boundary.** `Segment Order` is not renumbered after the
   unpaired-row exclusion, so a gap marks a row that was removed rather than a row that never
   existed. Reading across a gap would put two turns next to each other in the model's window that
   were not adjacent in the conversation. Context therefore stops at a gap, and the example simply
   has less context.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "transcripts_all.csv"

# The data contract, CLAUDE.md. Only these are required: they are the fields this module reads.
KEY_COLUMNS = ["Student ID", "Conversation ID", "Segment Order", "Student Prompt", "LLM Response"]

# The label columns as the corpus file currently spells them. They are NOT required and are NOT
# read. They are named only so the report can say which of them a given file happens to carry;
# the names date from before the September revisions and do not describe the current instrument.
LEGACY_LABEL_COLUMNS = ["Student delegation", "Student cognition",
                        "AI delegation", "AI response type"]

DEFAULT_CONTEXT = 3          # "two to three preceding turns", Lin et al. 2024, methodology 3.2.2
DEFAULT_SEED = 42


# ----------------------------------------------------------------------------- loading

def load_corpus(path: Path | None = None) -> pd.DataFrame:
    """Read the corpus and put it in reading order. Label columns are neither required nor read."""
    path = Path(path) if path else CORPUS
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return normalise(df)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Sort into reading order and give the frame a fresh 0..n-1 index.

    Every function below addresses rows by position, so a frame that has kept the index of some
    larger frame it was sliced from would raise a KeyError. Call this on any subset before
    iterating it, or just pass the subset to `iter_examples`, which calls it.
    """
    out = df.copy()
    out["Segment Order"] = out["Segment Order"].astype(int)
    return out.sort_values(["Conversation ID", "Segment Order"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- context

def context_rows(df: pd.DataFrame, index: int, n_context: int = DEFAULT_CONTEXT) -> list[int]:
    """Indices of the turns that precede `index` and are genuinely adjacent to it.

    `df` must be normalised (see `normalise`): `index` is a position, not a label.

    Walks backwards while each step is exactly one lower in `Segment Order` and in the same
    conversation. Stops at a gap, at the start of the conversation's record, or once `n_context`
    turns have been collected. Returns oldest first.
    """
    out: list[int] = []
    conv = df.at[index, "Conversation ID"]
    expected = df.at[index, "Segment Order"] - 1
    j = index - 1
    while j >= 0 and len(out) < n_context:
        if df.at[j, "Conversation ID"] != conv:
            break
        if df.at[j, "Segment Order"] != expected:
            break                       # a removed row sits between these two turns
        out.append(j)
        expected -= 1
        j -= 1
    return list(reversed(out))


def iter_examples(df: pd.DataFrame, n_context: int = DEFAULT_CONTEXT):
    """Yield one dict per row: the row's identity, its turn pair, and its context turns.

    No label is read or emitted. When the label set settles, the consumer adds the label fields
    it needs; nothing in this function has to change.
    """
    df = normalise(df)
    first_seg = df.groupby("Conversation ID")["Segment Order"].min().to_dict()
    for i in range(len(df)):
        ctx = context_rows(df, i, n_context)
        yield {
            "student_id": df.at[i, "Student ID"],
            "conversation_id": df.at[i, "Conversation ID"],
            "segment_order": int(df.at[i, "Segment Order"]),
            "student_prompt": df.at[i, "Student Prompt"],
            "llm_response": df.at[i, "LLM Response"],
            "context": [
                {"segment_order": int(df.at[j, "Segment Order"]),
                 "student_prompt": df.at[j, "Student Prompt"],
                 "llm_response": df.at[j, "LLM Response"]}
                for j in ctx
            ],
            "context_available": len(ctx),
            "context_short_because": _why_short(df, i, ctx, n_context, first_seg),
        }


def _why_short(df: pd.DataFrame, index: int, ctx: list[int], n_context: int,
               first_seg: dict) -> str | None:
    """Why this row has fewer than `n_context` preceding turns. None when it does not.

    Three different situations are collapsed into "short window" if they are not separated, and
    they mean different things for what the missing context is:

      "conversation opens"   the walk reached the first turn of the conversation and that turn is
                             Segment Order 1, so there is genuinely nothing earlier.
      "record starts late"   the walk reached the earliest row this corpus holds for the
                             conversation, but that row is not Segment Order 1. Earlier turns
                             existed and are not in the corpus. The context is missing, not absent.
      "gap"                  a row between two retained turns was removed by the unpaired-row
                             exclusion, so the walk stopped rather than read across it.
    """
    if len(ctx) >= n_context:
        return None
    conv = df.at[index, "Conversation ID"]
    oldest = ctx[0] if ctx else index
    j = oldest - 1
    if j >= 0 and df.at[j, "Conversation ID"] == conv:
        return "gap"
    return "conversation opens" if first_seg[conv] == 1 else "record starts late"


# ----------------------------------------------------------------------------- splitting

def grouped_split(df: pd.DataFrame,
                  fractions: dict[str, float] | None = None,
                  seed: int = DEFAULT_SEED) -> dict[str, pd.DataFrame]:
    """Split by student, never by row.

    Student sizes are extremely skewed, so students are assigned largest first, each to whichever
    split is furthest below its target row count. That keeps the row proportions close to the
    targets without ever putting one student on both sides.

    **This is a balanced deterministic split, not a randomised repeatable one, and it is not the
    experimental split.** Two limits follow from the algorithm and must not be forgotten when it is
    reported. First, `seed` only permutes students of equal size, so a student whose row count is
    unique never moves: the largest student lands in the largest split under every seed. Repeated
    runs therefore do not give independent splits and must not be used as replicates. Second, this
    splits whatever frame it is given. Run over the whole unlabelled corpus it is a demonstration
    that the grouping holds, nothing more. The experimental split has to be formed over the
    labelled subset, has to keep students already seen during development out of the independent test
    set (they may still train), and its student roster has to be written down and kept rather than
    recomputed.
    """
    fractions = fractions or {"train": 0.7, "val": 0.15, "test": 0.15}
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {total}")

    counts = df.groupby("Student ID").size().to_dict()
    rng = __import__("random").Random(seed)
    students = sorted(counts, key=lambda s: (-counts[s], s))
    # shuffle within equal-size runs only, so the seed matters but the size order does not move
    i = 0
    while i < len(students):
        j = i
        while j + 1 < len(students) and counts[students[j + 1]] == counts[students[i]]:
            j += 1
        block = students[i:j + 1]
        rng.shuffle(block)
        students[i:j + 1] = block
        i = j + 1

    n = len(df)
    targets = {k: v * n for k, v in fractions.items()}
    assigned: dict[str, list[str]] = {k: [] for k in fractions}
    filled = {k: 0 for k in fractions}
    for s in students:
        k = max(fractions, key=lambda k: targets[k] - filled[k])
        assigned[k].append(s)
        filled[k] += counts[s]

    return {k: df[df["Student ID"].isin(v)].reset_index(drop=True) for k, v in assigned.items()}


def check_no_leak(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any student or conversation appears in more than one split."""
    for field in ("Student ID", "Conversation ID"):
        seen: dict[str, str] = {}
        for name, part in splits.items():
            for v in part[field].unique():
                if v in seen:
                    raise AssertionError(
                        f"{field} {v!r} is in both {seen[v]!r} and {name!r}")
                seen[v] = name


# ----------------------------------------------------------------------------- report

def main() -> None:
    df = load_corpus()
    print(f"  corpus            {len(df)} rows, {df['Conversation ID'].nunique()} conversations, "
          f"{df['Student ID'].nunique()} students")
    sizes = df.groupby("Student ID").size()
    print(f"  rows per student  median {int(sizes.median())}, mean {sizes.mean():.1f}, "
          f"max {sizes.max()}  (the max is why the split is grouped)")

    avail: dict[int, int] = {}
    why: dict[str, int] = {}
    for ex in iter_examples(df):
        avail[ex["context_available"]] = avail.get(ex["context_available"], 0) + 1
        r = ex["context_short_because"]
        if r:
            why[r] = why.get(r, 0) + 1
    print(f"\n  preceding turns available, window n={DEFAULT_CONTEXT}")
    for k in sorted(avail):
        print(f"    {k}   {avail[k]:>5}")
    print("  why a window is short")
    for k in sorted(why):
        print(f"    {k:<20} {why[k]:>5}")
    print("    (the three are different things: an opening has no earlier turn, a late-starting")
    print("     record is missing one, and a gap is a turn this corpus removed)")

    splits = grouped_split(df)
    check_no_leak(splits)
    print(f"\n  grouped split over the WHOLE UNLABELLED CORPUS, seed {DEFAULT_SEED}")
    print("    a demonstration that the grouping holds, not the experimental split: that one is")
    print("    formed over the labelled subset, must keep students already used in development out")
    print("    of the independent test set, and must have its roster written down.")
    print("    See grouped_split's docstring on why repeated seeds are not replicates.")
    for k, part in splits.items():
        print(f"    {k:<6} {len(part):>5} rows  {len(part)/len(df):>6.1%}  "
              f"{part['Student ID'].nunique():>4} students  "
              f"{part['Conversation ID'].nunique():>5} conversations")
    print("    no student and no conversation appears in more than one split")


if __name__ == "__main__":
    main()
