#!/usr/bin/env python3
"""
comparison/score.py - score prediction files against the D-190 labels, for all 485 and for each part (150, 135, 200).

    python comparison/score.py --out-dir data/results/llm-terra-20261028/scores-all \
        --pred LLM-A=data/results/llm-terra-20261028/predictions/predictions-LLM-A-all.jsonl \
        --pred LLM-B=data/results/llm-terra-20261028/predictions/predictions-LLM-B-all.jsonl \
        --pred C1=data/results/dw485-c1-20261028/oof-predictions/predictions-C1-485-oof.jsonl \
        --compare LLM-B:LLM-A --compare LLM-A:C1 --compare LLM-B:C1 \
        --note "C1 here is the D-186 trial, trained on labels that differ from D-190 in 26 cells"

    --group 335=135+200 scores the 135 and 200 together as one part named 335 (D-215: the 150 coded by both coders,
    the 335 added interactions coded by the researcher alone).

Measures (spec section 5):
  * per code, simple agreement C / max(A, B): A interactions with the code in the reference, B in the model, C in both
    (the supervisor's measure, D-179); for Solve-Provide and Explain also the absent side (the same formula on the
    interactions without the code);
  * the C1 report's measures from finetuning/student_bootstrap.py: exact-set rate (accuracy for the student
    interaction type), macro and micro F1, paired counts;
  * intervals: 95% percentile, 10,000 replicates that resample students (seed 20261026), and paired differences on the
    same replicates.
Rows: an axis is scored on the interactions where the reference and every listed model have a value (null labels,
truncated responses and failed calls are left out, and the counts say how many). No interaction text is read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, ROOT / "coding", ROOT / "finetuning", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import labels as L  # noqa: E402
import student_bootstrap as SB  # noqa: E402

ABSENT_SIDE = {"AI response type": ("Solve-Provide", "Explain")}
TARGETS = (("student interaction type", "Guided generation"), ("AI response type", "Evaluate"),
           ("AI delegation", "hand-back: directive"), ("AI delegation", "hand-back: asks the student to supply"))
SHORT = {"Providing material / responding": "PMR", "Information retrieval": "IR", "Task automation": "TA",
         "Guided generation": "GG", "tooling or environment": "tooling", "conversation management": "CM",
         "hand-back: directive": "directive", "hand-back: asks the student to supply": "asks",
         "hand-back: Socratic": "Socratic", "none of the four applies": "none", "Solve-Provide": "SP",
         "Correct-Debug": "CD"}


def part_files(labels_path: Path, parts: dict, out_dir: Path) -> dict:
    """Write each part's reference lines (byte copies of the label file's lines) for student_bootstrap."""
    lines = [l for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    d = out_dir / "part-labels"
    d.mkdir(parents=True, exist_ok=True)
    for part in sorted(set(parts.values())):
        keep = [l for l in lines if parts[(json.loads(l)["conversation_id"], json.loads(l)["segment_order"])] == part]
        path = d / f"{labels_path.stem}-part-{part}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(keep) + "\n")
        out[part] = path
    return out


def student_draws(students: list[str], n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros((n_boot, len(students)))
    for b in range(n_boot):
        m[b] = np.bincount(rng.integers(0, len(students), len(students)), minlength=len(students))
    return m


def _agree(c, a, b):
    den = np.maximum(a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, c / np.where(den > 0, den, 1), np.nan)


def per_code(ref: dict, preds: dict, keys: list, compare: list, n_boot: int, seed: int) -> dict:
    names = list(preds)
    students = sorted({SB.student(k[0]) for k in keys})
    s_index = {s: i for i, s in enumerate(students)}
    draws = student_draws(students, n_boot, seed)
    out = {}
    for axis in L.AXES:
        rows = [k for k in keys if ref[k][axis] is not None and all(preds[n][k][axis] is not None for n in names)]
        excluded = {n: sum(1 for k in keys if ref[k][axis] is not None and preds[n][k][axis] is None) for n in names}
        w = draws[:, [s_index[SB.student(k[0])] for k in rows]]
        ax = {"rows": len(rows), "students": len({SB.student(k[0]) for k in rows}),
              "excluded_for_missing_prediction": excluded, "codes": {}}
        sides = [(c, False) for c in L.DIRECT_CODES[axis]] + [(c, True) for c in ABSENT_SIDE.get(axis, ())]
        for code, absent in sides:
            y = np.array([(code in ref[k][axis]) != absent for k in rows], dtype=float)
            entry = {"reference": int(y.sum()), "models": {}, "comparisons": {}}
            boot = {}
            for n in names:
                p = np.array([(code in preds[n][k][axis]) != absent for k in rows], dtype=float)
                a, b, c = y.sum(), p.sum(), (y * p).sum()
                ab = _agree(w @ (y * p), w @ y, w @ p)
                boot[n] = ab
                point = float(_agree(np.array(c), np.array(a), np.array(b)))
                entry["models"][n] = {"B": int(b), "C": int(c), "agreement": None if np.isnan(point) else point,
                                      "ci": [float(np.nanpercentile(ab, 2.5)), float(np.nanpercentile(ab, 97.5))]
                                      if not np.all(np.isnan(ab)) else None}
            for m1, m2 in compare:
                p1, p2 = entry["models"][m1]["agreement"], entry["models"][m2]["agreement"]
                d = boot[m1] - boot[m2]
                entry["comparisons"][f"{m1} - {m2}"] = {
                    "difference": None if p1 is None or p2 is None else p1 - p2,
                    "ci": None if np.all(np.isnan(d)) else [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
            ax["codes"][code + (" (absent)" if absent else "")] = entry
        out[axis] = ax
    return out


def pct(x):
    return "" if x is None else f"{100 * x:.0f}%"


def tables(result: dict, names: list, compare: list, note: str) -> str:
    out = [f"# Scores against {result['reference']}", ""]
    if note:
        out += [note, ""]
    bs = result["bootstrap"]
    out += [f"Per code: C / max(A, B), with B and C, on the rows scored for that dimension. Intervals: students "
            f"resampled, {bs['replicates']:,} replicates, seed {bs['seed']}.", ""]
    for part, pr in result["parts"].items():
        out += [f"## {part}", ""]
        out += ["| Dimension | Code | A | " + " | ".join(f"{n}: C/max (B, C)" for n in names) + " |",
                "|---|---|---|" + "---|" * len(names)]
        for axis, ax in pr["per_code"].items():
            for code, e in ax["codes"].items():
                cells = [f"{pct(e['models'][n]['agreement'])} ({e['models'][n]['B']}, {e['models'][n]['C']})" for n in names]
                out.append(f"| {axis} ({ax['rows']} rows) | {code} | {e['reference']} | " + " | ".join(cells) + " |")
        out += ["", "Differences on the four target codes (percentage points, 95% interval):", "",
                "| Code | " + " | ".join(f"{a} - {b}" for a, b in compare) + " |", "|---|" + "---|" * len(compare)]
        for axis, code in TARGETS:
            e = pr["per_code"][axis]["codes"][code]
            cells = []
            for a, b in compare:
                c = e["comparisons"][f"{a} - {b}"]
                cells.append("" if c["difference"] is None else
                             f"{100 * c['difference']:+.0f} [{100 * c['ci'][0]:+.0f}, {100 * c['ci'][1]:+.0f}]")
            out.append(f"| {SHORT.get(code, code)} | " + " | ".join(cells) + " |")
        out += ["", "Exact-set rate (accuracy for the student interaction type), 95% interval:", "",
                "| Dimension | " + " | ".join(names) + " | " + " | ".join(f"{a} - {b}" for a, b in compare) + " |",
                "|---|" + "---|" * (len(names) + len(compare))]
        for axis, ax in pr["bootstrap"]["axes"].items():
            m = ax["models"]
            cells = [f"{100 * m[n]['exact']['value']:.0f}% [{100 * m[n]['exact']['ci'][0]:.0f}, {100 * m[n]['exact']['ci'][1]:.0f}]" for n in names]
            for a, b in compare:
                c = ax["comparisons"][f"{a} - {b}"]["exact"]
                cells.append(f"{100 * c['difference']:+.0f} [{100 * c['ci'][0]:+.0f}, {100 * c['ci'][1]:+.0f}]")
            out.append(f"| {axis} ({ax['rows']}) | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    import llm_compare as LC
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, default=LC.LABELS)
    ap.add_argument("--pred", action="append", required=True, help="NAME=path to a model_prediction JSONL")
    ap.add_argument("--compare", action="append", default=[], help="NAME1:NAME2, difference NAME1 minus NAME2")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--note", default="")
    ap.add_argument("--group", action="append", default=[],
                    help="NAME=PART+PART: score these parts together as one part, e.g. 335=135+200")
    a = ap.parse_args(argv)
    if a.out_dir.exists():
        sys.exit(f"REFUSED: {a.out_dir} exists")
    preds_paths = dict(x.split("=", 1) for x in a.pred)
    compare = [tuple(x.split(":", 1)) for x in a.compare]
    names = list(preds_paths)
    for pair in compare:
        if any(n not in names for n in pair):
            sys.exit(f"REFUSED: unknown name in --compare {pair}")
    parts = LC.part_map()
    for g in a.group:
        name, _, members = g.partition("=")
        members = set(members.split("+"))
        if not name or len(members) < 2 or members - set(parts.values()):
            sys.exit(f"REFUSED: --group {g}: give NAME=PART+PART with existing parts {sorted(set(parts.values()))}")
        parts = {k: (name if v in members else v) for k, v in parts.items()}
    ref = SB.load_reference(a.reference)
    if set(ref) != set(parts):
        sys.exit("REFUSED: the reference and the part files do not hold the same interactions")
    preds = {}
    for n, p in preds_paths.items():
        loaded, _ = SB.load_predictions(p)
        missing = set(ref) - set(loaded)
        if missing:
            sys.exit(f"REFUSED: {n}: {len(missing)} reference interactions have no prediction")
        preds[n] = loaded
    a.out_dir.mkdir(parents=True)
    pfiles = part_files(a.reference, parts, a.out_dir)
    result = {"reference": L.rel(a.reference), "reference_sha256": L.sha256_file(a.reference),
              "predictions": {n: {"path": L.rel(p), "sha256": L.sha256_file(p)} for n, p in preds_paths.items()},
              "note": a.note, "groups": a.group,
              "bootstrap": {"replicates": a.replicates, "seed": a.seed, "unit": "student"}, "parts": {}}
    order = [p for p in ("150", "135", "200") if p in pfiles] + sorted(set(pfiles) - {"150", "135", "200"})
    scopes = [("485", sorted(ref), a.reference)] + [(p, sorted(k for k in ref if parts[k] == p), pfiles[p])
                                                     for p in order]
    for name, keys, ref_path in scopes:
        boot = SB.run(ref_path, preds_paths, compare, a.replicates, a.seed)
        result["parts"][name] = {"interactions": len(keys),
                                 "per_code": per_code(ref, preds, keys, compare, a.replicates, a.seed),
                                 "bootstrap": boot}
        print(f"scored {name}: {len(keys)} interactions", flush=True)
    (a.out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out_dir / "tables.md").write_text(tables(result, names, compare, a.note), encoding="utf-8")
    print(f"written {L.rel(a.out_dir / 'metrics.json')} and tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
