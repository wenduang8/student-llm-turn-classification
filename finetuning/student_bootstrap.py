#!/usr/bin/env python3
"""
finetuning/student_bootstrap.py - score saved out-of-fold predictions against reference labels and give
intervals that resample students, not interactions.

    python finetuning/student_bootstrap.py --reference data/labels/dw150-reviewed-v9.jsonl \
        --pred C1=data/results/v9-cv-20261022/oof-predictions/predictions-C1-at0.5-oof.jsonl \
        --pred E=data/results/v9-encoder-20261025/oof-predictions/predictions-E-oof.jsonl \
        --compare E-C1 --out <result.json>

WHY. One student can contribute many interactions (one student contributes 40 of 150), and interactions of the
same student are not independent. Tests and intervals that treat each interaction as independent can be
too narrow. Here each bootstrap replicate draws the students with replacement and keeps all interactions
of every drawn student; the metric, and the paired difference between two prediction sets on the same
replicate, are recomputed. 95% percentile intervals.

WHAT IS SCORED, per dimension (null labels and null predictions are excluded, as in the CV reports):
  exact: single-select equality; multi-select equality of the whole set
  macro_f1: mean F1 over codes with support in the reference (single- and multi-select)
  micro_f1: multi-select only; TP, FP and FN summed over all codes, including ND and none of the four applies
  pooled_f1: macro_f1 for Student interaction type, micro_f1 otherwise (the F1 used by the D-164 rule)
  paired counts (first right and second wrong / the reverse) with an exact two-sided binomial p
The student of an interaction is the part of its conversation ID before the first underscore.

It reads saved files only: no model, no training, no transcript text. It produces no label.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent), str(_HERE.parent / "coding")]
import labels as L  # noqa: E402

AXES = L.AXES
POOLED = {a: ("macro_f1" if a in L.SINGLE else "micro_f1") for a in AXES}


def student(cid: str) -> str:
    return cid.split("_", 1)[0]


def as_set(axis, value):
    if value is None:
        return None
    return frozenset([value]) if axis in L.SINGLE else frozenset(value)


def load_reference(path):
    """The reviewed reader for a reviewed file, the human-record reader otherwise."""
    first = next(json.loads(l) for l in Path(path).read_text(encoding="utf-8-sig").splitlines() if l.strip())
    items = L.read_reviewed_direct_jsonl(path) if first.get("record_kind") == L.DEV_REVIEW_KIND else L.read_direct(path)
    return {i.key: {a: as_set(a, i.labels[a]) for a in AXES} for i in items}


def load_predictions(path):
    recs = L.read_predictions(path)
    runs = {r["run_id"] for r in recs}
    if len(runs) != 1:
        raise ValueError(f"{path}: expected one run, found {sorted(runs)}")
    return {(r["conversation_id"], r["segment_order"]): {a: as_set(a, r["labels"][a]) for a in AXES} for r in recs}, runs.pop()


def axis_arrays(axis, keys, ref, preds):
    """Rows usable for this axis, with a 0/1 matrix per source: rows x codes."""
    codes = list(L.DIRECT_CODES[axis])
    rows = [k for k in keys if ref[k][axis] is not None and all(p[k][axis] is not None for p in preds)]
    def mat(src):
        return np.array([[c in src[k][axis] for c in codes] for k in rows], dtype=bool)
    return rows, mat(ref), [mat(p) for p in preds]


def metrics(y, p, single: bool, w=None):
    """y, p: rows x codes booleans; w: row weights (bootstrap multiplicities)."""
    w = np.ones(len(y)) if w is None else w
    out = {"exact": float((w * (y == p).all(1)).sum() / w.sum())}
    tp = (w[:, None] * (y & p)).sum(0); fp = (w[:, None] * (~y & p)).sum(0); fn = (w[:, None] * (y & ~p)).sum(0)
    support = (w[:, None] * y).sum(0) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    out["macro_f1"] = float(f1[support].mean()) if support.any() else float("nan")
    if not single:
        out["micro_f1"] = float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-12))
    return out


def binom_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def run(reference, preds: dict, compare: list[tuple[str, str]], n_boot=10000, seed=20261026):
    ref = load_reference(reference)
    loaded = {name: load_predictions(path) for name, path in preds.items()}
    keys = sorted(ref)
    for name, (p, _) in loaded.items():
        missing = set(keys) - set(p)
        if missing:
            raise ValueError(f"{name}: {len(missing)} reference interactions have no prediction")
    rng = np.random.default_rng(seed)
    result = {"reference": str(reference), "predictions": {n: {"path": str(preds[n]), "run_id": r} for n, (_, r) in loaded.items()},
              "bootstrap": {"replicates": n_boot, "seed": seed, "unit": "student", "interval": "95% percentile"},
              "axes": {}}
    names = list(loaded)
    for axis in AXES:
        single = axis in L.SINGLE
        rows, y, ps = axis_arrays(axis, keys, ref, [loaded[n][0] for n in names])
        studs = sorted({student(k[0]) for k in rows})
        idx = {s: [i for i, k in enumerate(rows) if student(k[0]) == s] for s in studs}
        counts = np.zeros((n_boot, len(rows)))
        for b in range(n_boot):
            drawn = rng.integers(0, len(studs), len(studs))
            for d in drawn:
                counts[b, idx[studs[d]]] += 1
        ax = {"rows": len(rows), "students": len(studs), "models": {}, "comparisons": {}}
        point = {n: metrics(y, p, single) for n, p in zip(names, ps)}
        boots = {n: [metrics(y, p, single, counts[b]) for b in range(n_boot)] for n, p in zip(names, ps)}
        for n in names:
            ax["models"][n] = {m: {"value": v, "ci": [float(np.nanpercentile([bb[m] for bb in boots[n]], 2.5)),
                                                      float(np.nanpercentile([bb[m] for bb in boots[n]], 97.5))]}
                               for m, v in point[n].items()}
        for a_name, b_name in compare:
            ia, ib = names.index(a_name), names.index(b_name)
            ra, rb = (y == ps[ia]).all(1), (y == ps[ib]).all(1)
            first_only, second_only = int((ra & ~rb).sum()), int((~ra & rb).sum())
            comp = {"paired": {f"{a_name} right, {b_name} wrong": first_only, f"{b_name} right, {a_name} wrong": second_only,
                               "both right": int((ra & rb).sum()), "both wrong": int((~ra & ~rb).sum()),
                               "exact_binomial_p_per_interaction": binom_two_sided(first_only, second_only)}}
            for m in point[a_name]:
                d = np.array([boots[a_name][b][m] - boots[b_name][b][m] for b in range(n_boot)])
                comp[m] = {"difference": point[a_name][m] - point[b_name][m],
                           "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
            comp["pooled_f1"] = dict(comp[POOLED[axis]], metric=POOLED[axis])
            ax["comparisons"][f"{a_name} - {b_name}"] = comp
        result["axes"][axis] = ax
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--pred", action="append", required=True, help="NAME=path to a model_prediction JSONL")
    ap.add_argument("--compare", action="append", default=[], help="A-B: difference A minus B")
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    preds = dict(x.split("=", 1) for x in a.pred)
    compare = [tuple(x.split("-", 1)) for x in a.compare]
    res = run(a.reference, preds, compare, a.replicates, a.seed)
    if a.out:
        out = Path(a.out)
        if out.exists():
            sys.exit(f"REFUSED: {out} exists")
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for axis, ax in res["axes"].items():
        print(f"\n{axis} ({ax['rows']} rows, {ax['students']} students)")
        for n, m in ax["models"].items():
            print("  " + n + ": " + "; ".join(f"{k} {v['value']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" for k, v in m.items()))
        for c, comp in ax["comparisons"].items():
            pr = comp["paired"]
            print(f"  {c}: paired {list(pr.values())[:4]}, p per interaction {pr['exact_binomial_p_per_interaction']:.3g}; "
                  f"exact diff {comp['exact']['difference']:+.3f} [{comp['exact']['ci'][0]:+.3f}, {comp['exact']['ci'][1]:+.3f}]; "
                  f"{comp['pooled_f1']['metric']} diff {comp['pooled_f1']['difference']:+.3f} [{comp['pooled_f1']['ci'][0]:+.3f}, {comp['pooled_f1']['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
