#!/usr/bin/env python3
"""Checks and student-grouped 5-fold cross-validation for the v9 labels (D-106 preliminary
development; not a formal evaluation). Every score comes from finetuning/evaluation.py; the
multi-select micro-F1 and Hamming accuracy are derived from that module's per-category TP/FP/FN/TN.
Prints only counts and verdicts, never interaction text.

    .venv/bin/python -I data/results/v9-cv-20261022/cv.py step1          # checks, no training
    .venv/bin/python -I data/results/v9-cv-20261022/cv.py plan           # folds, inner splits, configs, estimate
    .venv/bin/python -I data/results/v9-cv-20261022/cv.py run            # train/resume every run; resumable
    .venv/bin/python -I data/results/v9-cv-20261022/cv.py report         # metrics, summary, confusion matrices
    .venv/bin/python -I data/results/v9-cv-20261022/cv.py smoke          # the same pipeline on 10 interactions

Learning curve (added 2026-10-22 at DW's request, after the CV above had finished; the CV code is unchanged):
    .venv/bin/python -I data/results/v9-cv-20261022/cv.py plan-lc        # half-size training folds, same test folds
    .venv/bin/python -I data/results/v9-cv-20261022/cv.py run-lc         # C1 only (inner run, thresholds, final run); resumable
    .venv/bin/python -I data/results/v9-cv-20261022/cv.py report-lc      # C1 on about half the data vs the full-data CV
    .venv/bin/python -I data/results/v9-cv-20261022/cv.py smoke-lc       # the same on the 10-interaction smoke data
  lc      the test folds are exactly the CV's. Each training fold is cut to about half its interactions by
          whole students: students in a seeded random order (random.Random(20261017 + 7000 + fold)) join
          while the sample holds fewer than 50% of the training fold's interactions and would stay within
          55%. Everything else is C1 as run in the CV: 8 epochs, head lr, class weights, whole responses,
          the inner-holdout thresholds (the holdout is drawn from the smaller sample), seed 20261017.

Design (fixed before any CV run; written to progress.md and plan.json):
  folds   students sorted by ID, shuffled with random.Random(20261017), then stably sorted by number
          of interactions, largest first; in that order each student joins the fold with the fewest
          interactions so far (ties: lower fold number). 5 folds; each interaction is predicted once,
          by the model of the fold that holds it out. (Amended before any CV run: without the size
          sort the 40-interaction student came late and the folds held 29/23/24/21/53 interactions.)
  C0      the current configuration (config-v9.json): the training settings of B.
  C1      C0 plus: 8 epochs (C0: 4); heads at learning rate 1e-3 (adapter stays at 2e-4); class
          weights training_part_balanced_v1 (train_multihead.class_weights); response kept whole:
          response_cap 3072 >= the longest response (3028 tokens) and max_tokens 7168 = 6144 + 1024,
          so the student-side budget (max_tokens minus the response reservation) is unchanged;
          per-axis thresholds for the three multi-select axes, chosen on an inner holdout of about
          20% of the training fold's interactions (whole students, GS.make_split with seed
          20261017 + fold, max 30%): a C1 model trained on the rest of the training fold predicts the
          holdout; per axis the threshold in 0.05..0.95 (step 0.05) with the highest exact-set match
          wins (ties: higher micro-F1, then nearer 0.5, then lower). The final C1 model is then trained
          on the whole training fold and its test-fold probabilities are decoded with those thresholds
          (labels.decode_scores per axis). The test fold is never used for any choice.
  baseline majority code / most frequent complete code set of each training fold.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # this copy is in finetuning/
sys.path.insert(0, str(ROOT / "finetuning"))
import evaluation as EV  # noqa: E402
import grouped_split as GS  # noqa: E402
import labels as L  # noqa: E402

HERE = ROOT / "data" / "results" / "v9-cv-20261022"              # outputs, where the original ran
PY = ROOT / ".venv" / "bin" / "python"
TRAIN = ROOT / "finetuning" / "train_multihead.py"
PREPARE = ROOT / "finetuning" / "prepare_data.py"
CORPUS = ROOT / "data" / "processed" / "dw150-context.csv"
V9 = ROOT / "data" / "training" / "dw150-reviewed-v9"
V9_LABELS = ROOT / "data" / "labels" / "dw150-reviewed-v9.jsonl"
V9_SPLIT = ROOT / "data" / "splits" / "dw150-reviewed-v9-20261021.json"
B_RUN = ROOT / "data" / "runs" / "v9-20261021-qwen3-0.6b-lora"
AB_REPORTS = ROOT / "data" / "results" / "v9-run-20261021" / "reports"
BASE_CONFIG = ROOT / "data" / "results" / "v9-run-20261021" / "config-v9.json"

SEED, FOLDS = 20261017, 5
C1_EPOCHS, C1_HEAD_LR, C1_RESPONSE_CAP, C1_MAX_TOKENS = 8, 1e-3, 3072, 7168
LC_FRACTION, LC_CAP_FRACTION, LC_SEED = 0.5, 0.55, SEED + 7000
GRID = [round(0.05 * i, 2) for i in range(1, 20)]
INNER_TARGET, INNER_MAX = 0.2, 0.3
MULTI = tuple(a for a in L.AXES if a not in L.SINGLE)
CV_PURPOSE = ("Cross-validation test fold (student-grouped 5-fold, cv-20261022): held-out students used only to produce "
              "this fold's out-of-fold predictions, scored once against the v9 labels; the per-epoch loss on them is "
              "descriptive only. Not used to choose epochs, checkpoints, thresholds, input policy or models.")
INNER_PURPOSE = ("Inner holdout of a cross-validation training fold (cv-20261022, configuration C1): about 20% of the "
                 "training fold's interactions (whole students). The CV script uses its predictions only to choose C1's "
                 "per-axis multi-select thresholds; the trainer selects nothing on it; the test fold is not in this data.")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def roundtrip(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False))


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def key_text(key) -> str:
    return GS.key_text(*key)


# ----------------------------------------------------------------------------- baselines and metrics

def majority_labels(train: list) -> dict:
    """Majority baseline from training labels: the most frequent code (single-select) or the most
    frequent complete code set (multi-select); ties go to the earlier code / code set in schema order."""
    out = {}
    for axis in L.AXES:
        codes = L.DIRECT_CODES[axis]
        given = [v for v in (t[axis] for t in train) if v is not None]
        if axis in L.SINGLE:
            counts = Counter(given)
            out[axis] = min(codes, key=lambda c: (-counts.get(c, 0), codes.index(c)))
        else:
            counts = Counter(tuple(sorted(v, key=codes.index)) for v in given)
            out[axis] = list(min(counts.items(), key=lambda kv: (-kv[1], [codes.index(c) for c in kv[0]]))[0])
    if L.check_labels(out, allow_null=False):
        raise AssertionError(f"invalid majority labels: {L.check_labels(out, allow_null=False)}")
    return out


def per_code_majority_labels(train: list) -> dict:
    """The trainer's label-prior reading aid (train_multihead.prior_reference): majority code, or
    each code positive in at least half of the labelled training rows, then labels.decode_scores."""
    scores = {}
    for axis in L.AXES:
        given = [v for v in (t[axis] for t in train) if v is not None]
        codes = L.DIRECT_CODES[axis]
        scores[axis] = ([sum(v == c for v in given) / len(given) for c in codes] if axis in L.SINGLE
                        else [sum(c in v for v in given) / len(given) for c in codes])
    return L.decode_scores(scores, 0.5)


def multi_extra(axis_report: dict) -> dict:
    """Exact-set match (the module's row accuracy), label-level micro-F1 and Hamming accuracy, both
    from the module's per-category counts: micro-F1 = 2*sum TP / (2*sum TP + sum FP + sum FN);
    Hamming accuracy = (sum TP + sum TN) / (n * number of codes)."""
    per, n = axis_report["per_category"], axis_report["n"]
    tp, fp, fn, tn = (sum(m[k] for m in per.values()) for k in ("tp", "fp", "fn", "tn"))
    return {"n": n, "codes": len(per), "exact_set_match": axis_report["row_accuracy"],
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "hamming_accuracy": (tp + tn) / (n * len(per)) if n else None}


def headline(report: dict) -> dict:
    out = {}
    for axis, r in report["axes"].items():
        out[axis] = {"n": r["n"], "accuracy": r["row_accuracy"], "macro_f1": r["macro_f1"]}
        if r["selection"] == "multiple":
            out[axis].update({k: v for k, v in multi_extra(r).items() if k in ("micro_f1", "hamming_accuracy")})
    out["all_axis_exact_match"] = {k: report["all_axis_exact_match"][k] for k in ("n", "matches", "rate")}
    return out


def truncation(report_path: Path, part_of: dict) -> dict:
    rows = json.loads(report_path.read_text(encoding="utf-8"))["per_interaction"]
    out = {}
    for part in ("train", "validation", None):
        chosen = [v for k, v in rows.items() if part is None or part_of[k] == part]
        cut = [v["response_tokens_omitted"] for v in chosen if v["response_tokens_omitted"] > 0]
        pcut = [v["prompt_tokens_omitted"] for v in chosen if v["prompt_tokens_omitted"] > 0]
        out[part or "all"] = {
            "interactions": len(chosen), "responses_clipped": len(cut),
            "response_tokens_omitted_mean_over_clipped": sum(cut) / len(cut) if cut else 0,
            "response_tokens_omitted_mean_over_all": sum(cut) / len(chosen),
            "response_tokens_omitted_max": max(cut, default=0),
            "longest_response_tokens": max(v["response_tokens"] for v in chosen),
            "prompts_clipped": len(pcut), "prompt_tokens_omitted_mean_over_clipped": sum(pcut) / len(pcut) if pcut else 0}
    return out


def step1() -> int:
    split = json.loads(V9_SPLIT.read_text(encoding="utf-8"))
    part_of = {k: p for p in split["parts"] for k in split["parts"][p]["interactions"]}
    refs, _ = EV.references_from_prepared(V9)
    _, keyed = GS.load_split(V9_SPLIT, V9)
    train = [refs[k] for k, p in keyed.items() if p == "train"]
    val = [k for k, p in keyed.items() if p == "validation"]
    base = {"majority": majority_labels(train), "per_code_majority": per_code_majority_labels(train)}
    trainer_prior, _ = EV.predictions_from_file(B_RUN / "outputs" / "predictions-validation-prior-reference.jsonl")
    reports = {name: roundtrip(EV.evaluate(refs, {k: labels for k in val}, val)) for name, labels in base.items()}
    for name in ("A_vs_v9", "B_vs_v9"):
        reports[name] = json.loads((AB_REPORTS / f"{name}.json").read_text(encoding="utf-8"))
    out = {"what": "step 1 checks without training (2026-10-22); existing 117/33 split, v9 reference",
           "response_truncation_current_policy": truncation(B_RUN / "inputs-report.json", part_of),
           "policy": json.loads((B_RUN / "inputs-report.json").read_text(encoding="utf-8"))["policy"],
           "baseline_labels": base,
           "per_code_majority_equals_trainer_prior_aid": all(trainer_prior[k] == base["per_code_majority"] for k in val),
           "headline": {name: headline(r) for name, r in reports.items()},
           "definitions": {"majority": " ".join(majority_labels.__doc__.split()),
                           "multi_select_extra": " ".join(multi_extra.__doc__.split())}}
    write_json(HERE / "step1-checks.json", out)
    print(json.dumps({"prior_aid_reproduced": out["per_code_majority_equals_trainer_prior_aid"]}))
    return 0


# ----------------------------------------------------------------------------- the CV workspace

@dataclass
class Space:
    name: str                    # cv-20261022, or cv-20261022-smoke
    prepared: Path               # prepared data holding every interaction of this CV
    labels: Path                 # the review JSONL that prepared data was made from
    results: Path
    epochs: dict = field(default_factory=lambda: {"C0": 4, "C1": C1_EPOCHS})
    sample_fraction: float | None = None    # learning curve: share of each training fold kept (whole students)
    run_c0: bool = True                     # the learning curve runs C1 only

    @property
    def splits(self) -> Path:
        return ROOT / "data" / "splits" / self.name

    @property
    def training(self) -> Path:
        return ROOT / "data" / "training" / self.name

    @property
    def runs(self) -> Path:
        return ROOT / "data" / "runs" / self.name

    def fold_split(self, k: int) -> Path:
        return self.splits / f"fold{k}.json"

    def inner_split(self, k: int) -> Path:
        return self.splits / f"fold{k}-inner.json"

    def trainpart(self, k: int) -> Path:
        return self.training / f"fold{k}-trainpart"

    def fold_prepared(self, k: int) -> Path:
        """Prepared data holding exactly fold k's train and validation interactions."""
        return self.prepared if self.sample_fraction is None else self.training / f"fold{k}-sample"

    def config(self, name: str) -> Path:
        return self.results / "configs" / f"{name}.json"

    def run_dir(self, name: str) -> Path:
        return self.runs / name

    def c1_predictions(self, k: int) -> Path:
        return self.results / "predictions" / f"predictions-C1-fold{k}.jsonl"

    def thresholds(self, k: int) -> Path:
        return self.results / "thresholds" / f"fold{k}.json"


REAL = Space("cv-20261022", V9, V9_LABELS, HERE)
LC = Space("cv-20261022-lc", V9, V9_LABELS, HERE / "lc", sample_fraction=LC_FRACTION, run_c0=False)


def assign_folds(examples: list) -> dict:
    rows = Counter(e["student_id"] for e in examples)
    students = sorted(rows)
    random.Random(SEED).shuffle(students)
    students.sort(key=lambda s: -rows[s])        # largest first; the seeded shuffle orders equal sizes
    load, fold_of = [0] * FOLDS, {}
    for s in students:
        k = min(range(FOLDS), key=lambda i: (load[i], i))
        fold_of[s] = k + 1
        load[k] += rows[s]
    return {"fold_of": fold_of, "shuffled_student_order": students}


def part_record(chosen: list) -> dict:
    out = {"students": sorted({e["student_id"] for e in chosen}),
           "conversations": sorted({e["conversation_id"] for e in chosen}),
           "interactions": [key_text((e["conversation_id"], e["segment_order"]))
                            for e in sorted(chosen, key=lambda e: (e["conversation_id"], e["segment_order"]))]}
    out["counts"] = {k: len(v) for k, v in out.items()}
    return out


def choose_sample(space: Space, k: int, train_fold: list) -> set:
    """Learning curve: the students kept from fold k's training fold. Students in a seeded random order
    join while the sample holds fewer than sample_fraction of the training fold's interactions and would
    stay within LC_CAP_FRACTION of them (a large student that would overshoot is skipped)."""
    rows = Counter(e["student_id"] for e in train_fold)
    n = len(train_fold)
    target, cap = round(space.sample_fraction * n), round(LC_CAP_FRACTION * n)
    order = sorted(rows)
    random.Random(LC_SEED + k).shuffle(order)
    chosen, filled = set(), 0
    for s in order:
        if filled < target and filled + rows[s] <= cap:
            chosen.add(s)
            filled += rows[s]
    if not 0.4 * n <= filled <= LC_CAP_FRACTION * n + 1:
        raise ValueError(f"fold {k}: the sample holds {filled} of {n} interactions, not about {space.sample_fraction:.0%}")
    return chosen


def subset_prepared(source: Path, keep: set, labels_path: Path, out: Path, what: str) -> None:
    """Prepared data for `keep` only, from a byte-identical subset of the review file `source`."""
    lines = [line for line in source.read_bytes().split(b"\n") if line.strip()]
    chosen = [line for line in lines if (lambda r: (r["conversation_id"], r["segment_order"]))(json.loads(line)) in keep]
    if len(chosen) != len(keep):
        raise ValueError(f"{what}: {len(keep)} interactions but {len(chosen)} review records")
    data = b"\n".join(chosen) + b"\n"
    if labels_path.exists():
        if labels_path.read_bytes() != data:
            raise ValueError(f"{labels_path} exists with different content")
    else:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_bytes(data)
    if not (out / "manifest.json").exists():
        r = subprocess.run([str(PY), str(PREPARE), "--reviewed-direct", str(labels_path), "--corpus", str(CORPUS),
                            "--output-dir", str(out), "--context", "3"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"prepare_data failed for {what}: {r.stderr[-500:]}")


def prepare_sample(space: Space, k: int, examples: list, folds: dict) -> None:
    """Learning curve: prepared data of the sampled training students plus fold k's test students."""
    test = [e for e in examples if folds["fold_of"][e["student_id"]] == k]
    train_fold = [e for e in examples if folds["fold_of"][e["student_id"]] != k]
    kept = choose_sample(space, k, train_fold)
    keep = {(e["conversation_id"], e["segment_order"]) for e in test} | {
        (e["conversation_id"], e["segment_order"]) for e in train_fold if e["student_id"] in kept}
    subset_prepared(space.labels, keep, space.training / f"fold{k}-sample-labels.jsonl", space.fold_prepared(k),
                    f"fold {k} sample")


def write_fold_split(space: Space, k: int, examples: list, folds: dict) -> dict:
    test = [e for e in examples if folds["fold_of"][e["student_id"]] == k]
    train_fold = [e for e in examples if folds["fold_of"][e["student_id"]] != k]
    sampled = None if space.sample_fraction is None else choose_sample(space, k, train_fold)
    train = train_fold if sampled is None else [e for e in train_fold if e["student_id"] in sampled]
    parts = {"train": part_record(train), "validation": part_record(test)}
    overlap = {f: len(set(parts["train"][f]) & set(parts["validation"][f])) for f in ("students", "conversations")}
    if any(overlap.values()) or len(train_fold) + len(test) != len(examples):
        raise AssertionError(f"fold {k} split checks failed: {overlap}")
    source = ("the other folds" if sampled is None else
              f"a student-grouped sample (about {space.sample_fraction:.0%}) of the other folds")
    record = {"record_kind": GS.RECORD_KIND,
              "status": f"{space.name} fold {k} of {FOLDS}: train = {source}, validation = this test fold; "
                        "provisional development protocol (D-106), not an adopted experimental protocol",
              "validation_purpose": CV_PURPOSE, "created_utc": utc(),
              "split_module_sha256": L.sha256_file(GS.__file__), "procedure_script_sha256": L.sha256_file(__file__),
              "prepared": GS.prepared_fingerprint(space.fold_prepared(k)), "seed": SEED, "fold": k, "folds": FOLDS,
              "procedure": "students sorted by Student ID, shuffled with random.Random(seed), then stably sorted by "
                           "number of interactions, largest first; in that order each student joins the fold with the "
                           "fewest interactions so far (ties: lower fold number)",
              "parts": parts, "label_support": {"train": GS.label_support(train), "validation": GS.label_support(test)},
              "checks": {"student_overlap": overlap["students"], "conversation_overlap": overlap["conversations"],
                         "every_interaction_once": True, "shuffled_student_order": folds["shuffled_student_order"]}}
    if sampled is not None:
        rows = Counter(e["student_id"] for e in train_fold)
        largest = max(sorted(rows), key=lambda s: rows[s])
        record["sample"] = {"fraction": space.sample_fraction, "cap_fraction": LC_CAP_FRACTION, "seed": LC_SEED + k,
                            "training_fold_interactions": len(train_fold), "training_fold_students": len(rows),
                            "sampled_interactions": len(train), "sampled_students": len(sampled),
                            "share": len(train) / len(train_fold),
                            "largest_student_in_training_fold": {"student": largest, "interactions": rows[largest],
                                                                 "in_sample": largest in sampled},
                            "rule": "seeded random student order; a student joins while the sample is below the target "
                                    "and would stay within the cap"}
    path = space.fold_split(k)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["parts"] != parts or old["prepared"] != record["prepared"]:
            raise ValueError(f"{path} exists with a different roster or prepared data")
        return old
    write_json(path, record)
    return record


def write_trainpart(space: Space, k: int, fold_record: dict) -> None:
    """Prepared data holding only the training fold (from a byte-identical subset of the review file)."""
    out = space.trainpart(k)
    labels_path = space.training / f"fold{k}-trainpart-labels.jsonl"
    keep = {L.parse_key(t) for t in fold_record["parts"]["train"]["interactions"]}
    lines = [line for line in space.labels.read_bytes().split(b"\n") if line.strip()]
    chosen = [line for line in lines if (lambda r: (r["conversation_id"], r["segment_order"]))(json.loads(line)) in keep]
    if len(chosen) != len(keep):
        raise ValueError(f"fold {k}: {len(keep)} training interactions but {len(chosen)} review records")
    data = b"\n".join(chosen) + b"\n"
    if labels_path.exists():
        if labels_path.read_bytes() != data:
            raise ValueError(f"{labels_path} exists with different content")
    else:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_bytes(data)
    if not (out / "manifest.json").exists():
        r = subprocess.run([str(PY), str(PREPARE), "--reviewed-direct", str(labels_path), "--corpus", str(CORPUS),
                            "--output-dir", str(out), "--context", "3"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"prepare_data failed for fold {k}: {r.stderr[-500:]}")


def write_inner_split(space: Space, k: int) -> dict:
    path = space.inner_split(k)
    if not path.exists():
        n = len(GS.load_examples(space.trainpart(k)))
        GS.write_split(space.trainpart(k), path, SEED + k, max(1, round(INNER_TARGET * n)), max(1, round(INNER_MAX * n)),
                       INNER_PURPOSE, f"{space.name} fold {k} inner holdout for C1 thresholds (D-106 development)")
    return json.loads(path.read_text(encoding="utf-8"))


def make_configs(space: Space, k: int) -> dict:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    note = (f"{space.name}: student-grouped 5-fold cross-validation (D-106 preliminary development; not a formal "
            "evaluation). Labels: the v9 AI-assisted review (D-144); not independent coding; not joint adjudication.")
    if space.sample_fraction is not None:
        note += (f" Learning-curve variant: each training fold is cut to about {space.sample_fraction:.0%} of its "
                 f"interactions by whole students (seed {LC_SEED} + fold); the test folds are the CV's.")
    c0 = copy.deepcopy(base)
    c0["status"] = f"CV C0 fold {k} ({space.name}): the current configuration (config-v9.json training settings). {note}"
    c0["data"].update(prepared_dir=L.rel(space.fold_prepared(k)), split_file=L.rel(space.fold_split(k)))
    c0["validation"]["purpose"] = CV_PURPOSE
    c0["schedule"]["epochs"] = space.epochs["C0"]
    c1 = copy.deepcopy(c0)
    c1["status"] = (f"CV C1 fold {k} ({space.name}): C0 plus {space.epochs['C1']} epochs, head_lr {C1_HEAD_LR}, class "
                    f"weights, response kept whole (response_cap {C1_RESPONSE_CAP}, max_tokens {C1_MAX_TOKENS}); "
                    f"per-axis multi-select thresholds are applied by the CV script. {note}")
    c1["schedule"]["epochs"] = space.epochs["C1"]
    c1["optim"]["head_lr"] = C1_HEAD_LR
    c1["loss"]["class_weights"] = "training_part_balanced_v1"
    c1["input"].update(response_cap=C1_RESPONSE_CAP, max_tokens=C1_MAX_TOKENS)
    inner = copy.deepcopy(c1)
    inner["status"] = c1["status"].replace(f"CV C1 fold {k}", f"CV C1 fold {k} inner run")
    inner["data"].update(prepared_dir=L.rel(space.trainpart(k)), split_file=L.rel(space.inner_split(k)))
    inner["validation"]["purpose"] = INNER_PURPOSE
    configs = ({f"C0-fold{k}": c0} if space.run_c0 else {}) | {f"C1-fold{k}-inner": inner, f"C1-fold{k}": c1}
    for name, cfg in configs.items():
        path = space.config(name)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != cfg:
                raise ValueError(f"{path} exists with different content")
        else:
            write_json(path, cfg)
    return configs


def time_estimate(space: Space, folds: list) -> dict:
    """From B's run on this Mac: seconds per training interaction-pass, per validation interaction and
    per end-of-run prediction, plus a fixed start-up allowance per run."""
    log = [json.loads(line) for _, line in L.jsonl_lines(B_RUN / "train_log.jsonl")]
    kept = {}
    for e in log:
        if e["event"] == "step":
            kept[e["step"]] = e
    per_train = sum(s["seconds"] for s in kept.values()) / sum(s["examples"] for s in kept.values())
    t = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp()  # noqa: E731
    gaps, last_step = [], None
    for e in log:
        if e["event"] == "step":
            last_step = e
        elif e["event"] == "epoch_end" and last_step is not None:
            gaps.append(t(e["utc"]) - t(last_step["utc"]))
    per_val = sorted(gaps)[len(gaps) // 2] / 33
    out_event = next(e for e in log if e["event"] == "outputs_written")
    last_epoch = [e for e in log if e["event"] == "epoch_end"][-1]
    per_out = (t(out_event["utc"]) - t(last_epoch["utc"])) / (117 + 33 + 33)
    startup = 40.0
    runs = {}
    for f in folds:
        k, n_train, n_test, n_inner_train, n_inner = f["fold"], f["train"], f["test"], f["inner_train"], f["inner_holdout"]
        for name, ntr, nva, ep in (([(f"C0-fold{k}", n_train, n_test, space.epochs["C0"])] if space.run_c0 else [])
                                   + [(f"C1-fold{k}-inner", n_inner_train, n_inner, space.epochs["C1"]),
                                      (f"C1-fold{k}", n_train, n_test, space.epochs["C1"])]):
            runs[name] = ntr * ep * per_train + nva * ep * per_val + (ntr + 2 * nva) * per_out + startup
    total = sum(runs.values())
    return {"seconds_per_training_interaction_pass": round(per_train, 2), "seconds_per_validation_interaction": round(per_val, 2),
            "seconds_per_output_prediction": round(per_out, 2), "startup_seconds_per_run": startup,
            "per_run_seconds": {k: round(v) for k, v in runs.items()}, "total_hours": round(total / 3600, 1),
            "basis": "B (data/runs/v9-20261021-qwen3-0.6b-lora) on this Mac; C1's longer response reservation only "
                     "lengthens the one interaction whose response was clipped"}


def check_inputs(space: Space) -> dict:
    """Configs pass the trainer's checks; under C1 no response is clipped and every interaction's
    student side (tokens up to the student pool position) is identical to C0's."""
    import model_inputs as MI
    import train_multihead as T
    c0 = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))        # C0's input policy is the base config's
    c1, inner = (json.loads(space.config(n).read_text(encoding="utf-8")) for n in ("C1-fold1", "C1-fold1-inner"))
    for cfg in (c0, c1, inner):
        T.check_config(cfg)
    encode = T.tokenizer_encoder(T.model_snapshot(c0["model"])[0])
    policies = {n: MI.InputPolicy.from_config(cfg["input"]) for n, cfg in (("C0", c0), ("C1", c1))}
    costs = {n: p.check_with(encode) for n, p in policies.items()}
    data = T.load_data(space.fold_prepared(1), space.fold_split(1))
    built = {n: T.build_inputs(data, encode, p)[0] for n, p in policies.items()}
    student_same = sum(built["C0"][k]["input_ids"][:built["C0"][k]["student_pool"] + 1]
                       == built["C1"][k]["input_ids"][:built["C1"][k]["student_pool"] + 1] for k in built["C0"])
    out = {"student_side_budget": {n: c["student_side_budget"] for n, c in costs.items()},
           "responses_clipped": {n: sum(b["report"]["response_tokens_omitted"] > 0 for b in v.values()) for n, v in built.items()},
           "max_tokens_in_any_input": {n: max(b["report"]["tokens"] for b in v.values()) for n, v in built.items()},
           "interactions": len(built["C0"]), "student_side_identical": student_same,
           "inputs_changed_by_c1": sum(built["C0"][k]["input_ids"] != built["C1"][k]["input_ids"] for k in built["C0"])}
    ok = (out["student_side_budget"]["C0"] == out["student_side_budget"]["C1"] and out["responses_clipped"]["C1"] == 0
          and student_same == len(built["C0"]))
    if not ok:
        raise AssertionError(f"C1 input change is not what the plan states: {out}")
    return out


def plan(space: Space) -> dict:
    space.results.mkdir(parents=True, exist_ok=True)
    examples = GS.load_examples(space.prepared)
    folds = assign_folds(examples)
    summary = []
    for k in range(1, FOLDS + 1):
        if space.sample_fraction is not None:
            prepare_sample(space, k, examples, folds)
        record = write_fold_split(space, k, examples, folds)
        write_trainpart(space, k, record)
        inner = write_inner_split(space, k)
        make_configs(space, k)
        summary.append({"fold": k, "train": record["parts"]["train"]["counts"]["interactions"],
                        "test": record["parts"]["validation"]["counts"]["interactions"],
                        "test_students": record["parts"]["validation"]["counts"]["students"],
                        "inner_train": inner["parts"]["train"]["counts"]["interactions"],
                        "inner_holdout": inner["parts"]["validation"]["counts"]["interactions"],
                        "inner_holdout_students": inner["parts"]["validation"]["counts"]["students"],
                        **({"sample": record["sample"]} if "sample" in record else {})})
    covered = sorted(t for k in range(1, FOLDS + 1)
                     for t in json.loads(space.fold_split(k).read_text(encoding="utf-8"))["parts"]["validation"]["interactions"])
    if covered != sorted(key_text((e["conversation_id"], e["segment_order"])) for e in examples):
        raise AssertionError("the test folds do not cover every interaction exactly once")
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout  # noqa: E731
    diff = git("diff", "HEAD", "--", "finetuning")
    record = {"what": f"{space.name} plan (fixed before the runs)", "created_utc": utc(), "seed": SEED, "folds": summary,
              "test_fold_rosters": {k: json.loads(space.fold_split(k).read_text(encoding="utf-8"))["parts"]["validation"]
                                    for k in range(1, FOLDS + 1)},
              "c1": {"epochs": space.epochs["C1"], "head_lr": C1_HEAD_LR, "class_weights": "training_part_balanced_v1",
                     "response_cap": C1_RESPONSE_CAP, "max_tokens": C1_MAX_TOKENS, "threshold_grid": GRID,
                     "inner_target_fraction": INNER_TARGET, "inner_max_fraction": INNER_MAX},
              "c0_epochs": space.epochs["C0"], "c1_input_check": check_inputs(space),
              "code": {"head": git("rev-parse", "HEAD").strip(), "uncommitted_finetuning_diff_sha256":
                       hashlib.sha256(diff.encode()).hexdigest() if diff else None},
              "estimate": time_estimate(space, summary), "procedure_script_sha256": L.sha256_file(__file__)}
    if diff:
        (space.results / "code-diff-vs-head.patch").write_text(diff, encoding="utf-8")
    write_json(space.results / "plan.json", record)
    print(json.dumps({"folds": summary, "estimate_hours": record["estimate"]["total_hours"],
                      "per_run_seconds": record["estimate"]["per_run_seconds"],
                      "c1_input_check": record["c1_input_check"]}, indent=1))
    return record


# ----------------------------------------------------------------------------- running

def train_or_resume(space: Space, name: str, status: dict) -> None:
    run_dir, log = space.run_dir(name), space.runs / f"{name}.out"
    if (run_dir / "outputs" / "summary.json").exists():
        return
    space.runs.mkdir(parents=True, exist_ok=True)
    command = ([str(PY), str(TRAIN), "resume", "--run-dir", str(run_dir)] if (run_dir / "run.json").exists()
               else [str(PY), str(TRAIN), "train", "--config", str(space.config(name)), "--run-dir", str(run_dir)])
    started = time.time()
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n== {utc()} {command[2]} ==\n")
        out.flush()
        code = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    status["runs"].setdefault(name, []).append({"command": command[2], "exit": code, "seconds": round(time.time() - started),
                                               "finished_utc": utc()})
    write_json(space.results / "run-status.json", status)
    if code != 0 or not (run_dir / "outputs" / "summary.json").exists():
        raise RuntimeError(f"{name}: trainer exited {code}; see {log}")


def read_scores(path: Path) -> dict:
    out = {}
    for _, line in L.jsonl_lines(path):
        r = json.loads(line)
        if r["codes"] != {a: L.DIRECT_CODES[a] for a in L.AXES}:
            raise ValueError(f"{path.name}: scores use a different code order")
        out[(r["conversation_id"], r["segment_order"])] = r
    return out


def decode(probabilities: dict, thresholds: dict) -> dict:
    labels = L.decode_scores(probabilities, 0.5)
    for axis, t in thresholds.items():
        labels[axis] = L.decode_scores(probabilities, t)[axis]
    errors = L.check_labels(labels, allow_null=False)
    if errors:
        raise AssertionError("; ".join(errors))
    return labels


def choose_thresholds(space: Space, k: int) -> dict:
    run = space.run_dir(f"C1-fold{k}-inner")
    scores = read_scores(run / "outputs" / "scores-validation.jsonl")
    refs, _ = EV.references_from_prepared(space.trainpart(k))
    _, part_of = GS.load_split(space.inner_split(k), space.trainpart(k))
    holdout = [key for key, p in part_of.items() if p == "validation"]
    if set(scores) != set(holdout):
        raise ValueError(f"fold {k}: inner predictions do not cover exactly the inner holdout")
    chosen, table = {}, {}
    for axis in MULTI:
        rows = []
        for t in GRID:
            preds = {key: decode(scores[key]["probabilities"], {axis: t}) for key in holdout}
            r = EV.evaluate(refs, preds, holdout)["axes"][axis]
            rows.append({"threshold": t, **{m: multi_extra(r)[m] for m in ("n", "exact_set_match", "micro_f1")}})
        valid = [x for x in rows if x["n"]]
        best = max(valid, key=lambda x: (x["exact_set_match"], x["micro_f1"] or 0.0, -abs(x["threshold"] - 0.5),
                                         -x["threshold"])) if valid else {"threshold": 0.5}
        chosen[axis], table[axis] = best["threshold"], rows
    record = {"fold": k, "thresholds": chosen, "inner_run": L.rel(run), "inner_holdout": len(holdout),
              "rule": "max exact-set match on the inner holdout; ties: higher micro-F1, nearer 0.5, lower",
              "grid_results": table, "chosen_utc": utc()}
    write_json(space.thresholds(k), record)
    return record


def write_c1_predictions(space: Space, k: int, thresholds: dict) -> Path:
    run = space.run_dir(f"C1-fold{k}")
    out = space.c1_predictions(k)
    scores = read_scores(run / "outputs" / "scores-validation.jsonl")
    first = next(iter(scores.values()))
    route = f"{first['model_route']}+axis-thresholds:{canonical_sha(thresholds)[:12]}"
    info = {"run_id": first["run_id"], "model_route": route, "config_sha256": first["config_sha256"],
            "dataset_manifest_sha256": first["dataset_manifest_sha256"]}
    records = [L.prediction_record(cid, seg, decode(r["probabilities"], thresholds), info)
               for (cid, seg), r in sorted(scores.items())]
    if out.exists():
        if [json.loads(line) for _, line in L.jsonl_lines(out)] != roundtrip(records):
            raise ValueError(f"{out} exists with different predictions")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    L.write_predictions(out, records)
    return out


def run_all(space: Space) -> int:
    path = space.results / "run-status.json"
    status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": {}, "thresholds": {}}
    status["driver_started_utc"] = utc()
    write_json(path, status)
    for k in range(1, FOLDS + 1):
        if space.run_c0:
            train_or_resume(space, f"C0-fold{k}", status)
        train_or_resume(space, f"C1-fold{k}-inner", status)
        chosen = (json.loads(space.thresholds(k).read_text(encoding="utf-8")) if space.thresholds(k).exists()
                  else choose_thresholds(space, k))
        status["thresholds"][str(k)] = chosen["thresholds"]
        train_or_resume(space, f"C1-fold{k}", status)
        write_c1_predictions(space, k, chosen["thresholds"])
        status[f"fold{k}_done_utc"] = utc()
        write_json(path, status)
    status["driver_finished_utc"] = utc()
    write_json(path, status)
    return 0


# ----------------------------------------------------------------------------- report

def mcnemar(b: int, c: int) -> float | None:
    """Exact two-sided McNemar p (binomial on the discordant pairs); descriptive, uncorrected."""
    n = b + c
    if not n:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def correct(axis: str, ref, pred):
    return None if ref is None or pred is None else L.canonical(axis, ref) == L.canonical(axis, pred)


def report(space: Space) -> int:
    refs, fp = EV.references_from_prepared(space.prepared)
    statement = EV.statements(fp)["validation"]
    folds = {}
    for k in range(1, FOLDS + 1):
        _, part_of = GS.load_split(space.fold_split(k), space.prepared)
        folds[k] = {"test": [key for key, p in part_of.items() if p == "validation"],
                    "train": [key for key, p in part_of.items() if p == "train"]}
    sources = {"C0": {k: space.run_dir(f"C0-fold{k}") / "outputs" / "predictions-validation.jsonl" for k in folds},
               "C1": {k: space.c1_predictions(k) for k in folds}}
    per_fold, pooled_preds, files = {"C0": {}, "C1": {}, "majority": {}, "per_code_majority": {}}, {}, {}
    for config, paths in sources.items():
        merged = {}
        for k, path in paths.items():
            rep = roundtrip(EV.run(space.prepared, path, f"{config} fold {k}", f"{space.name} {config} fold {k}",
                                   space.fold_split(k), "validation"))
            per_fold[config][k] = rep
            preds, _ = EV.predictions_from_file(path)
            merged.update(preds)
        pooled_preds[config] = merged
    for name, fn in (("majority", majority_labels), ("per_code_majority", per_code_majority_labels)):
        merged = {}
        for k, f in folds.items():
            labels = fn([refs[key] for key in f["train"]])
            preds = {key: labels for key in f["test"]}
            per_fold[name][k] = roundtrip(EV.evaluate(refs, preds, f["test"]))
            merged.update(preds)
        pooled_preds[name] = merged
    all_keys = sorted(refs)
    pooled = {name: roundtrip(EV.evaluate(refs, preds, all_keys)) for name, preds in pooled_preds.items()}
    out_dir = space.results / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rep in pooled.items():
        full = {"title": f"{space.name} pooled out-of-fold: {name}", "statement": statement,
                "note": "pooled over the 5 test folds; every interaction predicted once", "reference": fp,
                "predictions": {"source": name}, "selection": {"part": "all interactions, out of fold"}, **rep}
        for suffix, text in ((".json", json.dumps(full, ensure_ascii=False, indent=1) + "\n"), (".md", EV.markdown(full))):
            path = out_dir / f"pooled-{name}{suffix}"
            path.write_text(text, encoding="utf-8")
            files[L.rel(path)] = L.sha256_file(path)

    def table(name):
        rows = {k: headline(r) for k, r in per_fold[name].items()}
        metrics = {}
        for axis in list(L.AXES) + ["all_axis_exact_match"]:
            keys = ["rate"] if axis == "all_axis_exact_match" else [m for m in ("accuracy", "macro_f1", "micro_f1",
                                                                                "hamming_accuracy") if m in rows[1][axis]]
            metrics[axis] = {}
            for m in keys:
                vals = [rows[k][axis][m] for k in sorted(rows)]
                present = [v for v in vals if v is not None]
                mean = sum(present) / len(present) if present else None
                sd = (math.sqrt(sum((v - mean) ** 2 for v in present) / (len(present) - 1))
                      if len(present) > 1 else None)
                metrics[axis][m] = {"per_fold": vals, "mean": mean, "sd": sd,
                                    "pooled": headline(pooled[name])[axis][m]}
        return metrics
    tables = {name: table(name) for name in per_fold}
    paired = {}
    for axis in L.AXES:
        b = sum(1 for key in all_keys if correct(axis, refs[key][axis], pooled_preds["C1"][key][axis]) is True
                and correct(axis, refs[key][axis], pooled_preds["C0"][key][axis]) is False)
        c = sum(1 for key in all_keys if correct(axis, refs[key][axis], pooled_preds["C1"][key][axis]) is False
                and correct(axis, refs[key][axis], pooled_preds["C0"][key][axis]) is True)
        paired[axis] = {"C1_right_C0_wrong": b, "C0_right_C1_wrong": c, "mcnemar_exact_p": mcnemar(b, c)}
    thresholds = {k: json.loads(space.thresholds(k).read_text(encoding="utf-8"))["thresholds"] for k in folds if space.thresholds(k).exists()}
    metrics = {"what": f"{space.name}: C0 vs C1, student-grouped 5-fold CV over all interactions, v9 reference; D-106 "
                       "preliminary development; development validation, not final RQ3 performance, not human reliability",
               "statement": statement, "tables": tables, "paired_C1_vs_C0_pooled": paired, "thresholds": thresholds,
               "fold_sizes": {k: len(f["test"]) for k, f in folds.items()},
               "per_fold_reports": per_fold, "pooled_reports": pooled, "files": files}
    write_json(space.results / "cv-metrics.json", metrics)
    (space.results / "cv-summary.md").write_text(summary_md(space, metrics), encoding="utf-8")
    print(json.dumps({name: {axis: {m: (round(v["mean"], 3) if v["mean"] is not None else None, round(v["pooled"], 3)
                                       if v["pooled"] is not None else None) for m, v in t[axis].items()}
                             for axis in t} for name, t in tables.items()}, ensure_ascii=False))
    return 0


def f3(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def summary_md(space: Space, m: dict) -> str:
    t, names = m["tables"], ("majority", "C0", "C1")
    out = [f"# {space.name}: C0 与 C1 的学生分组 5 折交叉验证（参照 v9；D-106 初步开发，非正式评价）", "",
           f"性质：{m['statement']}", "",
           f"每条互动只被预测一次（由不含它的那一折的模型预测）。各折测试集大小：{m['fold_sizes']}。", "",
           "## 各维度结果（逐折均值 ± 标准差；括号内为 150 条合并计算）", "",
           "| 维度 | 指标 | 多数类基线 | C0 | C1 |", "|---|---|---|---|---|"]
    for axis in list(L.AXES) + ["all_axis_exact_match"]:
        for metric in t["C0"][axis]:
            cells = [f"{f3(t[n][axis][metric]['mean'])} ± {f3(t[n][axis][metric]['sd'])} ({f3(t[n][axis][metric]['pooled'])})"
                     for n in names]
            label = "四维全对" if axis == "all_axis_exact_match" else axis
            out.append(f"| {label} | {metric} | " + " | ".join(cells) + " |")
    out += ["", "## 逐折结果（准确率；四维全对为比例）", "", "| 维度 | 配置 | " + " | ".join(f"折{k}" for k in m["fold_sizes"]) + " |",
            "|---|---|" + "---|" * len(m["fold_sizes"])]
    for axis in list(L.AXES) + ["all_axis_exact_match"]:
        metric = "rate" if axis == "all_axis_exact_match" else "accuracy"
        for n in names:
            out.append(f"| {'四维全对' if metric == 'rate' else axis} | {n} | "
                       + " | ".join(f3(v) for v in t[n][axis][metric]["per_fold"]) + " |")
    out += ["", "## C1 对 C0 的逐条配对比较（150 条合并；精确 McNemar，未校正，仅供参考）", "",
            "| 维度 | C1 对、C0 错 | C0 对、C1 错 | p |", "|---|---|---|---|"]
    for axis, p in m["paired_C1_vs_C0_pooled"].items():
        out.append(f"| {axis} | {p['C1_right_C0_wrong']} | {p['C0_right_C1_wrong']} | {f3(p['mcnemar_exact_p'])} |")
    out += ["", "## C1 各折选出的阈值（内部留出集选出，未看测试折）", ""]
    out += [f"- 折{k}：" + "，".join(f"{a} {v}" for a, v in th.items()) for k, th in m["thresholds"].items()]
    out += ["", "学生互动类型的混淆矩阵、各类别 precision/recall 和四维组合列联表见 reports/pooled-*.md 与 cv-metrics.json。", "",
            "## 文件", ""] + [f"- `{k}` sha256 `{v}`" for k, v in m["files"].items()] + [""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- learning curve

def fold_table(per_fold: dict, pooled_report: dict) -> dict:
    """Per axis and metric: the per-fold values, their mean and SD, and the pooled (150-row) value."""
    rows = {k: headline(r) for k, r in per_fold.items()}
    pooled = headline(pooled_report)
    out = {}
    for axis in list(L.AXES) + ["all_axis_exact_match"]:
        first = rows[min(rows)][axis]
        keys = ["rate"] if axis == "all_axis_exact_match" else [m for m in ("accuracy", "macro_f1", "micro_f1", "hamming_accuracy") if m in first]
        out[axis] = {}
        for m in keys:
            vals = [rows[k][axis][m] for k in sorted(rows)]
            present = [v for v in vals if v is not None]
            mean = sum(present) / len(present) if present else None
            sd = math.sqrt(sum((v - mean) ** 2 for v in present) / (len(present) - 1)) if len(present) > 1 else None
            out[axis][m] = {"per_fold": vals, "mean": mean, "sd": sd, "pooled": pooled[axis][m]}
    return out


def lc_report(lc: Space, full: Space) -> int:
    """C1 on about half of each training fold (lc) against the same C1 pipeline and C0 on the whole
    training fold (full), on the same test folds, every interaction predicted once."""
    refs, fp = EV.references_from_prepared(full.prepared)
    statement = EV.statements(fp)["validation"]
    all_keys = sorted(refs)
    for k in range(1, FOLDS + 1):
        if (json.loads(lc.fold_split(k).read_text(encoding="utf-8"))["parts"]["validation"]
                != json.loads(full.fold_split(k).read_text(encoding="utf-8"))["parts"]["validation"]):
            raise ValueError(f"fold {k}: the learning-curve test fold is not the full CV's test fold")
    variants = {
        "C0 full": (full, lambda k: full.run_dir(f"C0-fold{k}") / "outputs" / "predictions-validation.jsonl"),
        "C1 full": (full, lambda k: full.c1_predictions(k)),
        "C1 full @0.5": (full, lambda k: full.run_dir(f"C1-fold{k}") / "outputs" / "predictions-validation.jsonl"),
        "C1 half": (lc, lambda k: lc.c1_predictions(k)),
        "C1 half @0.5": (lc, lambda k: lc.run_dir(f"C1-fold{k}") / "outputs" / "predictions-validation.jsonl"),
    }
    per_fold, preds_of = {}, {}
    for name, (space, path_of) in variants.items():
        per_fold[name], preds_of[name] = {}, {}
        for k in range(1, FOLDS + 1):
            path = path_of(k)
            per_fold[name][k] = roundtrip(EV.run(space.fold_prepared(k), path, name, f"{lc.name} {name} fold {k}",
                                                 space.fold_split(k), "validation"))
            preds, _ = EV.predictions_from_file(path)
            preds_of[name].update(preds)
        if sorted(preds_of[name]) != all_keys:
            raise ValueError(f"{name}: the five test folds do not give one prediction for each interaction")
    pooled = {name: roundtrip(EV.evaluate(refs, preds, all_keys)) for name, preds in preds_of.items()}
    tables = {name: fold_table(per_fold[name], pooled[name]) for name in variants}
    paired = {}
    for axis in L.AXES:
        right = {n: {k for k in all_keys if correct(axis, refs[k][axis], preds_of[n][k][axis]) is True}
                 for n in ("C1 half", "C1 full")}
        b, c = len(right["C1 half"] - right["C1 full"]), len(right["C1 full"] - right["C1 half"])
        paired[axis] = {"half_right_full_wrong": b, "full_right_half_wrong": c, "mcnemar_exact_p": mcnemar(b, c)}
    samples = {k: json.loads(lc.fold_split(k).read_text(encoding="utf-8"))["sample"] for k in range(1, FOLDS + 1)}
    thresholds = {n: {k: json.loads(sp.thresholds(k).read_text(encoding="utf-8"))["thresholds"] for k in range(1, FOLDS + 1)}
                  for n, sp in (("C1 full", full), ("C1 half", lc))}
    # Declared before the runs: a fold whose sample is more than half one student is also shown left out.
    dominated = [k for k, s in samples.items() if s["largest_student_in_training_fold"]["in_sample"]
                 and s["largest_student_in_training_fold"]["interactions"] > 0.5 * s["sampled_interactions"]]
    keep = [i for i in range(FOLDS) if (i + 1) not in dominated]
    sensitivity = {name: {axis: {m: (sum(v["per_fold"][i] for i in keep) / len(keep)
                                     if all(v["per_fold"][i] is not None for i in keep) else None)
                                 for m, v in ms.items()} for axis, ms in tables[name].items()}
                   for name in ("C1 full", "C1 half", "C1 full @0.5", "C1 half @0.5")}
    out_dir = lc.results / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("C1 half", "C1 half @0.5"):
        rep = {"title": f"{lc.name} pooled out-of-fold: {name}", "statement": statement,
               "note": "pooled over the 5 test folds; every interaction predicted once; training folds cut to about half",
               "reference": fp, "predictions": {"source": name}, "selection": {"part": "all interactions, out of fold"},
               **pooled[name]}
        for suffix, text in ((".json", json.dumps(rep, ensure_ascii=False, indent=1) + "\n"), (".md", EV.markdown(rep))):
            path = out_dir / f"pooled-{name.replace(' ', '_').replace('@', 'at')}{suffix}"
            path.write_text(text, encoding="utf-8")
            files[L.rel(path)] = L.sha256_file(path)
    metrics = {"what": f"{lc.name}: C1 on about half of each training fold vs the full-data CV (same test folds, v9 "
                       "reference); D-106 preliminary development; development validation, not final RQ3 performance, "
                       "not human reliability", "statement": statement, "samples": samples, "tables": tables,
               "paired_half_vs_full_pooled": paired, "thresholds": thresholds, "files": files,
               "sensitivity_without_folds": {"folds": dominated, "fold_mean": sensitivity,
                                             "why": "the sample of these folds is more than half one student"},
               "per_fold_reports": per_fold, "pooled_reports": pooled}
    write_json(lc.results / "lc-metrics.json", metrics)
    (lc.results / "lc-summary.md").write_text(lc_summary_md(lc, metrics), encoding="utf-8")
    print(json.dumps({n: {a: round(t[a]["rate" if a == "all_axis_exact_match" else "accuracy"]["mean"], 3)
                          for a in t} for n, t in tables.items()}))
    return 0


def lc_summary_md(lc: Space, m: dict) -> str:
    t, names = m["tables"], ("C0 full", "C1 full", "C1 half", "C1 full @0.5", "C1 half @0.5")
    out = [f"# 学习曲线：C1 用约一半训练数据 与 全量（{lc.name}；参照 v9；D-106 初步开发，非正式评价）", "",
           f"性质：{m['statement']}", "",
           "测试折与交叉验证完全相同，每条互动只被预测一次。“半量”= 每个训练折按学生（整名学生）随机抽约一半互动（固定种子），"
           "其余设置与 C1 完全相同，包括在更小的训练样本里留出约 20% 选阈值。“@0.5”= 同一批模型不用各折阈值、直接按 0.5 解码。", "",
           "## 各折训练样本", "", "| 折 | 训练折互动数 | 抽样互动数 | 占比 | 抽到学生数 / 训练折学生数 | 训练折最大学生（条数） | 是否抽到 |",
           "|---|---|---|---|---|---|---|"]
    for k, s in m["samples"].items():
        big = s["largest_student_in_training_fold"]
        out.append(f"| {k} | {s['training_fold_interactions']} | {s['sampled_interactions']} | {s['share']:.2f} | "
                   f"{s['sampled_students']} / {s['training_fold_students']} | {big['student']}（{big['interactions']}） | "
                   f"{'是' if big['in_sample'] else '否'} |")
    out += ["", "## 各维度结果（逐折均值 ± 标准差；括号内为 150 条合并计算；多选维度的准确率即整组完全一致率）", "",
            "| 维度 | 指标 | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for axis in list(L.AXES) + ["all_axis_exact_match"]:
        for metric in t["C1 half"][axis]:
            cells = [f"{f3(t[n][axis][metric]['mean'])} ± {f3(t[n][axis][metric]['sd'])} ({f3(t[n][axis][metric]['pooled'])})"
                     for n in names]
            out.append(f"| {'四维全对' if axis == 'all_axis_exact_match' else axis} | {metric} | " + " | ".join(cells) + " |")
    out += ["", "## 逐折准确率（四维全对为比例）", "", "| 维度 | 配置 | " + " | ".join(f"折{k}" for k in m["samples"]) + " |",
            "|---|---|" + "---|" * len(m["samples"])]
    for axis in list(L.AXES) + ["all_axis_exact_match"]:
        metric = "rate" if axis == "all_axis_exact_match" else "accuracy"
        for n in ("C1 full", "C1 half"):
            out.append(f"| {'四维全对' if metric == 'rate' else axis} | {n} | "
                       + " | ".join(f3(v) for v in t[n][axis][metric]["per_fold"]) + " |")
    sens = m["sensitivity_without_folds"]
    out += ["", "## 敏感性检查（运行前声明）：去掉样本被单个学生占据的折后的逐折均值", ""]
    if sens["folds"]:
        names2 = ("C1 full", "C1 half", "C1 full @0.5", "C1 half @0.5")
        out += [f"被去掉的折：{sens['folds']}（它们的抽样里单个学生占了一半以上），其余折的均值：", "",
                "| 维度 | 指标 | C1 全量 | C1 半量 | 半量−全量 | C1 全量@0.5 | C1 半量@0.5 |", "|---|---|---|---|---|---|---|"]
        for axis in list(L.AXES) + ["all_axis_exact_match"]:
            for metric in ([("rate")] if axis == "all_axis_exact_match" else ("accuracy", "macro_f1")):
                v = {n: sens["fold_mean"][n][axis][metric] for n in names2}
                diff = None if v["C1 half"] is None or v["C1 full"] is None else v["C1 half"] - v["C1 full"]
                out.append(f"| {'四维全对' if axis == 'all_axis_exact_match' else axis} | {metric} | {f3(v['C1 full'])} | "
                           f"{f3(v['C1 half'])} | {f3(diff)} | {f3(v['C1 full @0.5'])} | {f3(v['C1 half @0.5'])} |")
    else:
        out.append("没有这样的折。")
    out += ["", "## 半量对全量的逐条配对比较（150 条合并；精确 McNemar，未校正，仅供参考）", "",
            "| 维度 | 半量对、全量错 | 全量对、半量错 | p |", "|---|---|---|---|"]
    for axis, p in m["paired_half_vs_full_pooled"].items():
        out.append(f"| {axis} | {p['half_right_full_wrong']} | {p['full_right_half_wrong']} | {f3(p['mcnemar_exact_p'])} |")
    out += ["", "## 各折选出的阈值（内部留出集选出，未看测试折）", ""]
    for n in ("C1 full", "C1 half"):
        out += [f"- {n} 折{k}：" + "，".join(f"{a} {v}" for a, v in th.items()) for k, th in m["thresholds"][n].items()]
    out += ["", "混淆矩阵见 reports/pooled-C1_half.md；全量的见上级目录 reports/pooled-C1.md。", "", "## 文件", ""]
    out += [f"- `{k}` sha256 `{v}`" for k, v in m["files"].items()] + [""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- smoke

def smoke(learning_curve: bool = False) -> int:
    """The whole pipeline on 10 interactions from 10 students (including the one whose response C0
    clips), 1 epoch for C0 and 2 for C1, so every step runs once before the real CV. With
    learning_curve, the half-size C1 pipeline on the same data, compared with that smoke CV."""
    reports = json.loads((B_RUN / "inputs-report.json").read_text(encoding="utf-8"))["per_interaction"]
    long_key = next(L.parse_key(k) for k, v in reports.items() if v["response_tokens_omitted"] > 0)
    examples = GS.load_examples(V9)
    owner = {(e["conversation_id"], e["segment_order"]): e["student_id"] for e in examples}
    first = {}
    for e in sorted(examples, key=lambda e: (e["student_id"], e["conversation_id"], e["segment_order"])):
        first.setdefault(e["student_id"], (e["conversation_id"], e["segment_order"]))
    chosen = {long_key} | {key for s, key in sorted(first.items()) if s != owner[long_key]}
    chosen = {long_key} | set(sorted(chosen - {long_key})[:9])
    base = ROOT / "data" / "training" / "cv-20261022-smoke"
    labels_path, prepared = base / "smoke-labels.jsonl", base / "smoke-prepared"
    if not (prepared / "manifest.json").exists():
        base.mkdir(parents=True, exist_ok=True)
        lines = [line for line in V9_LABELS.read_bytes().split(b"\n") if line.strip()]
        keep = [line for line in lines if (lambda r: (r["conversation_id"], r["segment_order"]))(json.loads(line)) in chosen]
        labels_path.write_bytes(b"\n".join(keep) + b"\n")
        r = subprocess.run([str(PY), str(PREPARE), "--reviewed-direct", str(labels_path), "--corpus", str(CORPUS),
                            "--output-dir", str(prepared), "--context", "3"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-500:])
    space = Space("cv-20261022-smoke", prepared, labels_path, HERE / "smoke", {"C0": 1, "C1": 2})
    if learning_curve:
        sample = Space("cv-20261022-lc-smoke", prepared, labels_path, HERE / "smoke-lc", {"C0": 1, "C1": 2},
                       sample_fraction=LC_FRACTION, run_c0=False)
        plan(sample)
        run_all(sample)
        return lc_report(sample, space)
    space.results.mkdir(parents=True, exist_ok=True)
    plan(space)
    run_all(space)
    return report(space)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("step1", "plan", "run", "report", "smoke", "plan-lc", "run-lc", "report-lc", "smoke-lc"):
        sub.add_parser(name)
    a = p.parse_args(argv)
    if a.command == "step1":
        return step1()
    if a.command == "smoke":
        return smoke()
    if a.command == "smoke-lc":
        return smoke(learning_curve=True)
    if a.command == "plan":
        plan(REAL)
        return 0
    if a.command == "plan-lc":
        plan(LC)
        return 0
    if a.command == "run-lc":
        return run_all(LC)
    if a.command == "report-lc":
        return lc_report(LC, REAL)
    return run_all(REAL) if a.command == "run" else report(REAL)


if __name__ == "__main__":
    raise SystemExit(main())
