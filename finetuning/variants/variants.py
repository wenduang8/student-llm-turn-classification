#!/usr/bin/env python3
"""Exploratory C1 variants V1 (focal loss) and V2 (six context turns) on folds 3 and 4 of the final run's split (D-194).
Spec: docs/planning/c1-variants-spec-20261028.md (version c, D-202); master README
docs/planning/c1-variants-master-readme-20261028c.txt (V3 is not run, D-195; the adoption rule of spec section 4 is confirmed
by DW, its criterion 2 changed to per code by D-202 before any variant ran). C1 is the reported model; these runs are
exploratory. Prints counts and verdicts only, never interaction text.

The variants run the code copy in ./finetuning: the working-tree finetuning/ with its repository root and fingerprinted
file list pointed at the copy, plus V1's focal loss (code-diff-vs-finetuning.patch), so nothing the running final job may
reload on resume is touched. This driver scores with the unchanged main evaluation module (through c485.py's imports)
and runs the copy only as a subprocess.

    .venv/bin/python -I data/results/c1-variants-f34-20261028/variants.py plan        # CPU: data, splits, configs, inputs, estimate
    .venv/bin/python -I data/results/c1-variants-f34-20261028/variants.py cpu-smoke   # CPU: both variants on a few short real interactions
    .venv/bin/python -I data/results/c1-variants-f34-20261028/variants.py run         # MPS, after the final run: V1 folds 3, 4; the V2
                                                                                      #   memory smoke; V2 folds 3, 4. Resumable.
    .venv/bin/python -I data/results/c1-variants-f34-20261028/variants.py rule-check  # CPU: D-202's criterion-2 codes vs the labels
    .venv/bin/python -I data/results/c1-variants-f34-20261028/variants.py report      # spec section 3 comparison and the section 4 rule

Design (fixed by the spec before any variant result):
  folds   3 and 4 of data/splits/cv485-20261028d (the D-186 rosters; 194 test interactions; D-193), 388 training each.
  V1      C1 of the final run with one change: focal loss, gamma 2, every head, C1's class weights as the alpha terms.
  V2      C1 of the final run with six preceding interactions (prepared with --context 6; a Segment Order gap still ends
          the context) and max_tokens 10,240; prompt cap 1,536 and the 3,072 response reservation unchanged. Before the
          runs, on MPS: a memory smoke test on the 10 longest V2 inputs (the trainer on those 10 as one training part of two
          optimizer steps, 1 epoch); projected peak = the largest MPS driver memory at a step end (the measure the runs log,
          before each step's cache release). Above 37 GiB, or out of memory: five interactions and 9,216 instead, recorded.
  both    labels D-190, seed 20261017, 8 epochs, threshold 0.5, budgets as above, memory settings as D-186; per-code scores
          saved (each run's outputs/scores-validation.jsonl, pooled with the predicted labels by `report`).
  report  per code only (D-201, D-202): simple agreement C / max(A, B) with A, B and C, and the paired differences, from
          per_code() of comparison/score.py on the 194 test keys (10,000 student replicates, seed 20261026); no exact-set rate.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]           # this copy is in finetuning/variants/
HERE = Path(__file__).resolve().parent
CODE = HERE / "finetuning"
D186 = ROOT / "data" / "results" / "dw485-c1-20261028"
FINAL = ROOT / "data" / "results" / "dw485d-c1-final-20261028"
sys.path.insert(0, str(D186))
import c485 as B  # noqa: E402  (helpers; its imports load the unchanged main finetuning modules)

cv, EV, GS, L = B.cv, B.EV, B.GS, B.L
utc, sha, read_json, write_json, roundtrip = B.utc, B.sha, B.read_json, B.write_json, B.roundtrip
PY = ROOT / ".venv" / "bin" / "python"
TRAIN_V, PREPARE_V, INPUTS_V = CODE / "train_multihead.py", CODE / "prepare_data.py", CODE / "variant_inputs.py"
SPEC = ROOT / "docs" / "planning" / "c1-variants-spec-20261028.md"
MASTER = ROOT / "docs" / "planning" / "c1-variants-master-readme-20261028c.txt"
SCORE = ROOT / "comparison" / "score.py"
LABELS = ROOT / "data" / "labels" / "dw485-reviewed-20261028d.jsonl"
CORPUS = ROOT / "data" / "processed" / "transcripts-485-conversations-20261028.csv"
PREPARED = {3: ROOT / "data" / "training" / "dw485-reviewed-20261028d",
            6: ROOT / "data" / "training" / "dw485-reviewed-20261028d-ctx6",
            5: ROOT / "data" / "training" / "dw485-reviewed-20261028d-ctx5"}
FINAL_SPLITS = ROOT / "data" / "splits" / "cv485-20261028d"
FINAL_RUNS = ROOT / "data" / "runs" / "cv485-20261028d"
NAME = "c1-variants-f34-20261028"
SPLITS, RUNS, TRAINING = (ROOT / "data" / d / NAME for d in ("splits", "runs", "training"))
FOLDS = (3, 4)
V2_OPTIONS = {"ctx6": {"context": 6, "max_tokens": 10240}, "ctx5": {"context": 5, "max_tokens": 9216}}
MEMORY_LIMIT_GIB, GAMMA = 37.0, 2.0
TARGETS = (("student interaction type", "Guided generation"), ("AI response type", "Evaluate"),
           ("AI delegation", "hand-back: directive"), ("AI delegation", "hand-back: asks the student to supply"))
TARGET_MIN_RAISED, TARGET_MIN_MEAN_GAIN = 3, Fraction(5, 100)                  # criterion 1 (unchanged)
# criterion 2 (D-202): the other codes with at least 40 reference cases in folds 3 and 4 (D-190 labels), with those counts
GUARDS = {("student interaction type", "Information retrieval"): 88, ("student interaction type", "Task automation"): 46,
          ("topic", "code"): 134, ("AI delegation", "ND"): 89, ("AI delegation", "offer"): 86,
          ("AI response type", "Solve-Provide"): 191, ("AI response type", "Explain"): 162,
          ("AI response type", "Correct-Debug"): 46}
GUARD_MIN_CASES, GUARD_MAX_LOSS = 40, Fraction(5, 100)
BOOT_SEED, BOOT_N = 20261026, 10000
FOCAL_NAMES = {"single_select": "focal_cross_entropy", "multi_select": "focal_bce_with_logits_mean_over_codes"}
V1_ALLOWED = ("status", "validation.purpose", "data.split_file", "loss.single_select", "loss.multi_select", "loss.focal_gamma",
              "loss.notes")
V2_ALLOWED = ("status", "validation.purpose", "data.prepared_dir", "data.split_file", "data.label_source_note",
              "input.context_interactions", "input.max_tokens")
PURPOSE = ("Test fold {k} of the final run's split (cv485-20261028d = the D-186 rosters); D-194 exploratory variant {v}: "
           "held-out students used only to produce this fold's predictions at threshold 0.5, compared once with C1's "
           "final-run predictions on folds 3 and 4 under the adoption rule fixed before any variant result. No per-epoch loss "
           "(per_epoch_loss false). Not used to choose epochs, checkpoints, thresholds or the input policy.")
# the trainer requires the split's validation purpose to equal the config's (train_multihead.run_training)
SMOKE_PURPOSE = f"{NAME} smoke subset: pipeline or memory check only; nothing is chosen on it."


def note(text: str) -> None:
    with (HERE / "progress.md").open("a", encoding="utf-8") as f:
        f.write(f"- {utc()} {text}\n")


def keys_of(prepared: Path) -> list:
    return [(e["conversation_id"], e["segment_order"]) for e in GS.load_examples(prepared)]


# ----------------------------------------------------------------------------- data, splits, configs

def prepare(context: int, labels: Path, out: Path) -> None:
    if (out / "manifest.json").exists():
        return
    r = subprocess.run([str(PY), str(PREPARE_V), "--reviewed-direct", str(labels), "--corpus", str(CORPUS),
                        "--output-dir", str(out), "--context", str(context)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"prepare_data (copy) failed for {L.rel(out)}: {r.stderr[-800:]}")


def compare_prepared(n: int) -> dict:
    """The context-n preparation against the context-3 one: same interactions, students, labels and targets; only the
    stored context differs, and its last three earlier interactions are the context-3 ones."""
    a = {(e["conversation_id"], e["segment_order"]): e for e in GS.load_examples(PREPARED[3])}
    b = {(e["conversation_id"], e["segment_order"]): e for e in GS.load_examples(PREPARED[n])}
    same_core = all(a[k][f] == b[k][f] for k in a for f in ("student_id", "student_prompt", "llm_response", "labels", "targets"))
    tail = sum(b[k]["context"][-len(a[k]["context"]):] == a[k]["context"] if a[k]["context"] else True for k in a)
    longer = Counter(len(b[k]["context"]) - len(a[k]["context"]) for k in a)
    return {"interactions": len(a), "same_keys": set(a) == set(b), "same_students_texts_labels_targets": same_core,
            "context3_is_the_tail": tail, "extra_earlier_interactions": dict(sorted(longer.items())),
            "with_more_context_available": sum(v for d, v in longer.items() if d > 0),
            "context_lengths": dict(sorted(Counter(len(b[k]["context"]) for k in b).items()))}


def bind_split(prepared: Path, roster_from: Path, out: Path, status: str, purpose: str) -> dict:
    """A split file holding the roster of `roster_from` unchanged, bound to `prepared` (grouped_split.load_split) and
    stating the variant run's validation purpose (the trainer requires it to equal the config's)."""
    old = read_json(roster_from)
    examples = {(e["conversation_id"], e["segment_order"]): e for e in GS.load_examples(prepared)}
    record = copy.deepcopy(old)
    rows = {n: [examples[L.parse_key(t)] for t in old["parts"][n]["interactions"]] for n in ("train", "validation")}
    record.update({"status": status, "validation_purpose": purpose, "created_utc": utc(), "prepared": GS.prepared_fingerprint(prepared),
                   "driver_script_sha256": sha(Path(__file__)), "reused_roster_from": {"path": L.rel(roster_from), "sha256": sha(roster_from)},
                   "label_support": {n: GS.label_support(r) for n, r in rows.items()}})
    assert record["parts"] == old["parts"] and sum(len(r) for r in rows.values()) == len(examples)
    if out.exists():
        existing = read_json(out)
        if existing["parts"] != record["parts"] or existing["prepared"] != record["prepared"]:
            raise ValueError(f"{out} exists with a different roster or prepared data")
        return existing
    write_json(out, record, exclusive=True)
    return record


def subset_split(prepared: Path, train: list, validation: list, out: Path, purpose: str) -> None:
    """An explicit roster for a smoke subset (the prepared subset holds exactly these interactions)."""
    examples = {(e["conversation_id"], e["segment_order"]): e for e in GS.load_examples(prepared)}
    assert set(examples) == set(train) | set(validation) and not set(train) & set(validation)
    parts = {n: cv.part_record([examples[k] for k in sorted(ks)]) for n, ks in (("train", train), ("validation", validation))}
    assert not set(parts["train"]["students"]) & set(parts["validation"]["students"])
    record = {"record_kind": GS.RECORD_KIND, "status": f"{NAME} smoke subset (pipeline or memory check only; not a result)",
              "validation_purpose": purpose, "seed": None, "procedure": "explicit roster chosen by input length (smoke only)",
              "created_utc": utc(), "split_module_sha256": sha(Path(GS.__file__)),
              "driver_script_sha256": sha(Path(__file__)), "prepared": GS.prepared_fingerprint(prepared), "parts": parts,
              "label_support": {n: GS.label_support([examples[k] for k in ks]) for n, ks in (("train", train), ("validation", validation))}}
    if out.exists():
        if read_json(out)["parts"] != parts:
            raise ValueError(f"{out} exists with a different roster")
        return
    write_json(out, record, exclusive=True)


def subset_labels(keys: set, out: Path) -> Path:
    lines = [line for line in LABELS.read_bytes().split(b"\n") if line.strip()]
    chosen = [line for line in lines if B.record_key(line) in keys]
    assert len(chosen) == len(keys)
    data = b"\n".join(chosen) + b"\n"
    if out.exists():
        assert out.read_bytes() == data
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    return out


def config_path(name: str) -> Path:
    return HERE / "configs" / f"{name}.json"


def save_config(name: str, cfg: dict) -> None:
    path = config_path(name)
    if path.exists():
        if read_json(path) != cfg:
            raise ValueError(f"{path} exists with different content")
    else:
        write_json(path, cfg, exclusive=True)


def v1_config(k: int) -> tuple[dict, list]:
    template = read_json(FINAL / "configs" / f"C1-fold{k}.json")
    cfg = copy.deepcopy(template)
    cfg["status"] = (f"V1 fold {k} ({NAME}; D-194 exploratory variant; D-106 preliminary development): C1 of the final run "
                     "D-191 with one change, focal loss (gamma 2) on every head with C1's class weights as the alpha terms. "
                     "Labels: D-190 (the researcher's labels; not joint adjudication).")
    cfg["loss"].update(FOCAL_NAMES, focal_gamma=GAMMA,
                       notes="D-194 V1: data/results/c1-variants-f34-20261028/finetuning/multihead.py example_loss with gamma 2: "
                             "single-select (1 - p)^2 times the class-weighted cross-entropy of the observed code; multi-select "
                             "positive terms pos_weight * (1 - p)^2 * -log p and negative terms p^2 * -log(1 - p), mean over codes. "
                             "ND and 'none of the four applies' are ordinary observed codes. Validation loss (off) stays plain.")
    cfg["data"]["split_file"] = L.rel(SPLITS / f"V1-fold{k}.json")
    cfg["validation"]["purpose"] = PURPOSE.format(k=k, v="V1")
    changes = B.changed_keys(template, cfg)
    if B.within(changes, V1_ALLOWED):
        raise AssertionError(f"V1 fold {k}: unexpected config changes {B.within(changes, V1_ALLOWED)}")
    return cfg, changes


def v2_config(k: int, option: str) -> tuple[dict, list]:
    o = V2_OPTIONS[option]
    template = read_json(FINAL / "configs" / f"C1-fold{k}.json")
    cfg = copy.deepcopy(template)
    cfg["status"] = (f"V2 fold {k} ({NAME}; D-194 exploratory variant, option {option}; D-106 preliminary development): C1 of the "
                     f"final run D-191 with one change, {o['context']} preceding interactions instead of three, max_tokens "
                     f"{o['max_tokens']} so the extra context is not cut (prompt cap 1536 and response reservation 3072 unchanged). "
                     "Labels: D-190 (the researcher's labels; not joint adjudication).")
    cfg["data"].update(prepared_dir=L.rel(PREPARED[o["context"]]), split_file=L.rel(SPLITS / f"{option}-fold{k}.json"),
                       label_source_note=template["data"]["label_source_note"]
                       + f" D-194 V2: prepared with {o['context']} preceding interactions (a Segment Order gap still ends the context).")
    cfg["input"].update(context_interactions=o["context"], max_tokens=o["max_tokens"])
    cfg["validation"]["purpose"] = PURPOSE.format(k=k, v=f"V2 ({option})")
    changes = B.changed_keys(template, cfg)
    if B.within(changes, V2_ALLOWED):
        raise AssertionError(f"V2 fold {k}: unexpected config changes {B.within(changes, V2_ALLOWED)}")
    return cfg, changes


def run_inputs_check(out: Path, configs: dict) -> dict:
    r = subprocess.run([str(PY), str(INPUTS_V), str(out), *(f"{n}={p}" for n, p in configs.items())], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    if r.returncode != 0:
        raise SystemExit(f"variant_inputs.py failed: {r.stderr[-1200:]}")
    return read_json(out)


def summarise_inputs(rec: dict, keys: list | None = None) -> dict:
    per = rec["per_interaction"]
    chosen = [per[t] for t in (keys or per)]
    tokens = sorted(v["tokens"] for v in chosen)
    return {"interactions": len(chosen), "loses_context": sum(v["context_tokens_omitted"] > 0 for v in chosen),
            "loses_prompt": sum(v["prompt_tokens_omitted"] > 0 for v in chosen),
            "loses_response": sum(v["response_tokens_omitted"] > 0 for v in chosen),
            "tokens": {"median": tokens[len(tokens) // 2], "p90": tokens[int(0.9 * (len(tokens) - 1))], "max": tokens[-1]},
            "context_included": dict(sorted(Counter(v["context_included"] for v in chosen).items()))}


def plan() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    for n in (6, 5):
        prepare(n, LABELS, PREPARED[n])
    prepared_checks = {n: compare_prepared(n) for n in (6, 5)}
    if not all(c["same_keys"] and c["same_students_texts_labels_targets"] and c["context3_is_the_tail"] == c["interactions"]
               for c in prepared_checks.values()):
        raise AssertionError(f"the context-6/5 preparations differ from context 3 in more than the context: {prepared_checks}")
    splits, configs = {}, {}
    for k in FOLDS:
        roster = FINAL_SPLITS / f"fold{k}.json"
        out = SPLITS / f"V1-fold{k}.json"
        bind_split(PREPARED[3], roster, out, f"{NAME} V1 fold {k}: the final run's roster ({L.rel(roster)}) unchanged (D-194 V1)",
                   PURPOSE.format(k=k, v="V1"))
        splits[f"V1-fold{k}"] = {"path": L.rel(out), "sha256": sha(out)}
        for option, o in V2_OPTIONS.items():
            out = SPLITS / f"{option}-fold{k}.json"
            bind_split(PREPARED[o["context"]], roster, out,
                       f"{NAME} {option} fold {k}: the final run's roster ({L.rel(roster)}) unchanged, "
                       f"bound to the context-{o['context']} preparation (D-194 V2)", PURPOSE.format(k=k, v=f"V2 ({option})"))
            splits[f"{option}-fold{k}"] = {"path": L.rel(out), "sha256": sha(out)}
        cfg, changes = v1_config(k)
        save_config(f"V1-fold{k}", cfg)
        configs[f"V1-fold{k}"] = {"path": L.rel(config_path(f"V1-fold{k}")), "changed_keys_vs_final_c1": changes}
        for option in V2_OPTIONS:
            cfg, changes = v2_config(k, option)
            save_config(f"V2-{option}-fold{k}", cfg)
            configs[f"V2-{option}-fold{k}"] = {"path": L.rel(config_path(f"V2-{option}-fold{k}")), "changed_keys_vs_final_c1": changes}
    for name in configs:
        configs[name]["sha256"] = sha(config_path(name))
    check = {"C1-fold3": FINAL / "configs" / "C1-fold3.json", "C1-fold4": FINAL / "configs" / "C1-fold4.json",
             **{n: config_path(n) for n in configs}}
    inputs = run_inputs_check(HERE / "input-check.json", {n: str(p) for n, p in check.items()})
    test = {k: read_json(FINAL_SPLITS / f"fold{k}.json")["parts"]["validation"]["interactions"] for k in FOLDS}
    train = {k: read_json(FINAL_SPLITS / f"fold{k}.json")["parts"]["train"]["interactions"] for k in FOLDS}
    summary = {}
    for name in ("C1-fold3", "V1-fold3", "V2-ctx6-fold3", "V2-ctx5-fold3"):
        summary[name.rsplit("-fold", 1)[0]] = {"all_485": summarise_inputs(inputs[name]),
                                               "test_folds_3_4": summarise_inputs(inputs[name], test[3] + test[4])}
    c1 = inputs["C1-fold3"]["per_interaction"]
    for option in V2_OPTIONS:
        per = inputs[f"V2-{option}-fold3"]["per_interaction"]
        summary[f"V2-{option}"]["more_context_than_c1"] = sum(per[t]["context_included"] > c1[t]["context_included"] for t in per)
        summary[f"V2-{option}"]["student_side_budget"] = inputs[f"V2-{option}-fold3"]["student_side_budget"]
    summary["C1"]["student_side_budget"] = inputs["C1-fold3"]["student_side_budget"]
    identity = {f"V1-fold{k}": inputs[f"V1-fold{k}"]["inputs_sha256"] == inputs[f"C1-fold{k}"]["inputs_sha256"] for k in FOLDS}
    if not all(identity.values()):
        raise AssertionError("V1's inputs differ from C1's")
    copy_code = {n: inputs["V1-fold3"]["code_sha256"][n] for n in inputs["V1-fold3"]["code_sha256"]}
    # the 10 longest inputs of each V2 option (memory smoke) and their validation partners
    longest = {}
    for option in V2_OPTIONS:
        per = inputs[f"V2-{option}-fold3"]["per_interaction"]
        longest[option] = sorted(per, key=lambda t: (-per[t]["tokens"], t))[:10]
    estimate = time_estimate(inputs, train, test)
    record = {"what": f"{NAME} plan (fixed before any variant run; D-194 exploratory variants V1 and V2 on folds 3 and 4)",
              "created_utc": utc(), "spec": {"path": L.rel(SPEC), "sha256": sha(SPEC)}, "master": {"path": L.rel(MASTER), "sha256": sha(MASTER)},
              "labels": {"path": L.rel(LABELS), "sha256": sha(LABELS)}, "folds": list(FOLDS),
              "test_interactions": {k: len(test[k]) for k in FOLDS}, "train_interactions": {k: len(train[k]) for k in FOLDS},
              "prepared": {n: {"path": L.rel(p), "manifest_sha256": sha(p / "manifest.json")} for n, p in PREPARED.items()},
              "prepared_checks": prepared_checks, "splits": splits, "configs": configs, "v1_inputs_equal_c1": identity,
              "inputs": summary, "v2_memory_smoke_inputs": {o: [{"interaction": t, "tokens": inputs[f"V2-{o}-fold3"]["per_interaction"][t]["tokens"]}
                                                                for t in longest[o]] for o in V2_OPTIONS},
              "memory_rule": f"projected peak = max MPS driver memory at a step end in the smoke; above {MEMORY_LIMIT_GIB} GiB or out of "
                             "memory: option ctx5 (five interactions, 9,216 tokens)",
              "adoption_rule": "spec section 4, confirmed by DW (master README step 5)", "estimate": estimate,
              "code_copy": {"dir": L.rel(CODE), "fingerprinted_files_sha256": copy_code,
                            "diff_vs_working_tree_finetuning": L.rel(HERE / "code-diff-vs-finetuning.patch")},
              "driver_script_sha256": sha(Path(__file__))}
    diff = subprocess.run(["git", "diff", "--no-index", "--", "finetuning", L.rel(CODE)], cwd=ROOT, capture_output=True, text=True).stdout
    (HERE / "code-diff-vs-finetuning.patch").write_text(diff, encoding="utf-8")
    record["code_copy"]["diff_sha256"] = L.sha256_bytes(diff.encode("utf-8"))
    path = HERE / "plan.json"
    if path.exists():
        old = read_json(path)
        if old["configs"] != roundtrip(record["configs"]) or old["splits"] != roundtrip(record["splits"]):
            raise ValueError(f"{path} exists with a different plan")
        print(f"{L.rel(path)} exists and matches; not rewritten")
    else:
        write_json(path, record, exclusive=True)
    print(json.dumps({"prepared_checks": prepared_checks, "v1_inputs_equal_c1": identity, "inputs": summary,
                      "config_changes": {n: v["changed_keys_vs_final_c1"] for n, v in configs.items() if n.endswith("fold3")},
                      "v2_memory_smoke_longest_tokens": {o: [x["tokens"] for x in v] for o, v in record["v2_memory_smoke_inputs"].items()},
                      "estimate": estimate}, ensure_ascii=False, indent=1))
    return 0


def time_estimate(inputs: dict, train: dict, test: dict) -> dict:
    """V1: the D-186 run's measured wall time of the same fold (same inputs and settings; the focal terms add nothing
    measurable). V2: the D-186 fold's optimizer-step time scaled by the mean per-pass cost (profile cost model) of the
    V2 training inputs over C1's, and the rest of the run by the same ratio over train + test inputs."""
    status = read_json(D186 / "run-status.json")["runs"]
    c1 = inputs["C1-fold3"]["per_interaction"]
    out = {"V1": {}, "V2-ctx6": {}, "V2-ctx5": {}}
    for k in FOLDS:
        log = [json.loads(line) for _, line in L.jsonl_lines(B.REAL.run_dir(k) / "train_log.jsonl")]
        step_s = sum(e["seconds"] for e in {e["step"]: e for e in log if e["event"] == "step"}.values())
        wall = sum(r["seconds"] for r in status[f"C1-fold{k}"])
        out["V1"][k] = wall
        for option in V2_OPTIONS:
            per = inputs[f"V2-{option}-fold{k}"]["per_interaction"]
            r_train = sum(B.COST(per[t]["tokens"]) for t in train[k]) / sum(B.COST(c1[t]["tokens"]) for t in train[k])
            r_all = (sum(B.COST(per[t]["tokens"]) for t in train[k] + test[k])
                     / sum(B.COST(c1[t]["tokens"]) for t in train[k] + test[k]))
            out[f"V2-{option}"][k] = round(step_s * r_train + (wall - step_s) * r_all)
    return {"per_fold_seconds": out, "hours": {v: round(sum(x.values()) / 3600, 2) for v, x in out.items()},
            "sequence_hours_V1_then_V2ctx6": round((sum(out["V1"].values()) + sum(out["V2-ctx6"].values())) / 3600, 2),
            "basis": "D-186 folds 3 and 4 measured (data/results/dw485-c1-20261028/run-status.json, train_log.jsonl); cost model "
                     "1.007 ms/token + 0.383 us/token^2"}


# ----------------------------------------------------------------------------- smoke tests

def smoke_subset(tag: str, train: list, validation: list, contexts: tuple) -> dict:
    """Prepared subsets (one per context size) and their explicit rosters; returns {context: (prepared, split)}."""
    keys = set(train) | set(validation)
    labels = subset_labels(keys, TRAINING / tag / "labels.jsonl")
    out = {}
    for n in contexts:
        prepared = TRAINING / tag / f"prepared-ctx{n}"
        prepare(n, labels, prepared)
        split = TRAINING / tag / f"split-ctx{n}.json"
        subset_split(prepared, train, validation, split, SMOKE_PURPOSE)
        out[n] = (prepared, split)
    return out


def pick(per: dict, n_train: int, n_val: int, longest: bool) -> tuple[list, list]:
    """Training interactions by input length (longest or shortest first), then validation interactions, shortest first,
    from students not in the training part."""
    student = lambda t: t.split("_", 1)[0]  # noqa: E731
    order = sorted(per, key=lambda t: ((-1 if longest else 1) * per[t]["tokens"], t))
    train, seen = [], set()
    for t in order:
        if len(train) < n_train and (longest or student(t) not in seen):
            train.append(t)
            seen.add(student(t))
    val = [t for t in sorted(per, key=lambda t: (per[t]["tokens"], t)) if student(t) not in seen][:n_val]
    return [L.parse_key(t) for t in train], [L.parse_key(t) for t in val]


def smoke_config(base: dict, prepared: Path, split: Path, device: str, status: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg.update(device=device, status=status)
    cfg["data"].update(prepared_dir=L.rel(prepared), split_file=L.rel(split))
    cfg["schedule"]["epochs"] = 1
    cfg["validation"]["purpose"] = SMOKE_PURPOSE
    return cfg


def run_trainer(name: str, config: Path, run_dir: Path, log: Path, env: dict | None = None) -> int:
    command = ([str(PY), str(TRAIN_V), "resume", "--run-dir", str(run_dir)] if (run_dir / "run.json").exists()
               else [str(PY), str(TRAIN_V), "train", "--config", str(config), "--run-dir", str(run_dir)])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n== {utc()} {command[2]} ==\n")
        out.flush()
        proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env)
        forward = lambda signum, _frame: proc.send_signal(signum)  # noqa: E731
        old = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            return proc.wait()
        finally:
            for s, h in old.items():
                signal.signal(s, h)


def cpu_smoke() -> int:
    """Both variants with the real model and tokenizer on CPU: 3 short training interactions (distinct students) and
    1 short validation interaction, 1 epoch (one optimizer step). Pipeline check only (no MPS)."""
    plan_record = read_json(HERE / "plan.json")
    per = read_json(HERE / "input-check.json")["V2-ctx6-fold3"]["per_interaction"]
    train, val = pick(per, 3, 1, longest=False)
    sub = smoke_subset("cpu-smoke", train, val, (3, 6))
    env = {**os.environ, "OMP_NUM_THREADS": "4"}
    out = {}
    for name, base, n in (("V1", read_json(config_path("V1-fold3")), 3), ("V2", read_json(config_path("V2-ctx6-fold3")), 6)):
        cfg = smoke_config(base, *sub[n], "cpu", f"{NAME} CPU smoke of {name} (pipeline check only; not a result)")
        path = HERE / "smoke" / f"cpu-{name}.json"
        if not path.exists():
            write_json(path, cfg, exclusive=True)
        run_dir = RUNS / "smoke" / f"cpu-{name}"
        started = time.time()
        code = run_trainer(name, path, run_dir, RUNS / "smoke" / f"cpu-{name}.out", env)
        log = [json.loads(line) for _, line in L.jsonl_lines(run_dir / "train_log.jsonl")] if (run_dir / "train_log.jsonl").exists() else []
        steps = [e for e in log if e["event"] == "step"]
        run = read_json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
        out[name] = {"exit": code, "seconds": round(time.time() - started), "outputs": (run_dir / "outputs" / "summary.json").exists(),
                     "steps": len(steps), "loss": [s["loss_mean"] for s in steps],
                     "fingerprint_names_the_copy": all(p.startswith(L.rel(CODE)) for p in run.get("fingerprint", {}).get("code_sha256", {})
                                                       if p.endswith(".py") and "/" in p and not p.startswith("coding/")),
                     "scores_written": (run_dir / "outputs" / "scores-validation.jsonl").exists()}
    write_json(HERE / "smoke" / "cpu-smoke.json", {"what": "CPU smoke of V1 and V2 (real model, 4 interactions, 1 step)", "utc": utc(),
                                                   "plan_sha256": sha(HERE / "plan.json"), "runs": out})
    print(json.dumps(out, indent=1))
    return 0 if all(v["exit"] == 0 and v["outputs"] for v in out.values()) else 1


def v2_memory(option: str) -> dict:
    """The trainer (MPS) on the 10 longest inputs of a V2 option as one training part (two optimizer steps, 1 epoch)."""
    per = read_json(HERE / "input-check.json")[f"V2-{option}-fold3"]["per_interaction"]
    train, val = pick(per, 10, 2, longest=True)
    n = V2_OPTIONS[option]["context"]
    sub = smoke_subset(f"v2-memory-{option}", train, val, (n,))
    cfg = smoke_config(read_json(config_path(f"V2-{option}-fold3")), *sub[n], "mps",
                       f"{NAME} V2 memory smoke ({option}; the 10 longest inputs; not a result)")
    path = HERE / "smoke" / f"v2-memory-{option}.json"
    if not path.exists():
        write_json(path, cfg, exclusive=True)
    run_dir = RUNS / "smoke" / f"v2-memory-{option}"
    started = time.time()
    code = run_trainer(f"v2-memory-{option}", path, run_dir, RUNS / "smoke" / f"v2-memory-{option}.out")
    log = [json.loads(line) for _, line in L.jsonl_lines(run_dir / "train_log.jsonl")] if (run_dir / "train_log.jsonl").exists() else []
    steps = [e for e in log if e["event"] == "step"]
    peak = max((s.get("mps_driver_gib", 0.0) for s in steps), default=None)
    tokens = [per[GS.key_text(*k)]["tokens"] for k in train]
    return {"option": option, "exit": code, "seconds": round(time.time() - started), "steps": len(steps),
            "step_seconds": [s["seconds"] for s in steps], "mps_driver_gib_at_step_end": [s.get("mps_driver_gib") for s in steps],
            "projected_peak_gib": peak, "train_tokens": tokens, "within_limit": code == 0 and peak is not None and peak <= MEMORY_LIMIT_GIB}


def decide_v2() -> dict:
    path = HERE / "v2-memory.json"
    if path.exists():
        return read_json(path)
    tried = [v2_memory("ctx6")]
    option = "ctx6" if tried[0]["within_limit"] else None
    if option is None:
        tried.append(v2_memory("ctx5"))
        option = "ctx5" if tried[1]["within_limit"] else None
    record = {"rule": f"ctx6 unless its projected peak exceeds {MEMORY_LIMIT_GIB} GiB or it fails; then ctx5; if ctx5 fails too, V2 is not run",
              "tried": tried, "option": option, "decided_utc": utc()}
    write_json(path, record, exclusive=True)
    note(f"V2 memory smoke: {[(t['option'], t['projected_peak_gib'], t['exit']) for t in tried]} -> option {option}")
    return record


# ----------------------------------------------------------------------------- runs

def train_or_resume(name: str, config: Path, status: dict) -> None:
    run_dir = RUNS / name
    if (run_dir / "outputs" / "summary.json").exists():
        return
    note(f"{name}: started ({L.rel(config)})")
    started = time.time()
    code = run_trainer(name, config, run_dir, RUNS / f"{name}.out")
    status["runs"].setdefault(name, []).append({"config": L.rel(config), "exit": code, "seconds": round(time.time() - started),
                                               "finished_utc": utc()})
    write_json(HERE / "run-status.json", status)
    done = (run_dir / "outputs" / "summary.json").exists()
    note(f"{name}: exited {code} after {round(time.time() - started)} s; outputs {'written' if done else 'not written'}")
    if code != 0 or not done:
        raise SystemExit(f"{name}: trainer exited {code}; see {L.rel(RUNS / f'{name}.out')}")


def final_run_done() -> bool:
    return all((FINAL_RUNS / f"C1-fold{k}" / "outputs" / "summary.json").exists() for k in range(1, 6))


def run_all() -> int:
    if not final_run_done():
        raise SystemExit("the final run has not finished: no variant uses MPS before it ends (master README step 3)")
    path = HERE / "run-status.json"
    status = read_json(path) if path.exists() else {"runs": {}}
    status["driver_started_utc"] = utc()
    write_json(path, status)
    for k in FOLDS:
        train_or_resume(f"V1-fold{k}", config_path(f"V1-fold{k}"), status)
    decision = decide_v2()
    status["v2_option"] = decision["option"]
    write_json(path, status)
    if decision["option"] is None:
        note("V2 not run: both memory smoke tests failed or exceeded the limit")
        return 1
    for k in FOLDS:
        train_or_resume(f"V2-fold{k}", config_path(f"V2-{decision['option']}-fold{k}"), status)
    status["driver_finished_utc"] = utc()
    write_json(path, status)
    note("V1 and V2 runs complete")
    return 0


# ----------------------------------------------------------------------------- report (spec sections 3 and 4; D-202)

def load_score():
    """comparison/score.py (D-202), imported from its path. It must run on the main finetuning labels and student_bootstrap
    modules (the ones this driver scores with), never on the variant copy."""
    spec = importlib.util.spec_from_file_location("comparison_score", SCORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    used = {n: Path(getattr(module, m).__file__).resolve() for n, m in (("labels", "L"), ("student_bootstrap", "SB"))}
    if module.L is not L or used != {n: ROOT / "finetuning" / f"{n}.py" for n in used}:
        raise RuntimeError(f"comparison/score.py imported {used}, not the main finetuning modules")
    return module


def test_keys() -> list:
    return sorted(L.parse_key(t) for k in FOLDS
                  for t in read_json(FINAL_SPLITS / f"fold{k}.json")["parts"]["validation"]["interactions"])


def guard_check(cases: dict) -> dict:
    """D-202's criterion-2 list against the reference: the codes other than the four targets with at least 40 reference
    cases on the test keys, and their counts."""
    found = {f"{a} | {c}": n for (a, c), n in cases.items() if n >= GUARD_MIN_CASES and (a, c) not in TARGETS}
    listed = {f"{a} | {c}": n for (a, c), n in GUARDS.items()}
    return {"listed_by_D202": listed, "found": found, "match": found == listed}


def rule_check() -> int:
    """Before any variant result: D-202's criterion-2 codes and their case counts against the D-190 labels on the test keys
    of folds 3 and 4 (rows with a reference label on the dimension)."""
    S = load_score()
    ref, keys = S.SB.load_reference(LABELS), test_keys()
    cases = {(a, c): sum(c in ref[k][a] for k in keys if ref[k][a] is not None) for a in L.AXES for c in L.DIRECT_CODES[a]}
    check = guard_check(cases)
    record = {"what": f"{NAME}: D-202 criterion-2 codes against the D-190 labels on the test keys of folds 3 and 4 (before any "
                      "variant result)", "created_utc": utc(), "labels": {"path": L.rel(LABELS), "sha256": sha(LABELS)},
              "interactions": len(keys), "students": len({S.SB.student(k[0]) for k in keys}),
              "rows_per_dimension": {a: sum(ref[k][a] is not None for k in keys) for a in L.AXES},
              "reference_cases": {f"{a} | {c}": n for (a, c), n in cases.items()},
              "targets": {f"{a} | {c}": cases[(a, c)] for a, c in TARGETS}, "guard_check": check,
              "score_py": {"path": L.rel(SCORE), "sha256": sha(SCORE)}, "spec": {"path": L.rel(SPEC), "sha256": sha(SPEC)},
              "driver_script_sha256": sha(Path(__file__))}
    write_json(HERE / "rule-check.json", record)
    print(json.dumps({k: record[k] for k in ("interactions", "students", "rows_per_dimension", "targets")}
                     | {"guards_found": check["found"], "match": check["match"]}, ensure_ascii=False, indent=1))
    return 0 if check["match"] else 1


def exact_agreement(entry: dict, name: str) -> Fraction | None:
    top = max(entry["reference"], entry["models"][name]["B"])
    return Fraction(entry["models"][name]["C"], top) if top else None


def apply_rule(scores: dict, v: str) -> tuple[dict, Fraction]:
    """Spec section 4 for variant v against C1, criterion 2 as changed by D-202. Decided in exact fractions of A, B and C
    (a loss of exactly 5 points does not fail), checked against per_code()'s agreement values; the intervals are
    per_code()'s paired differences."""
    def change(axis: str, code: str) -> Fraction:
        e = scores[axis]["codes"][code]
        first, second = exact_agreement(e, v), exact_agreement(e, "C1")
        if first is None or second is None:
            raise ValueError(f"{code}: agreement undefined (no reference case and no prediction)")
        for x, n in ((first, v), (second, "C1")):
            if abs(float(x) - e["models"][n]["agreement"]) > 1e-12:
                raise AssertionError(f"{code}: per_code() agreement of {n} differs from C / max(A, B)")
        return first - second

    def interval(axis: str, code: str):
        return scores[axis]["codes"][code]["comparisons"][f"{v} - C1"]["ci"]

    gains = {(a, c): change(a, c) for a, c in TARGETS}
    guards = {(a, c): change(a, c) for a, c in GUARDS}
    mean_gain = sum(gains.values()) / len(gains)
    raised = [c for (_, c), g in gains.items() if g > 0]
    lowest = min(guards, key=lambda k: guards[k])
    rule1 = len(raised) >= TARGET_MIN_RAISED and mean_gain >= TARGET_MIN_MEAN_GAIN
    rule2 = all(g >= -GUARD_MAX_LOSS for g in guards.values())
    return {"criterion_1": {"changes": {c: float(g) for (_, c), g in gains.items()},
                            "change_ci": {c: interval(a, c) for a, c in TARGETS}, "raised_on": raised,
                            "mean_gain": float(mean_gain), "mean_gain_exact": str(mean_gain), "passes": rule1},
            "criterion_2": {"changes": {c: float(g) for (_, c), g in guards.items()},
                            "change_ci": {c: interval(a, c) for a, c in GUARDS},
                            "lowest_change": {"code": lowest[1], "change": float(guards[lowest])},
                            "losing_more_than_5_points": [c for (_, c), g in guards.items() if g < -GUARD_MAX_LOSS],
                            "passes": rule2},
            "qualifies": rule1 and rule2}, mean_gain


def evaluation_check(scores: dict, refs: dict, raw: dict, keys: list) -> dict:
    """per_code()'s rows and A, B, C (both sides) against the main evaluation module's per-category counts."""
    out = {}
    for n, r in raw.items():
        rep = roundtrip(EV.evaluate(refs, r, keys))
        ok = True
        for axis in L.AXES:
            ev, sc = rep["axes"][axis], scores[axis]
            ok = ok and sc["rows"] == ev["n"]
            for code, c in ev["per_category"].items():
                e = sc["codes"][code]
                ok = ok and (e["reference"], e["models"][n]["B"], e["models"][n]["C"]) == (c["support"], c["predicted"], c["tp"])
                if f"{code} (absent)" in sc["codes"]:
                    e = sc["codes"][f"{code} (absent)"]
                    ok = ok and ((e["reference"], e["models"][n]["B"], e["models"][n]["C"])
                                 == (ev["n"] - c["support"], ev["n"] - c["predicted"], c["tn"]))
        out[n] = ok
    return out


def pool(name: str, paths: list, out_dir: Path) -> tuple[Path, Path]:
    """One model_prediction file (and one scores file with the predicted labels) for a prediction set made by the two fold
    models; run fields describe the pooled set, the origin of each record is kept."""
    records, scores, firsts = [], [], []
    for path in paths:
        source = L.read_predictions(path)
        assert len({tuple(r[f] for f in L.RUN_FIELDS) for r in source}) == 1, f"{path}: mixed provenance"
        firsts.append(source[0])
        sc = {(s["conversation_id"], s["segment_order"]): s for s in (json.loads(line) for _, line in L.jsonl_lines(path.with_name(
            path.name.replace("predictions-", "scores-"))))}
        for r in source:
            key = (r["conversation_id"], r["segment_order"])
            records.append((key, r, sc[key]))
    assert len({f["dataset_manifest_sha256"] for f in firsts}) == 1, f"{name}: the folds used different prepared data"
    run = {"run_id": f"{NAME}-{name}-f34", "model_route": firsts[0]["model_route"].partition("#")[0] + "#final-checkpoints-of-folds-3-4",
           "config_sha256": cv.canonical_sha([f["config_sha256"] for f in firsts]), "dataset_manifest_sha256": firsts[0]["dataset_manifest_sha256"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path, score_path = out_dir / f"predictions-{name}-f34.jsonl", out_dir / f"scores-{name}-f34.jsonl"
    for p in (pred_path, score_path):
        if p.exists():
            p.unlink()                  # rebuilt from the run outputs on every report
    L.write_predictions(pred_path, [L.prediction_record(k[0], k[1], r["labels"], run, r["not_predicted"]) for k, r, _ in sorted(records)])
    with score_path.open("x", encoding="utf-8", newline="\n") as f:
        for k, r, s in sorted(records):
            f.write(json.dumps({"record_kind": "model_scores_with_labels", **run, "original_run_id": r["run_id"],
                                "conversation_id": k[0], "segment_order": k[1], "codes": s["codes"],
                                "probabilities": s["probabilities"], "predicted_labels": r["labels"], "threshold": 0.5},
                               ensure_ascii=False) + "\n")
    return pred_path, score_path


def run_info(name: str) -> dict:
    status = read_json(HERE / "run-status.json")["runs"] if (HERE / "run-status.json").exists() else {}
    out = {}
    for k in FOLDS:
        log_path = RUNS / f"{name}-fold{k}" / "train_log.jsonl"
        if not log_path.exists():
            continue
        log = [json.loads(line) for _, line in L.jsonl_lines(log_path)]
        steps = {e["step"]: e for e in log if e["event"] == "step"}
        gib = [s["mps_driver_gib"] for s in steps.values() if "mps_driver_gib" in s]
        out[k] = {"optimizer_steps": len(steps), "resumes": sum(e["event"] == "resume" for e in log),
                  "wall_seconds": sum(r["seconds"] for r in status.get(f"{name}-fold{k}", [])),
                  "mps_driver_gib_at_step_end_max": max(gib) if gib else None}
    return out


RULE = ("spec section 4 (confirmed by DW; criterion 2 per code, D-202, changed before any variant ran): a variant is adopted "
        "only if, against C1 on folds 3 and 4, (1) it raises simple agreement on at least three of Guided generation, Evaluate, "
        "hand-back: directive and hand-back: asks the student to supply, with a mean gain over the four of at least 5 percentage "
        "points, and (2) none of Information retrieval, Task automation, code, ND, offer, Solve-Provide, Explain and "
        "Correct-Debug (the other codes with at least 40 reference cases in folds 3 and 4) loses more than 5 percentage points "
        "of simple agreement; if more than one qualifies, the largest mean gain; no combinations; if none qualifies, C1's final "
        "run is the result")


def report(test_keys_override=None, sources_override=None, out_dir: Path | None = None) -> int:
    """Each variant against C1 from the final run on the 194 test interactions of folds 3 and 4 (spec section 3 as changed by
    D-202): per-code simple agreement C / max(A, B) with A, B and C, and per-code paired differences, both from per_code()
    of comparison/score.py; then the adoption rule of section 4 (criterion 2 per code). No exact-set rate is reported.
    Overrides are for a dry run on earlier outputs."""
    out_dir = out_dir or HERE
    S = load_score()
    keys = test_keys_override or test_keys()
    sources = sources_override or {
        "C1": [FINAL_RUNS / f"C1-fold{k}" / "outputs" / "predictions-validation.jsonl" for k in FOLDS],
        **{v: [RUNS / f"{v}-fold{k}" / "outputs" / "predictions-validation.jsonl" for k in FOLDS] for v in ("V1", "V2")}}
    sources = {n: [Path(p) for p in ps] for n, ps in sources.items() if all(Path(p).exists() for p in ps)}
    assert "C1" in sources, "C1's fold 3 and 4 predictions are missing"
    pooled = {n: pool(n, ps, out_dir / "oof") for n, ps in sources.items()}
    ref = S.SB.load_reference(LABELS)
    preds = {n: S.SB.load_predictions(pooled[n][0])[0] for n in sources}
    for n, p in preds.items():
        assert set(keys) == set(p), f"{n}: predictions do not cover exactly the {len(keys)} test interactions"
    names = list(preds)
    variants = [n for n in names if n != "C1"]
    scores = roundtrip(S.per_code(ref, preds, keys, [(v, "C1") for v in variants], BOOT_N, BOOT_SEED))
    refs, fingerprint = EV.references_from_prepared(PREPARED[3])
    raw = {n: EV.predictions_from_file(pooled[n][0])[0] for n in names}
    cases = {(a, c): e["reference"] for a, ax in scores.items() for c, e in ax["codes"].items() if not c.endswith(" (absent)")}
    checks = {"guard_list": guard_check(cases),
              "no_prediction_left_out": all(not any(ax["excluded_for_missing_prediction"].values()) for ax in scores.values()),
              "per_code_matches_evaluation_module": evaluation_check(scores, refs, raw, keys),
              "score_py": {"path": L.rel(SCORE), "sha256": sha(SCORE), "labels_module": L.rel(Path(S.L.__file__)),
                           "student_bootstrap_module": L.rel(Path(S.SB.__file__))}}
    applied = (checks["guard_list"]["match"] and checks["no_prediction_left_out"]
               and all(checks["per_code_matches_evaluation_module"].values()))
    comparisons, mean_gain = {}, {}
    for v in variants:
        comparisons[v], mean_gain[v] = apply_rule(scores, v)
    qualified = [v for v in variants if comparisons[v]["qualifies"]]
    best = max((mean_gain[v] for v in qualified), default=None)
    leaders = [v for v in qualified if mean_gain[v] == best]
    adopted = leaders[0] if applied and len(leaders) == 1 else None
    files = {L.rel(p): sha(p) for ps in pooled.values() for p in ps}
    record = {"what": f"{NAME}: V1 and V2 against C1 (final run D-191) on the test interactions of folds 3 and 4; exploratory; C1 "
                      "remains the reported model unless the adoption rule selects a variant", "created_utc": utc(),
              "interactions": len(keys), "students": len({S.SB.student(k[0]) for k in keys}), "models": names,
              "statement": EV.statements(fingerprint)["validation"], "reference": {"path": L.rel(LABELS), "sha256": sha(LABELS)},
              "sources": {n: [L.rel(p) for p in ps] for n, ps in sources.items()},
              "measure": "per_code() of comparison/score.py: per code, simple agreement C / max(A, B) (absent side for "
                         "Solve-Provide and Explain) and the paired differences, students resampled, 10,000 replicates, seed "
                         "20261026, 95% percentile intervals; no exact-set rate or accuracy per dimension (D-201, D-202)",
              "per_code": scores, "rule": RULE, "comparisons": comparisons, "checks": checks, "rule_applied": applied,
              "tied": leaders if len(leaders) > 1 else [], "adopted": adopted,
              "runs": {v: run_info(v) for v in variants},
              "v2_memory": read_json(HERE / "v2-memory.json") if (HERE / "v2-memory.json").exists() else None,
              "spec": {"path": L.rel(SPEC), "sha256": sha(SPEC)}, "master": {"path": L.rel(MASTER), "sha256": sha(MASTER)},
              "driver_script_sha256": sha(Path(__file__)), "files": files}
    write_json(out_dir / "comparison.json", record)
    (out_dir / "summary.md").write_text(summary_md(record), encoding="utf-8")
    print(json.dumps({"interactions": len(keys), "students": record["students"],
                      "checks": {"guard_list": checks["guard_list"]["match"], "no_prediction_left_out": checks["no_prediction_left_out"],
                                 "per_code_matches_evaluation_module": checks["per_code_matches_evaluation_module"]},
                      "targets": {n: {c: scores[a]["codes"][c]["models"][n]["agreement"] for a, c in TARGETS} for n in names},
                      "verdicts": {v: {"raised_on": c["criterion_1"]["raised_on"], "mean_gain": c["criterion_1"]["mean_gain"],
                                       "criterion_2_lowest": c["criterion_2"]["lowest_change"], "qualifies": c["qualifies"]}
                                   for v, c in comparisons.items()},
                      "rule_applied": applied, "tied": record["tied"], "adopted": adopted}, ensure_ascii=False, indent=1))
    if out_dir == HERE:
        note(f"report written (per code, D-202): comparison.json, summary.md; rule applied: {applied}; adopted: {adopted}")
    return 0


def pc(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def pts(x) -> str:
    return "—" if x is None else f"{100 * x:+.1f}"


def ci_pts(ci) -> str:
    return "—" if ci is None else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def summary_md(r: dict) -> str:
    sc, comp, names = r["per_code"], r["comparisons"], r["models"]
    variants = [n for n in names if n != "C1"]
    out = [f"# {NAME}：C1 变体 V1、V2 与 C1（最终运行）在第 3、4 折上的比较（D-194；规则 2 按 D-202；探索性）", "",
           f"性质：{r['statement']}", "",
           "**说明：**",
           "- 探索性比较；除非下面的采用规则选中某个变体，C1 仍是报告的模型。",
           f"- 第 3、4 折按只用标签的规则事先选定（D-193）；{r['interactions']} 条测试互动、{r['students']} 名学生，每折训练 388 条；"
           "与最终运行的 C1 逐条配对。",
           "- 标签为 D-190（研究者的标签；150 部分为讨论后的标签，不是共同裁定；135 与 200 部分为单人编码）。",
           "- 只报告每个代码的简单一致率 C / max(A, B)（A：参考中有该代码的互动数；B：模型中有；C：两者都有）及其配对差"
           "（D-201、D-202），不报告整组一致率或各维度准确率。数值由 comparison/score.py 的 per_code() 计算：学生重抽样 10,000 次"
           "（种子 20261026），95% 百分位区间。",
           "- V3 不运行（D-195）。采用规则在任何变体结果之前固定并经 DW 确认；规则 2 于 10 月 28 日 21:14 由 DW 改为按代码（D-202），"
           "当时还没有任何变体运行。",
           "- 若某变体被采用：选择用到了第 3、4 折，它们也是五折中的两折，所以五折合并结果略偏乐观；届时另报第 1、2、5 折"
           "（291 条，未用于选择）作为干净估计。", "",
           "## 设置", "",
           "- V1：每个头用 focal loss（gamma 2），C1 的类别权重作为 alpha 项；其余与 C1 相同。"]
    d = r.get("v2_memory")
    if d:
        tried = "；".join(f"{x['option']}：峰值 {x['projected_peak_gib']} GiB，退出码 {x['exit']}" for x in d["tried"])
        o = V2_OPTIONS.get(d["option"])
        used = f"实际使用前文 {o['context']} 轮、max_tokens {o['max_tokens']:,}" if o else "两项都未通过，V2 未运行"
        out.append(f"- V2：规格为前文 6 轮、max_tokens 10,240（提示上限 1,536 与回复预留 3,072 不变）；超过 {MEMORY_LIMIT_GIB:g} GiB "
                   f"则改用 5 轮、9,216。显存冒烟测试（最长的 10 条输入）：{tried}；{used}。")
    else:
        out.append("- V2：前文 6 轮、max_tokens 10,240（未运行或无显存冒烟记录）。")

    def rows(codes, part: str) -> list:
        lines = ["| 代码 | A | " + " | ".join(f"{n} 一致率 (B, C)" for n in names) + " | "
                 + " | ".join(f"{v} − C1（百分点）[95% 区间]" for v in variants) + " |",
                 "|---|---|" + "---|" * (len(names) + len(variants))]
        for a, c in codes:
            e = sc[a]["codes"][c]
            lines.append(f"| {c} | {e['reference']} | "
                         + " | ".join(f"{pc(e['models'][n]['agreement'])} ({e['models'][n]['B']}, {e['models'][n]['C']})" for n in names)
                         + " | " + " | ".join(f"{pts(comp[v][part]['changes'][c])} {ci_pts(comp[v][part]['change_ci'][c])}"
                                              for v in variants) + " |")
        return lines

    out += ["", "## 采用规则（第 4 节；规则 2 按 D-202）", "",
            "规则 1：四个目标代码中至少 3 个的一致率提高，且四个的平均增益至少 5 个百分点。", ""] + rows(TARGETS, "criterion_1")
    out += ["", "规则 2（D-202）：其他至少有 40 个参考案例的代码，一致率都不得下降超过 5 个百分点。", ""] + rows(GUARDS, "criterion_2")
    out += ["", "| 变体 | 提高的目标代码 | 平均增益（百分点） | 规则 1 | 规则 2 中变化最小的代码 | 规则 2 | 是否合格 |",
            "|---|---|---|---|---|---|---|"]
    for v in variants:
        c1, c2 = comp[v]["criterion_1"], comp[v]["criterion_2"]
        out.append(f"| {v} | {len(c1['raised_on'])} / 4 | {pts(c1['mean_gain'])} | {'成立' if c1['passes'] else '不成立'} | "
                   f"{c2['lowest_change']['code']} {pts(c2['lowest_change']['change'])} | {'成立' if c2['passes'] else '不成立'} | "
                   f"{'合格' if comp[v]['qualifies'] else '不合格'} |")
    if not r["rule_applied"]:
        verdict = "未作判定：下面的核对没有全部通过，需要检查。"
    elif r["tied"]:
        verdict = f"{'、'.join(r['tied'])} 都合格且平均增益相同；规则没有规定如何取舍，需要 DW 决定。"
    elif r["adopted"]:
        verdict = f"采用 {r['adopted']}（按规则，平均增益最大的合格变体）。"
    else:
        verdict = "没有变体合格；C1 的最终运行即为结果。"
    out += ["", f"**判定：{verdict}**", "",
            "## 每个代码的简单一致率与配对差（全部代码；Solve-Provide 与 Explain 另给“没有该代码”一侧）", "",
            "一致率后括号内为 (B, C)；维度后括号内为计分的行数。", "",
            "| 维度 | 代码 | A | " + " | ".join(names) + " | " + " | ".join(f"{v} − C1（百分点）[95% 区间]" for v in variants) + " |",
            "|---|---|---|" + "---|" * (len(names) + len(variants))]
    for a, ax in sc.items():
        for code, e in ax["codes"].items():
            cells = [f"{pc(e['models'][n]['agreement'])} ({e['models'][n]['B']}, {e['models'][n]['C']})" for n in names]
            diffs = [f"{pts(e['comparisons'][f'{v} - C1']['difference'])} {ci_pts(e['comparisons'][f'{v} - C1']['ci'])}" for v in variants]
            out.append(f"| {a}（{ax['rows']}） | {code.replace(' (absent)', '（没有该代码）')} | {e['reference']} | "
                       + " | ".join(cells + diffs) + " |")
    out += ["", "## 运行时间与内存", "", "| 变体 | 折 | 优化步 | 恢复次数 | 用时（秒） | 每步结束时 MPS 驱动内存峰值（GiB） |",
            "|---|---|---|---|---|---|"]
    for v in variants:
        for k, x in r["runs"].get(v, {}).items():
            out.append(f"| {v} | {k} | {x['optimizer_steps']} | {x['resumes']} | {x['wall_seconds']} | {x['mps_driver_gib_at_step_end_max']} |")
    ck = r["checks"]
    yes = lambda b: "是" if b else "否"  # noqa: E731
    out += ["", "## 核对", "",
            f"- 规则 2 的代码清单与参考标签相符（除四个目标代码外，参考案例至少 40 个的代码及其案例数与 D-202 所列完全相同）：{yes(ck['guard_list']['match'])}。",
            f"- 没有互动因缺少预测而被排除：{yes(ck['no_prediction_left_out'])}。",
            "- per_code() 的行数与 A、B、C（含“没有该代码”一侧）与评估模块 finetuning/evaluation.py 的逐类计数一致："
            + "，".join(f"{n} {yes(ok)}" for n, ok in ck["per_code_matches_evaluation_module"].items()) + "。",
            "- 规则按 A、B、C 的精确分数判定（恰好下降 5 个百分点不算超过），并与 per_code() 的一致率核对；区间是 per_code() 的配对差。",
            f"- comparison/score.py sha256 `{ck['score_py']['sha256']}`，它导入的是 `{ck['score_py']['labels_module']}` 与 "
            f"`{ck['score_py']['student_bootstrap_module']}`（主代码，不是变体副本）。",
            "- 每个代码的概率：`oof/scores-<变体>-f34.jsonl`（每条测试互动、每个头、每个代码的概率，附 0.5 下的预测标签）。",
            "- 变体代码在本文件夹的 finetuning/ 副本中（与工作区 finetuning/ 的差异：code-diff-vs-finetuning.patch），没有改动正在运行的作业会重新加载的文件。",
            "", "## 文件", ""] + [f"- `{k}` sha256 `{v}`" for k, v in r["files"].items()] + [""]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("plan", "cpu-smoke", "run", "rule-check", "report"))
    a = p.parse_args(argv)
    return {"plan": plan, "cpu-smoke": cpu_smoke, "run": run_all, "rule-check": rule_check, "report": report}[a.command]()


if __name__ == "__main__":
    raise SystemExit(main())
