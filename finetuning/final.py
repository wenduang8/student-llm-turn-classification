#!/usr/bin/env python3
"""Final C1 run on the 485 interactions with the D-190 training labels (D-191; D-106 preliminary development). Not the
reported evaluation protocol's last word; development validation against the researcher's labels.
Spec: README.txt of mac-training-485-final-20261028.zip (saved as docs/planning/c1-485-final-readme-20261028.txt), with
step 1 of mac-variants-spec-20261028b.zip's README (saved as docs/planning/c1-variants-master-readme-20261028b.txt): its
step 8 (one model on all 485) is skipped for now. The design is the D-186 run's (data/results/dw485-c1-20261028/c485.py,
whose helpers this imports unchanged): only the labels change. Prints counts and verdicts only, never interaction text.

    .venv/bin/python -I data/results/dw485d-c1-final-20261028/final.py check     # transfer, patch, labels vs D-186, parts
    .venv/bin/python -I data/results/dw485d-c1-final-20261028/final.py prepare   # prepare_data.py with the D-186 corpus
    .venv/bin/python -I data/results/dw485d-c1-final-20261028/final.py plan      # reused folds, configs, input identity, estimate
    .venv/bin/python -I data/results/dw485d-c1-final-20261028/final.py run       # train/resume the five folds; resumable

  folds    the D-186 rosters (data/splits/cv485-20261028/fold<k>.json) rewritten as data/splits/cv485-20261028d/fold<k>.json,
           because a split file is bound to the hashes of the prepared data it was made from (grouped_split.load_split): the
           'parts' rosters are copied unchanged and checked equal, and cv.assign_folds on the new data reproduces them.
  configs  the D-186 configs with only data.prepared_dir, data.split_file, data.label_source_note, status and
           validation.purpose changed (checked key by key).
  inputs   every input must be token-for-token identical to its D-186 input (485/485), or the plan stops.
  scores   the trainer already writes each test interaction's per-code probabilities (outputs/scores-validation.jsonl,
           keyed by conversation_id and segment_order, beside predictions-validation.jsonl): no code change. The report
           script pools them with the predicted labels into one out-of-fold scores file.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # this copy is in finetuning/
HERE = ROOT / "data" / "results" / "dw485d-c1-final-20261028"     # outputs, where the original ran
D186 = ROOT / "data" / "results" / "dw485-c1-20261028"
sys.path.insert(0, str(Path(__file__).resolve().parent))         # c485.py is beside this copy
import c485 as B  # noqa: E402  (the D-186 driver: Space, run_all, part files, helpers)

cv, EV, GS, L = B.cv, B.EV, B.GS, B.L
TRANSFER_ZIP = ROOT / "mac-training-485-final-20261028.zip"
TRANSFER = ROOT / "data" / "transfer" / "mac-training-485-final-20261028"
MASTER_ZIP = ROOT / "mac-variants-spec-20261028b.zip"
MASTER = ROOT / "data" / "transfer" / "mac-variants-spec-20261028b"
PATCH = TRANSFER / "code-update" / "labels-accept-D187-D190.patch"
ALT_PATCH = TRANSFER / "code-update" / "labels-accept-D182-D190.patch"
SPEC = ROOT / "docs" / "planning" / "c1-485-final-readme-20261028.txt"
MASTER_SPEC = ROOT / "docs" / "planning" / "c1-variants-master-readme-20261028b.txt"
PLACED = {"data/labels/dw485-reviewed-20261028d.jsonl": "data/labels/dw485-reviewed-20261028d.jsonl",
          "data/labels/dw485-reviewed-20261028d.manifest.json": "data/labels/dw485-reviewed-20261028d.manifest.json",
          "data/processed/transcripts-485-conversations-20261028.csv": "data/processed/transcripts-485-conversations-20261028.csv",
          "README.txt": "docs/planning/c1-485-final-readme-20261028.txt"}
LABELS = ROOT / "data" / "labels" / "dw485-reviewed-20261028d.jsonl"
LABELS_MANIFEST = ROOT / "data" / "labels" / "dw485-reviewed-20261028d.manifest.json"
CORPUS = ROOT / "data" / "processed" / "transcripts-485-conversations-20261028.csv"
PREPARED = ROOT / "data" / "training" / "dw485-reviewed-20261028d"
NAME = "cv485-20261028d"          # splits data/splits/cv485-20261028d, runs data/runs/cv485-20261028d
PARTS, ALL = B.PARTS, B.ALL
PART_DECISIONS = {"150": {"D-168", "D-169", "D-182", "D-187", "D-188", "D-190"}, "135": {"D-166", "D-167", "D-187", "D-189"},
                  "200": {"D-185"}}
PURPOSE = ("Cross-validation test fold (student-grouped 5-fold, cv485-20261028d = the D-186 rosters; D-191 final C1 run on 485 "
           "interactions with the D-190 training labels): held-out students used only to produce this fold's out-of-fold "
           "predictions at threshold 0.5, scored once against the D-190 labels. No per-epoch loss is computed on them "
           "(per_epoch_loss false). Not used to choose epochs, checkpoints, thresholds, input policy or models.")
LABEL_NOTE = ("data/labels/dw485-reviewed-20261028d.jsonl (D-190; the training labels of the final run D-191): 150 the researcher's "
              "labels after discussion (v15; D-168, D-169, D-182, D-187, D-188, D-190; 43 students; where disagreement remains, "
              "the researcher's code), 135 researcher-coded and reviewed (v4; D-166, D-167, D-187, D-189; 41 students; single "
              "coder) and 200 researcher-coded with an AI-assisted check, reviewed (v1; D-185; 48 students; single coder); no "
              "student in two parts. Not independent coding, not joint adjudication. Report statements are derived from the "
              "prepared manifest's record kinds, not from this note.")
ALLOWED_VS_D186 = ("status", "data.prepared_dir", "data.split_file", "data.label_source_note", "validation.purpose")
REAL = B.Space(NAME, PREPARED, LABELS, HERE)
utc, sha, read_json, write_json, roundtrip, canon = B.utc, B.sha, B.read_json, B.write_json, B.roundtrip, B.canon


def sums_of(directory: Path) -> dict:
    out = {}
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            out[p.strip()] = h
    return out


def check() -> int:
    sums, master_sums = sums_of(TRANSFER), sums_of(MASTER)
    extracted = {p: sha(TRANSFER / p) == h for p, h in sums.items()}
    master_extracted = {p: sha(MASTER / p) == h for p, h in master_sums.items()}
    placed = {dest: (ROOT / dest).exists() and sha(ROOT / dest) == sums[src] for src, dest in PLACED.items()}
    patch = {"D187-D190": subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=ROOT, capture_output=True).returncode == 0}
    d186_check = read_json(D186 / "check.json")
    items = L.read_reviewed_direct_jsonl(LABELS)
    keys = [i.key for i in items]
    by_key = {i.key: i for i in items}
    old = {i.key: i for i in L.read_reviewed_direct_jsonl(B.LABELS)}
    part = {L.parse_key(t): p for p in PARTS for t in d186_check["part_keys"][p]}
    decisions = {p: dict(Counter(i.source_record["review_provenance"]["decision"] for i in items if part[i.key] == p)) for p in PARTS}
    manifest = read_json(LABELS_MANIFEST)
    cells = {p: {a: sum(canon(by_key[k].labels)[a] != canon(old[k].labels)[a] for k in keys if part[k] == p) for a in L.AXES}
             for p in PARTS}
    rows_changed = {p: sum(canon(by_key[k].labels) != canon(old[k].labels) for k in keys if part[k] == p) for p in PARTS}
    nulls = {p: {a: {"D-190": sum(by_key[k].labels[a] is None for k in keys if part[k] == p),
                     "D-186": sum(old[k].labels[a] is None for k in keys if part[k] == p)} for a in L.AXES} for p in PARTS}
    corpus = L.load_corpus(CORPUS)
    ex, missing = L.attach(items, corpus, L.DEFAULT_CONTEXT)
    ex = {(e["conversation_id"], e["segment_order"]): e for e in ex}
    students = {p: sorted({ex[k]["student_id"] for k in keys if part[k] == p}) for p in PARTS}
    conversations = {p: sorted({k[0] for k in keys if part[k] == p}) for p in PARTS}
    out = {
        "what": "D-191 input checks (2026-10-28): transfer, patch, D-190 labels vs the D-186 labels, parts, corpus; counts only",
        "created_utc": utc(),
        "transfer": {"zip": L.rel(TRANSFER_ZIP), "zip_sha256": sha(TRANSFER_ZIP), "extracted_to": L.rel(TRANSFER),
                     "files_match_SHA256SUMS": extracted, "placed_copies_match": placed, "patch_applied": patch,
                     "master_zip": L.rel(MASTER_ZIP), "master_zip_sha256": sha(MASTER_ZIP), "master_files_match_SHA256SUMS": master_extracted},
        "labels": {"path": L.rel(LABELS), "sha256": sha(LABELS), "records": len(items), "distinct_keys": len(set(keys)),
                   "manifest_sha256_matches_file": manifest["sha256"] == sha(LABELS), "manifest_note": manifest.get("note"),
                   "same_interactions_as_d186": set(keys) == set(old)},
        "parts": {p: {"interactions": sum(part[k] == p for k in keys), "students": len(students[p]),
                      "conversations": len(conversations[p]), "review_decisions": decisions[p],
                      "manifest": next(m for m in manifest["parts"] if m["part"] == p)} for p in PARTS},
        "part_rule": "the D-186 parts (data/results/dw485-c1-20261028/check.json part_keys)",
        "cells_changed_vs_d186": {"by_part_and_axis": cells, "total": sum(v for c in cells.values() for v in c.values()),
                                  "interactions_changed": rows_changed},
        "null_references": nulls,
        "corpus": {"path": L.rel(CORPUS), "sha256": sha(CORPUS), "equals_d186_corpus": d186_check["corpus_used"]["sha256"] == sha(CORPUS),
                   "missing_of_485": len(missing)},
        "students_in_two_parts": sum(n > 1 for n in Counter(s for p in PARTS for s in students[p]).values()),
    }
    ok = (all(extracted.values()) and all(master_extracted.values()) and all(placed.values()) and patch["D187-D190"]
          and len(set(keys)) == len(keys) == 485 and out["labels"]["manifest_sha256_matches_file"] and out["labels"]["same_interactions_as_d186"]
          and all(out["parts"][p]["interactions"] == out["parts"][p]["manifest"]["records"]
                  and out["parts"][p]["students"] == out["parts"][p]["manifest"]["students"]
                  and out["parts"][p]["conversations"] == out["parts"][p]["manifest"]["conversations"]
                  and set(decisions[p]) <= PART_DECISIONS[p] for p in PARTS)
          and out["corpus"]["equals_d186_corpus"] and not missing and out["students_in_two_parts"] == 0)
    out["all_checks_pass"] = ok
    write_json(HERE / "check.json", out)
    show = {k: v for k, v in out.items() if k not in ("what", "created_utc")}
    show["labels"] = {k: v for k, v in show["labels"].items() if k != "manifest_note"}
    show["parts"] = {p: {k: v for k, v in show["parts"][p].items() if k != "manifest"} for p in PARTS}
    print(json.dumps(show, ensure_ascii=False, indent=1))
    return 0 if ok else 1


def prepare() -> int:
    if (PREPARED / "manifest.json").exists():
        print(f"{L.rel(PREPARED)} already prepared")
        return 0
    report = B.run_prepare(LABELS, PREPARED, HERE / "prepare-report.json")       # the D-186 corpus (B's check.json)
    print(json.dumps({k: report[k] for k in ("interactions", "students", "conversations", "command")}, indent=1))
    return 0


def write_fold_split(k: int, examples: list, part_of: dict, reproduced: dict) -> dict:
    old_path = B.REAL.fold_split(k)
    old = read_json(old_path)
    by_key = {(e["conversation_id"], e["segment_order"]): e for e in examples}
    rows = {n: [by_key[L.parse_key(t)] for t in old["parts"][n]["interactions"]] for n in ("train", "validation")}
    if reproduced[k] != old["parts"]["validation"]["students"]:
        raise AssertionError(f"fold {k}: cv.assign_folds on the new data does not reproduce the D-186 test students")
    record = copy.deepcopy(old)
    record.update({"status": f"{NAME} fold {k} of {B.FOLDS}: the D-186 roster of {L.rel(old_path)} unchanged, bound to the D-190 "
                             "prepared data; D-191 final run (D-106 preliminary development)",
                   "validation_purpose": PURPOSE, "created_utc": utc(), "prepared": GS.prepared_fingerprint(PREPARED),
                   "driver_script_sha256": sha(Path(__file__)), "split_module_sha256": sha(Path(GS.__file__)),
                   "reused_roster_from": {"path": L.rel(old_path), "sha256": sha(old_path)},
                   "label_parts": {n: dict(Counter(part_of[(e["conversation_id"], e["segment_order"])] for e in r)) for n, r in rows.items()},
                   "label_support": {n: GS.label_support(r) for n, r in rows.items()}})
    assert record["parts"] == old["parts"]
    path = REAL.fold_split(k)
    if path.exists():
        existing = read_json(path)
        if existing["parts"] != record["parts"] or existing["prepared"] != record["prepared"]:
            raise ValueError(f"{path} exists with a different roster or prepared data")
        return existing
    write_json(path, record, exclusive=True)
    return record


def make_config(k: int) -> tuple[dict, list]:
    template = read_json(B.REAL.config(k))
    cfg = copy.deepcopy(template)
    cfg["status"] = (f"C1 fold {k} ({NAME}; D-191 final run on 485 interactions with the D-190 training labels; D-106 preliminary "
                     "development): the D-186 configuration unchanged (C1 of cv-20261022: 8 epochs, head_lr 0.001, class weights "
                     "training_part_balanced_v1, budgets 1536/3072/7168 with the response reservation, seed 20261017), decoded at "
                     "0.5; no per-epoch test-fold loss; MPS cache released after every optimizer step. Labels: the researcher's "
                     "labels (D-190); not joint adjudication.")
    cfg["data"].update(prepared_dir=L.rel(PREPARED), split_file=L.rel(REAL.fold_split(k)), label_source_note=LABEL_NOTE)
    cfg["validation"]["purpose"] = PURPOSE
    changes = B.changed_keys(template, cfg)
    stray = B.within(changes, ALLOWED_VS_D186)
    if stray:
        raise AssertionError(f"fold {k}: the config differs from D-186's in {stray}")
    path = REAL.config(k)
    if path.exists():
        if read_json(path) != cfg:
            raise ValueError(f"{path} exists with different content")
    else:
        write_json(path, cfg, exclusive=True)
    return cfg, changes


def input_identity(part_of: dict) -> dict:
    """Tokenizer only (CPU): every D-191 input against its D-186 input, and what C1's budgets cut."""
    import model_inputs as MI
    import train_multihead as T
    cfg = read_json(REAL.config(1))
    encode = T.tokenizer_encoder(T.model_snapshot(cfg["model"])[0])
    policy = MI.InputPolicy.from_config(cfg["input"])
    if policy != MI.InputPolicy.from_config(read_json(B.REAL.config(1))["input"]):
        raise AssertionError("the input policy differs from D-186's")
    new = T.build_inputs(T.load_data(PREPARED, REAL.fold_split(1)), encode, policy)[0]
    old = T.build_inputs(T.load_data(B.PREPARED, B.REAL.fold_split(1)), encode, policy)[0]
    same = sum(new[k]["input_ids"] == old[k]["input_ids"] and new[k]["student_pool"] == old[k]["student_pool"]
               and new[k]["ai_pool"] == old[k]["ai_pool"] for k in old)
    out = {"compared": len(old), "identical_inputs": same, "keys_equal": set(new) == set(old)}
    for name, ks in ((ALL, list(new)), *((p, [k for k in new if part_of[k] == p]) for p in PARTS)):
        flags = [B.loses(new[k]["report"]) for k in ks]
        out[name] = {"interactions": len(ks), **{f"loses_{x}": sum(f[x] for f in flags) for x in ("context", "prompt", "response")},
                     "loses_any": sum(any(f.values()) for f in flags)}
    out["truncated_responses"] = [{"interaction": GS.key_text(*k), "part": part_of[k], "response_tokens": new[k]["report"]["response_tokens"],
                                   "response_tokens_omitted": new[k]["report"]["response_tokens_omitted"]}
                                  for k in sorted(new) if new[k]["report"]["response_tokens_omitted"] > 0]
    if same != len(old) or not out["keys_equal"]:
        raise AssertionError(f"STOP: inputs differ from D-186's ({same}/{len(old)} identical); the labels must be the only change")
    return out


def plan() -> dict:
    HERE.mkdir(parents=True, exist_ok=True)
    examples = GS.load_examples(PREPARED)
    old_examples = GS.load_examples(B.PREPARED)
    new_fp, old_fp = GS.prepared_fingerprint(PREPARED), GS.prepared_fingerprint(B.PREPARED)
    if new_fp["keys_sha256"] != old_fp["keys_sha256"] or len(examples) != len(old_examples):
        raise AssertionError("the prepared interactions differ from D-186's")
    if [(e["conversation_id"], e["segment_order"], e["student_id"]) for e in examples] != \
            [(e["conversation_id"], e["segment_order"], e["student_id"]) for e in old_examples]:
        raise AssertionError("interaction keys or students differ from D-186's")
    part_of = B.part_map()
    folds = cv.assign_folds(examples)
    reproduced = {k: sorted(s for s, f in folds["fold_of"].items() if f == k) for k in range(1, B.FOLDS + 1)}
    splits, configs = {}, {}
    for k in range(1, B.FOLDS + 1):
        write_fold_split(k, examples, part_of, reproduced)
        _, changes = make_config(k)
        splits[k] = {"path": L.rel(REAL.fold_split(k)), "sha256": sha(REAL.fold_split(k)),
                     "roster_from": L.rel(B.REAL.fold_split(k)), "roster_identical": True}
        configs[k] = {"path": L.rel(REAL.config(k)), "sha256": sha(REAL.config(k)), "template": L.rel(B.REAL.config(k)),
                      "changed_keys_vs_d186": changes}
    import train_multihead as T
    for k in range(1, B.FOLDS + 1):
        T.check_config(read_json(REAL.config(k)))
    part_files = B.write_part_labels(REAL, part_of)
    inputs = input_identity(part_of)
    d186_status = read_json(D186 / "run-status.json")["runs"]
    per_fold = {k: sum(r["seconds"] for r in d186_status[f"C1-fold{k}"]) for k in range(1, B.FOLDS + 1)}
    estimate = {"basis": "the D-186 runs' measured wall time per fold (same inputs, configuration and code apart from the labels)",
                "per_fold_seconds": per_fold, "total_hours": round(sum(per_fold.values()) / 3600, 2)}
    diff = B.code_patch()
    if diff:
        (HERE / "code-diff-vs-head.patch").write_text(diff, encoding="utf-8")
    record = {"what": f"{NAME} plan (fixed before the runs; D-191 final C1 run, D-106 preliminary development)", "created_utc": utc(),
              "spec": {"readme": {"path": L.rel(SPEC), "sha256": sha(SPEC)}, "master_readme": {"path": L.rel(MASTER_SPEC), "sha256": sha(MASTER_SPEC)},
                       "skipped": "README step 8 (a model on all 485), per the master README step 1"},
              "dataset": {"prepared": L.rel(PREPARED), "manifest_sha256": sha(PREPARED / "manifest.json"), "interactions": len(examples),
                          "students": len({e["student_id"] for e in examples}), "conversations": len({e["conversation_id"] for e in examples}),
                          "labels": {"path": L.rel(LABELS), "sha256": sha(LABELS)}, "keys_sha256": new_fp["keys_sha256"]},
              "parts": {p: sorted(GS.key_text(*k) for k, v in part_of.items() if v == p) for p in PARTS},
              "part_label_files": part_files, "splits": splits, "configs": configs,
              "folds": [{"fold": k, "train": len(read_json(REAL.fold_split(k))["parts"]["train"]["interactions"]),
                         "test": len(read_json(REAL.fold_split(k))["parts"]["validation"]["interactions"]),
                         "test_by_part": read_json(REAL.fold_split(k))["label_parts"]["validation"]} for k in range(1, B.FOLDS + 1)],
              "test_fold_rosters": {k: read_json(REAL.fold_split(k))["parts"]["validation"] for k in range(1, B.FOLDS + 1)},
              "bootstrap": {"script": L.rel(B.BOOTSTRAP), "replicates": B.BOOT_REPLICATES, "seed": B.BOOT_SEED,
                            "references": {s: L.rel(REAL.reference(s)) for s in B.SETS}},
              "input_check": inputs, "estimate": estimate,
              "code": {"head": B.git("rev-parse", "HEAD").strip(), "patch": L.rel(HERE / "code-diff-vs-head.patch") if diff else None,
                       "patch_sha256": L.sha256_bytes(diff.encode("utf-8")) if diff else None, "trainer_sha256": sha(B.TRAIN),
                       "labels_module_sha256": sha(ROOT / "finetuning" / "labels.py"),
                       "scores": "no trainer change: outputs/scores-validation.jsonl already holds per-code probabilities per test interaction"},
              "driver_script_sha256": sha(Path(__file__)), "base_driver_sha256": sha(Path(B.__file__))}
    path = HERE / "plan.json"
    if path.exists():
        old = read_json(path)
        if old["test_fold_rosters"] != roundtrip(record["test_fold_rosters"]) or old["configs"] != roundtrip(record["configs"]):
            raise ValueError(f"{path} exists with a different plan")
        print(f"{L.rel(path)} exists and matches; not rewritten")
    else:
        write_json(path, record, exclusive=True)
    print(json.dumps({"dataset": {k: v for k, v in record["dataset"].items() if k != "labels"}, "folds": record["folds"],
                      "config_changed_keys_vs_d186": sorted({c for v in configs.values() for c in v["changed_keys_vs_d186"]}),
                      "input_check": {k: v for k, v in inputs.items() if k != "truncated_responses"},
                      "truncated_responses": [{k: v for k, v in r.items() if k != "interaction"} for r in inputs["truncated_responses"]],
                      "estimate_hours": estimate["total_hours"]}, ensure_ascii=False, indent=1))
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("check", "prepare", "plan", "run"))
    a = p.parse_args(argv)
    if a.command == "check":
        return check()
    if a.command == "prepare":
        return prepare()
    if a.command == "plan":
        plan()
        return 0
    if not (HERE / "plan.json").exists():
        raise SystemExit("no plan.json: run `plan` first")
    return B.run_all(REAL)


if __name__ == "__main__":
    raise SystemExit(main())
