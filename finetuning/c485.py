#!/usr/bin/env python3
"""C1 trial run on the 485 reviewed interactions (D-186; D-106 preliminary development). Not the reported evaluation
and not the final ground truth. Spec: README.txt of mac-training-485-20261028.zip (saved verbatim as
docs/planning/c1-485-trial-readme-20261028.txt) on the base procedure docs/planning/c1-285-devrun-spec-20261026.md
(D-170, whose driver data/results/dw285-c1-20261026/c285.py this extends to three parts). Prints counts and verdicts
only, never interaction text.

    .venv/bin/python -I data/results/dw485-c1-20261028/c485.py check      # transfer, labels, parts, corpus coverage
    .venv/bin/python -I data/results/dw485-c1-20261028/c485.py prepare    # finetuning/prepare_data.py, context 3
    .venv/bin/python -I data/results/dw485-c1-20261028/c485.py plan       # folds, configs, part files, inputs, estimate
    .venv/bin/python -I data/results/dw485-c1-20261028/c485.py run        # train/resume the five folds; resumable
    .venv/bin/python -I data/results/dw485-c1-20261028/c485.py assemble   # one out-of-fold model_prediction file
    .venv/bin/python -I data/results/dw485-c1-20261028/c485.py report     # metrics.json, summary.md, bootstrap files
    .venv/bin/python -I data/results/dw485-c1-20261028/c485.py smoke      # the same chain on a few students, 1 epoch

Design (fixed by the spec; no search; the test folds are never used for any choice):
  folds   as D-170: cv-20261022's procedure and seed by calling that CV's own code (cv.assign_folds in
          data/results/v9-cv-20261022/cv.py): students sorted by ID, shuffled with random.Random(20261017), stably
          sorted by number of interactions (largest first); each joins the fold with the fewest interactions so far
          (ties: lower fold). 5 folds; no inner holdout.
  model   C1 as in D-170: cv-20261022's C1-fold<k>.json with only the data paths, the label note, the status, the
          validation purpose, validation.per_epoch_loss = false and memory.mps_empty_cache_every_steps = 1 changed
          (checked key by key; against D-170's own config only paths, note, status and purpose may differ).
  decode  threshold 0.5: the trainer's own outputs/predictions-validation.jsonl of each fold.
  parts   "150": the interactions of data/labels/dw150-reviewed-v9.jsonl (reviewed again as v12, D-182);
          "135": the other interactions of data/labels/dw285-reviewed-20261026.jsonl (v2, D-166/D-167);
          "200": the rest (researcher-coded with an AI-assisted check, reviewed v1, D-185). Checked against the manifest.
  CIs     finetuning/student_bootstrap.py, 10,000 replicates, seed 20261026, for the 485 (reference: the label file) and
          for each part (reference: a byte-identical subset of the label file holding that part's records).
  time    estimated before the run and reported (README item 6); no automatic stop.
"""
from __future__ import annotations

import argparse
import copy
import json
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # this copy is in finetuning/
HERE = ROOT / "data" / "results" / "dw485-c1-20261028"           # outputs, where the original ran
CVDIR = ROOT / "data" / "results" / "v9-cv-20261022"
D170 = ROOT / "data" / "results" / "dw285-c1-20261026"
sys.path[:0] = [str(ROOT / "finetuning"), str(CVDIR)]
import cv  # noqa: E402  (the cv-20261022 procedure: assign_folds, part_record, majority_labels, headline, fold_table)

EV, GS, L = cv.EV, cv.GS, cv.L
PY = ROOT / ".venv" / "bin" / "python"
TRAIN = ROOT / "finetuning" / "train_multihead.py"
PREPARE = ROOT / "finetuning" / "prepare_data.py"
BOOTSTRAP = ROOT / "finetuning" / "student_bootstrap.py"
SPEC = ROOT / "docs" / "planning" / "c1-485-trial-readme-20261028.txt"
BASE_SPEC = ROOT / "docs" / "planning" / "c1-285-devrun-spec-20261026.md"
TRANSFER_ZIP = ROOT / "mac-training-485-20261028.zip"
TRANSFER = ROOT / "data" / "transfer" / "mac-training-485-20261028"
PATCH = TRANSFER / "code-update" / "labels-accept-D182-D185.patch"
PLACED = {p: p for p in ("data/labels/dw485-reviewed-20261028.jsonl", "data/labels/dw485-reviewed-20261028.manifest.json",
                         "data/processed/transcripts-485-conversations-20261028.csv")}
PLACED["README.txt"] = "docs/planning/c1-485-trial-readme-20261028.txt"
LABELS = ROOT / "data" / "labels" / "dw485-reviewed-20261028.jsonl"
LABELS_MANIFEST = ROOT / "data" / "labels" / "dw485-reviewed-20261028.manifest.json"
V9_LABELS = ROOT / "data" / "labels" / "dw150-reviewed-v9.jsonl"
LABELS_285 = ROOT / "data" / "labels" / "dw285-reviewed-20261026.jsonl"
CORPORA = (("mac-150", ROOT / "data" / "processed" / "dw150-context.csv"),
           ("mac-285", ROOT / "data" / "processed" / "transcripts-285-conversations-20261026.csv"),
           ("transfer-485", ROOT / "data" / "processed" / "transcripts-485-conversations-20261028.csv"))
PREPARED = ROOT / "data" / "training" / "dw485-reviewed-20261028"
V9_PREPARED = ROOT / "data" / "training" / "dw150-reviewed-v9"
D170_PREPARED = ROOT / "data" / "training" / "dw285-reviewed-20261026"
BASES = {"cv-20261022": {"runs": ROOT / "data" / "runs" / "cv-20261022", "splits": ROOT / "data" / "splits" / "cv-20261022",
                         "status": CVDIR / "run-status.json", "per_epoch_validation": True},
         "D-170": {"runs": ROOT / "data" / "runs" / "cv285-20261026", "splits": ROOT / "data" / "splits" / "cv285-20261026",
                   "status": D170 / "run-status.json", "per_epoch_validation": False}}

NAME, SEED, FOLDS, EPOCHS = "cv485-20261028", cv.SEED, cv.FOLDS, cv.C1_EPOCHS
BOOT_SEED, BOOT_REPLICATES = 20261026, 10000
ALL, PARTS = "485", ("150", "135", "200")
SETS = (ALL,) + PARTS
PART_DECISIONS = {"150": {"D-168", "D-169", "D-182"}, "135": {"D-166", "D-167"}, "200": {"D-185"}}
PART_ZH = {"150": "研究者复核（v12）", "135": "单人编码（v2）", "200": "单人编码，经 AI 辅助检查（v1）"}
OOF_NAME = "predictions-C1-485-oof.jsonl"
# forward + backward time per pass on this Mac (cv-20261022 profile, progress.md 2026-10-23); used only to scale
COST = lambda t: 1.007e-3 * t + 0.383e-6 * t * t  # noqa: E731
PURPOSE = ("Cross-validation test fold (student-grouped 5-fold, cv485-20261028; D-186 trial run on 485 interactions): "
           "held-out students used only to produce this fold's out-of-fold predictions at threshold 0.5, scored once "
           "against the 485 reviewed labels. No per-epoch loss is computed on them (per_epoch_loss false). Not used to "
           "choose epochs, checkpoints, thresholds, input policy or models.")
LABEL_NOTE = ("data/labels/dw485-reviewed-20261028.jsonl (D-186): 150 researcher-reviewed interactions (v12; D-168, D-169, "
              "D-182; 43 students), 135 researcher-coded and reviewed interactions (v2; D-166, D-167; 41 students) and 200 "
              "researcher-coded interactions with an AI-assisted check, reviewed (v1; D-185; 48 students); no student in two "
              "parts. The researcher's labels in the AI-assisted review format: not independent coding, not joint "
              "adjudication; trial labels, not the final ground truth. Report statements are derived from the prepared "
              "manifest's record kinds, not from this note.")
ALLOWED_CHANGES = ("status", "data.prepared_dir", "data.split_file", "data.label_source_note", "validation.purpose",
                   "validation.per_epoch_loss", "memory")
D170_ALLOWED = ("status", "data.prepared_dir", "data.split_file", "data.label_source_note", "validation.purpose")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def sha(path: Path) -> str:
    return L.sha256_file(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def roundtrip(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False))


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def changed_keys(a: dict, b: dict) -> list:
    fa, fb = flat(a), flat(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>"))


def within(changes: list, allowed: tuple) -> list:
    return [c for c in changes if not any(c == a or c.startswith(a + ".") for a in allowed)]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout


def record_key(line: bytes) -> tuple:
    r = json.loads(line)
    return r["conversation_id"], r["segment_order"]


@dataclass
class Space:
    name: str
    prepared: Path
    labels: Path
    results: Path
    epochs: int = EPOCHS
    replicates: int = BOOT_REPLICATES
    full: bool = True            # the real run: input check, time estimate, progress.md notes

    @property
    def splits(self) -> Path:
        return ROOT / "data" / "splits" / self.name

    @property
    def runs(self) -> Path:
        return ROOT / "data" / "runs" / self.name

    def fold_split(self, k: int) -> Path:
        return self.splits / f"fold{k}.json"

    def config(self, k: int) -> Path:
        return self.results / "configs" / f"C1-fold{k}.json"

    def run_dir(self, k: int) -> Path:
        return self.runs / f"C1-fold{k}"

    @property
    def oof(self) -> Path:
        return self.results / "oof-predictions" / OOF_NAME

    def reference(self, s: str) -> Path:
        """The bootstrap reference of set s: the label file itself, or a byte-identical subset holding part s."""
        return self.labels if s == ALL else self.results / "part-labels" / f"{self.labels.stem}-part-{s}.jsonl"

    def note(self, text: str) -> None:
        path = self.results / ("progress.md" if self.full else "run-log.md")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"- {utc()} {text}\n")


REAL = Space(NAME, PREPARED, LABELS, HERE)


# ----------------------------------------------------------------------------- check and prepare

def canon(labels: dict) -> dict:
    return {a: L.canonical(a, labels[a]) for a in L.AXES}


def check() -> int:
    sums = {}
    for line in (TRANSFER / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            sums[p.strip()] = h
    extracted = {p: sha(TRANSFER / p) == h for p, h in sums.items()}
    placed = {dest: sha(ROOT / dest) == sums[src] for src, dest in PLACED.items()}
    patch_applied = subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=ROOT,
                                   capture_output=True).returncode == 0
    items = L.read_reviewed_direct_jsonl(LABELS)
    keys = [i.key for i in items]
    by_key = {i.key: i for i in items}
    k150 = {i.key for i in L.read_reviewed_direct_jsonl(V9_LABELS)}
    old285 = {i.key: i for i in L.read_reviewed_direct_jsonl(LABELS_285)}
    part = {k: ("150" if k in k150 else "135" if k in old285 else "200") for k in keys}
    decisions = {p: dict(Counter(i.source_record["review_provenance"]["decision"] for i in items if part[i.key] == p))
                 for p in PARTS}
    manifest = read_json(LABELS_MANIFEST)
    vs285 = {}
    for p in ("150", "135"):
        ks = [k for k in keys if part[k] == p and k in old285]
        vs285[p] = {"interactions": sum(part[k] == p for k in keys), "in_285_file": len(ks),
                    "labels_identical": sum(canon(by_key[k].labels) == canon(old285[k].labels) for k in ks),
                    "records_identical": sum(json.dumps(by_key[k].source_record, sort_keys=True)
                                             == json.dumps(old285[k].source_record, sort_keys=True) for k in ks),
                    "changed_by_axis": {a: sum(canon(by_key[k].labels)[a] != canon(old285[k].labels)[a] for k in ks)
                                        for a in L.AXES}}
    corpora, examples = {}, {}
    for name, path in CORPORA:
        corpus = L.load_corpus(path)
        ex, missing = L.attach(items, corpus, L.DEFAULT_CONTEXT)
        examples[name] = {(e["conversation_id"], e["segment_order"]): e for e in ex}
        corpora[name] = {"path": L.rel(path), "sha256": sha(path), "rows": int(len(corpus)),
                         "conversations": int(corpus["Conversation ID"].nunique()), "missing_of_485": len(missing),
                         "missing_by_part": dict(Counter(part[k] for k in missing))}
    used = next((n for n, _ in CORPORA if corpora[n]["missing_of_485"] == 0), None)
    if used is None:
        raise SystemExit("STOP: some of the 485 interactions are missing from every corpus")
    ex = examples[used]
    identical = {}
    for name, _ in CORPORA:
        if name != used:
            both = [k for k in examples[name] if k in ex]
            identical[name] = {"compared": len(both), "identical_examples": sum(
                json.dumps(examples[name][k], sort_keys=True) == json.dumps(ex[k], sort_keys=True) for k in both)}
    students = {p: sorted({ex[k]["student_id"] for k in keys if part[k] == p}) for p in PARTS}
    conversations = {p: sorted({k[0] for k in keys if part[k] == p}) for p in PARTS}
    in_parts = Counter(s for p in PARTS for s in students[p])
    owner = {}
    for e in ex.values():
        owner.setdefault(e["conversation_id"], set()).add(e["student_id"])
    out = {
        "what": "D-186 input checks (2026-10-28): transfer, label file, parts and corpus coverage; counts only",
        "created_utc": utc(),
        "transfer": {"zip": L.rel(TRANSFER_ZIP), "zip_sha256": sha(TRANSFER_ZIP), "extracted_to": L.rel(TRANSFER),
                     "files_match_SHA256SUMS": extracted, "placed_copies_match": placed, "patch_applied": patch_applied},
        "labels": {"path": L.rel(LABELS), "sha256": sha(LABELS), "records": len(items), "distinct_keys": len(set(keys)),
                   "manifest_sha256_matches_file": manifest["sha256"] == sha(LABELS), "manifest_records": manifest["records"],
                   "manifest_note": manifest.get("note")},
        "parts": {p: {"interactions": sum(part[k] == p for k in keys), "students": len(students[p]),
                      "conversations": len(conversations[p]), "review_decisions": decisions[p],
                      "manifest": next(m for m in manifest["parts"] if m["part"] == p)} for p in PARTS},
        "part_rule": ("150 = the interactions of data/labels/dw150-reviewed-v9.jsonl; 135 = the other interactions of "
                      "data/labels/dw285-reviewed-20261026.jsonl; 200 = the rest"),
        "labels_vs_d170_file": vs285,
        "students_in_two_parts": sum(n > 1 for n in in_parts.values()),
        "conversations_with_more_than_one_student": sum(len(v) > 1 for v in owner.values()),
        "student_id_equals_conversation_prefix": all(e["student_id"] == e["conversation_id"].split("_", 1)[0] for e in ex.values()),
        "corpora": corpora, "corpus_used": {"name": used, **{k: corpora[used][k] for k in ("path", "sha256")}},
        "corpus_rule": "the Mac's corpora (dw150-context.csv, then the D-170 transfer) unless they lack any of the 485 (README)",
        "examples_identical_to_other_corpora": identical,
        "part_keys": {p: [GS.key_text(*k) for k in sorted(keys) if part[k] == p] for p in PARTS},
    }
    ok = (all(extracted.values()) and all(placed.values()) and patch_applied and len(set(keys)) == len(keys) == 485
          and out["labels"]["manifest_sha256_matches_file"]
          and all(out["parts"][p]["interactions"] == out["parts"][p]["manifest"]["records"]
                  and len(students[p]) == out["parts"][p]["manifest"]["students"]
                  and len(conversations[p]) == out["parts"][p]["manifest"]["conversations"]
                  and set(decisions[p]) <= PART_DECISIONS[p] for p in PARTS)
          and all(v["in_285_file"] == v["interactions"] for v in vs285.values())
          and out["students_in_two_parts"] == 0 and out["conversations_with_more_than_one_student"] == 0
          and out["student_id_equals_conversation_prefix"]
          and all(v["compared"] == v["identical_examples"] for v in identical.values()))
    out["all_checks_pass"] = ok
    write_json(HERE / "check.json", out)
    show = {k: v for k, v in out.items() if k not in ("part_keys", "what", "created_utc")}
    show["labels"] = {k: v for k, v in show["labels"].items() if k != "manifest_note"}
    show["parts"] = {p: {k: v for k, v in show["parts"][p].items() if k != "manifest"} for p in PARTS}
    print(json.dumps(show, ensure_ascii=False, indent=1))
    return 0 if ok else 1


def run_prepare(labels: Path, out_dir: Path, report_path: Path | None) -> dict:
    corpus = ROOT / read_json(HERE / "check.json")["corpus_used"]["path"]
    command = [str(PY), str(PREPARE), "--reviewed-direct", str(labels), "--corpus", str(corpus),
               "--output-dir", str(out_dir), "--context", str(L.DEFAULT_CONTEXT)]
    r = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"prepare_data failed: {r.stderr[-800:]}")
    report = json.loads(r.stdout)
    report["command"] = " ".join(L.rel(Path(c)) if c.startswith(str(ROOT)) else c for c in command)
    if report_path is not None:
        write_json(report_path, report, exclusive=True)
    return report


def prepare() -> int:
    if (PREPARED / "manifest.json").exists():
        print(f"{L.rel(PREPARED)} already prepared")
        return 0
    report = run_prepare(LABELS, PREPARED, HERE / "prepare-report.json")
    print(json.dumps({k: report[k] for k in ("interactions", "students", "conversations", "command")}, indent=1))
    return 0


# ----------------------------------------------------------------------------- plan

def part_map() -> dict:
    keys = read_json(HERE / "check.json")["part_keys"]
    return {L.parse_key(t): p for p in PARTS for t in keys[p]}


def write_fold_split(space: Space, k: int, examples: list, folds: dict, part_of: dict) -> dict:
    test = [e for e in examples if folds["fold_of"][e["student_id"]] == k]
    train = [e for e in examples if folds["fold_of"][e["student_id"]] != k]
    parts = {"train": cv.part_record(train), "validation": cv.part_record(test)}
    overlap = {f: len(set(parts["train"][f]) & set(parts["validation"][f])) for f in ("students", "conversations")}
    if any(overlap.values()) or len(train) + len(test) != len(examples):
        raise AssertionError(f"fold {k} split checks failed: {overlap}")
    record = {"record_kind": GS.RECORD_KIND,
              "status": f"{space.name} fold {k} of {FOLDS}: train = the other folds, validation = this test fold; D-186 "
                        "trial run (D-106 preliminary development), not an adopted experimental protocol",
              "validation_purpose": PURPOSE, "created_utc": utc(),
              "split_module_sha256": sha(Path(GS.__file__)), "procedure_script_sha256": sha(Path(cv.__file__)),
              "driver_script_sha256": sha(Path(__file__)), "prepared": GS.prepared_fingerprint(space.prepared),
              "seed": SEED, "fold": k, "folds": FOLDS,
              "procedure": "cv-20261022's: students sorted by Student ID, shuffled with random.Random(seed), then stably "
                           "sorted by number of interactions, largest first; in that order each student joins the fold "
                           "with the fewest interactions so far (ties: lower fold number)",
              "parts": parts,
              "label_parts": {n: dict(Counter(part_of[(e["conversation_id"], e["segment_order"])] for e in rows))
                              for n, rows in (("train", train), ("validation", test))},
              "label_support": {"train": GS.label_support(train), "validation": GS.label_support(test)},
              "checks": {"student_overlap": overlap["students"], "conversation_overlap": overlap["conversations"],
                         "every_interaction_once": True, "shuffled_student_order": folds["shuffled_student_order"]}}
    path = space.fold_split(k)
    if path.exists():
        old = read_json(path)
        if old["parts"] != parts or old["prepared"] != record["prepared"]:
            raise ValueError(f"{path} exists with a different roster or prepared data")
        return old
    write_json(path, record, exclusive=True)
    return record


def make_config(space: Space, k: int) -> tuple[dict, list, list]:
    template = read_json(CVDIR / "configs" / f"C1-fold{k}.json")
    cfg = copy.deepcopy(template)
    cfg["status"] = (f"C1 fold {k} ({space.name}; D-186 trial run on 485 interactions; D-106 preliminary development; not a "
                     "formal evaluation): C1 of cv-20261022 as run in D-170 (8 epochs, head_lr 0.001, class weights "
                     "training_part_balanced_v1, budgets 1536/3072/7168, seed 20261017), decoded at 0.5 (no inner "
                     "thresholds); no per-epoch test-fold loss; MPS cache released after every optimizer step. Labels: the "
                     "researcher's reviewed labels (D-166 to D-169, D-182, D-185); not joint adjudication; not the final "
                     "ground truth.")
    cfg["data"].update(prepared_dir=L.rel(space.prepared), split_file=L.rel(space.fold_split(k)), label_source_note=LABEL_NOTE)
    cfg["validation"].update(purpose=PURPOSE, per_epoch_loss=False)
    cfg["memory"] = {"mps_empty_cache_every_steps": 1}
    allowed, d170_allowed = ALLOWED_CHANGES, D170_ALLOWED
    if space.epochs != EPOCHS:                 # the smoke only
        cfg["schedule"]["epochs"] = space.epochs
        allowed, d170_allowed = allowed + ("schedule.epochs",), d170_allowed + ("schedule.epochs",)
    changes = changed_keys(template, cfg)
    vs_d170 = changed_keys(read_json(D170 / "configs" / f"C1-fold{k}.json"), cfg)
    stray = within(changes, allowed) + within(vs_d170, d170_allowed)
    if stray:
        raise AssertionError(f"fold {k}: the config differs from cv-20261022's C1 or D-170's in {stray}")
    path = space.config(k)
    if path.exists():
        if read_json(path) != cfg:
            raise ValueError(f"{path} exists with different content")
    else:
        write_json(path, cfg, exclusive=True)
    return cfg, changes, vs_d170


def template_consistency() -> list:
    """The five cv-20261022 C1 fold configs differ only in status and split file."""
    first = read_json(CVDIR / "configs" / "C1-fold1.json")
    return sorted({c for k in range(2, FOLDS + 1) for c in changed_keys(first, read_json(CVDIR / "configs" / f"C1-fold{k}.json"))})


def write_part_labels(space: Space, part_of: dict) -> dict:
    """Byte-identical subsets of the label file, one per part (bootstrap references)."""
    lines = [line for line in space.labels.read_bytes().split(b"\n") if line.strip()]
    out = {}
    for p in PARTS:
        chosen = [line for line in lines if part_of.get(record_key(line)) == p]
        if len(chosen) != sum(v == p for v in part_of.values()):
            raise ValueError(f"part {p}: {sum(v == p for v in part_of.values())} interactions but {len(chosen)} records")
        data = b"\n".join(chosen) + b"\n"
        path = space.reference(p)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"{path} exists with different content")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as f:
                f.write(data)
        out[p] = {"path": L.rel(path), "sha256": sha(path), "records": len(chosen)}
    return out


def loses(r: dict) -> dict:
    return {"context": r["context_tokens_omitted"] > 0, "prompt": r["prompt_tokens_omitted"] > 0,
            "response": r["response_tokens_omitted"] > 0}


def check_inputs(space: Space, part_of: dict) -> tuple[dict, dict]:
    """C1's input policy on the 485 inputs (tokenizer only): what is cut, by part; and whether the 150 and 285 inputs
    equal those the cv-20261022 and D-170 runs were trained on."""
    import model_inputs as MI
    import train_multihead as T
    cfg = read_json(space.config(1))
    T.check_config(cfg)
    encode = T.tokenizer_encoder(T.model_snapshot(cfg["model"])[0])
    policy = MI.InputPolicy.from_config(cfg["input"])
    built = T.build_inputs(T.load_data(space.prepared, space.fold_split(1)), encode, policy)[0]
    old150 = T.build_inputs(T.load_data(V9_PREPARED, BASES["cv-20261022"]["splits"] / "fold1.json"), encode, policy)[0]
    old285 = T.build_inputs(T.load_data(D170_PREPARED, BASES["D-170"]["splits"] / "fold1.json"), encode, policy)[0]
    out = {"policy": cfg["input"], "student_side_budget": policy.check_with(encode)["student_side_budget"],
           "interactions": len(built), "summary": MI.summarise([b["report"] for b in built.values()])}
    for name, keys in ((ALL, list(built)), *((p, [k for k in built if part_of[k] == p]) for p in PARTS)):
        flags = [loses(built[k]["report"]) for k in keys]
        tokens = sorted(built[k]["report"]["tokens"] for k in keys)
        out[name] = {"interactions": len(keys), **{f"loses_{x}": sum(f[x] for f in flags) for x in ("context", "prompt", "response")},
                     "loses_any": sum(any(f.values()) for f in flags),
                     "tokens": {"median": tokens[len(tokens) // 2], "p90": tokens[int(0.9 * (len(tokens) - 1))], "max": tokens[-1]}}

    def same(old):
        return {"compared": len(old), "identical_input_ids": sum(
            built[k]["input_ids"] == old[k]["input_ids"] and built[k]["student_pool"] == old[k]["student_pool"] for k in old)}
    out["the_150_equal_cv20261022_inputs"] = same(old150)
    out["the_285_equal_d170_inputs"] = same(old285)
    out["definition"] = ("loses context: context_tokens_omitted > 0 (an earlier interaction clipped or dropped for length); "
                         "loses prompt / response: tokens of the current prompt / response omitted")
    return out, {GS.key_text(*k): b["report"]["tokens"] for k, b in built.items()}


def basis_estimate(name: str, folds: list, new_tokens: dict) -> dict:
    """From one earlier set of C1 runs on this Mac: seconds per training interaction-pass (optimizer-step time over
    interactions), seconds per end-of-run prediction (last epoch end to outputs written, over train + test
    interactions) and the rest of each run (start-up and checkpoint writes; where those runs computed a per-epoch
    test-fold loss, it is taken out at the prediction rate). The length-adjusted variant scales the training and
    prediction rates by the mean per-pass cost 1.007 ms/token + 0.383 us/token^2 (profile) of the new inputs over the
    inputs of those runs."""
    b = BASES[name]
    status = read_json(b["status"])["runs"]
    step_total = examples = out_gap = out_n = 0.0
    overheads, old_train_cost, old_train_n, old_pred_cost, old_pred_n = [], 0.0, 0, 0.0, 0
    per_out_parts = []
    for k in range(1, FOLDS + 1):
        run = b["runs"] / f"C1-fold{k}"
        log = [json.loads(line) for _, line in L.jsonl_lines(run / "train_log.jsonl")]
        steps = {e["step"]: e for e in log if e["event"] == "step"}
        step_s = sum(s["seconds"] for s in steps.values())
        step_total += step_s
        examples += sum(s["examples"] for s in steps.values())
        split = read_json(b["splits"] / f"fold{k}.json")["parts"]
        tr, te = split["train"]["interactions"], split["validation"]["interactions"]
        gap = stamp(next(e for e in log if e["event"] == "outputs_written")["utc"]) - stamp([e for e in log if e["event"] == "epoch_end"][-1]["utc"])
        out_gap += gap
        out_n += len(tr) + len(te)
        per_out_parts.append((sum(r["seconds"] for r in status[f"C1-fold{k}"]), step_s, gap, len(te)))
        tokens = {t: v["tokens"] for t, v in read_json(run / "inputs-report.json")["per_interaction"].items()}
        old_train_cost += sum(COST(tokens[t]) for t in tr)
        old_train_n += len(tr)
        old_pred_cost += sum(COST(tokens[t]) for t in tr + te)
        old_pred_n += len(tr) + len(te)
    per_train, per_out = step_total / examples, out_gap / out_n
    for total, step_s, gap, n_test in per_out_parts:
        overheads.append(total - step_s - gap - (EPOCHS * n_test * per_out if b["per_epoch_validation"] else 0))
    overhead = sum(overheads) / len(overheads)
    new_train = [COST(new_tokens[GS.key_text(*x)]) for f in folds for x in f["train_keys"]]
    new_pred = [COST(new_tokens[GS.key_text(*x)]) for f in folds for x in f["train_keys"] + f["test_keys"]]
    r_train = (sum(new_train) / len(new_train)) / (old_train_cost / old_train_n)
    r_pred = (sum(new_pred) / len(new_pred)) / (old_pred_cost / old_pred_n)
    plain = {f["fold"]: f["train"] * EPOCHS * per_train + (f["train"] + f["test"]) * per_out + overhead for f in folds}
    adjusted = {f["fold"]: f["train"] * EPOCHS * per_train * r_train + (f["train"] + f["test"]) * per_out * r_pred + overhead
                for f in folds}
    return {"basis": f"{name} C1 final runs C1-fold1..5 on this Mac ({L.rel(b['runs'])})",
            "seconds_per_training_interaction_pass": round(per_train, 3), "seconds_per_output_prediction": round(per_out, 3),
            "other_seconds_per_run": round(overhead, 1), "other_seconds_per_run_by_fold": [round(x, 1) for x in overheads],
            "per_fold_seconds": {k: round(v) for k, v in plain.items()}, "total_hours": round(sum(plain.values()) / 3600, 2),
            "length_adjusted": {"train_pass_cost_ratio_new_over_old": round(r_train, 3),
                                "prediction_cost_ratio_new_over_old": round(r_pred, 3),
                                "per_fold_seconds": {k: round(v) for k, v in adjusted.items()},
                                "total_hours": round(sum(adjusted.values()) / 3600, 2)}}


def time_estimate(folds: list, new_tokens: dict) -> dict:
    est = {name: basis_estimate(name, folds, new_tokens) for name in BASES}
    return {"primary": "cv-20261022 (the README's basis: about 3.57 s per training interaction-pass)",
            "total_hours": est["cv-20261022"]["total_hours"], "per_fold_seconds": est["cv-20261022"]["per_fold_seconds"],
            "bases": est, "rule": "README item 6: report the estimate before starting; no automatic stop"}


def code_patch() -> str:
    """git diff HEAD for finetuning/, plus every untracked file there as a new-file diff."""
    diff = git("diff", "HEAD", "--", "finetuning")
    for path in sorted(git("ls-files", "--others", "--exclude-standard", "finetuning").split()):
        diff += subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", path], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8").stdout
    return diff


def plan(space: Space) -> dict:
    space.results.mkdir(parents=True, exist_ok=True)
    examples = GS.load_examples(space.prepared)
    present = {(e["conversation_id"], e["segment_order"]) for e in examples}
    part_of = {k: v for k, v in part_map().items() if k in present}
    if set(part_of) != present:
        raise AssertionError("some prepared interactions have no part")
    folds = cv.assign_folds(examples)
    template_diff = template_consistency()
    if set(template_diff) - {"status", "data.split_file"}:
        raise AssertionError(f"the cv-20261022 C1 fold configs differ in {template_diff}")
    summary, configs = [], {}
    for k in range(1, FOLDS + 1):
        record = write_fold_split(space, k, examples, folds, part_of)
        _, changes, vs_d170 = make_config(space, k)
        configs[k] = {"path": L.rel(space.config(k)), "sha256": sha(space.config(k)),
                      "template": L.rel(CVDIR / "configs" / f"C1-fold{k}.json"), "changed_keys": changes,
                      "changed_keys_vs_d170": vs_d170}
        test, train = record["parts"]["validation"], record["parts"]["train"]
        summary.append({"fold": k, "train": train["counts"]["interactions"], "test": test["counts"]["interactions"],
                        "test_students": test["counts"]["students"], "test_conversations": test["counts"]["conversations"],
                        "test_by_part": record["label_parts"]["validation"], "train_by_part": record["label_parts"]["train"],
                        "train_keys": [L.parse_key(t) for t in train["interactions"]],
                        "test_keys": [L.parse_key(t) for t in test["interactions"]]})
    covered = sorted(t for k in range(1, FOLDS + 1) for t in read_json(space.fold_split(k))["parts"]["validation"]["interactions"])
    if covered != sorted(GS.key_text(e["conversation_id"], e["segment_order"]) for e in examples):
        raise AssertionError("the test folds do not cover every interaction exactly once")
    import train_multihead as T
    for k in range(1, FOLDS + 1):
        T.check_config(read_json(space.config(k)))
    part_files = write_part_labels(space, part_of)
    inputs = estimate = None
    if space.full:
        inputs, new_tokens = check_inputs(space, part_of)
        estimate = time_estimate(summary, new_tokens)
    diff = code_patch()
    if diff:
        (space.results / "code-diff-vs-head.patch").write_text(diff, encoding="utf-8")
    fold_rows = [{k: v for k, v in f.items() if k not in ("train_keys", "test_keys")} for f in summary]
    record = {"what": f"{space.name} plan (fixed before the runs; D-186 trial run, D-106 preliminary development)",
              "created_utc": utc(),
              "spec": {"readme": {"path": L.rel(SPEC), "sha256": sha(SPEC)},
                       "base": {"path": L.rel(BASE_SPEC), "sha256": sha(BASE_SPEC)}} if space.full else None,
              "seed": SEED, "folds": fold_rows,
              "dataset": {"prepared": L.rel(space.prepared), "manifest_sha256": sha(space.prepared / "manifest.json"),
                          "interactions": len(examples), "students": len({e["student_id"] for e in examples}),
                          "conversations": len({e["conversation_id"] for e in examples}),
                          "labels": {"path": L.rel(space.labels), "sha256": sha(space.labels)}},
              "parts": {p: sorted(GS.key_text(*k) for k, v in part_of.items() if v == p) for p in PARTS},
              "part_label_files": part_files,
              "test_fold_rosters": {k: read_json(space.fold_split(k))["parts"]["validation"] for k in range(1, FOLDS + 1)},
              "fold_of_student": folds["fold_of"], "shuffled_student_order": folds["shuffled_student_order"],
              "splits": {k: {"path": L.rel(space.fold_split(k)), "sha256": sha(space.fold_split(k))} for k in range(1, FOLDS + 1)},
              "configs": configs, "template_configs_differ_only_in": template_diff,
              "c1": {"epochs": space.epochs, "head_lr": cv.C1_HEAD_LR, "class_weights": "training_part_balanced_v1",
                     "prompt_cap": 1536, "response_cap": cv.C1_RESPONSE_CAP, "max_tokens": cv.C1_MAX_TOKENS,
                     "decode_threshold": 0.5, "per_epoch_loss": False, "mps_empty_cache_every_steps": 1, "inner_holdout": None},
              "bootstrap": {"script": L.rel(BOOTSTRAP), "replicates": space.replicates, "seed": BOOT_SEED,
                            "references": {s: L.rel(space.reference(s)) for s in SETS}},
              "input_check": inputs, "estimate": estimate,
              "code": {"head": git("rev-parse", "HEAD").strip(), "patch": L.rel(space.results / "code-diff-vs-head.patch") if diff else None,
                       "patch_sha256": L.sha256_bytes(diff.encode("utf-8")) if diff else None,
                       "trainer_sha256": sha(TRAIN), "labels_module_sha256": sha(ROOT / "finetuning" / "labels.py")},
              "procedure_script_sha256": sha(Path(cv.__file__)), "driver_script_sha256": sha(Path(__file__))}
    path = space.results / "plan.json"
    if path.exists():
        old = read_json(path)
        if old["test_fold_rosters"] != roundtrip(record["test_fold_rosters"]) or old["configs"] != roundtrip(record["configs"]):
            raise ValueError(f"{path} exists with a different plan")
        print(f"{L.rel(path)} exists and matches; not rewritten")
    else:
        write_json(path, record, exclusive=True)
    print(json.dumps({"dataset": {k: v for k, v in record["dataset"].items() if k != "labels"}, "folds": fold_rows,
                      "config_changed_keys": sorted({c for v in configs.values() for c in v["changed_keys"]}),
                      "config_changed_keys_vs_d170": sorted({c for v in configs.values() for c in v["changed_keys_vs_d170"]}),
                      "part_label_files": {p: v["records"] for p, v in part_files.items()},
                      "input_check": None if inputs is None else {n: inputs[n] for n in (*SETS, "the_150_equal_cv20261022_inputs",
                                                                                          "the_285_equal_d170_inputs")},
                      "estimate": None if estimate is None else {
                          "primary_total_hours": estimate["total_hours"],
                          **{n: {k: v[k] for k in ("seconds_per_training_interaction_pass", "seconds_per_output_prediction",
                                                   "other_seconds_per_run", "per_fold_seconds", "total_hours", "length_adjusted")}
                             for n, v in estimate["bases"].items()}}},
                     ensure_ascii=False, indent=1))
    return record


# ----------------------------------------------------------------------------- run

def train_or_resume(space: Space, k: int, status: dict) -> None:
    name, run_dir, log = f"C1-fold{k}", space.run_dir(k), space.runs / f"C1-fold{k}.out"
    if (run_dir / "outputs" / "summary.json").exists():
        return
    space.runs.mkdir(parents=True, exist_ok=True)
    command = ([str(PY), str(TRAIN), "resume", "--run-dir", str(run_dir)] if (run_dir / "run.json").exists()
               else [str(PY), str(TRAIN), "train", "--config", str(space.config(k)), "--run-dir", str(run_dir)])
    space.note(f"fold {k}: {command[2]} started")
    started = time.time()
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n== {utc()} {command[2]} ==\n")
        out.flush()
        proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        forward = lambda signum, _frame: proc.send_signal(signum)  # noqa: E731  (the trainer checkpoints and exits)
        old = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            code = proc.wait()
        finally:
            for s, h in old.items():
                signal.signal(s, h)
    status["runs"].setdefault(name, []).append({"command": command[2], "exit": code, "seconds": round(time.time() - started),
                                               "finished_utc": utc()})
    write_json(space.results / "run-status.json", status)
    done = (run_dir / "outputs" / "summary.json").exists()
    space.note(f"fold {k}: {command[2]} exited {code} after {round(time.time() - started)} s; outputs {'written' if done else 'not written'}")
    if code != 0 or not done:
        raise SystemExit(f"{name}: trainer exited {code}; see {L.rel(log)}")


def run_all(space: Space) -> int:
    path = space.results / "run-status.json"
    status = read_json(path) if path.exists() else {"runs": {}}
    status["driver_started_utc"] = utc()
    write_json(path, status)
    for k in range(1, FOLDS + 1):
        train_or_resume(space, k, status)
        status[f"fold{k}_done_utc"] = status.get(f"fold{k}_done_utc") or utc()
        write_json(path, status)
    status["driver_finished_utc"] = utc()
    write_json(path, status)
    space.note("all five folds complete")
    return 0


# ----------------------------------------------------------------------------- assemble

def fold_keys(space: Space) -> dict:
    out = {}
    for k in range(1, FOLDS + 1):
        _, part_of = GS.load_split(space.fold_split(k), space.prepared)
        out[k] = {"test": sorted(x for x, p in part_of.items() if p == "validation"),
                  "train": sorted(x for x, p in part_of.items() if p == "train")}
    return out


def assemble(space: Space) -> int:
    refs, fingerprint = EV.references_from_prepared(space.prepared)
    all_keys = sorted(refs)
    manifest_sha = sha(space.prepared / "manifest.json")
    folds = fold_keys(space)
    rosters = {k: set(f["test"]) for k, f in folds.items()}
    assert sorted(set().union(*rosters.values())) == all_keys and sum(map(len, rosters.values())) == len(all_keys)
    sources = {k: space.run_dir(k) / "outputs" / "predictions-validation.jsonl" for k in folds}
    before = {k: sha(p) for k, p in sources.items()}
    origin, per_fold = {}, {}
    for k, path in sources.items():
        source = L.read_predictions(path)
        keys = {(r["conversation_id"], r["segment_order"]) for r in source}
        assert keys == rosters[k] and len(source) == len(keys), f"fold {k}: not exactly the fold's test interactions"
        assert len({tuple(r[f] for f in L.RUN_FIELDS) for r in source}) == 1, f"fold {k}: mixed provenance"
        first = source[0]
        assert first["dataset_manifest_sha256"] == manifest_sha, f"fold {k}: a different prepared dataset"
        route, _, tail = first["model_route"].partition("#")
        per_fold[k] = {"source": L.rel(path), "source_sha256": before[k], "records": len(source),
                       "original_run_id": first["run_id"], "original_model_route": first["model_route"],
                       "original_config_sha256": first["config_sha256"], "model": route, "final_checkpoint": tail,
                       "config": L.rel(space.config(k)), "config_file_sha256": sha(space.config(k)),
                       "split": L.rel(space.fold_split(k)), "split_sha256": sha(space.fold_split(k)), "decoding": "threshold 0.5"}
        for r in source:
            origin[(r["conversation_id"], r["segment_order"])] = (k, r)
    models = {v["model"] for v in per_fold.values()}
    assert len(models) == 1, f"the folds name different models {models}"
    run = {"run_id": f"{space.name}-C1-oof", "model_route": f"{models.pop()}#out-of-fold-final-checkpoints-of-folds-1-5",
           "config_sha256": cv.canonical_sha([per_fold[k]["original_config_sha256"] for k in sorted(per_fold)]),
           "dataset_manifest_sha256": manifest_sha}
    records = [L.prediction_record(key[0], key[1], origin[key][1]["labels"], run, origin[key][1]["not_predicted"]) for key in all_keys]
    space.oof.parent.mkdir(parents=True, exist_ok=True)
    L.write_predictions(space.oof, records)                     # exclusive: never overwrites
    written = L.read_predictions(space.oof)
    preds, info = EV.predictions_from_file(space.oof, reference=fingerprint, reference_keys=set(refs))
    same_as_run_reports = {}
    for k in folds:
        mine = roundtrip(EV.evaluate(refs, preds, folds[k]["test"]))
        theirs = read_json(space.run_dir(k) / "outputs" / "evaluation-validation.json")
        same_as_run_reports[k] = mine["axes"] == theirs["axes"] and mine["all_axis_exact_match"] == theirs["all_axis_exact_match"]
    checks = {"records": len(written), "distinct_interactions": len({(r["conversation_id"], r["segment_order"]) for r in written}),
              "covers_every_interaction": sorted((r["conversation_id"], r["segment_order"]) for r in written) == all_keys,
              "labels_and_not_predicted_identical_to_sources": all(
                  (r["labels"], r["not_predicted"]) == (origin[(r["conversation_id"], r["segment_order"])][1]["labels"],
                                                        origin[(r["conversation_id"], r["segment_order"])][1]["not_predicted"]) for r in written),
              "each_from_the_fold_that_held_it_out": all((r["conversation_id"], r["segment_order"]) in
                                                         rosters[origin[(r["conversation_id"], r["segment_order"])][0]] for r in written),
              "passes_evaluation_provenance_check": info["provenance_consistent_records"] == len(all_keys) and info["dataset_manifest_matches_reference"],
              "per_fold_scores_equal_each_run_report": same_as_run_reports,
              "sources_unchanged": {k: sha(p) == before[k] for k, p in sources.items()}}
    manifest = {"what": f"out-of-fold model_prediction file of {space.name} (C1 at threshold 0.5, five folds, "
                        f"{len(all_keys)} interactions; D-186 trial run, not final performance)",
                "created_utc": utc(),
                "how": "the saved per-fold test predictions (each run's outputs/predictions-validation.jsonl) were read and "
                       "rewritten with labels.write_predictions; no model was run",
                "reading_note": "the file pools five models; run_id, model_route and config_sha256 describe the pooled set "
                                "(config_sha256 = SHA-256 of the JSON list of the five per-fold config_sha256 values, in fold "
                                "order); the origin of every record is under 'folds' and 'interaction_fold'",
                "file": {"path": L.rel(space.oof), "sha256": sha(space.oof), "records": len(written), "run_fields": run},
                "dataset": {"prepared": L.rel(space.prepared), "manifest_sha256": manifest_sha,
                            "examples_sha256": fingerprint["examples_sha256"], "interactions": len(all_keys)},
                "reference_labels": {"path": L.rel(space.labels), "sha256": sha(space.labels)},
                "folds": per_fold, "interaction_fold": {GS.key_text(*k): origin[k][0] for k in all_keys}, "checks": checks}
    write_json(space.oof.parent / "oof-predictions-manifest.json", manifest, exclusive=True)
    ok = (checks["records"] == checks["distinct_interactions"] == len(all_keys) and checks["covers_every_interaction"]
          and checks["labels_and_not_predicted_identical_to_sources"] and checks["each_from_the_fold_that_held_it_out"]
          and checks["passes_evaluation_provenance_check"] and all(same_as_run_reports.values()) and all(checks["sources_unchanged"].values()))
    print(json.dumps({**{k: v for k, v in checks.items()}, "sha256": manifest["file"]["sha256"], "all_checks_pass": ok}, indent=1))
    if space.full:
        space.note(f"assembled {L.rel(space.oof)} ({len(written)} records, checks pass: {ok})")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- report

METRICS = ("accuracy", "macro_f1", "micro_f1")
BOOT_KEY = {"accuracy": "exact", "macro_f1": "macro_f1", "micro_f1": "micro_f1"}


def set_tables(refs: dict, preds: dict, keys: list, folds: dict) -> dict:
    pooled = roundtrip(EV.evaluate(refs, preds, keys))
    keyset = set(keys)
    per_fold = {k: roundtrip(EV.evaluate(refs, preds, sorted(keyset & set(f["test"])))) for k, f in folds.items()
                if keyset & set(f["test"])}
    return {"pooled": pooled, "per_fold": per_fold, "table": cv.fold_table(per_fold, pooled),
            "fold_sizes": {k: len(keyset & set(f["test"])) for k, f in folds.items()}}


def run_record(space: Space, k: int, status: dict, estimate: dict | None) -> dict:
    log = [json.loads(line) for _, line in L.jsonl_lines(space.run_dir(k) / "train_log.jsonl")]
    steps = {e["step"]: e for e in log if e["event"] == "step"}
    gib = [s["mps_driver_gib"] for s in steps.values() if "mps_driver_gib" in s]
    runs = status["runs"].get(f"C1-fold{k}", [])
    return {"optimizer_steps": len(steps), "trainer_invocations": len(runs), "resumes": sum(e["event"] == "resume" for e in log),
            "stops": sum(e["event"] == "stopped" for e in log), "wall_seconds": sum(r["seconds"] for r in runs),
            "step_seconds": round(sum(s["seconds"] for s in steps.values()), 1),
            "seconds_per_training_interaction_pass": round(sum(s["seconds"] for s in steps.values()) / sum(s["examples"] for s in steps.values()), 3),
            "mps_driver_gib_at_step_end": {"max": max(gib), "median": sorted(gib)[len(gib) // 2]} if gib else None,
            "estimated_seconds": None if estimate is None else estimate["per_fold_seconds"][str(k)]}


def bootstrap(space: Space, s: str, plan_record: dict) -> tuple[dict, list]:
    ref = space.reference(s)
    if s != ALL and sha(ref) != plan_record["part_label_files"][s]["sha256"]:
        raise ValueError(f"{ref} is not the part file the plan recorded")
    out_json, out_txt = space.results / "bootstrap" / f"bootstrap-{s}.json", space.results / "bootstrap" / f"bootstrap-{s}.txt"
    if not out_json.exists():
        out_json.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([str(PY), "-I", str(BOOTSTRAP), "--reference", str(ref), "--pred", f"C1={space.oof}",
                            "--replicates", str(space.replicates), "--seed", str(BOOT_SEED), "--out", str(out_json)],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise SystemExit(f"student_bootstrap.py failed for set {s}: {r.stderr[-800:]}")
        out_txt.write_text(r.stdout, encoding="utf-8")
    return read_json(out_json), [out_json, out_txt]


def report(space: Space) -> int:
    refs, fingerprint = EV.references_from_prepared(space.prepared)
    statement = EV.statements(fingerprint)["validation"]
    plan_record = read_json(space.results / "plan.json")
    all_keys = sorted(refs)
    parts = {p: sorted(L.parse_key(t) for t in plan_record["parts"][p]) for p in PARTS}
    assert sorted(k for p in PARTS for k in parts[p]) == all_keys
    sets = {ALL: all_keys, **parts}
    folds = fold_keys(space)
    c1, info = EV.predictions_from_file(space.oof, reference=fingerprint, reference_keys=set(refs))
    majority, majority_labels = {}, {}
    for k, f in folds.items():
        majority_labels[k] = cv.majority_labels([refs[x] for x in f["train"]])
        majority.update({x: majority_labels[k] for x in f["test"]})
    sources = {"C1": c1, "majority": majority}
    results = {s: {n: set_tables(refs, p, keys, folds) for n, p in sources.items()} for s, keys in sets.items()}
    out_dir = space.results / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for s in sets:
        for n in sources:
            full = {"title": f"{space.name} pooled out-of-fold, set {s}: {n}", "statement": statement,
                    "note": f"set {s} of the 485 (parts reported separately); pooled over the 5 test folds; every interaction "
                            "predicted once" + ("; majority = most frequent code / code set of each training fold" if n == "majority" else ""),
                    "reference": fingerprint, "predictions": info if n == "C1" else {"source": "majority baseline per training fold"},
                    "selection": {"part": f"set {s}, out of fold"}, **results[s][n]["pooled"]}
            for suffix, text in ((".json", json.dumps(full, ensure_ascii=False, indent=1) + "\n"), (".md", EV.markdown(full))):
                path = out_dir / f"pooled-{n}-{s}{suffix}"
                path.write_text(text, encoding="utf-8")
                files[L.rel(path)] = sha(path)
    boots, agree = {}, {}
    for s, keys in sets.items():
        boots[s], paths = bootstrap(space, s, plan_record)
        for p in paths:
            files[L.rel(p)] = sha(p)
        pooled = results[s]["C1"]["pooled"]
        agree[s] = {}
        for axis in L.AXES:
            m, r = boots[s]["axes"][axis]["models"]["C1"], pooled["axes"][axis]
            pairs = [(m["exact"]["value"], r["row_accuracy"]), (m["macro_f1"]["value"], r["macro_f1"])]
            if axis not in L.SINGLE:
                pairs.append((m["micro_f1"]["value"], cv.multi_extra(r)["micro_f1"]))
            agree[s][axis] = boots[s]["axes"][axis]["rows"] == r["n"] and all(abs(a - b) < 1e-9 for a, b in pairs)
    status = read_json(space.results / "run-status.json")
    runs = {k: run_record(space, k, status, plan_record.get("estimate")) for k in folds}
    trainfit = {}
    for k, f in folds.items():
        preds, _ = EV.predictions_from_file(space.run_dir(k) / "outputs" / "predictions-train.jsonl")
        trainfit[k] = cv.headline(roundtrip(EV.evaluate(refs, preds, f["train"])))
    files[L.rel(space.oof)] = sha(space.oof)
    files[L.rel(space.oof.parent / "oof-predictions-manifest.json")] = sha(space.oof.parent / "oof-predictions-manifest.json")
    metrics = {"what": f"{space.name}: C1 (threshold 0.5), student-grouped 5-fold CV over the 485 reviewed interactions; D-186 "
                       "trial run (D-106 preliminary development); development validation, not the reported evaluation, "
                       "not the final ground truth, not human reliability",
               "statement": statement,
               "caveats": ["the labels are the researcher's (AI-assisted review format), not joint adjudication; trial labels, "
                           "not the final ground truth",
                           "pending (label manifest): " + str(read_json(LABELS_MANIFEST).get("note")) if space.full else "smoke",
                           "C1's settings were chosen on the v9 folds of cv-20261022, which include the 150 interactions",
                           "not comparable one to one with cv-20261022 or D-170: the training size, the folds and the labels changed",
                           "parts: 150 researcher-reviewed (v12); 135 single-coded (v2); 200 single-coded with an AI-assisted check (v1)"],
               "sets": {s: len(k) for s, k in sets.items()},
               "fold_sizes": {s: results[s]["C1"]["fold_sizes"] for s in sets},
               "tables": {s: {n: results[s][n]["table"] for n in sources} for s in sets},
               "per_code": {s: {n: {a: {c: {kk: v[kk] for kk in ("support", "predicted", "tp", "precision", "recall", "f1")}
                                        for c, v in results[s][n]["pooled"]["axes"][a]["per_category"].items()} for a in L.AXES}
                                for n in sources} for s in sets},
               "all_four_right": {s: {n: results[s][n]["pooled"]["all_axis_exact_match"] for n in sources} for s in sets},
               "bootstrap": {s: {"file": L.rel(space.results / "bootstrap" / f"bootstrap-{s}.json"),
                                 "reference": L.rel(space.reference(s)), "reference_sha256": sha(space.reference(s)),
                                 "replicates": boots[s]["bootstrap"]["replicates"], "seed": boots[s]["bootstrap"]["seed"],
                                 "axes": {a: {"rows": v["rows"], "students": v["students"], "C1": v["models"]["C1"]}
                                          for a, v in boots[s]["axes"].items()},
                                 "point_values_equal_evaluation_module": agree[s]} for s in sets},
               "majority_labels_per_training_fold": majority_labels,
               "training_fold_fit": trainfit, "runs": runs,
               "predictions": info, "per_fold_reports": {s: {n: results[s][n]["per_fold"] for n in sources} for s in sets},
               "pooled_reports": {s: {n: results[s][n]["pooled"] for n in sources} for s in sets}, "files": files}
    write_json(space.results / "metrics.json", metrics)
    (space.results / "summary.md").write_text(summary_md(space, metrics, plan_record), encoding="utf-8")
    print(json.dumps({s: {a: {m: round(v["pooled"], 3) for m, v in metrics["tables"][s]["C1"][a].items()
                              if m in ("accuracy", "macro_f1", "micro_f1", "rate") and v["pooled"] is not None}
                          for a in metrics["tables"][s]["C1"]} for s in sets}, ensure_ascii=False))
    print("bootstrap point values equal evaluation module:", {s: all(v.values()) for s, v in agree.items()})
    if space.full:
        space.note("report written: metrics.json, summary.md, bootstrap/, reports/")
    return 0


def f3(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def summary_md(space: Space, m: dict, plan_record: dict) -> str:
    t = m["tables"]
    zh_metric = {"accuracy": "准确率（多选为整组一致率）", "macro_f1": "macro-F1", "micro_f1": "micro-F1"}
    fs = m["fold_sizes"]
    ic, est = plan_record.get("input_check") or {}, plan_record.get("estimate") or {}
    out = [f"# {space.name}：C1 在 485 条复核互动上的学生分组 5 折交叉验证（阈值 0.5；D-186 试运行；D-106 初步开发，非正式评价）", "",
           f"性质：{m['statement']}", "",
           "**说明（随结果一并陈述）：**",
           "- 标签是研究者的标签（AI 辅助复核格式），不是共同裁定；是试运行标签，不是最终基准标签。",
           "- 标签清单列出的待定事项：第二编码员对规则 11、12、15 的确认（D-180）；135 中 17 条无 Explain 的行按放宽的规则 13 "
           "复查，以及 P264_P2_06 #1 的 topic；150 中交接清单所列的未定单元格。",
           "- C1 的设置是在 cv-20261022 的 v9 折上选定的，而那些折包含这 150 条互动。",
           "- 这些数字不能与 cv-20261022 或 D-170 一一对比：训练规模、折和标签都变了。",
           "- 三部分分开报告：" + "；".join(f"{p} 部分为{PART_ZH[p]}" for p in PARTS) + "。485 合并不是同一种参照标签。", "",
           f"每条互动只被预测一次（由不含其学生的那一折的模型预测）。各折测试集大小（485 / 150 / 135 / 200）："
           + "；".join(f"折{k} " + " / ".join(str(fs[s].get(k, 0)) for s in SETS) for k in fs[ALL]) + "。", "",
           "## 各维度结果", "",
           "每格：C1 合并值 [学生重抽样 95% 区间]（多数类基线；基线来自各训练折，仅作背景）。", "",
           "| 维度 | 指标 | " + " | ".join(SETS) + " |", "|---|---|" + "---|" * len(SETS)]
    for axis in L.AXES:
        for metric in METRICS:
            if metric not in t[ALL]["C1"][axis] or (axis in L.SINGLE and metric == "micro_f1"):
                continue
            cells = []
            for s in SETS:
                ci = m["bootstrap"][s]["axes"][axis]["C1"][BOOT_KEY[metric]]["ci"]
                cells.append(f"{f3(t[s]['C1'][axis][metric]['pooled'])} [{f3(ci[0])}, {f3(ci[1])}]（{f3(t[s]['majority'][axis][metric]['pooled'])}）")
            out.append(f"| {axis} | {zh_metric[metric]} | " + " | ".join(cells) + " |")
    a4 = m["all_four_right"]
    out.append("| 四维全对 | 条数 / n | " + " | ".join(f"{a4[s]['C1']['matches']}/{a4[s]['C1']['n']}（{a4[s]['majority']['matches']}）"
                                                 for s in SETS) + " |")
    out += ["", "## 485 的逐折结果（C1；均值 ± 标准差，以及各折值）", "",
            "| 维度 | 指标 | 均值 ± 标准差 | " + " | ".join(f"折{k}" for k in fs[ALL]) + " |", "|---|---|---|" + "---|" * len(fs[ALL])]
    for axis in list(L.AXES) + ["all_axis_exact_match"]:
        for metric, v in t[ALL]["C1"][axis].items():
            if metric == "hamming_accuracy" or (axis in L.SINGLE and metric == "micro_f1"):
                continue
            label = "四维全对" if axis == "all_axis_exact_match" else axis
            name = "比例" if metric == "rate" else zh_metric[metric]
            out.append(f"| {label} | {name} | {f3(v['mean'])} ± {f3(v['sd'])} | " + " | ".join(f3(x) for x in v["per_fold"]) + " |")
    out += ["", "## 每个代码的 F1（合并）", "",
            "| 维度 | 代码 | 支持数 " + " / ".join(SETS) + " | " + " | ".join(f"C1 F1 {s}" for s in SETS) + " | 多数类 F1 485 |",
            "|---|---|---|" + "---|" * len(SETS) + "---|"]
    pc = m["per_code"]
    for axis in L.AXES:
        for code in L.DIRECT_CODES[axis]:
            sup = " / ".join(str(pc[s]["C1"][axis][code]["support"]) for s in SETS)
            out.append(f"| {axis} | {code} | {sup} | " + " | ".join(f3(pc[s]["C1"][axis][code]["f1"]) for s in SETS)
                       + f" | {f3(pc[ALL]['majority'][axis][code]['f1'])} |")
    out += ["", "支持数 < 5 的代码（稀有）的 F1 不稳定；macro-F1 只平均在该集合中有支持的代码，所以某个稀有代码在一次重抽样中"
                "缺席时 macro-F1 会跳动，其区间可能偏离点估计。", "",
            "## 训练拟合（诊断，不是表现）与运行记录", "",
            "| 折 | 训练 / 测试 | 优化步 | 恢复次数 | 训练折准确率（四维） | 训练折四维全对 | 用时（秒） | 估计（秒） | 每步结束时 MPS 驱动内存峰值（GiB） |",
            "|---|---|---|---|---|---|---|---|---|"]
    folds = {f["fold"]: f for f in plan_record["folds"]}
    for k, r in m["runs"].items():
        fit = m["training_fold_fit"][k]
        acc = " / ".join(f3(fit[a]["accuracy"]) for a in L.AXES)
        mem = r["mps_driver_gib_at_step_end"]["max"] if r["mps_driver_gib_at_step_end"] else "—"
        out.append(f"| {k} | {folds[int(k)]['train']} / {folds[int(k)]['test']} | {r['optimizer_steps']} | {r['resumes']} | {acc} | "
                   f"{f3(fit['all_axis_exact_match']['rate'])} | {r['wall_seconds']} | {r['estimated_seconds']} | {mem} |")
    out += ["", "## 实现说明（设计未变；详见 plan.json）", ""]
    if ic:
        out.append(f"- 输入（C1 的预算 1,536 / 3,072 / 7,168）：485 条中上下文被截 {ic[ALL]['loses_context']} 条、当前提示被截 "
                   f"{ic[ALL]['loses_prompt']} 条、当前回复被截 {ic[ALL]['loses_response']} 条；"
                   + "；".join(f"{p}：{ic[p]['loses_context']} / {ic[p]['loses_prompt']} / {ic[p]['loses_response']}" for p in PARTS)
                   + f"。与此前运行逐 token 相同的输入：150 条对 cv-20261022 {ic['the_150_equal_cv20261022_inputs']['identical_input_ids']}/"
                   f"{ic['the_150_equal_cv20261022_inputs']['compared']}，285 条对 D-170 {ic['the_285_equal_d170_inputs']['identical_input_ids']}/"
                   f"{ic['the_285_equal_d170_inputs']['compared']}。")
    if est:
        b = est["bases"]
        out.append(f"- 用时：按 cv-20261022 C1 的记录速度估计 {b['cv-20261022']['total_hours']} 小时（按输入长度调整 "
                   f"{b['cv-20261022']['length_adjusted']['total_hours']} 小时；按 D-170 的速度 {b['D-170']['total_hours']} / "
                   f"{b['D-170']['length_adjusted']['total_hours']} 小时），实际 {round(sum(r['wall_seconds'] for r in m['runs'].values()) / 3600, 2)} 小时。")
    out += ["- 折：cv-20261022 的同一程序和种子（直接调用其 cv.assign_folds），无内部留出集；解码阈值固定 0.5，无阈值选择。",
            "- 配置：cv-20261022 的 C1-fold<k>.json，仅改数据路径、标签说明、状态、验证用途、validation.per_epoch_loss = false、"
            "memory.mps_empty_cache_every_steps = 1；与 D-170 的配置相比只差路径、标签说明、状态和验证用途（逐键核对，见 plan.json configs）。",
            f"- 学生重抽样区间：finetuning/student_bootstrap.py，{m['bootstrap'][ALL]['replicates']} 次，种子 {m['bootstrap'][ALL]['seed']}；"
            f"485 的参照为 {L.rel(space.labels)}，各部分的参照为该文件中该部分记录的逐字节子集（part-labels/）。点估计与评价模块一致："
            + "，".join(f"{s} {all(m['bootstrap'][s]['point_values_equal_evaluation_module'].values())}" for s in SETS) + "。", "",
            "混淆矩阵、各代码的 precision/recall 与四维组合分布见 `reports/pooled-*.md` 与 `metrics.json`。", "", "## 文件", ""]
    out += [f"- `{k}` sha256 `{v}`" for k, v in m["files"].items() if "/pooled-" not in k]
    out += [f"- `reports/pooled-{{C1,majority}}-{{{','.join(SETS)}}}.json/.md`（{sum(1 for k in m['files'] if '/pooled-' in k)} 个文件；"
            "sha256 见 metrics.json files）", ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- smoke

def smoke() -> int:
    """The whole chain (plan, run, assemble, report) on the students with the fewest interactions, three from each
    part, 1 epoch, 200 bootstrap replicates; everything under *-smoke names. Pipeline check only."""
    base = ROOT / "data" / "training" / f"{NAME}-smoke"
    labels_path, prepared = base / "smoke-labels.jsonl", base / "smoke-prepared"
    examples = GS.load_examples(PREPARED)
    part_of = part_map()
    rows = Counter(e["student_id"] for e in examples)
    part_of_student = {e["student_id"]: part_of[(e["conversation_id"], e["segment_order"])] for e in examples}
    chosen = set()
    for p in PARTS:
        chosen |= set(sorted((s for s in rows if part_of_student[s] == p), key=lambda s: (rows[s], s))[:3])
    keep = {(e["conversation_id"], e["segment_order"]) for e in examples if e["student_id"] in chosen}
    lines = [line for line in LABELS.read_bytes().split(b"\n") if line.strip()]
    picked = [line for line in lines if record_key(line) in keep]
    assert len(picked) == len(keep)
    data = b"\n".join(picked) + b"\n"
    if labels_path.exists():
        assert labels_path.read_bytes() == data
    else:
        base.mkdir(parents=True, exist_ok=True)
        labels_path.write_bytes(data)
    if not (prepared / "manifest.json").exists():
        run_prepare(labels_path, prepared, None)
    space = Space(f"{NAME}-smoke", prepared, labels_path, HERE / "smoke", epochs=1, replicates=200, full=False)
    print(f"smoke: {len(keep)} interactions, {len(chosen)} students")
    plan(space)
    run_all(space)
    if not space.oof.exists():
        assemble(space)
    report(space)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("check", "prepare", "plan", "run", "assemble", "report", "smoke"))
    a = p.parse_args(argv)
    if a.command == "check":
        return check()
    if a.command == "prepare":
        return prepare()
    if a.command == "plan":
        plan(REAL)
        return 0
    if a.command == "run":
        if not (HERE / "plan.json").exists():
            raise SystemExit("no plan.json: run `plan` first")
        return run_all(REAL)
    if a.command == "assemble":
        return assemble(REAL)
    if a.command == "report":
        return report(REAL)
    return smoke()


if __name__ == "__main__":
    raise SystemExit(main())
