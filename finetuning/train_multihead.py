#!/usr/bin/env python3
"""Train, resume and predict with the four-axis multi-head LoRA classifier.

    python finetuning/train_multihead.py train   --config <config.json> --run-dir data/runs/<run_id>
    python finetuning/train_multihead.py resume  --run-dir data/runs/<run_id>
    python finetuning/train_multihead.py predict --run-dir data/runs/<run_id> --part validation --out <file>
    python finetuning/train_multihead.py train --synthetic --config <config.json> --run-dir tmp/<run_id>

Before the first step, run.json records the full config and the hashes of everything the run
reads: prepared examples and manifest (and through them the selected label source), the split
roster, the base-model files, the built token inputs, the training, split and evaluation code
and the package versions. `resume` and `predict` recompute them and refuse on any difference.
Predictions go to the given path and probabilities beside it (predictions-<x> -> scores-<x>,
otherwise <stem>.scores.jsonl); for a run on private data both must be below data/.

Every checkpoint is a complete training state, written to a temporary directory, hashed and then
renamed into place: all trainable tensors under their exact names (LoRA adapter and heads), the
PEFT adapter and the heads in their own files, the optimizer, the scheduler, the Python, NumPy,
torch CPU and accelerator RNG streams, and progress (epoch, group position within that epoch's
stored data order, optimizer step, per-step and per-epoch history). The frozen base model is
identified by revision and file hashes. An adapter alone is not a resumable state.

SIGINT or SIGTERM stops at the next safe point (exit code 3): after the current optimizer step,
before the next step or phase, or between interactions of a validation or prediction pass, whose
partial results are discarded. Training then leaves a checkpoint to resume from, and resume
recomputes the discarded pass; outputs are written to a partial directory and discarded on a
stop, so only complete outputs are ever published. A request that arrives after the last check
before publishing, or once the predict command has begun its last interaction, lets the complete
outputs be written and the run ends normally (exit code 0). Console output is best effort, so a closed pipe
cannot prevent that checkpoint. A hard kill loses at most the steps since the last checkpoint;
resume recomputes them from the saved RNG state and data position. Pointers and records are
replaced atomically; a start that died before run.json was written leaves no run, and `train`
moves its leftovers aside and starts again.

A non-finite training loss or gradient stops the run before that optimizer step is taken, and an
update that leaves a parameter or optimizer moment non-finite stops it before that step is kept
(exit code 4). No checkpoint is written, latest.json keeps naming the last valid checkpoint, and
the failure is logged and described under failures/ in records whose numbers only grow (if none
can be written, the failure is still logged); no checkpoint is ever written from a non-finite
state. Resume replays from that checkpoint, so a deterministic failure recurs there.

Predictions carry the run_id recorded in run.json (the run directory's name when the run started;
a blank name is refused), also after the directory is renamed. A --synthetic run keeps its
generated inputs inside its directory, so after a rename their recorded paths differ and resume
and predict refuse.

Validation is only what the config says it is for. Each epoch's validation loss is recorded as a
description; nothing here selects a checkpoint, an epoch, a threshold or a hyperparameter from it.
The final step is evaluated once with finetuning/evaluation.py. --synthetic replaces the private
data with generated text and labels and is for pipeline checks only.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import uuid
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Model files come from a project-local cache, and nothing is fetched or reported during a run.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation as EV  # noqa: E402
import grouped_split as GS  # noqa: E402
import labels as L  # noqa: E402
import model_inputs as MI  # noqa: E402
import multihead as M  # noqa: E402

CONFIG_KIND = "multihead_lora_training"
RUN_KIND = "multihead_training_run"
EXIT_INTERRUPTED = 3
EXIT_FAILED = 4
TRAINING_CODE = ("codebook_schema.json", "codebook_schema.py", "coding/direct_coding.py", "finetuning/data.py",
                 "finetuning/evaluation.py", "finetuning/grouped_split.py", "finetuning/labels.py",
                 "finetuning/model_inputs.py", "finetuning/multihead.py", "finetuning/train_multihead.py")
# Descriptive config fields must name what is implemented; anything else is refused, so a run
# record can never claim a setting the code did not apply.
IMPLEMENTED = {
    ("optim", "name"): "AdamW",
    ("loss", "single_select"): "cross_entropy",
    ("loss", "multi_select"): "bce_with_logits_mean_over_codes",
    ("loss", "axis_weights"): "equal_sum_over_labelled_axes",
    ("loss", "null_labels"): "axis_removed_before_loss",
    # ("loss", "class_weights") is null or CLASS_WEIGHT_SCHEME, checked in check_config
    ("schedule", "decay"): "linear_after_warmup",
    ("decode", "single_select"): "argmax",
    ("decode", "multi_select"): "threshold_else_single_highest",
    ("decode", "exclusive_negative"): "labels.decode_scores",
    ("validation", "evaluated_state"): "final_optimizer_step",
    ("model", "dtype"): "float32",
}
PACKAGES = ("torch", "transformers", "peft", "safetensors", "tokenizers", "huggingface_hub", "accelerate",
            "numpy", "pandas")
CLASS_WEIGHT_SCHEME = "training_part_balanced_v1"     # see class_weights()
CLASS_WEIGHT_MAX = 10.0
statements = EV.statements          # report statements live with the evaluation module


class _Stop:
    signal = None


class Interrupted(Exception):
    """A stop request reached a pass over interactions; that partial pass is discarded."""


class NumericalFailure(Exception):
    """A non-finite loss, gradient, update or state. Raised before that state is kept or saved."""

    def __init__(self, detail: dict):
        super().__init__(detail["what"])
        self.detail = detail


def release_cache(device: torch.device) -> None:
    """Hand unused cached accelerator memory back to the system (MPS only; a no-op elsewhere).

    The MPS allocator keeps every freed buffer of each size it has met, and over a long run with
    varying input lengths it holds several times what a step needs (the 0.6B runs of cv-20261022
    grew from 10 GiB to 42 GiB). This touches no tensor and no RNG state, so it cannot change a
    number; the cache is simply rebuilt on demand. Allocator watermarks are no substitute: a cap
    below the cache size raises out-of-memory although the live set is far smaller."""
    if device.type == "mps":
        torch.mps.empty_cache()


def _check_stop() -> None:
    if _Stop.signal is not None:
        raise Interrupted(_Stop.signal)


def _number(x: float):
    """JSON-safe: a non-finite value is written as its name ('nan', 'inf', '-inf')."""
    return x if math.isfinite(x) else repr(x)


def say(text: str, stream=None) -> None:
    """Console progress is best effort: a closed pipe must never stop training or a checkpoint."""
    target = stream or sys.stdout
    try:
        print(text, file=target, flush=True)
    except RuntimeError:
        pass        # the signal handler interrupted a write to the same stream ("reentrant call")
    except (BrokenPipeError, OSError, ValueError):
        # The reader is gone. Point the descriptor at /dev/null so the interpreter's final flush
        # cannot fail and replace the exit status (Python signal docs, note on SIGPIPE).
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, target.fileno())
            os.close(devnull)
        except (OSError, ValueError, AttributeError):
            pass


def _on_signal(signum, _frame):
    _Stop.signal = signum
    say(f"\n[signal {signum}] stopping at the next safe point: after the current optimizer step (with a "
        "checkpoint) or before the next interaction", sys.stderr)


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dir_name(run_dir: Path) -> str:
    """The run directory's own name as stored on disk, also when it is reached as '.', through '..'
    or a symbolic link, or in another letter case or Unicode normalisation on a file system that
    treats those as one name."""
    real = Path(os.path.realpath(run_dir))
    if real.is_dir():
        try:
            entries = list(real.parent.iterdir())
        except OSError:
            entries = []                    # an unreadable parent: the name as reached
        for entry in entries:               # matches other letter case or Unicode normalisation too
            try:
                if not (entry.is_symlink() or getattr(entry, "is_junction", bool)()) and os.path.samefile(entry, real):
                    return entry.name
            except OSError:
                continue
    return real.name                        # not created yet: the name as given


def _nonfinite(named) -> list:
    """Names of the (name, tensor) pairs holding a NaN or an infinity; one sync per device."""
    by_device, bad = {}, []
    for name, t in named:
        by_device.setdefault(t.device, []).append((name, t))
    for items in by_device.values():
        finite = torch.stack([torch.isfinite(t).all() for _, t in items])
        if not bool(finite.all()):
            bad += [name for (name, _), ok in zip(items, finite.tolist()) if not ok]
    return bad


def _moments(named: list, optimizer) -> list:
    """The optimizer's floating-point state tensors, named after their parameters."""
    return [(f"{name}:{k}", v) for name, p in named for k, v in optimizer.state.get(p, {}).items()
            if torch.is_tensor(v) and v.is_floating_point()]


def canonical_sha256(obj) -> str:
    return L.sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def key_text(key: tuple) -> str:
    return f"{key[0]}#{key[1]}"


def write_json(path: Path, obj, exclusive: bool = True) -> None:
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def write_json_atomic(path: Path, obj) -> None:
    """Write beside the target and rename over it, so a reader never sees a partial file."""
    path = Path(path)
    partial = path.with_name(f".{path.name}.partial")
    write_json(partial, obj, exclusive=False)
    os.replace(partial, path)


# ----------------------------------------------------------------------------- config

def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    check_config(cfg)
    return cfg


def check_config(cfg: dict) -> None:
    required = {"config_kind", "status", "seed", "device", "data", "model", "input", "lora", "heads", "loss",
                "optim", "schedule", "checkpoint", "validation", "decode"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"config is missing {sorted(missing)}")
    if cfg["config_kind"] != CONFIG_KIND:
        raise ValueError(f"config_kind must be {CONFIG_KIND}")
    MI.InputPolicy.from_config(cfg["input"])
    s, o, c = cfg["schedule"], cfg["optim"], cfg["checkpoint"]
    for name, value in (("seed", cfg["seed"]), ("epochs", s["epochs"]), ("grad_accumulation", s["grad_accumulation"]),
                        ("every_steps", c["every_steps"]), ("keep_last", c["keep_last"])):
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a positive integer")
    if s.get("micro_batch") != 1:
        raise ValueError("this implementation trains one unpadded interaction per forward pass (micro_batch 1)")
    if not 0 <= s["warmup_fraction"] < 1 or o["lr"] <= 0 or o["grad_clip_norm"] <= 0:
        raise ValueError("invalid warmup_fraction, lr or grad_clip_norm")
    for (section, name), value in IMPLEMENTED.items():
        if cfg[section].get(name, "<missing>") != value:
            raise ValueError(f"{section}.{name} must be {value!r}: that is what this trainer implements "
                             "(bfloat16 heads and other variants are not implemented)")
    if cfg["loss"].get("class_weights", "<missing>") not in (None, CLASS_WEIGHT_SCHEME):
        raise ValueError(f"loss.class_weights must be null or {CLASS_WEIGHT_SCHEME!r}: that is what this trainer implements")
    head_lr = o.get("head_lr")
    if head_lr is not None and (type(head_lr) not in (int, float) or not math.isfinite(head_lr) or head_lr <= 0):
        raise ValueError("optim.head_lr must be a positive number, or absent for one learning rate")
    if cfg["device"] == "mps" and cfg["model"]["gradient_checkpointing"] and cfg["lora"]["dropout"] > 0:
        # torch 2.14 activation checkpointing does not stash the MPS RNG (torch.mps has no
        # `_initialized`), so dropout inside a checkpointed layer would be recomputed with a
        # different mask and the gradient would not match the forward pass. Verified on device.
        raise ValueError("on MPS with gradient checkpointing, LoRA dropout must be 0 (masks are not replayed)")
    memory = cfg.get("memory")
    if memory is not None:
        every = memory.get("mps_empty_cache_every_steps") if isinstance(memory, dict) else "<not an object>"
        if not isinstance(memory, dict) or set(memory) - {"mps_empty_cache_every_steps"} or (
                every is not None and (type(every) is not int or every < 1)):
            raise ValueError("memory, if present, holds only mps_empty_cache_every_steps: a positive integer or null")
    if cfg["validation"].get("used_for_selection") is not False:
        raise ValueError("this trainer never selects on validation; the config must say used_for_selection: false")
    if not 0 < cfg["decode"]["multi_threshold"] <= 1:
        raise ValueError("decode.multi_threshold must be in (0, 1]")


def device_from(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("config asks for mps but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be mps or cpu")


# ----------------------------------------------------------------------------- data

@dataclass
class RunData:
    examples: dict          # key -> prepared example
    part_of: dict           # key -> "train" | "validation"
    targets: dict           # key -> labels.encode_targets(labels), recomputed here
    fingerprint: dict       # hashes of prepared data, split and label sources
    summary: dict           # per-part counts and label support

    def keys(self, part: str) -> list:
        return sorted(k for k, p in self.part_of.items() if p == part)


def check_data(examples: list, part_of: dict) -> tuple[dict, dict, dict]:
    """Refuse unless every interaction is assigned once, nothing crosses the split, context is
    earlier and from the same conversation, and stored targets/masks match the labels."""
    by_key = {}
    for e in examples:
        key = (e["conversation_id"], e["segment_order"])
        if key in by_key:
            raise ValueError(f"duplicate interaction {key_text(key)}")
        by_key[key] = e
    if set(by_key) != set(part_of):
        raise ValueError("the split roster and the prepared examples cover different interactions")
    owner, targets = {}, {}
    for key, e in by_key.items():
        part = part_of[key]
        for ident in (("student", e["student_id"]), ("conversation", e["conversation_id"])):
            if owner.setdefault(ident, part) != part:
                raise ValueError(f"{ident[0]} {ident[1]} appears in more than one part")
        for c in e["context"]:
            ckey = (e["conversation_id"], c["segment_order"])
            if c["segment_order"] >= e["segment_order"]:
                raise ValueError(f"{key_text(key)}: context includes a later or the same interaction")
            if ckey in part_of and part_of[ckey] != part:
                raise ValueError(f"{key_text(key)}: context row {key_text(ckey)} is in another part")
        targets[key] = L.encode_targets(e["labels"])
        if "targets" in e and e["targets"] != targets[key]:
            raise ValueError(f"{key_text(key)}: stored targets do not match the labels")
        if sorted(e.get("masked_axes", [])) != sorted(a for a in L.AXES if e["labels"][a] is None):
            raise ValueError(f"{key_text(key)}: masked_axes do not match the null labels")
    summary = {}
    for part in GS.PARTS:
        chosen = [by_key[k] for k in sorted(by_key) if part_of[k] == part]
        summary[part] = {"interactions": len(chosen), "students": len({e["student_id"] for e in chosen}),
                         "conversations": len({e["conversation_id"] for e in chosen}),
                         "label_support": GS.label_support(chosen)}
    return by_key, targets, summary


def load_data(prepared_dir: Path, split_file: Path) -> RunData:
    fingerprint = GS.prepared_fingerprint(prepared_dir)
    split_record, part_of = GS.load_split(split_file, prepared_dir)
    examples, targets, summary = check_data(GS.load_examples(prepared_dir), part_of)
    fingerprint = {"prepared": fingerprint,
                   "split": {"path": L.rel(split_file), "sha256": L.sha256_file(split_file),
                             "seed": split_record["seed"], "status": split_record["status"],
                             "validation_purpose": split_record["validation_purpose"]}}
    return RunData(examples, part_of, targets, fingerprint, summary)


# ----------------------------------------------------------------------------- synthetic inputs

_WORDS = ("loop", "pointer", "array", "struct", "malloc", "thread", "socket", "buffer", "signal", "file",
          "parse", "queue", "mutex", "fork", "pipe", "header", "linker", "stack", "heap", "offset")
_STUDENT_CUE = {"Providing material / responding": "here is my output", "Information retrieval": "what does it mean",
                "Task automation": "please write it for me", "Guided generation": "it must follow my rule"}


def synthetic_examples(seed: int = 7, sizes=(9, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1), long_text: bool = True) -> list:
    """Generated conversations with schema-valid labels. Cue words make the labels learnable; one
    row is very long, one has a truncated capture (AI labels null) and every row carries canary
    annotation fields that must never reach a model input. Context comes from finetuning/data.py."""
    rng = random.Random(seed)
    student, topic, aidel, airesp = L.AXES

    def words(n):
        return " ".join(rng.choice(_WORDS) for _ in range(n))

    rows, labelled = [], {}
    for s, size in enumerate(sizes):
        sid, cid = f"SYN_S{s:02d}", f"SYN_C{s:02d}"
        for seg in range(1, size + 1):
            kind = rng.choice(L.DIRECT_CODES[student])
            topics = sorted(rng.sample(L.DIRECT_CODES[topic], rng.choice((1, 1, 2))), key=L.DIRECT_CODES[topic].index)
            delegation = (["ND"] if rng.random() < 0.3 else
                          sorted(rng.sample(L.DIRECT_CODES[aidel][1:], rng.choice((1, 2))), key=L.DIRECT_CODES[aidel].index))
            response_types = (["none of the four applies"] if rng.random() < 0.05 else
                              sorted(rng.sample(L.DIRECT_CODES[airesp][:4], rng.choice((1, 2, 3))),
                                     key=L.DIRECT_CODES[airesp].index))
            truncated = (s, seg) == (1, 2)
            lab = {student: kind, topic: topics, aidel: None if truncated else delegation,
                   airesp: None if truncated else response_types}
            prompt = f"{_STUDENT_CUE[kind]} about {' and '.join(topics)}: {words(rng.randint(5, 40))}"
            response = f"{' / '.join(delegation)} {' / '.join(response_types)}: {words(rng.randint(20, 120))}"
            rows.append([sid, cid, seg, prompt, response])
            labelled[(cid, seg)] = (lab, truncated)
    if long_text:                     # far beyond every cap and the whole budget
        rows[3][3] += " " + words(9000)
        rows[3][4] += " " + words(6000)
        rows[4][4] += " " + words(5000)
    corpus = pd.DataFrame(rows, columns=L.corpus_data.KEY_COLUMNS)
    out = []
    for ex in L.corpus_data.iter_examples(corpus, 3):
        lab, truncated = labelled[(ex["conversation_id"], ex["segment_order"])]
        masked = [a for a in L.AXES if lab[a] is None]
        out.append({**ex, "labels": lab, "targets": L.encode_targets(lab), "masked_axes": masked,
                    "mask_reasons": {a: L.TRUNCATED for a in masked},
                    "annotation_record_kind": "synthetic",
                    "notes": "SYNTHETIC-CANARY-NOTE",
                    "source_record": {"notes": "SYNTHETIC-CANARY-SOURCE-NOTE", "labels": lab,
                                      "flags": [L.TRUNCATED] if truncated else [],
                                      "review_history": [{"original_notes": "SYNTHETIC-CANARY-HISTORY"}]}})
    return out


def write_synthetic_inputs(directory: Path, examples: list, seed: int, validation_target: int,
                           validation_max: int) -> tuple[Path, Path]:
    """A prepared-format directory and split record for synthetic examples (no private data).
    Generation is deterministic, so a retry after an aborted start reuses identical inputs."""
    prepared, split_path = directory / "synthetic-prepared", directory / "synthetic-split.json"
    text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
    split = GS.make_split(examples, seed, validation_target, validation_max)
    if prepared.exists() or split_path.exists():
        try:
            recorded = json.loads(split_path.read_text(encoding="utf-8"))
            reuse = ((prepared / "examples.jsonl").read_text(encoding="utf-8") == text
                     and {k: recorded.get(k) for k in split} == json.loads(json.dumps(split))
                     and GS.prepared_fingerprint(prepared) == recorded["prepared"])
        except (OSError, ValueError, KeyError):
            reuse = False
        if reuse:
            return prepared, split_path
        split_path.unlink(missing_ok=True)        # split first, so a half-removed set is never reused
        if prepared.exists():
            shutil.rmtree(prepared)
    prepared.mkdir(parents=True)
    with (prepared / "examples.jsonl").open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    keys = sorted((e["conversation_id"], e["segment_order"]) for e in examples)
    write_json(prepared / "manifest.json", {
        "format_version": "synthetic", "scope": "synthetic pipeline check only",
        "prepared_file": {"name": "examples.jsonl", "sha256": L.sha256_bytes(text.encode("utf-8")), "records": len(examples)},
        "keys_sha256": L.sha256_bytes(json.dumps(keys).encode("utf-8")), "vocabulary_sha256": L.vocabulary_sha256(),
        "sources": [{"path": "synthetic", "sha256": None, "record_kind": "synthetic", "records": len(examples)}],
        "source_selection": "synthetic", "input_rendering": {"context_interactions": 3}})
    write_json_atomic(split_path, {"record_kind": GS.RECORD_KIND, "status": "synthetic pipeline check only",
                            "validation_purpose": "synthetic pipeline check only",
                            "prepared": GS.prepared_fingerprint(prepared), **split})
    return prepared, split_path


# ----------------------------------------------------------------------------- inputs and identity

def build_inputs(data: RunData, encode, policy: MI.InputPolicy) -> tuple[dict, str]:
    built = {k: MI.build_input(data.examples[k], encode, policy) for k in sorted(data.examples)}
    digest = canonical_sha256([[key_text(k), b["input_ids"], b["student_pool"], b["ai_pool"]]
                               for k, b in sorted(built.items())])
    return built, digest


def tokenizer_encoder(model_dir: Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.model_max_length = 10 ** 9      # inputs are clipped by the policy, not by the tokenizer
    return lambda text: tok(text, add_special_tokens=False)["input_ids"]


def model_snapshot(model_cfg: dict) -> tuple[Path, dict]:
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(model_cfg["id"], revision=model_cfg["revision"], local_files_only=True))
    files = {p.name: L.sha256_file(p) for p in sorted(path.iterdir()) if p.is_file()}
    if files.get("model.safetensors") != model_cfg["weights_sha256"]:
        raise ValueError("model.safetensors does not match the configured weights_sha256")
    return path, {"id": model_cfg["id"], "revision": model_cfg["revision"], "files_sha256": files}


def environment(device: torch.device) -> dict:
    def sysctl(name):
        try:
            return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5, encoding="utf-8").stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
            "device": device.type,
            "platform": {"system": platform.system(), "machine": platform.machine(), "mac_ver": platform.mac_ver()[0],
                         "chip": sysctl("machdep.cpu.brand_string"), "memory_bytes": sysctl("hw.memsize"),
                         "mps_available": torch.backends.mps.is_available()}}


def fingerprint(cfg: dict, data: RunData, model_identity: dict, inputs_sha256: str, env: dict) -> dict:
    """Everything resume must find unchanged. The platform is recorded in run.json, not compared."""
    return {"config_sha256": canonical_sha256(cfg), "data": data.fingerprint, "model": model_identity,
            "inputs_sha256": inputs_sha256, "python": env["python"], "packages": env["packages"],
            "device": env["device"], "code_sha256": {p: L.sha256_file(ROOT / p) for p in TRAINING_CODE}}


def fingerprint_differences(a: dict, b: dict) -> list:
    diffs = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
                diffs += [f"{k}.{d}" for d in fingerprint_differences(a[k], b[k])]
            else:
                diffs.append(k)
    return diffs


# ----------------------------------------------------------------------------- model, optimizer, schedule

def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def linear_warmup_decay(warmup: int, total: int):
    def factor(step: int) -> float:              # step = optimizer steps already taken
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))
    return factor


def build_training_objects(cfg: dict, backbone_factory, device: torch.device, n_train: int):
    seed_everything(cfg["seed"], device)
    backbone = backbone_factory()
    hidden = backbone.config.hidden_size
    model = M.MultiAxisClassifier(M.attach_lora(backbone, cfg["lora"]), hidden,
                                  cfg["heads"]["dropout"], cfg["heads"]["init_std"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    o, s = cfg["optim"], cfg["schedule"]
    if o.get("head_lr") is None:
        groups = params
    else:                                       # the adapter at lr, the four heads at head_lr; same schedule
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        groups = [{"params": [p for n, p in named if not n.startswith("heads.")], "lr": o["lr"]},
                  {"params": [p for n, p in named if n.startswith("heads.")], "lr": o["head_lr"]}]
    optimizer = torch.optim.AdamW(groups, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                                  weight_decay=o["weight_decay"])
    steps_per_epoch = math.ceil(n_train / s["grad_accumulation"])
    total = steps_per_epoch * s["epochs"]
    warmup = max(1, math.ceil(s["warmup_fraction"] * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_decay(warmup, total))
    plan = {"train_examples": n_train, "steps_per_epoch": steps_per_epoch, "total_steps": total,
            "warmup_steps": warmup, "trainable_parameters": sum(p.numel() for p in params),
            "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    return model, optimizer, scheduler, plan


def epoch_order(seed: int, epoch: int, keys: list) -> list:
    """This epoch's data order: its own generator, so it never depends on the global RNG."""
    keys = sorted(keys)
    return [key_text(keys[i]) for i in np.random.default_rng([seed, epoch]).permutation(len(keys))]


# ----------------------------------------------------------------------------- checkpoints

def _rng_capture(device: torch.device) -> tuple[dict, dict]:
    py = random.getstate()
    npy = np.random.get_state()
    tensors = {"torch_cpu": torch.get_rng_state()}
    if device.type == "mps":
        tensors["torch_mps"] = torch.mps.get_rng_state()
    return ({"python": [py[0], list(py[1]), py[2]],
             "numpy": [npy[0], npy[1].tolist(), int(npy[2]), int(npy[3]), float(npy[4])]}, tensors)


def _rng_restore(state: dict, tensors: dict, device: torch.device) -> None:
    py, npy = state["python"], state["numpy"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    np.random.set_state((npy[0], np.array(npy[1], dtype=np.uint32), npy[2], npy[3], npy[4]))
    torch.set_rng_state(tensors["torch_cpu"])
    if device.type == "mps":
        if "torch_mps" not in tensors:
            raise ValueError("checkpoint has no MPS RNG state")
        torch.mps.set_rng_state(tensors["torch_mps"])


def checkpoint_tag(progress: dict) -> str:
    return f"s{progress['global_step']:06d}-e{progress['epoch']:02d}-g{progress['group']:03d}"


def save_checkpoint(run_dir: Path, model, optimizer, scheduler, progress: dict, device: torch.device,
                    keep_last: int) -> str:
    from safetensors.torch import save_file
    tag = checkpoint_tag(progress)
    state = M.trainable_state(model)
    bad = _nonfinite(list(state.items()) + _moments(list(model.named_parameters()), optimizer))
    if bad:                                            # refused before anything is written
        raise NumericalFailure({"what": f"non-finite state at optimizer step {progress['global_step']}; "
                                        f"checkpoint {tag} not written", "kind": "state",
                                "optimizer_step": progress["global_step"], "nonfinite_tensors": bad})
    root = run_dir / "checkpoints"
    root.mkdir(exist_ok=True)
    final, partial = root / tag, root / f".{tag}.partial"
    if final.exists():
        raise FileExistsError(f"checkpoint {tag} already exists")
    if partial.exists():
        shutil.rmtree(partial)                         # left by a kill during an earlier save
    partial.mkdir()
    save_file(state, partial / "trainable.safetensors")
    save_file({n: t for n, t in state.items() if n.startswith("heads.")}, partial / "heads.safetensors")
    model.backbone.save_pretrained(partial / "adapter", save_embedding_layers=False)
    torch.save(optimizer.state_dict(), partial / "optimizer.pt")
    torch.save(scheduler.state_dict(), partial / "scheduler.pt")
    rng_json, rng_tensors = _rng_capture(device)
    write_json(partial / "rng.json", rng_json)
    save_file(rng_tensors, partial / "rng.safetensors")
    write_json(partial / "progress.json", progress)
    files = {p.relative_to(partial).as_posix(): L.sha256_file(p) for p in sorted(partial.rglob("*")) if p.is_file()}
    write_json(partial / "COMPLETE.json", {"tag": tag, "global_step": progress["global_step"], "saved_utc": utc(),
                                           "run_fingerprint_sha256": progress["run_fingerprint_sha256"],
                                           "run_nonce": progress["run_nonce"], "files": files})
    os.rename(partial, final)
    write_json_atomic(root / "latest.json", {"tag": tag, "global_step": progress["global_step"],
                                             "complete": progress["complete"]})
    protected = {tag}
    if (root / "final.json").exists():
        protected.add(json.loads((root / "final.json").read_text(encoding="utf-8"))["tag"])
    saved = sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith("s")), key=lambda p: p.name)
    for old in saved[:-keep_last]:
        if old.name not in protected:
            shutil.rmtree(old)
    return tag


def _os_metadata(path: Path) -> bool:
    """Files the OS adds to folders it displays (Finder), never part of a checkpoint."""
    return path.name == ".DS_Store" or path.name.startswith("._")


def verify_checkpoint(ckpt: Path) -> dict:
    complete = json.loads((ckpt / "COMPLETE.json").read_text(encoding="utf-8"))
    present = {p.relative_to(ckpt).as_posix() for p in ckpt.rglob("*")
               if p.is_file() and not _os_metadata(p)} - {"COMPLETE.json"}
    if present != set(complete["files"]):
        raise ValueError(f"checkpoint {ckpt.name}: files differ from its record "
                         f"(missing {sorted(set(complete['files']) - present)}, extra {sorted(present - set(complete['files']))})")
    for name, digest in complete["files"].items():
        if L.sha256_file(ckpt / name) != digest:
            raise ValueError(f"checkpoint {ckpt.name}: {name} does not match its recorded hash")
    if complete.get("tag") != ckpt.name:
        raise ValueError(f"checkpoint directory {ckpt.name} records the tag {complete.get('tag')!r}")
    return complete


def restore_checkpoint(ckpt: Path, model, optimizer, scheduler, device: torch.device) -> dict:
    """Load a complete state into freshly built objects. RNG is restored last, after building."""
    from safetensors.torch import load_file
    verify_checkpoint(ckpt)
    M.load_trainable_state(model, load_file(ckpt / "trainable.safetensors"))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=True))
        scheduler.load_state_dict(torch.load(ckpt / "scheduler.pt", map_location="cpu", weights_only=True))
        _rng_restore(json.loads((ckpt / "rng.json").read_text(encoding="utf-8")), load_file(ckpt / "rng.safetensors"), device)
    return json.loads((ckpt / "progress.json").read_text(encoding="utf-8"))


TAG = re.compile(r"^s\d{6,}-e\d{2,}-g\d{3,}$")      # checkpoint_tag's widths are minimums


def checkpoint_path(run_dir: Path, which: str) -> Path:
    """'latest', 'final' or a bare tag of THIS run; any other path is refused."""
    root = run_dir / "checkpoints"
    if which in ("latest", "final"):
        pointer = root / f"{which}.json"
        if not pointer.exists():
            raise FileNotFoundError(f"no {which} checkpoint in {run_dir}")
        which = json.loads(pointer.read_text(encoding="utf-8"))["tag"]
    if not isinstance(which, str) or not TAG.match(which) or not (root / which).is_dir():
        raise ValueError(f"{which!r} is not a checkpoint of {run_dir}")
    return root / which


# ----------------------------------------------------------------------------- training loop

def log_event(run_dir: Path, event: dict) -> None:
    path, torn = run_dir / "train_log.jsonl", False
    try:                                    # a kill during an earlier append can leave a torn last line
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
    except OSError:                         # no log yet, or an empty one
        pass
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(("\n" if torn else "") + json.dumps({"utc": utc(), **event}, ensure_ascii=False) + "\n")


def mean_loss(model, keys: list, built: dict, targets: dict, device: torch.device) -> dict:
    """Masked loss without gradient or dropout. Consumes no RNG, so it cannot shift training.
    A stop request is honoured between interactions (Interrupted); the partial pass is discarded."""
    model.eval()
    sums, counts, total = {a: 0.0 for a in L.AXES}, {a: 0 for a in L.AXES}, 0.0
    try:
        with torch.no_grad():
            for key in keys:
                _check_stop()
                b = built[key]
                logits = model(torch.tensor([b["input_ids"]], device=device), b["student_pool"], b["ai_pool"])
                loss, parts = M.example_loss(logits, targets[key])
                total += 0.0 if loss is None else loss.item()
                for a, v in parts.items():
                    if v is not None:
                        sums[a] += v.item()
                        counts[a] += 1
    finally:
        model.train()
    # loss_mean uses the training objective's definition without class weights, so it stays comparable
    # across configs (per interaction: the sum over its labelled axes; mean over interactions).
    # Compare parts through axis_loss_mean, because the
    # number of labelled axes per interaction can differ between parts.
    return {"examples": len(keys), "loss_mean": total / len(keys) if keys else None,
            "axis_loss_mean": {a: sums[a] / counts[a] if counts[a] else None for a in L.AXES}, "axis_rows": counts}


def _nonfinite_loss(progress: dict, epoch: int, group: int, key: tuple, value: float, parts: dict) -> dict:
    return {"what": f"non-finite loss before optimizer step {progress['global_step'] + 1}; the step was not taken",
            "kind": "loss", "optimizer_step": progress["global_step"] + 1, "epoch": epoch, "group": group,
            "interaction": key_text(key), "loss": _number(value),
            "axis_loss": {a: None if v is None else _number(v.item()) for a, v in parts.items()}}


def _nonfinite_gradient(progress: dict, epoch: int, group: int, named: list, norm: float) -> dict:
    bad = {n: int((~torch.isfinite(p.grad)).sum()) for n, p in named
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())}
    return {"what": f"non-finite gradient norm before optimizer step {progress['global_step'] + 1}; the step "
                    "was not taken", "kind": "gradient", "optimizer_step": progress["global_step"] + 1, "epoch": epoch,
            "group": group, "grad_norm": _number(norm), "nonfinite_gradient_elements": bad,
            "note": None if bad else "every gradient element is finite; their norm overflowed"}


def _nonfinite_update(progress: dict, epoch: int, group: int, bad: list, loss_sum: float, norm: float) -> dict:
    return {"what": f"optimizer step {progress['global_step'] + 1} left non-finite values after a finite loss and "
                    "gradient; the step is not kept", "kind": "update", "optimizer_step": progress["global_step"] + 1,
            "epoch": epoch, "group": group, "loss_sum": loss_sum, "grad_norm": norm, "nonfinite_tensors": bad}


def train_loop(run_dir: Path, cfg: dict, data: RunData, built: dict, model, optimizer, scheduler, progress: dict,
               device: torch.device, stop_after_steps: int | None = None) -> str:
    """Train from `progress` to the end of the schedule. Returns "complete" or "interrupted"; raises
    NumericalFailure before an optimizer step whose loss or gradient is not finite, and before a step
    whose update left a non-finite value is counted, logged or saved."""
    s = cfg["schedule"]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    train_keys, val_keys = data.keys("train"), data.keys("validation")
    keep = cfg["checkpoint"]["keep_last"]
    weights = progress["plan"].get("class_weights")     # training loss only; validation loss stays unweighted
    release_every = (cfg.get("memory") or {}).get("mps_empty_cache_every_steps")   # null/absent: never
    saved = True                    # the state is the latest checkpoint's: just started or just restored

    def stop(phase: str) -> str:
        nonlocal saved
        tag = checkpoint_tag(progress) if saved else save_checkpoint(run_dir, model, optimizer, scheduler,
                                                                     progress, device, keep)
        saved = True
        log_event(run_dir, {"event": "stopped", "phase": phase, "checkpoint": tag, "signal": _Stop.signal})
        say(f"stopped with checkpoint {tag}")
        return "interrupted"

    model.train()
    while progress["epoch"] < s["epochs"]:
        epoch = progress["epoch"]
        expected = epoch_order(cfg["seed"], epoch, train_keys)
        if progress["order_epoch"] != epoch:
            progress["order"], progress["order_epoch"] = expected, epoch
        if progress["order"] != expected:
            raise ValueError("stored data order differs from its definition")
        groups = [progress["order"][i:i + s["grad_accumulation"]]
                  for i in range(0, len(progress["order"]), s["grad_accumulation"])]
        for g in range(progress["group"], len(groups)):
            if _Stop.signal is not None:            # requested between steps or phases: take no further step
                return stop("before optimizer step")
            started = time.time()
            sums, counts, total, used = {a: 0.0 for a in L.AXES}, {a: 0 for a in L.AXES}, 0.0, 0
            for text in groups[g]:
                key = L.parse_key(text)
                b = built[key]
                logits = model(torch.tensor([b["input_ids"]], device=device), b["student_pool"], b["ai_pool"])
                loss, parts = (M.example_loss(logits, data.targets[key]) if weights is None
                               else M.example_loss(logits, data.targets[key], weights))
                if loss is None:
                    continue
                value = loss.item()
                if not math.isfinite(value):
                    raise NumericalFailure(_nonfinite_loss(progress, epoch, g, key, value, parts))
                (loss / len(groups[g])).backward()
                total, used = total + value, used + 1
                for a, v in parts.items():
                    if v is not None:
                        sums[a] += v.item()
                        counts[a] += 1
            # clip_grad_norm_ is exactly these two calls; between them the norm is checked, so a
            # non-finite gradient stops the run before it reaches the parameters or the optimizer.
            norm = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])
            grad_norm = float(norm)
            if not math.isfinite(grad_norm):
                raise NumericalFailure(_nonfinite_gradient(progress, epoch, g, named, grad_norm))
            torch.nn.utils.clip_grads_with_norm_(params, cfg["optim"]["grad_clip_norm"], norm)
            lrs = scheduler.get_last_lr()
            lr = lrs[0]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            saved = False
            bad = _nonfinite(named + _moments(named, optimizer))
            if bad:                                 # e.g. AdamW dividing 0 by 0: not counted, logged or kept
                raise NumericalFailure(_nonfinite_update(progress, epoch, g, bad, total, grad_norm))
            progress["global_step"] += 1
            progress["group"] = g + 1
            step = {"step": progress["global_step"], "epoch": epoch, "group": g, "examples": len(groups[g]),
                    "examples_with_labels": used, "loss_sum": total,
                    "loss_mean": total / len(groups[g]),          # exactly the objective backpropagated
                    "lr": lr, "grad_norm": grad_norm, "axis_loss_sum": sums,
                    "axis_loss_mean": {a: sums[a] / counts[a] if counts[a] else None for a in L.AXES},
                    "axis_rows": counts, "seconds": round(time.time() - started, 3)}
            if len(lrs) > 1:
                step["head_lr"] = lrs[1]
            if device.type == "mps":
                step["mps_driver_gib"] = round(torch.mps.driver_allocated_memory() / 2 ** 30, 2)
            progress["steps"].append(step)
            log_event(run_dir, {"event": "step", **step})
            line = (f"step {step['step']:4d}/{progress['plan']['total_steps']}  epoch {epoch + 1}  "
                    f"loss {step['loss_mean']:.4f}  lr {lr:.2e}  {step['seconds']:.1f}s")
            say(line)
            if release_every and progress["global_step"] % release_every == 0:
                release_cache(device)
            if _Stop.signal is not None or (stop_after_steps is not None and progress["global_step"] >= stop_after_steps):
                return stop("after optimizer step")
            if g + 1 < len(groups) and progress["global_step"] % cfg["checkpoint"]["every_steps"] == 0:
                tag = save_checkpoint(run_dir, model, optimizer, scheduler, progress, device, keep)
                saved = True
                log_event(run_dir, {"event": "checkpoint", "checkpoint": tag})
        epoch_steps = [x for x in progress["steps"] if x["epoch"] == epoch]
        rows = sum(x["examples"] for x in epoch_steps)
        axis_rows = {a: sum(x["axis_rows"][a] for x in epoch_steps) for a in L.AXES}
        record = {"epoch": epoch, "train_during_epoch": {
            "examples": rows, "loss_mean": sum(x["loss_sum"] for x in epoch_steps) / rows if rows else None,
            "axis_loss_mean": {a: (sum(x["axis_loss_sum"][a] for x in epoch_steps) / axis_rows[a]) if axis_rows[a] else None
                               for a in L.AXES},
            "axis_rows": axis_rows, "note": "accumulated while the weights changed, with head dropout on"}}
        if cfg["validation"]["per_epoch_loss"] and val_keys:
            try:
                record["validation"] = mean_loss(model, val_keys, built, data.targets, device)
            except Interrupted:             # the epoch's steps are kept; resume repeats this validation
                return stop("validation")
        progress["epochs"].append(record)
        if release_every:
            release_cache(device)
        progress.update(epoch=epoch + 1, group=0, order=None, order_epoch=None)
        progress["complete"] = progress["epoch"] >= s["epochs"]
        tag = save_checkpoint(run_dir, model, optimizer, scheduler, progress, device, keep)
        saved = True
        log_event(run_dir, {"event": "epoch_end", "checkpoint": tag, **record})
        say(f"epoch {epoch + 1} done; validation loss (descriptive only): "
            f"{record.get('validation', {}).get('loss_mean')}")
        if _Stop.signal is not None:        # before the next epoch, or before the outputs after the last
            return stop("after epoch end")
    return "complete"


# ----------------------------------------------------------------------------- prediction and evaluation

def predict(model, keys: list, built: dict, device: torch.device, threshold: float) -> tuple[dict, dict]:
    """Labels and probabilities for every key, or Interrupted if a stop request arrives first."""
    model.eval()
    labels, probs = {}, {}
    with torch.no_grad():
        for key in keys:
            _check_stop()
            b = built[key]
            logits = model(torch.tensor([b["input_ids"]], device=device), b["student_pool"], b["ai_pool"])
            probs[key] = M.scores(logits)
            labels[key] = L.decode_scores(probs[key], threshold)
    return labels, probs


def class_weights(cfg: dict, data: RunData) -> dict | None:
    """Per-axis loss weights from the training part's labels only; None when the config sets none.

    training_part_balanced_v1, over the labelled (non-null) training rows of each axis:
      single-select code c: n / (k * n_c), k = number of codes present in the training part
        (a code absent there gets 1; it is never a target);
      multi-select code c: BCE pos_weight negatives_c / positives_c, floored at 1 (frequent codes are
        not down-weighted; a code with no positive gets 1).
    Every weight is capped at CLASS_WEIGHT_MAX."""
    scheme = cfg["loss"].get("class_weights")
    if scheme is None:
        return None
    if scheme != CLASS_WEIGHT_SCHEME:
        raise ValueError(f"unknown class weight scheme {scheme!r}")
    train = [data.examples[k]["labels"] for k in data.keys("train")]
    out = {}
    for axis in L.AXES:
        codes = L.DIRECT_CODES[axis]
        given = [v for v in (t[axis] for t in train) if v is not None]
        if axis in L.SINGLE:
            counts = [sum(v == c for v in given) for c in codes]
            present = sum(1 for n in counts if n)
            out[axis] = [min(CLASS_WEIGHT_MAX, len(given) / (present * n)) if n else 1.0 for n in counts]
        else:
            positives = [sum(c in v for v in given) for c in codes]
            out[axis] = [min(CLASS_WEIGHT_MAX, max(1.0, (len(given) - p) / p)) if p else 1.0 for p in positives]
    return out


def prior_reference(data: RunData) -> dict:
    """Training-label priors as labels: the majority code, or each code positive in at least half
    of the labelled training rows (the most frequent if none), then the exclusive-negative rule."""
    train = [data.examples[k]["labels"] for k in data.keys("train")]
    scores = {}
    for axis in L.AXES:
        given = [v for v in (t[axis] for t in train) if v is not None]
        codes = L.DIRECT_CODES[axis]
        if axis in L.SINGLE:
            scores[axis] = [sum(v == c for v in given) / len(given) for c in codes]
        else:
            scores[axis] = [sum(c in v for v in given) / len(given) for c in codes]
    labels = L.decode_scores(scores, 0.5)
    return {k: labels for k in data.keys("validation")}


def scores_path(predictions: Path) -> Path:
    """predictions-<x>.jsonl -> scores-<x>.jsonl; any other name -> <stem>.scores<suffix>."""
    name = predictions.name
    if name.startswith("predictions-"):
        return predictions.with_name("scores-" + name[len("predictions-"):])
    return predictions.with_name(predictions.stem + ".scores" + predictions.suffix)


def _write_predictions(path: Path, labels: dict, probs: dict | None, run_info: dict) -> None:
    targets = [path] + ([scores_path(path)] if probs is not None else [])
    taken = [str(t) for t in targets if t.exists()]
    if taken:
        raise FileExistsError(f"refusing to overwrite {taken}")
    records = [L.prediction_record(k[0], k[1], labels[k], run_info) for k in sorted(labels)]
    L.write_predictions(path, records)
    if probs is not None:
        identity = {"record_kind": "model_scores", "schema_version": L.DIRECT_VERSION,
                    "vocabulary_sha256": L.vocabulary_sha256(), **{f: run_info[f] for f in L.RUN_FIELDS},
                    "codes": {a: list(L.DIRECT_CODES[a]) for a in L.AXES}}
        with scores_path(path).open("x", encoding="utf-8", newline="\n") as f:
            for k in sorted(probs):
                f.write(json.dumps({**identity, "conversation_id": k[0], "segment_order": k[1],
                                    "probabilities": probs[k]}, ensure_ascii=False) + "\n")


def _evaluation(out_dir: Path, final_dir: Path, name: str, data: RunData, keys: list, statement: str,
                title: str, predictions_path: Path, part: str) -> dict:
    """Evaluate the prediction file as written (re-read and re-checked), recording where it will live."""
    labels, info = EV.predictions_from_file(predictions_path)
    info["path"] = L.rel(final_dir / predictions_path.name)
    report = EV.evaluate({k: data.examples[k]["labels"] for k in keys}, labels, keys)
    report = {"title": title, "statement": statement, "reference": data.fingerprint["prepared"],
              "predictions": info,
              "selection": {"split": data.fingerprint["split"]["path"], "split_sha256": data.fingerprint["split"]["sha256"],
                            "part": part}, **report}
    write_json(out_dir / f"{name}.json", report)
    with (out_dir / f"{name}.md").open("x", encoding="utf-8", newline="\n") as f:
        f.write(EV.markdown(report))
    return report


def post_training(run_dir: Path, cfg: dict, data: RunData, built: dict, model, device: torch.device,
                  run_info: dict, synthetic: bool) -> dict | None:
    """Predict with the final state, then evaluate once. Written to a partial directory and renamed.
    A stop request during a prediction pass or between phases, up to the last check before the
    outputs are published, discards the partial directory and returns None; resume writes them again
    from the final checkpoint. Only writing the summary, the rename and the outputs_written log
    entry follow that last check, so a request arriving then lets the complete outputs stand."""
    out, partial = run_dir / "outputs", run_dir / ".outputs.partial"
    if out.exists():
        return json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    threshold = cfg["decode"]["multi_threshold"]
    said = statements(data.fingerprint["prepared"])
    summary = {"status": cfg["status"], "final_checkpoint": json.loads((run_dir / "checkpoints" / "final.json").read_text(encoding="utf-8"))["tag"],
               "reference_record_kinds": sorted({s["record_kind"] for s in data.fingerprint["prepared"]["sources"]})}
    phase = None
    try:
        for part, title in (("validation", "Development validation (final step)"),
                            ("train", "Training-fit diagnostic (final step)")):
            keys = data.keys(part)
            phase = f"{part} predictions"
            labels, probs = predict(model, keys, built, device, threshold)      # checks between interactions
            path = partial / f"predictions-{part}.jsonl"
            _write_predictions(path, labels, probs, run_info)
            phase = f"{part} evaluation"
            _check_stop()
            name = "evaluation-validation" if part == "validation" else "evaluation-train-fit"
            report = _evaluation(partial, out, name, data, keys, said[part], title, path, part)
            summary[name] = _headline(report)
        phase = "label-prior reading aid"
        _check_stop()
        prior = prior_reference(data)
        path = partial / "predictions-validation-prior-reference.jsonl"
        _write_predictions(path, prior, None, {**run_info, "model_route": "label-prior-reference (no model)"})
        report = _evaluation(partial, out, "evaluation-validation-prior-reference", data, data.keys("validation"),
                             said["prior"], "Label-prior reading aid on validation (no model)", path, "validation")
        summary["evaluation-validation-prior-reference"] = _headline(report)
        phase = "publishing the outputs"
        _check_stop()
    except Interrupted:
        shutil.rmtree(partial)
        log_event(run_dir, {"event": "stopped", "phase": f"outputs ({phase})", "signal": _Stop.signal})
        say("stopped before the outputs were complete; nothing was published (resume writes them)")
        return None
    write_json(partial / "summary.json", summary)
    os.rename(partial, out)
    log_event(run_dir, {"event": "outputs_written", "dir": "outputs"})
    return summary


def _headline(report: dict) -> dict:
    return {"statement": report["statement"], "rows": report["rows"], "all_axis_exact_match": report["all_axis_exact_match"],
            "axes": {a: {"n": r["n"], "excluded": r["excluded"], "row_accuracy": r["row_accuracy"],
                         "macro_f1": r["macro_f1"], "macro_f1_categories": len(r["macro_f1_categories"]),
                         "macro_f1_union": r["macro_f1_union"], "macro_f1_union_categories": len(r["macro_f1_union_categories"]),
                         "categories": len(r["per_category"]), "rare_categories": r["rare_categories"]}
                     for a, r in report["axes"].items()}}


# ----------------------------------------------------------------------------- run orchestration

def run_training(run_dir: Path, cfg: dict, data: RunData, model_identity: dict, backbone_factory, encode,
                 device: torch.device, *, resume: bool = False, stop_after_steps: int | None = None,
                 synthetic: bool = False, command=None) -> dict:
    check_config(cfg)
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy = MI.InputPolicy.from_config(cfg["input"])
    policy_costs = policy.check_with(encode)
    window = data.fingerprint["prepared"].get("context_interactions")
    if window is None or window < policy.context_interactions:
        raise ValueError(f"prepared context windows hold {window} earlier interactions; the policy asks for "
                         f"{policy.context_interactions}")
    if not data.keys("validation") or not data.keys("train"):
        raise ValueError("the split must give both training and validation interactions")
    if not synthetic and data.fingerprint["split"]["validation_purpose"] != cfg["validation"]["purpose"]:
        raise ValueError("the split records a different validation purpose from the config's")
    statements(data.fingerprint["prepared"])          # refuse an undescribed reference before training
    unlabelled = [a for a in L.AXES if not data.summary["train"]["label_support"][a]["labelled"]]
    if unlabelled:
        raise ValueError(f"no training interaction has a label on {unlabelled}")
    built, inputs_sha = build_inputs(data, encode, policy)
    weights = class_weights(cfg, data)          # recorded in the plan, so a resume must recompute the same
    env = environment(device)
    fp = fingerprint(cfg, data, model_identity, inputs_sha, env)
    fp_sha = canonical_sha256(fp)
    run_file = run_dir / "run.json"
    run_id = _dir_name(run_dir)         # recorded once in run.json; that record is the run's identity
    if not resume:
        if not run_id.strip():
            raise ValueError(f"the run directory's name becomes the run_id and must not be blank: {run_dir}")
        if run_file.exists():
            raise FileExistsError(f"{run_dir} already holds a run; use resume or a new run directory")
        run_dir.mkdir(parents=True, exist_ok=True)
        # Without run.json a run never finished starting; keep its leftovers for audit, then start again.
        leftovers = [run_dir / n for n in ("checkpoints", "inputs-report.json", "train_log.jsonl", "outputs",
                                           ".outputs.partial", ".run.json.partial", ".inputs-report.json.partial")
                     if (run_dir / n).exists()]
        if leftovers:
            aside = run_dir / f".aborted-start-{utc().replace(':', '')}"
            aside.mkdir()
            for path in leftovers:
                os.rename(path, aside / path.name)
        model, optimizer, scheduler, plan = build_training_objects(cfg, backbone_factory, device, len(data.keys("train")))
        if weights is not None:
            plan["class_weights"] = weights
        reports = {key_text(k): b["report"] for k, b in sorted(built.items())}
        write_json_atomic(run_dir / "inputs-report.json", {"policy": cfg["input"], "summary": {
            part: MI.summarise([built[k]["report"] for k in data.keys(part)]) for part in GS.PARTS},
            "per_interaction": reports})
        nonce = uuid.uuid4().hex                  # identifies this run's checkpoints; no effect on training
        progress = {"run_fingerprint_sha256": fp_sha, "run_nonce": nonce, "plan": plan, "epoch": 0, "group": 0,
                    "global_step": 0, "order": None, "order_epoch": None, "steps": [], "epochs": [], "complete": False}
        tag = save_checkpoint(run_dir, model, optimizer, scheduler, progress, device, cfg["checkpoint"]["keep_last"])
        log_event(run_dir, {"event": "start", "checkpoint": tag, "fingerprint_sha256": fp_sha, "plan": plan,
                            "aborted_start_moved": [p.name for p in leftovers]})
        write_json_atomic(run_file, {"record_kind": RUN_KIND, "run_id": run_id, "created_utc": utc(),
                                     "status": cfg["status"], "synthetic": synthetic, "command": command or sys.argv,
                                     "config": cfg, "fingerprint": fp, "fingerprint_sha256": fp_sha,
                                     "run_nonce": nonce,
                                     "platform": env["platform"], "data_summary": data.summary, "plan": plan,
                                     "input_policy_costs": policy_costs,
                                     "inputs_report_sha256": L.sha256_file(run_dir / "inputs-report.json")})
    else:
        record = json.loads(run_file.read_text(encoding="utf-8"))
        diffs = fingerprint_differences(record["fingerprint"], fp)
        if diffs:
            raise ValueError("refusing to resume: these inputs changed since the run started: " + ", ".join(diffs))
        if record["platform"] != env["platform"]:
            log_event(run_dir, {"event": "platform_changed", "was": record["platform"], "now": env["platform"]})
        run_id = _recorded_run_id(record)
        if run_id != _dir_name(run_dir):
            log_event(run_dir, {"event": "run_dir_renamed", "run_id": run_id, "directory": _dir_name(run_dir)})
            say(f"note: directory {_dir_name(run_dir)} holds run {run_id}; predictions keep the recorded run_id")
        model, optimizer, scheduler, plan = build_training_objects(cfg, backbone_factory, device, len(data.keys("train")))
        if weights is not None:
            plan["class_weights"] = weights
        ckpt = checkpoint_path(run_dir, "latest")
        # A kill between renaming a checkpoint into place and updating latest.json leaves a newer,
        # unreferenced directory; the resumed trajectory recomputes it, so it is removed first.
        for d in sorted((run_dir / "checkpoints").iterdir()):
            orphan = d.is_dir() and d.name.startswith("s") and d.name > ckpt.name
            if orphan or d.name.endswith(".partial"):
                shutil.rmtree(d) if d.is_dir() else d.unlink()
                log_event(run_dir, {"event": "removed_unreferenced_checkpoint", "dir": d.name})
        progress = restore_checkpoint(ckpt, model, optimizer, scheduler, device)
        if (progress["run_fingerprint_sha256"] != fp_sha or progress["plan"] != plan
                or progress.get("run_nonce") != record.get("run_nonce")):
            raise ValueError("checkpoint belongs to a different run state")
        log_event(run_dir, {"event": "resume", "checkpoint": ckpt.name, "global_step": progress["global_step"]})
        say(f"resumed from {ckpt.name} (optimizer step {progress['global_step']}, epoch {progress['epoch'] + 1}, "
            f"group {progress['group']})")
    try:
        outcome = "complete" if progress["complete"] else train_loop(run_dir, cfg, data, built, model, optimizer,
                                                                      scheduler, progress, device, stop_after_steps)
    except NumericalFailure as failure:
        return _record_failure(run_dir, failure)
    result = {"run_dir": str(run_dir), "outcome": outcome, "global_step": progress["global_step"]}
    if outcome == "complete":
        final = run_dir / "checkpoints" / "final.json"
        try:
            tag = json.loads(final.read_text(encoding="utf-8"))["tag"] if final.exists() else None
        except (json.JSONDecodeError, KeyError, TypeError):
            tag = None
            log_event(run_dir, {"event": "rewrote_unreadable_final_pointer"})
        if tag is None:
            write_json_atomic(final, json.loads((run_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8")))
            tag = json.loads(final.read_text(encoding="utf-8"))["tag"]
        outputs = post_training(run_dir, cfg, data, built, model, device,
                                _run_info(run_id, fp, data, model_identity, tag), synthetic)
        if outputs is None:
            result["outcome"] = "interrupted"
        else:
            result["outputs"] = outputs
    return result


def _recorded_run_id(record: dict) -> str:
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"run.json records no usable run_id ({run_id!r}); its predictions could not be identified")
    return run_id


def _record_failure(run_dir: Path, failure: NumericalFailure) -> dict:
    """Describe the failure beside the checkpoints, which stay as they were. Record numbers only
    grow, past every number the log or failures/ has used, so no record is ever given the name of
    an earlier (even a deleted) one; if no record can be written the failure is still logged."""
    latest = json.loads((run_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
    detail = {**failure.detail, "utc": utc(), "checkpoint_written": False, "latest_checkpoint_kept": latest["tag"]}
    used, name = [], None
    try:                                    # never strict: a damaged log must not lose the failure
        text = (run_dir / "train_log.jsonl").read_bytes().decode("utf-8", errors="replace")
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        m = re.match(r"^failures/(\d+)-", str(event.get("details"))) if isinstance(event, dict) else None
        if m and event.get("event") == "failed":
            used.append(int(m[1]))
    try:
        where = run_dir / "failures"
        where.mkdir(exist_ok=True)
        try:
            names = [p.name for p in where.iterdir()]
        except OSError:                     # writable but not listable: the log's numbers still apply
            names = []
        used += [int(m[1]) for m in (re.match(r"^(\d+)-step\d+\.json$", x, re.IGNORECASE) for x in names) if m]
        n = max(used, default=0) + 1
        while name is None:
            candidate = f"failures/{n:03d}-step{failure.detail['optimizer_step']:06d}.json"
            try:
                write_json(run_dir / candidate, detail)
                name = candidate
            except FileExistsError:
                n += 1
    except OSError as exc:
        say(f"could not write the failure record ({exc}); the failure is still logged", sys.stderr)
    log_event(run_dir, {"event": "failed", "kind": failure.detail["kind"], "optimizer_step": failure.detail["optimizer_step"],
                        "latest_checkpoint_kept": latest["tag"], "details": name})
    say(f"failed: {failure}; no checkpoint was written; latest checkpoint stays {latest['tag']} ({name})", sys.stderr)
    return {"run_dir": str(run_dir), "outcome": "failed", "global_step": latest["global_step"],
            "latest_checkpoint": latest["tag"], "failure": name}


def predict_from_checkpoint(run_dir: Path, data: RunData, model_identity: dict, backbone_factory, encode,
                            device: torch.device, part: str, out: Path, which: str = "final") -> dict:
    """A fresh process reloads a saved state and predicts; it must find the run's inputs unchanged."""
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    cfg, run_id = record["config"], _recorded_run_id(record)
    built, inputs_sha = build_inputs(data, encode, MI.InputPolicy.from_config(cfg["input"]))
    fp = fingerprint(cfg, data, model_identity, inputs_sha, environment(device))
    diffs = fingerprint_differences(record["fingerprint"], fp)
    if diffs:
        raise ValueError("refusing to predict: these inputs changed since the run started: " + ", ".join(diffs))
    ckpt = checkpoint_path(run_dir, which)
    complete = verify_checkpoint(ckpt)
    if (complete["run_fingerprint_sha256"], complete.get("run_nonce")) != (record["fingerprint_sha256"],
                                                                          record.get("run_nonce")):
        raise ValueError(f"checkpoint {ckpt.name} does not belong to run {run_id} in {run_dir}")
    model, _, _, _ = build_training_objects(cfg, backbone_factory, device, len(data.keys("train")))
    progress = restore_checkpoint(ckpt, model, None, None, device)
    if (progress["run_fingerprint_sha256"], progress.get("run_nonce")) != (record["fingerprint_sha256"],
                                                                          record.get("run_nonce")):
        raise ValueError(f"checkpoint {ckpt.name} does not belong to run {run_id} in {run_dir}")
    keys = data.keys(part)
    try:
        labels, probs = predict(model, keys, built, device, cfg["decode"]["multi_threshold"])
    except Interrupted:
        say(f"stopped before every {part} interaction was predicted; nothing was written")
        return {"outcome": "interrupted", "checkpoint": ckpt.name, "part": part, "written": False}
    _write_predictions(out, labels, probs, _run_info(run_id, fp, data, model_identity, ckpt.name))
    result = {"outcome": "complete", "run_id": run_id, "checkpoint": ckpt.name, "part": part,
              "predictions": str(out), "scores": str(scores_path(out)), "records": len(keys)}
    if run_id != _dir_name(run_dir):
        result["run_dir_name"] = _dir_name(run_dir)    # the predictions carry the recorded run_id
    return result


def _run_info(run_id: str, fp: dict, data: RunData, model_identity: dict, tag: str) -> dict:
    """Prediction identity: the run recorded in run.json and the saved state that produced them."""
    return {"run_id": run_id, "config_sha256": fp["config_sha256"],
            "dataset_manifest_sha256": data.fingerprint["prepared"]["manifest_sha256"],
            "model_route": f"multihead-lora/{model_identity['id']}@{model_identity['revision'][:12]}#{tag}"}


def _real_inputs(cfg: dict, run_dir: Path, synthetic: bool, create: bool):
    if synthetic:
        if create:
            run_dir.mkdir(parents=True, exist_ok=True)
            n = cfg.get("synthetic", {})
            write_synthetic_inputs(run_dir, synthetic_examples(n.get("seed", 7)), cfg["seed"],
                                   n.get("validation_target", 8), n.get("validation_max", 10))
        data = load_data(run_dir / "synthetic-prepared", run_dir / "synthetic-split.json")
    else:
        data = load_data(ROOT / cfg["data"]["prepared_dir"], ROOT / cfg["data"]["split_file"])
    snapshot, identity = model_snapshot(cfg["model"])
    dtype = getattr(torch, cfg["model"]["dtype"])

    def factory():
        return M.load_backbone(snapshot, dtype, cfg["model"]["attn_implementation"], cfg["model"]["gradient_checkpointing"])
    return data, identity, factory, tokenizer_encoder(snapshot)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="start a new run")
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--run-dir", type=Path, required=True)
    t.add_argument("--synthetic", action="store_true", help="generated inputs instead of the private data")
    t.add_argument("--stop-after-steps", type=int, help="check: stop with a checkpoint after this optimizer step")
    r = sub.add_parser("resume", help="continue a run from its latest complete checkpoint")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--stop-after-steps", type=int)
    q = sub.add_parser("predict", help="reload a saved state in a fresh process and predict")
    q.add_argument("--run-dir", type=Path, required=True)
    q.add_argument("--part", choices=GS.PARTS, default="validation")
    q.add_argument("--checkpoint", default="final", help="final, latest or a checkpoint directory name")
    q.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    if a.command == "train":
        cfg = load_config(a.config)
        private = (ROOT / "data").resolve()
        if not a.synthetic and (a.run_dir.resolve() == private or not a.run_dir.resolve().is_relative_to(private)):
            p.error("a run on private data must be written below repository data/")
        if not _dir_name(a.run_dir).strip():
            p.error(f"the run directory's name becomes the run_id and must not be blank: {a.run_dir}")
        if (a.run_dir / "run.json").exists():
            p.error(f"{a.run_dir} already holds a run; use resume")
        device = device_from(cfg["device"])
        data, identity, factory, encode = _real_inputs(cfg, a.run_dir, a.synthetic, create=True)
        result = run_training(a.run_dir, cfg, data, identity, factory, encode, device,
                              stop_after_steps=a.stop_after_steps, synthetic=a.synthetic)
    else:
        if not (a.run_dir / "run.json").exists():
            p.error(f"{a.run_dir} has no run.json: the run never finished starting (use train to start it again)")
        record = json.loads((a.run_dir / "run.json").read_text(encoding="utf-8"))
        cfg = record["config"]
        if canonical_sha256(cfg) != record["fingerprint"]["config_sha256"]:
            p.error("run.json config does not match its recorded hash")
        if a.command == "predict" and not record["synthetic"]:
            private = (ROOT / "data").resolve()
            if not a.out.resolve().is_relative_to(private):
                p.error("predictions from a run on private data must be written below repository data/")
        device = device_from(cfg["device"])
        data, identity, factory, encode = _real_inputs(cfg, a.run_dir, record["synthetic"], create=False)
        if a.command == "resume":
            result = run_training(a.run_dir, cfg, data, identity, factory, encode, device, resume=True,
                                  stop_after_steps=a.stop_after_steps, synthetic=record["synthetic"])
        else:
            result = predict_from_checkpoint(a.run_dir, data, identity, factory, encode, device, a.part, a.out,
                                             a.checkpoint)
    say(json.dumps(result, ensure_ascii=False, indent=1))
    return {"interrupted": EXIT_INTERRUPTED, "failed": EXIT_FAILED}.get(result.get("outcome"), 0)


if __name__ == "__main__":
    raise SystemExit(main())
