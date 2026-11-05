#!/usr/bin/env python3
"""
finetuning/labels.py - the four-axis label path between human records and a model (D-85).

    python finetuning/labels.py --dev-review <labels-67-clean.xlsx> [--direct <records.jsonl> ...]
                                [--corpus <csv>] [--manifest <out.json>]

WHAT IT DOES

* Reads four-axis records and checks every value against `codebook_schema.json`
  `direct_instrument`. Nothing is inferred, mapped or filled in.
* Keeps the provenance of every label: record kind, coder, stage, source file and its SHA-256.
  A separate reviewed-JSONL reader preserves D-105's full history and non-independent status;
  finetuning/prepare_data.py explicitly selects it when preparing private development data.
* Joins records to the corpus on (Conversation ID, Segment Order) and attaches context with
  finetuning/data.py, which stops at a Segment Order gap.
* Encodes targets in forms both candidate routes can use (methodology 4.2 has not chosen one):
  a class index or multi-hot vector per axis with a mask (multi-head classifier), or one
  canonical JSON string (structured generation). Parses and decodes model output back into
  schema-valid label sets.
* Writes model predictions as their own record kind, which the human-record checker in
  coding/direct_coding.py rejects, so a prediction can never pass as a human label.

WHAT IT DELIBERATELY DOES NOT DO (D-79, D-85)

No split, cross-validation, model, loss, training loop, checkpointing, metric or score. The
decoding threshold and the input rendering below are provisional defaults for plumbing checks
only; the experiment plan sets the real ones.

THE RULE THAT MATTERS MOST

A null label is a missing or unavailable judgement (an unavailable reason, a truncated capture,
a blank export cell). It is masked. It is never encoded as the axis's negative code ("ND",
"none of the four applies"), and a masked multi-hot row must be ignored by the loss rather than
read as "no code applies".
"""
from __future__ import annotations

import argparse
import codecs
import copy
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "coding"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from codebook_schema import DIRECT_CODES, DIRECT_FIELDS, DIRECT_INSTRUMENT, DIRECT_VERSION  # noqa: E402
import direct_coding  # noqa: E402

# finetuning/data.py is loaded by path: a plain `import data` could resolve to the repository's
# data/ directory as a namespace package.
_spec = importlib.util.spec_from_file_location("finetuning_data", Path(__file__).resolve().parent / "data.py")
corpus_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(corpus_data)

DEFAULT_CORPUS = ROOT / "data" / "processed" / "transcripts_analysis.csv"   # current analysis corpus
DEFAULT_CONTEXT = corpus_data.DEFAULT_CONTEXT

AXES = list(DIRECT_CODES)                                   # schema order
SINGLE = {a for a in AXES if DIRECT_FIELDS[a]["selection"] == "single"}
NEGATIVES = dict(DIRECT_INSTRUMENT["negative_exclusive"])

HUMAN_KIND = {"calibration": "human_calibration",
              "independent": "human_independent",
              "adjudicated": "human_adjudicated"}
DEV_REVIEW_KIND = "AI_assisted_development_review"
# The AI-assisted review file was introduced by D-105; D-144 is the v9 lineage (after the D-110 to
# D-144 review). D-166 and D-167 are the reviewed versions of the 135 researcher-coded interactions,
# D-168 and D-169 the researcher's reviewed 150 (v10, v11) for the development run on about 285.
# D-182 is the reviewed 150 v12; D-185 the reviewed version of the 200 researcher-coded interactions;
# D-187 the reviewed 150 v13 and the 135 reviewed v3; D-188 the reviewed 150 v14;
# D-189 the 135 reviewed v4; D-190 the reviewed 150 v15.
# Only these are read; every other review version is refused.
ACCEPTED_REVIEW_DECISIONS = ("D-105", "D-144", "D-166", "D-167", "D-168", "D-169", "D-182", "D-185", "D-187", "D-188", "D-189", "D-190")
# Shape of review_history entries appended after the original-source entry (D-111 onwards).
LATER_HISTORY_FIELDS = ("source", "labels", "notes", "flags", "review_provenance")
PREDICTION_KIND = "model_prediction"
LABEL_KINDS = set(HUMAN_KIND.values()) | {DEV_REVIEW_KIND}

DEV_REVIEW_HEADER = ["Interaction", "Student interaction type", "topic", "AI delegation", "AI response type"]
DEV_REVIEW_BLANK = "blank in the development-review export"
TRUNCATED = "truncated capture"

PROVISIONAL_THRESHOLD = 0.5      # plumbing default, not a tuned value
RUN_FIELDS = ("run_id", "model_route", "config_sha256", "dataset_manifest_sha256")
PREDICTION_FIELDS = {"record_kind", "schema_version", "vocabulary_sha256", *RUN_FIELDS,
                     "conversation_id", "segment_order", "labels", "not_predicted"}
# A human record never carries these. Their presence means the record came from somewhere else,
# whatever coder or stage fields it also has.
NOT_HUMAN_FIELDS = ("record_kind", "vocabulary_sha256", "not_predicted", *RUN_FIELDS,
                    "review_history", "review_provenance", "stage_semantics",
                    "independent_coding", "joint_adjudication")


# ----------------------------------------------------------------------------- hashing and vocabulary

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def jsonl_lines(path) -> list[tuple[int, str]]:
    """(physical line number, text) for each nonblank line of a JSONL file (UTF-8, an optional BOM).
    Lines end at LF, CRLF or a lone CR, as in text-mode reading, and nowhere else: str.splitlines()
    also ends lines at U+0085, U+2028, U+2029 and a few control characters, the first three of
    which JSON may carry unescaped inside a string, so it would cut such a record."""
    raw = Path(path).read_bytes()
    skip = len(codecs.BOM_UTF8) if raw.startswith(codecs.BOM_UTF8) else 0
    try:
        text = raw[skip:].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{Path(path).name}: not UTF-8 text (byte {exc.start + skip} of the file)") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")      # what read_text's universal newlines do
    return [(n, line) for n, line in enumerate(text.split("\n"), 1) if line.strip()]


def json_record(line: str, where: str):
    """json.loads that names the file and physical line on an error; the decoder's own position
    would count within this one line only."""
    try:
        return json.loads(line)
    except ValueError as exc:               # a syntax error, or e.g. an integer too long to convert
        detail = f"{exc.msg} at column {exc.colno}" if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise ValueError(f"{where}: not valid JSON ({detail})") from exc


def rel(path) -> str:
    """Repository-relative path when possible, so manifests carry no machine-specific prefix."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def vocabulary() -> dict:
    """The label vocabulary a model is built against, in schema order."""
    return {"schema_version": DIRECT_VERSION,
            "axes": [{"name": a, "selection": DIRECT_FIELDS[a]["selection"], "codes": list(DIRECT_CODES[a]),
                      "negative_exclusive": NEGATIVES.get(a)} for a in AXES]}


def vocabulary_sha256() -> str:
    """Changes whenever an axis, a code, the code order or a selection type changes."""
    return sha256_bytes(json.dumps(vocabulary(), ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


# ----------------------------------------------------------------------------- label checks

def canonical(axis: str, value):
    """A single-select value as given; a multi-select value as a list in schema order."""
    if value is None or axis in SINGLE:
        return value
    return sorted(value, key=DIRECT_CODES[axis].index)


def check_labels(labels, allow_null: bool = True) -> list[str]:
    """Errors in a four-axis label set. A null is allowed only when `allow_null`."""
    if not isinstance(labels, dict):
        return ["labels must be an object"]
    errors = []
    if set(labels) != set(AXES):
        errors.append("exactly the four direct axes are required: " + ", ".join(AXES))
    for axis in AXES:
        if axis not in labels:
            continue
        v, codes = labels[axis], DIRECT_CODES[axis]
        if v is None:
            if not allow_null:
                errors.append(f"{axis}: no value")
            continue
        if axis in SINGLE:
            if not isinstance(v, str) or v not in codes:
                errors.append(f"{axis}: {v!r} is not one of its codes")
            continue
        if not isinstance(v, list) or not v:
            errors.append(f"{axis}: must be a non-empty list")
            continue
        bad = [x for x in v if not isinstance(x, str) or x not in codes]
        if bad:
            errors.append(f"{axis}: not codes: {bad!r}")
        elif len(v) != len(set(v)):
            errors.append(f"{axis}: duplicate selection")
        elif NEGATIVES.get(axis) in v and len(v) > 1:
            errors.append(f"{axis}: {NEGATIVES[axis]!r} cannot be combined with another code")
    return errors


# ----------------------------------------------------------------------------- records

@dataclass
class LabelledInteraction:
    conversation_id: str
    segment_order: int
    labels: dict
    record_kind: str
    source_path: str
    source_sha256: str
    coder_id: str | None = None
    stage: str | None = None
    unavailable: dict = field(default_factory=dict)
    flags: tuple = ()
    # Preserve the supplied record, including notes and review history, without promoting
    # its metadata into a claim that the production process has been independently verified.
    source_record: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.conversation_id, self.segment_order)

    def masked_axes(self) -> list[str]:
        return [a for a in AXES if self.labels[a] is None]

    def mask_reason(self, axis: str) -> str | None:
        if self.labels[axis] is not None:
            return None
        if axis in self.unavailable:
            return self.unavailable[axis]
        if TRUNCATED in self.flags and axis.startswith("AI "):
            return TRUNCATED
        return "null"


def read_direct(path) -> list[LabelledInteraction]:
    """Human records in the direct-20260924 format, checked by coding/direct_coding.py first.

    That checker refuses a blank judgement unless the record gives an unavailable reason or the
    capture is flagged truncated, so every null that reaches here has a stated reason.

    Before that, each line is refused if it carries a record kind or any model-run field: a
    prediction that also has coder and stage fields must not be relabelled as human coding."""
    path = Path(path)
    for n, line in jsonl_lines(path):
        raw = json_record(line, f"{path.name} line {n}")
        present = [f for f in NOT_HUMAN_FIELDS if isinstance(raw, dict) and f in raw]
        if present:
            raise ValueError(f"{path.name} line {n}: not a human record; it carries {present}")
    digest = sha256_file(path)
    out = []
    for r in direct_coding.read_records(path):
        out.append(LabelledInteraction(
            r["conversation_id"], r["segment_order"], {a: canonical(a, r["labels"][a]) for a in AXES},
            HUMAN_KIND[r["stage"]], rel(path), digest, coder_id=r["coder_id"], stage=r["stage"],
            unavailable=dict(r.get("unavailable") or {}), flags=tuple(r.get("flags") or ()),
            source_record=copy.deepcopy(r)))
    return out


def read_reviewed_direct_jsonl(path) -> list[LabelledInteraction]:
    """Read an explicitly selected AI-assisted review (D-105 or v9 under D-144), never as human data.

    The retained ``stage`` is the source allocation phase. The review must state that it is
    neither independent coding nor joint adjudication. Full supplied history/provenance stays
    in ``source_record``; the source file hash identifies the exact reviewed version. This
    validates the declaration and format, not the truth of its claimed production history.
    """
    path = Path(path)
    digest = sha256_file(path)
    out, seen = [], set()
    for n, line in jsonl_lines(path):
        r = json_record(line, f"{path.name} line {n}")
        if not isinstance(r, dict):
            raise ValueError(f"{path.name} line {n}: reviewed record must be an object")
        errors = direct_coding.validate(r)
        errors.extend(check_labels(r.get("labels")))
        if r.get("record_kind") != DEV_REVIEW_KIND:
            errors.append(f"record_kind must be {DEV_REVIEW_KIND}")
        if any(f in r for f in ("vocabulary_sha256", "not_predicted", *RUN_FIELDS)):
            errors.append("model-run fields cannot appear in a reviewed label record")
        for flag in ("independent_coding", "joint_adjudication"):
            if flag in r and r[flag] is not False:
                errors.append(f"{flag} must be false when present")
        provenance = r.get("review_provenance")
        if not isinstance(provenance, dict):
            errors.append("review_provenance must be an object")
        else:
            if provenance.get("decision") not in ACCEPTED_REVIEW_DECISIONS:
                errors.append("review_provenance must identify an accepted review decision: " + ", ".join(ACCEPTED_REVIEW_DECISIONS))
            for flag in ("independent_coding", "joint_adjudication"):
                if provenance.get(flag) is not False:
                    errors.append(f"review_provenance.{flag} must be false")
            original_hash = provenance.get("original_source_sha256")
            if not isinstance(original_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", original_hash):
                errors.append("original_source_sha256 must be a SHA-256 hex digest")
            for field_name in ("base_path", "revision", "instrument_release"):
                value = provenance.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"review_provenance.{field_name} is required")
        history = r.get("review_history")
        if not isinstance(history, list) or not history:
            errors.append("review_history must be a nonempty list")
        else:
            for index, entry in enumerate(history):
                if not isinstance(entry, dict):
                    errors.append(f"review_history[{index}] must be an object")
                    continue
                if index > 0 and "original_labels" not in entry:
                    # A later review batch: its own labels, notes, flags, source and provenance.
                    for error in check_labels(entry.get("labels")):
                        errors.append(f"review_history[{index}].labels: " + error)
                    for field_name in LATER_HISTORY_FIELDS:
                        if field_name not in entry:
                            errors.append(f"review_history[{index}].{field_name} is required")
                    if not isinstance(entry.get("source"), str) or not str(entry.get("source", "")).strip():
                        errors.append(f"review_history[{index}].source must be a nonempty string")
                    if not isinstance(entry.get("flags"), list):
                        errors.append(f"review_history[{index}].flags must be a list")
                    if not isinstance(entry.get("review_provenance"), dict):
                        errors.append(f"review_history[{index}].review_provenance must be an object")
                    continue
                for error in check_labels(entry.get("original_labels")):
                    errors.append(f"review_history[{index}].original_labels: " + error)
                for field_name in ("source_coder_id", "source_stage", "original_notes", "original_flags"):
                    if field_name not in entry:
                        errors.append(f"review_history[{index}].{field_name} is required")
        if not isinstance(r.get("stage_semantics"), str) or not r["stage_semantics"].strip():
            errors.append("stage_semantics must explain the retained allocation stage")
        cid, seg = r.get("conversation_id"), r.get("segment_order")
        if not isinstance(cid, str) or not cid.strip():
            errors.append("conversation_id must be a nonempty string")
        if errors:
            raise ValueError(f"{path.name} line {n}: " + "; ".join(errors))
        key = (cid, seg)
        if key in seen:
            raise ValueError(f"{path.name} line {n}: duplicate reviewed interaction {cid}#{seg}")
        seen.add(key)
        out.append(LabelledInteraction(
            cid, seg, {a: canonical(a, r["labels"][a]) for a in AXES}, DEV_REVIEW_KIND,
            rel(path), digest, coder_id=r["coder_id"], stage=r["stage"],
            unavailable=dict(r.get("unavailable") or {}), flags=tuple(r.get("flags") or ()),
            source_record=copy.deepcopy(r)))
    return out


_KEY = re.compile(r"^(?P<cid>[^#\s]+)#(?P<n>[1-9]\d*)$")


def parse_key(text) -> tuple[str, int]:
    m = _KEY.match(str(text).strip()) if text is not None else None
    if not m:
        raise ValueError(f"interaction key {text!r} is not of the form <conversation>#<segment>")
    return m["cid"], int(m["n"])


def read_dev_review_xlsx(path) -> list[LabelledInteraction]:
    """The labels-only development-review export (D-84): one sheet, five columns, `;` between
    multi-select codes. A blank cell is a masked label, not a negative one."""
    import openpyxl
    path = Path(path)
    digest = sha256_file(path)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if len(wb.worksheets) != 1:
            raise ValueError(f"{path.name}: expected exactly one worksheet")
        rows = list(wb.worksheets[0].iter_rows(values_only=True))
    finally:
        wb.close()
    if not rows:
        raise ValueError(f"{path.name}: empty worksheet")
    header = ["" if c is None else str(c).strip() for c in rows[0]]
    if header[:5] != DEV_REVIEW_HEADER or any(header[5:]):
        raise ValueError(f"{path.name}: unexpected header {header!r}")
    axis_of = {h: next(a for a in AXES if a.lower() == h.lower()) for h in DEV_REVIEW_HEADER[1:]}
    out, seen = [], set()
    for n, row in enumerate(rows[1:], start=2):
        row = list(row) + [None] * 5
        if all(c is None or str(c).strip() == "" for c in row[:5]):
            continue
        cid, seg = parse_key(row[0])
        labels, unavailable = {}, {}
        for h, cell in zip(DEV_REVIEW_HEADER[1:], row[1:5]):
            axis = axis_of[h]
            text = "" if cell is None else str(cell).strip()
            if not text:
                labels[axis], unavailable[axis] = None, DEV_REVIEW_BLANK
            elif axis in SINGLE:
                labels[axis] = text
            else:
                labels[axis] = [p.strip() for p in text.split(";") if p.strip()]
        errors = check_labels(labels)
        if errors:
            raise ValueError(f"{path.name} row {n} ({cid}#{seg}): " + "; ".join(errors))
        if (cid, seg) in seen:
            raise ValueError(f"{path.name} row {n}: duplicate interaction {cid}#{seg}")
        seen.add((cid, seg))
        out.append(LabelledInteraction(cid, seg, {a: canonical(a, labels[a]) for a in AXES},
                                       DEV_REVIEW_KIND, rel(path), digest, unavailable=unavailable))
    return out


def assemble(items, allowed_kinds) -> list[LabelledInteraction]:
    """One label record per interaction, from explicitly permitted record kinds only.

    It refuses rather than chooses: two records for one interaction (two coders, or a review and
    an adjudication) must be resolved by the caller, and a model prediction is never a label."""
    allowed = set(allowed_kinds)
    unknown = allowed - LABEL_KINDS
    if unknown:
        raise ValueError(f"not label record kinds: {sorted(unknown)}")
    present = {i.record_kind for i in items}
    if not present <= allowed:
        raise ValueError(f"record kinds not permitted here: {sorted(present - allowed)}")
    seen: dict = {}
    for i in items:
        if i.key in seen:
            raise ValueError(f"{i.conversation_id}#{i.segment_order} has more than one label record "
                             f"({seen[i.key]}; {i.record_kind} from {i.source_path}); choose one explicitly")
        seen[i.key] = f"{i.record_kind} from {i.source_path}"
    return sorted(items, key=lambda i: i.key)


# ----------------------------------------------------------------------------- corpus join

def load_corpus(path=None):
    return corpus_data.load_corpus(Path(path) if path else DEFAULT_CORPUS)


def attach(items, corpus, n_context: int = DEFAULT_CONTEXT) -> tuple[list[dict], list[tuple]]:
    """Join labelled interactions to corpus text and context. Returns (examples, missing keys).

    Only the conversations that carry a label are read; context never crosses a conversation,
    so this gives the same windows as reading the whole corpus."""
    wanted = {i.key: i for i in items}
    sub = corpus[corpus["Conversation ID"].isin({k[0] for k in wanted})]
    found: dict = {}
    for ex in corpus_data.iter_examples(sub, n_context):
        k = (ex["conversation_id"], ex["segment_order"])
        if k in wanted:
            if k in found:
                raise ValueError(f"corpus has more than one row for {k[0]}#{k[1]}")
            found[k] = ex
    examples = []
    for k in sorted(found):
        item, ex = wanted[k], found[k]
        examples.append({**ex, "labels": item.labels, "record_kind": item.record_kind,
                         "source_sha256": item.source_sha256, "masked_axes": item.masked_axes(),
                         "student_text_missing": not ex["student_prompt"].strip()})
    return examples, sorted(set(wanted) - set(found))


def render_input(example: dict) -> str:
    """PROVISIONAL plain-text rendering of one interaction and its context, for plumbing only.

    The current AI response is included because two axes code it. Format, truncation and token
    budget are for the experiment plan; nothing here is tuned."""
    lines = []
    if example.get("context_short_because") in ("record starts late", "gap"):
        lines.append("[earlier turns not available]")
    for c in example["context"]:
        lines += [f"[earlier interaction {c['segment_order']}]",
                  "Student: " + c["student_prompt"], "Assistant: " + c["llm_response"]]
    lines += ["[interaction to code]", "Student: " + example["student_prompt"],
              "Assistant: " + example["llm_response"]]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- targets

def encode_targets(labels: dict) -> dict:
    """Per-axis targets for a multi-head classifier. mask 0 means: ignore this axis in the loss."""
    errors = check_labels(labels)
    if errors:
        raise ValueError("; ".join(errors))
    out = {}
    for axis in AXES:
        v, codes = labels[axis], DIRECT_CODES[axis]
        if axis in SINGLE:
            out[axis] = {"mask": 0, "index": None} if v is None else {"mask": 1, "index": codes.index(v)}
        else:
            out[axis] = ({"mask": 0, "multi_hot": [0] * len(codes)} if v is None
                         else {"mask": 1, "multi_hot": [int(c in v) for c in codes]})
    return out


def decode_targets(targets: dict) -> dict:
    """Inverse of encode_targets."""
    labels = {}
    for axis in AXES:
        t, codes = targets[axis], DIRECT_CODES[axis]
        if not t["mask"]:
            labels[axis] = None
        elif axis in SINGLE:
            labels[axis] = codes[t["index"]]
        else:
            labels[axis] = [c for c, h in zip(codes, t["multi_hot"]) if h]
    errors = check_labels(labels)
    if errors:
        raise ValueError("; ".join(errors))
    return labels


def target_text(labels: dict) -> str:
    """Canonical JSON target for a structured-generation route. Nulls are written as null; the
    caller decides whether an example with a null axis is used."""
    errors = check_labels(labels)
    if errors:
        raise ValueError("; ".join(errors))
    return json.dumps({a: canonical(a, labels[a]) for a in AXES}, ensure_ascii=False)


def parse_generated(text, allow_null: bool = False) -> tuple[dict | None, list[str]]:
    """Parse generated output strictly. Returns (labels, []) or (None, errors); never repairs."""
    try:
        obj = json.loads(str(text).strip())
    except json.JSONDecodeError as e:
        return None, [f"not JSON: {e.msg}"]
    if not isinstance(obj, dict):
        return None, ["not a JSON object"]
    errors = check_labels(obj, allow_null=allow_null)
    if errors:
        return None, errors
    return {a: canonical(a, obj[a]) for a in AXES}, []


def decode_scores(scores: dict, threshold: float = PROVISIONAL_THRESHOLD) -> dict:
    """Turn per-axis scores from a multi-head model into a schema-valid label set.

    Single-select axes take the highest score (ties go to the earlier code). Multi-select axes
    take every code at or above `threshold`, or the single highest if none reaches it. If the
    exclusive negative code is selected with others, the side with the higher top score is kept:
    the negative alone, or the others without it. The threshold is a plumbing default.

    Scores must be finite, and multi-select scores must be probabilities in [0, 1]; a NaN, an
    infinity or a logit passed by mistake raises rather than turning into a normal label set."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
            or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError(f"threshold must be a number in (0, 1], got {threshold!r}")
    if not isinstance(scores, dict) or set(scores) != set(AXES):
        raise ValueError("scores are needed for exactly the four direct axes")
    labels = {}
    for axis in AXES:
        codes = DIRECT_CODES[axis]
        try:
            s = [float(x) for x in scores[axis]]
        except (TypeError, ValueError):
            raise ValueError(f"{axis}: scores must be numbers") from None
        if len(s) != len(codes):
            raise ValueError(f"{axis}: expected {len(codes)} scores, got {len(s)}")
        if not all(math.isfinite(x) for x in s):
            raise ValueError(f"{axis}: scores must be finite (NaN or infinity found)")
        if axis not in SINGLE and not all(0.0 <= x <= 1.0 for x in s):
            raise ValueError(f"{axis}: multi-select scores must be probabilities in [0, 1]")
        top = max(range(len(s)), key=lambda i: s[i])
        if axis in SINGLE:
            labels[axis] = codes[top]
            continue
        chosen = [c for c, x in zip(codes, s) if x >= threshold] or [codes[top]]
        neg = NEGATIVES.get(axis)
        if neg in chosen and len(chosen) > 1:
            neg_score = s[codes.index(neg)]
            best_other = max(s[codes.index(c)] for c in chosen if c != neg)
            chosen = [neg] if neg_score > best_other else [c for c in chosen if c != neg]
        labels[axis] = chosen
    errors = check_labels(labels, allow_null=False)
    if errors:                                   # unreachable unless the rule above is broken
        raise AssertionError("; ".join(errors))
    return labels


# ----------------------------------------------------------------------------- predictions

def prediction_errors(r) -> list[str]:
    """Every check a prediction record must pass, used both when it is made and when it is read."""
    if not isinstance(r, dict):
        return ["record must be an object"]
    errors = []
    if r.get("record_kind") != PREDICTION_KIND:
        errors.append("not a model_prediction record")
    if r.get("schema_version") != DIRECT_VERSION:
        errors.append("schema version differs from the current instrument")
    if r.get("vocabulary_sha256") != vocabulary_sha256():
        errors.append("label vocabulary differs from the current schema")
    errors += [f"run information missing: {f}" for f in RUN_FIELDS
               if not isinstance(r.get(f), str) or not r[f].strip()]
    cid, seg = r.get("conversation_id"), r.get("segment_order")
    if not isinstance(cid, str) or not cid.strip() or type(seg) is not int or seg < 1:
        errors.append("invalid interaction key: conversation_id must be text and segment_order an integer >= 1")
    labels = r.get("labels")
    errors += check_labels(labels)
    np_ = r.get("not_predicted")
    if not isinstance(np_, dict):
        errors.append("not_predicted must be an object")
    else:
        if any(not isinstance(v, str) or not v.strip() for v in np_.values()):
            errors.append("not_predicted reasons must be non-empty text")
        # also refuses an unknown axis in the reasons, since only axis names can be null axes
        if isinstance(labels, dict) and {a for a in AXES if a in labels and labels[a] is None} != set(np_):
            errors.append("each null axis needs a not_predicted reason, and only null axes may have one")
    extra = sorted(set(r) - PREDICTION_FIELDS)
    if extra:
        errors.append(f"unexpected fields: {extra}")
    return errors


def prediction_record(conversation_id: str, segment_order: int, labels: dict, run: dict,
                      not_predicted: dict | None = None) -> dict:
    """A model prediction as its own record kind. It has no coder and no stage, so the human
    checker in coding/direct_coding.py rejects it, and read_direct refuses it outright."""
    if not isinstance(run, dict):
        raise ValueError("run information must be an object")
    rec = {"record_kind": PREDICTION_KIND, "schema_version": DIRECT_VERSION,
           "vocabulary_sha256": vocabulary_sha256(), **{f: run.get(f) for f in RUN_FIELDS},
           "conversation_id": conversation_id, "segment_order": segment_order, "labels": labels,
           "not_predicted": {} if not_predicted is None else not_predicted}
    errors = prediction_errors(rec)
    if errors:
        raise ValueError("; ".join(errors))
    rec["labels"] = {a: canonical(a, labels[a]) for a in AXES}
    rec["not_predicted"] = dict(rec["not_predicted"])
    return rec


def write_predictions(path, records) -> Path:
    """Write prediction records as JSONL. Never overwrites an existing file."""
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_predictions(path) -> list[dict]:
    """Read prediction records back with the same checks used to make them, plus duplicates."""
    out, seen = [], set()
    for n, line in jsonl_lines(path):
        r = json_record(line, f"{Path(path).name} line {n}")
        errors = prediction_errors(r)
        key = (r.get("run_id"), r.get("conversation_id"), r.get("segment_order")) if isinstance(r, dict) else None
        if key in seen:
            errors.append("duplicate run/interaction")
        if errors:
            raise ValueError(f"line {n}: " + "; ".join(errors))
        seen.add(key)
        out.append(r)
    return out


# ----------------------------------------------------------------------------- manifest

def manifest(items, examples=None, corpus_path=None, corpus_rows=None) -> dict:
    """What a label set was built from, so a run can name exactly the labels it used."""
    sources = Counter((i.source_path, i.source_sha256, i.record_kind) for i in items)
    keys = sorted(i.key for i in items)
    m = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "labels_module_sha256": sha256_file(__file__),
        "schema_version": DIRECT_VERSION,
        "vocabulary_sha256": vocabulary_sha256(),
        "sources": [{"path": p, "sha256": h, "record_kind": k, "records": n}
                    for (p, h, k), n in sorted(sources.items())],
        "record_kinds": dict(sorted(Counter(i.record_kind for i in items).items())),
        "interactions": len(items),
        "conversations": len({k[0] for k in keys}),
        "keys_sha256": sha256_bytes(json.dumps(keys).encode("utf-8")),
        "per_axis": {a: {"labelled": sum(i.labels[a] is not None for i in items),
                         "masked": sum(i.labels[a] is None for i in items),
                         "mask_reasons": dict(Counter(i.mask_reason(a) for i in items
                                                      if i.labels[a] is None))} for a in AXES},
    }
    if corpus_path is not None:
        m["corpus"] = {"path": rel(corpus_path), "sha256": sha256_file(corpus_path), "rows": corpus_rows}
    if examples is not None:
        m["joined"] = len(examples)
        m["students"] = len({e["student_id"] for e in examples})
    return m


# ----------------------------------------------------------------------------- flow check

def path_checks(examples) -> dict:
    """Round trips through every conversion, on the examples' own labels. No model is involved:
    the "scores" are the targets themselves, so this checks plumbing and nothing else."""
    res = Counter()
    lengths = []
    for ex in examples:
        labels = ex["labels"]
        t = encode_targets(labels)
        res["targets_round_trip_ok"] += decode_targets(t) == labels
        parsed, errs = parse_generated(target_text(labels), allow_null=True)
        res["text_round_trip_ok"] += parsed == labels
        scores = {a: ([float(i == t[a]["index"]) for i in range(len(DIRECT_CODES[a]))] if a in SINGLE
                      else [float(h) for h in t[a]["multi_hot"]]) if t[a]["mask"]
                  else [0.0] * len(DIRECT_CODES[a]) for a in AXES}
        decoded = decode_scores(scores)
        res["score_decode_ok_on_labelled_axes"] += all(decoded[a] == labels[a] for a in AXES
                                                       if labels[a] is not None)
        not_pred = {a: "masked in source" for a in AXES if labels[a] is None}
        pred = prediction_record(ex["conversation_id"], ex["segment_order"],
                                 {a: (None if labels[a] is None else decoded[a]) for a in AXES},
                                 {f: "flow-check" for f in RUN_FIELDS}, not_pred)
        res["prediction_rejected_by_human_checker"] += bool(direct_coding.validate(pred))
        res["student_text_missing"] += ex["student_text_missing"]
        lengths.append(len(render_input(ex)))
    out = dict(res)
    out["examples"] = len(examples)
    if lengths:
        q = sorted(lengths)
        out["rendered_input_characters"] = {"min": q[0], "median": int(statistics.median(q)),
                                            "p90": q[int(0.9 * (len(q) - 1))], "max": q[-1],
                                            "note": "provisional rendering, characters not tokens"}
    out["context_short_because"] = dict(Counter(ex["context_short_because"] or "full window"
                                                for ex in examples))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Check a four-axis label set through the data path. "
                                            "No training, no weight update, no score.")
    p.add_argument("--dev-review", type=Path, action="append", default=[],
                   help="labels-only development-review export (.xlsx)")
    p.add_argument("--direct", type=Path, action="append", default=[], help="direct human records (.jsonl)")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    p.add_argument("--manifest", type=Path, help="write the manifest and check results here (must not exist)")
    a = p.parse_args()
    items = [i for f in a.dev_review for i in read_dev_review_xlsx(f)]
    items += [i for f in a.direct for i in read_direct(f)]
    if not items:
        p.error("give at least one --dev-review or --direct file")
    items = assemble(items, {i.record_kind for i in items})
    corpus = load_corpus(a.corpus)
    examples, missing = attach(items, corpus, a.context)
    report = {"scope": "data-path check only: no training, no weight update, no score",
              "manifest": manifest(items, examples, a.corpus, len(corpus)),
              "missing_from_corpus": [f"{c}#{n}" for c, n in missing],
              "checks": path_checks(examples)}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if a.manifest:
        with a.manifest.open("x", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
