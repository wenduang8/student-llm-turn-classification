#!/usr/bin/env python3
"""A student-grouped training/validation split over a prepared label set, written once and kept.

    python finetuning/grouped_split.py --prepared data/training/<version> --seed <int> \
        --validation-target <rows> --validation-max <rows> --purpose "<why validation exists>" \
        --out data/splits/<name>.json

Students are sorted by Student ID and shuffled with random.Random(seed). In that order each
student joins validation while validation holds fewer than `validation-target` interactions and
the student's interactions keep it at or below `validation-max`; every other student trains. A
student is never divided, so the realised sizes are reported rather than promised, and a student
with more interactions than `validation-max` always trains. Conversations follow their student,
and context rows come only from the target's own conversation, so no context crosses the split.

The file records the seed, the procedure, the exact roster (students, conversations, interaction
keys), the hashes of the prepared data it was made from, the reference labels, the stated
validation purpose and per-part label support. Readers verify those hashes before use. It is
written below data/ only, never overwritten, and its existence does not adopt a protocol.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L  # noqa: E402

PARTS = ("train", "validation")
RECORD_KIND = "student_grouped_split"
PROCEDURE = ("students sorted by Student ID then shuffled with random.Random(seed); in that order a "
             "student joins validation while validation has fewer than validation_target interactions "
             "and adding the student keeps it at or below validation_max; otherwise the student trains")


def key_text(cid: str, seg: int) -> str:
    return f"{cid}#{seg}"


def label_support(examples: list[dict]) -> dict:
    """Per axis: labelled rows, masked rows and positive count per code (schema order)."""
    out = {}
    for axis in L.AXES:
        values = [e["labels"][axis] for e in examples]
        given = [v for v in values if v is not None]
        counts = Counter(v for v in given) if axis in L.SINGLE else Counter(c for v in given for c in v)
        out[axis] = {"labelled": len(given), "masked": len(values) - len(given),
                     "positives": {c: counts.get(c, 0) for c in L.DIRECT_CODES[axis]}}
    return out


def make_split(examples: list[dict], seed: int, validation_target: int, validation_max: int) -> dict:
    """The roster and its checks. Pure: writes nothing."""
    if type(seed) is not int or type(validation_target) is not int or type(validation_max) is not int:
        raise ValueError("seed and sizes must be integers")
    if not 0 < validation_target <= validation_max < len(examples):
        raise ValueError("need 0 < validation_target <= validation_max < number of interactions")
    owner: dict = {}
    rows: Counter = Counter()
    for e in examples:
        if owner.setdefault(e["conversation_id"], e["student_id"]) != e["student_id"]:
            raise ValueError(f"conversation {e['conversation_id']} has more than one student")
        rows[e["student_id"]] += 1
    students = sorted(rows)
    random.Random(seed).shuffle(students)
    part_of, filled = {}, 0
    for s in students:
        joins = filled < validation_target and filled + rows[s] <= validation_max
        part_of[s] = "validation" if joins else "train"
        filled += rows[s] if joins else 0
    parts = {}
    for name in PARTS:
        chosen = [e for e in examples if part_of[e["student_id"]] == name]
        parts[name] = {
            "students": sorted({e["student_id"] for e in chosen}),
            "conversations": sorted({e["conversation_id"] for e in chosen}),
            "interactions": [key_text(e["conversation_id"], e["segment_order"])
                             for e in sorted(chosen, key=lambda e: (e["conversation_id"], e["segment_order"]))],
        }
        parts[name]["counts"] = {k: len(v) for k, v in parts[name].items()}
    support = {name: label_support([e for e in examples if part_of[e["student_id"]] == name]) for name in PARTS}
    overlap = {f: len(set(parts["train"][f]) & set(parts["validation"][f])) for f in ("students", "conversations")}
    assigned = parts["train"]["interactions"] + parts["validation"]["interactions"]
    checks = {"student_overlap": overlap["students"], "conversation_overlap": overlap["conversations"],
              "every_interaction_once": sorted(assigned) == sorted(key_text(e["conversation_id"], e["segment_order"])
                                                                   for e in examples),
              "students_larger_than_validation_max": sorted(s for s in rows if rows[s] > validation_max),
              "shuffled_student_order": students}
    if overlap["students"] or overlap["conversations"] or not checks["every_interaction_once"]:
        raise AssertionError(f"split checks failed: {checks}")
    if not parts["validation"]["interactions"]:
        raise ValueError("no student fits the validation capacity; validation would be empty")
    return {"seed": seed, "validation_target": validation_target, "validation_max": validation_max,
            "procedure": PROCEDURE, "parts": parts, "label_support": support, "checks": checks}


def prepared_fingerprint(prepared_dir: Path) -> dict:
    manifest_path, examples_path = prepared_dir / "manifest.json", prepared_dir / "examples.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples_sha = L.sha256_file(examples_path)
    if examples_sha != manifest["prepared_file"]["sha256"]:
        raise ValueError(f"{examples_path} does not match its manifest")
    if manifest["vocabulary_sha256"] != L.vocabulary_sha256():
        raise ValueError(f"{prepared_dir} was prepared against a different label vocabulary than the current schema")
    return {"dir": L.rel(prepared_dir), "manifest_sha256": L.sha256_file(manifest_path),
            "examples_sha256": examples_sha, "keys_sha256": manifest["keys_sha256"],
            "vocabulary_sha256": manifest["vocabulary_sha256"], "sources": manifest["sources"],
            "source_selection": manifest.get("source_selection"),
            "context_interactions": (manifest.get("input_rendering") or {}).get("context_interactions")}


def load_examples(prepared_dir: Path) -> list[dict]:
    prepared_fingerprint(prepared_dir)
    # Split on "\n" only: JSON text may legitimately contain U+2028/U+2029/U+0085, which
    # str.splitlines() would treat as line ends and cut a record in two.
    text = (prepared_dir / "examples.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def write_split(prepared_dir: Path, out: Path, seed: int, validation_target: int, validation_max: int,
                purpose: str, status: str) -> dict:
    out = Path(out).resolve()
    private = (L.ROOT / "data").resolve()
    if out == private or not out.is_relative_to(private):
        raise ValueError("a split roster names students and must be written below repository data/")
    if not purpose.strip() or not status.strip():
        raise ValueError("state the validation purpose and the status of this split")
    prepared_dir = Path(prepared_dir)
    fingerprint = prepared_fingerprint(prepared_dir)
    split = make_split(load_examples(prepared_dir), seed, validation_target, validation_max)
    record = {"record_kind": RECORD_KIND, "status": status, "validation_purpose": purpose,
              "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "split_module_sha256": L.sha256_file(__file__), "prepared": fingerprint, **split}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return record


def load_split(path: Path, prepared_dir: Path) -> tuple[dict, dict]:
    """(record, {(conversation_id, segment_order): part}) after checking it matches the prepared data."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("record_kind") != RECORD_KIND:
        raise ValueError(f"{path} is not a {RECORD_KIND} record")
    current = prepared_fingerprint(Path(prepared_dir))
    for field in ("manifest_sha256", "examples_sha256", "keys_sha256", "vocabulary_sha256"):
        if record["prepared"][field] != current[field]:
            raise ValueError(f"split {path} was made from different prepared data ({field} differs)")
    part_of = {}
    for name in PARTS:
        for text in record["parts"][name]["interactions"]:
            key = L.parse_key(text)
            if key in part_of:
                raise ValueError(f"{text} is assigned twice")
            part_of[key] = name
    return record, part_of


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--validation-target", type=int, required=True)
    p.add_argument("--validation-max", type=int, required=True)
    p.add_argument("--purpose", required=True, help="what validation is and is not used for")
    p.add_argument("--status", default="provisional pilot split; not an adopted experimental protocol")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        r = write_split(a.prepared, a.out, a.seed, a.validation_target, a.validation_max, a.purpose, a.status)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps({"out": str(a.out), "seed": r["seed"],
                      "counts": {k: r["parts"][k]["counts"] for k in PARTS},
                      "students_larger_than_validation_max": len(r["checks"]["students_larger_than_validation_max"]),
                      "status": r["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
