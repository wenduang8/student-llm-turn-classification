#!/usr/bin/env python3
"""Prepare an unsplit, private development dataset from one explicitly selected label source.

    python finetuning/prepare_data.py --direct path/to/original.jsonl
    python finetuning/prepare_data.py --reviewed-direct path/to/reviewed.jsonl

Outputs stay below data/ and a destination must not already exist. No model, split, training,
evaluation or choice of final reference labels is made. Context rendering is provisional.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L  # noqa: E402

PREPARED_KIND = "prepared_development_example"
FORMAT_VERSION = "unsplit-direct-development-v1"
DEFAULT_OUTPUT = L.ROOT / "data" / "training" / "prepared-dw150"


def private_destination(output_dir: Path) -> Path:
    """Resolve before creating anything; paths through symlinks must also remain in data/."""
    private_root = (L.ROOT / "data").resolve()
    destination = Path(output_dir).resolve()
    if destination == private_root or not destination.is_relative_to(private_root):
        raise ValueError("prepared data must be written to a new subdirectory of repository data/")
    if destination.exists():
        raise FileExistsError(f"destination already exists; choose a new version: {destination}")
    return destination


def prepare(*, source: Path, source_kind: str, output_dir: Path,
            corpus_path: Path = L.DEFAULT_CORPUS, n_context: int = L.DEFAULT_CONTEXT) -> dict:
    """Validate and join all targets before writing a new private preparation directory."""
    if source_kind not in ("direct", "reviewed-direct"):
        raise ValueError("choose direct or reviewed-direct explicitly")
    if type(n_context) is not int or n_context < 0:
        raise ValueError("context must be a nonnegative integer")
    destination = private_destination(output_dir)
    reader = L.read_direct if source_kind == "direct" else L.read_reviewed_direct_jsonl
    items = reader(source)
    if not items:
        raise ValueError("source has no label records")
    allowed = set(L.HUMAN_KIND.values()) if source_kind == "direct" else {L.DEV_REVIEW_KIND}
    items = L.assemble(items, allowed)
    corpus_path = Path(corpus_path)
    corpus = L.load_corpus(corpus_path)
    examples, missing = L.attach(items, corpus, n_context)
    if missing:
        keys = ", ".join(f"{cid}#{seg}" for cid, seg in missing)
        raise ValueError("labelled interactions missing from corpus: " + keys)

    # A conversation must have one student before any later student-grouped split can be safe.
    wanted = {i.conversation_id for i in items}
    relevant = corpus[corpus["Conversation ID"].isin(wanted)]
    if relevant["Student ID"].str.strip().eq("").any():
        raise ValueError("labelled conversations include an empty Student ID")
    if (relevant.groupby("Conversation ID")["Student ID"].nunique() != 1).any():
        raise ValueError("each labelled conversation must belong to exactly one student")
    if relevant.duplicated(["Conversation ID", "Segment Order"]).any():
        raise ValueError("labelled conversations include duplicate corpus interaction keys")

    by_key = {i.key: i for i in items}
    prepared = []
    for ex in examples:
        item = by_key[(ex["conversation_id"], ex["segment_order"])]
        prepared.append({
            **ex,
            "record_kind": PREPARED_KIND,
            "format_version": FORMAT_VERSION,
            "schema_version": L.DIRECT_VERSION,
            "vocabulary_sha256": L.vocabulary_sha256(),
            "annotation_record_kind": item.record_kind,
            "source": {"path": item.source_path, "sha256": item.source_sha256},
            "source_record": item.source_record,
            "input_text": L.render_input(ex),
            "targets": L.encode_targets(item.labels),
            "mask_reasons": {a: item.mask_reason(a) for a in item.masked_axes()},
        })

    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared)
    manifest = L.manifest(items, examples, corpus_path, len(corpus))
    manifest.update({
        "format_version": FORMAT_VERSION,
        "scope": "unsplit provisional development data; no training, scoring or final reference selection",
        "source_selection": source_kind,
        "provenance_note": "Record kinds retain supplied provenance; format checks do not verify independence.",
        "split": None,
        "input_rendering": {"context_interactions": n_context, "status": "provisional",
                            "gap_policy": "stop at a Segment Order gap", "token_truncation": None},
        "preparation_module_sha256": L.sha256_file(__file__),
        "schema_file_sha256": L.sha256_file(L.ROOT / "codebook_schema.json"),
        "prepared_file": {"name": "examples.jsonl", "sha256": L.sha256_bytes(text.encode("utf-8")),
                          "records": len(prepared)},
    })
    # Creating a fresh directory also refuses concurrent attempts to use the same version.
    # Consumers should require manifest.json, written last, before using the prepared file.
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "examples.jsonl").open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with (destination / "manifest.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return {"output_directory": str(destination), "interactions": len(prepared),
            "students": manifest["students"], "conversations": manifest["conversations"],
            "per_axis": manifest["per_axis"], "scope": manifest["scope"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--direct", type=Path, help="original direct human JSONL")
    source.add_argument("--reviewed-direct", type=Path,
                        help="explicitly selected AI-assisted reviewed JSONL (D-105, or v9 under D-144)")
    parser.add_argument("--corpus", type=Path, default=L.DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context", type=int, default=L.DEFAULT_CONTEXT,
                        help="preceding interactions for provisional rendering; not a model setting")
    args = parser.parse_args(argv)
    try:
        report = prepare(source=args.direct or args.reviewed_direct,
                         source_kind="direct" if args.direct else "reviewed-direct",
                         output_dir=args.output_dir, corpus_path=args.corpus, n_context=args.context)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
