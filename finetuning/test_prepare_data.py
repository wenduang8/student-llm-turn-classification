"""Synthetic-only checks for the private, unsplit development-data bootstrap. No training."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L
import prepare_data as P

STUDENT, TOPIC, AIDEL, AIRESP = L.AXES


def original(segment=1):
    return {"schema_version": L.DIRECT_VERSION, "coder_id": "Synthetic coder",
            "stage": "independent", "conversation_id": "SYN_A", "segment_order": segment,
            "labels": {STUDENT: "Task automation", TOPIC: ["code"], AIDEL: ["ND"],
                       AIRESP: ["Explain"]}, "flags": [], "notes": "Synthetic note", "unavailable": {}}


def reviewed(segment=1):
    r = original(segment)
    r.update(record_kind=L.DEV_REVIEW_KIND, coder_id="Synthetic AI-assisted review",
             stage_semantics="source allocation phase only; not independent coding",
             review_provenance={"decision": "D-105", "independent_coding": False,
                                "joint_adjudication": False, "original_source_sha256": "a" * 64,
                                "base_path": "data/synthetic-original.jsonl", "revision": "synthetic-v1",
                                "instrument_release": "direct-20261016", "extra_evidence": ["retained"]},
             review_history=[{"source_coder_id": "Synthetic coder", "source_stage": "independent",
                             "original_notes": "Original synthetic note", "original_flags": [],
                             "original_labels": copy.deepcopy(r["labels"])}])
    return r


def reviewed_v9(segment=1):
    """Synthetic record in the D-144 (v9) lineage: original-source entry plus a later batch entry."""
    r = reviewed(segment)
    r["review_provenance"].update(decision="D-144", revision="synthetic-v9")
    r["review_history"].append({"source": "synthetic later batch", "labels": copy.deepcopy(r["labels"]),
                                "notes": "Synthetic later note", "flags": [],
                                "review_provenance": {"decision": "D-143"}})
    return r


class Preparation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        # Only temporary synthetic files are ever created, even when testing path containment.
        (self.root / "codebook_schema.json").write_bytes((L.ROOT / "codebook_schema.json").read_bytes())
        self.root_patch = patch.object(L, "ROOT", self.root)
        self.root_patch.start()
        self.corpus = self.root / "data" / "corpus.csv"
        pd.DataFrame([
            ["SYN_S", "SYN_A", 1, "synthetic prompt one", "synthetic response one"],
            ["SYN_S", "SYN_A", 2, "synthetic prompt two", "synthetic response two"],
            ["SYN_S", "SYN_A", 4, "synthetic prompt four", "synthetic response four"],
        ], columns=L.corpus_data.KEY_COLUMNS).to_csv(self.corpus, index=False)

    def tearDown(self):
        self.root_patch.stop()
        self.tmp.cleanup()

    def write_records(self, rows, name="labels.jsonl"):
        path = self.root / "data" / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    def run_prepare(self, source, **overrides):
        args = {"source": source, "source_kind": "direct", "corpus_path": self.corpus,
                "output_dir": self.root / "data" / "prepared", "n_context": 3}
        args.update(overrides)
        return P.prepare(**args)

    def test_review_preserves_complete_provenance_and_never_passes_human_importer(self):
        row = reviewed()
        source = self.write_records([row])
        item = L.read_reviewed_direct_jsonl(source)[0]
        self.assertEqual(item.record_kind, L.DEV_REVIEW_KIND)
        self.assertEqual(item.source_record, row)
        self.assertEqual(item.source_sha256, L.sha256_file(source))
        self.assertIs(item.source_record["review_provenance"]["independent_coding"], False)
        self.assertIs(item.source_record["review_provenance"]["joint_adjudication"], False)
        with self.assertRaises(ValueError):
            L.read_direct(source)
        stripped = copy.deepcopy(row)
        del stripped["record_kind"]
        with self.assertRaises(ValueError):
            L.read_direct(self.write_records([stripped], "stripped.jsonl"))

    def test_review_rejects_wrong_kind_missing_provenance_and_false_independence_claims(self):
        cases = [original(), {**reviewed(), "record_kind": L.PREDICTION_KIND},
                 {**reviewed(), "review_history": None}, {**reviewed(), "review_history": []},
                 {**reviewed(), "review_history": {}}, {**reviewed(), "review_history": [None]},
                 {**reviewed(), "stage_semantics": ""},
                 {**reviewed(), "run_id": "prediction"}, {**reviewed(), "independent_coding": True}]
        for field in ("independent_coding", "joint_adjudication"):
            for invalid in (True, None, 0, "false"):
                row = reviewed()
                row["review_provenance"][field] = invalid
                cases.append(row)
        for field in ("original_source_sha256", "base_path", "revision", "instrument_release", "decision"):
            row = reviewed()
            row["review_provenance"][field] = ""
            cases.append(row)
        for index, row in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                L.read_reviewed_direct_jsonl(self.write_records([row]))
        with self.assertRaises(ValueError):
            L.read_reviewed_direct_jsonl(self.write_records([reviewed(), reviewed()]))

    def test_v9_review_lineage_is_read_and_prepared(self):
        source = self.write_records([reviewed_v9(1), reviewed_v9(2)])
        items = L.read_reviewed_direct_jsonl(source)
        self.assertEqual([i.source_record["review_provenance"]["decision"] for i in items], ["D-144"] * 2)
        result = self.run_prepare(source, source_kind="reviewed-direct")
        rows = (Path(result["output_directory"]) / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 2)

    def test_later_review_lineages_are_read_and_unlisted_ones_refused(self):
        for decision in ("D-166", "D-167", "D-168", "D-169", "D-182", "D-185", "D-187", "D-188", "D-189", "D-190"):
            row = reviewed_v9()
            row["review_provenance"]["decision"] = decision
            items = L.read_reviewed_direct_jsonl(self.write_records([row]))
            self.assertEqual(items[0].source_record["review_provenance"]["decision"], decision)
        row = reviewed_v9()
        row["review_provenance"]["decision"] = "D-170"
        with self.assertRaises(ValueError):
            L.read_reviewed_direct_jsonl(self.write_records([row]))

    def test_other_review_versions_and_malformed_later_history_are_refused(self):
        cases = []
        for decision in ("D-120", "D-143", "D-145"):
            row = reviewed_v9()
            row["review_provenance"]["decision"] = decision
            cases.append(row)
        for field in ("source", "labels", "notes", "flags", "review_provenance"):
            row = reviewed_v9()
            del row["review_history"][1][field]
            cases.append(row)
        row = reviewed_v9()
        row["review_history"][1]["labels"][TOPIC] = ["unknown"]
        cases.append(row)
        row = reviewed_v9()
        row["review_history"][1]["source"] = ""
        cases.append(row)
        row = reviewed_v9()
        row["review_history"] = [row["review_history"][1]]  # the first entry must be the original source
        cases.append(row)
        for index, row in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                L.read_reviewed_direct_jsonl(self.write_records([row]))

    def test_review_labels_still_require_schema_and_missing_reasons(self):
        for axis, invalid in ((TOPIC, ["unknown"]), (AIDEL, ["ND", "offer"]), (AIRESP, None)):
            row = reviewed()
            row["labels"][axis] = invalid
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                L.read_reviewed_direct_jsonl(self.write_records([row]))

    def test_preparation_is_unsplit_preserves_review_and_masks_truncated_axes(self):
        row = reviewed(2)
        row["flags"] = [L.TRUNCATED]
        row["labels"][AIDEL] = row["labels"][AIRESP] = None
        source = self.write_records([row, reviewed(4)])
        result = self.run_prepare(source, source_kind="reviewed-direct")
        out = Path(result["output_directory"])
        rows = [json.loads(line) for line in (out / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(result["interactions"], 2)
        self.assertEqual(result["students"], 1)
        self.assertIsNone(manifest["split"])
        self.assertEqual(manifest["prepared_file"]["sha256"], L.sha256_file(out / "examples.jsonl"))
        self.assertEqual(rows[0]["annotation_record_kind"], L.DEV_REVIEW_KIND)
        self.assertEqual(rows[0]["source_record"], row)
        self.assertEqual(rows[0]["source"]["sha256"], L.sha256_file(source))
        self.assertEqual(rows[0]["targets"][AIDEL]["mask"], 0)
        self.assertEqual(rows[0]["targets"][AIRESP]["mask"], 0)
        self.assertIsNone(rows[0]["labels"][AIDEL])
        self.assertEqual(rows[0]["mask_reasons"][AIDEL], L.TRUNCATED)
        self.assertEqual([c["segment_order"] for c in rows[0]["context"]], [1])
        self.assertEqual(rows[1]["context"], [])
        self.assertEqual(rows[1]["context_short_because"], "gap")
        self.assertIn("[earlier turns not available]", rows[1]["input_text"])
        # An observed ND has a real target and is not confused with the masked all-zero target.
        self.assertEqual(rows[1]["targets"][AIDEL]["mask"], 1)
        self.assertEqual(sum(rows[1]["targets"][AIDEL]["multi_hot"]), 1)

    def test_original_preserves_notes_without_asserting_independence_verification(self):
        row = original()
        result = self.run_prepare(self.write_records([row]))
        out = Path(result["output_directory"])
        saved = json.loads((out / "examples.jsonl").read_text(encoding="utf-8"))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["source_record"], row)
        self.assertEqual(saved["annotation_record_kind"], "human_independent")
        self.assertIn("do not verify independence", manifest["provenance_note"])

    def test_private_paths_and_existing_versions_are_refused_without_overwrite(self):
        source = self.write_records([original()])
        for invalid in (self.root / "public", self.root / "data", self.root / "data" / ".." / "public"):
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                self.run_prepare(source, output_dir=invalid)
        result = self.run_prepare(source)
        output = Path(result["output_directory"])
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.run_prepare(source)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_missing_duplicate_empty_inputs_fail_before_outputs(self):
        for rows in ([], [original(3)], [original(), original()]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                self.run_prepare(self.write_records(rows))
            self.assertFalse((self.root / "data" / "prepared").exists())

    def test_ambiguous_student_ownership_and_bad_context_are_refused(self):
        source = self.write_records([original()])
        for context in (-1, True, 1.5):
            with self.subTest(context=context), self.assertRaises(ValueError):
                self.run_prepare(source, n_context=context)
        frame = pd.read_csv(self.corpus)
        frame.loc[1, "Student ID"] = "SYN_OTHER"
        frame.to_csv(self.corpus, index=False)
        with self.assertRaises(ValueError):
            self.run_prepare(source)
        self.assertFalse((self.root / "data" / "prepared").exists())

    def test_cli_requires_explicit_source_and_writes_current_format(self):
        source = self.write_records([reviewed()])
        output = self.root / "data" / "cli-prepared"
        args = ["--reviewed-direct", str(source), "--corpus", str(self.corpus),
                "--output-dir", str(output), "--context", "0"]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(P.main(args), 0)
        self.assertEqual(json.loads(stream.getvalue())["interactions"], 1)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_selection"], "reviewed-direct")
        self.assertEqual(manifest["input_rendering"]["context_interactions"], 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            P.main(["--direct", str(source), "--reviewed-direct", str(source)])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
