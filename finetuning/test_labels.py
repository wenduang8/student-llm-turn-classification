"""Guards for finetuning/labels.py, tested with bad input as well as good.

Every record here is synthetic, on made-up keys; none is a student annotation. The one training
loop in this file (MaskedTrainingContract) is a small numpy model on synthetic features: it checks
the target and mask contract any multi-head implementation must honour, and is not the project's
training code (D-85: weight updates on synthetic data only).

    python finetuning/test_labels.py
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L  # noqa: E402

AX = L.AXES
STUDENT, TOPIC, AIDEL, AIRESP = AX


def lab(**over):
    base = {STUDENT: "Task automation", TOPIC: ["code"], AIDEL: ["ND"], AIRESP: ["Explain"]}
    base.update(over)
    return base


def human(cid, n, stage="independent", coder="X", **over):
    return {"schema_version": L.DIRECT_VERSION, "coder_id": coder, "stage": stage, "conversation_id": cid,
            "segment_order": n, "labels": lab(**over), "flags": [], "notes": "", "unavailable": {}}


def corpus(rows):
    """rows: (student, conversation, segment, prompt, response)"""
    return pd.DataFrame(rows, columns=["Student ID", "Conversation ID", "Segment Order",
                                       "Student Prompt", "LLM Response"]).astype(str)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def jsonl(self, name, rows):
        p = self.d / name
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return p

    def xlsx(self, name, rows, header=L.DEV_REVIEW_HEADER, sheets=1):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(header)
        for r in rows:
            ws.append(r)
        for k in range(sheets - 1):
            wb.create_sheet(f"extra{k}")
        p = self.d / name
        wb.save(p)
        return p


class Checks(unittest.TestCase):
    def test_valid_and_invalid_label_sets(self):
        self.assertEqual(L.check_labels(lab()), [])
        self.assertTrue(L.check_labels(lab(**{STUDENT: "Delegation"})))           # not a code
        self.assertTrue(L.check_labels(lab(**{TOPIC: "code"})))                   # multi needs a list
        self.assertTrue(L.check_labels(lab(**{TOPIC: []})))                       # empty set
        self.assertTrue(L.check_labels(lab(**{TOPIC: ["code", "code"]})))         # duplicate
        self.assertTrue(L.check_labels(lab(**{AIDEL: ["ND", "offer"]})))          # negative plus other
        self.assertTrue(L.check_labels({k: v for k, v in lab().items() if k != TOPIC}))
        self.assertTrue(L.check_labels({**lab(), "extra": 1}))
        self.assertEqual(L.check_labels(lab(**{AIDEL: None})), [])
        self.assertTrue(L.check_labels(lab(**{AIDEL: None}), allow_null=False))

    def test_multi_select_is_put_in_schema_order(self):
        codes = L.DIRECT_CODES[AIRESP]
        self.assertEqual(L.canonical(AIRESP, [codes[3], codes[0]]), [codes[0], codes[3]])

    def test_vocabulary_hash_tracks_the_schema(self):
        h = L.vocabulary_sha256()
        saved = copy.deepcopy(L.DIRECT_CODES[TOPIC])
        try:
            L.DIRECT_CODES[TOPIC].reverse()
            self.assertNotEqual(L.vocabulary_sha256(), h)
        finally:
            L.DIRECT_CODES[TOPIC][:] = saved
        self.assertEqual(L.vocabulary_sha256(), h)


class Targets(unittest.TestCase):
    def test_null_is_masked_and_never_the_negative_code(self):
        t = L.encode_targets(lab(**{AIDEL: None, AIRESP: None}))
        self.assertEqual(t[AIDEL]["mask"], 0)
        neg = L.encode_targets(lab(**{AIDEL: ["ND"]}))[AIDEL]
        self.assertEqual(neg["mask"], 1)
        self.assertEqual(sum(neg["multi_hot"]), 1)          # the negative is a real, positive code
        self.assertNotEqual(t[AIDEL], neg)

    def test_round_trip_every_code(self):
        for axis in AX:
            for code in L.DIRECT_CODES[axis]:
                v = code if axis in L.SINGLE else [code]
                self.assertEqual(L.decode_targets(L.encode_targets(lab(**{axis: v}))), lab(**{axis: v}))

    def test_invalid_labels_are_not_encoded(self):
        with self.assertRaises(ValueError):
            L.encode_targets(lab(**{AIDEL: ["ND", "offer"]}))

    def test_generation_text_round_trip_and_strict_parse(self):
        l = lab(**{TOPIC: ["concept", "code"]})
        self.assertEqual(L.parse_generated(L.target_text(l))[0], {a: L.canonical(a, l[a]) for a in AX})
        for bad in ["", "not json", "[]", json.dumps({**lab(), "extra": 1}),
                    json.dumps(lab(**{STUDENT: "Delegation"})), json.dumps(lab(**{AIDEL: ["ND", "offer"]})),
                    json.dumps(lab(**{AIDEL: None}))]:
            parsed, errors = L.parse_generated(bad)
            self.assertIsNone(parsed, bad)
            self.assertTrue(errors, bad)
        self.assertIsNotNone(L.parse_generated(json.dumps(lab(**{AIDEL: None})), allow_null=True)[0])


class Decoding(unittest.TestCase):
    def scores(self, **over):
        s = {a: [0.0] * len(L.DIRECT_CODES[a]) for a in AX}
        s[STUDENT][2] = 0.9
        s[TOPIC][0] = 0.8
        s[AIDEL][0] = 0.7
        s[AIRESP][1] = 0.6
        s.update(over)
        return s

    def test_argmax_threshold_and_fallback(self):
        out = L.decode_scores(self.scores(**{TOPIC: [0.2, 0.3, 0.1, 0.0]}))
        self.assertEqual(out[STUDENT], L.DIRECT_CODES[STUDENT][2])
        self.assertEqual(out[TOPIC], [L.DIRECT_CODES[TOPIC][1]])             # none above: top one
        out = L.decode_scores(self.scores(**{TOPIC: [0.9, 0.6, 0.1, 0.0]}))
        self.assertEqual(out[TOPIC], L.DIRECT_CODES[TOPIC][:2])

    def test_exclusive_negative_resolved_both_ways(self):
        codes = L.DIRECT_CODES[AIDEL]
        neg_wins = L.decode_scores(self.scores(**{AIDEL: [0.9, 0.6, 0.0, 0.0, 0.0]}))
        self.assertEqual(neg_wins[AIDEL], ["ND"])
        other_wins = L.decode_scores(self.scores(**{AIDEL: [0.6, 0.9, 0.7, 0.0, 0.0]}))
        self.assertEqual(other_wins[AIDEL], [codes[1], codes[2]])
        self.assertEqual(L.check_labels(other_wins, allow_null=False), [])

    def test_wrong_shapes_refused(self):
        with self.assertRaises(ValueError):
            L.decode_scores({k: v for k, v in self.scores().items() if k != TOPIC})
        with self.assertRaises(ValueError):
            L.decode_scores(self.scores(**{TOPIC: [0.9]}))


class Readers(Tmp):
    def test_direct_records_keep_provenance(self):
        p = self.jsonl("a.jsonl", [human("SYN_A", 1), human("SYN_A", 2, stage="calibration")])
        items = L.read_direct(p)
        self.assertEqual([i.record_kind for i in items], ["human_independent", "human_calibration"])
        self.assertEqual(items[0].source_sha256, L.sha256_file(p))
        self.assertEqual(items[0].coder_id, "X")

    def test_direct_blank_without_reason_refused_and_truncation_masked(self):
        with self.assertRaises(ValueError):
            L.read_direct(self.jsonl("b.jsonl", [human("SYN_A", 1, **{AIDEL: None})]))
        r = human("SYN_A", 1, **{AIDEL: None, AIRESP: None})
        r["flags"] = [L.TRUNCATED]
        item = L.read_direct(self.jsonl("c.jsonl", [r]))[0]
        self.assertEqual(item.masked_axes(), [AIDEL, AIRESP])
        self.assertEqual(item.mask_reason(AIDEL), L.TRUNCATED)

    def test_dev_review_export(self):
        p = self.xlsx("r.xlsx", [["SYN_A#1", "Task automation", "code; concept", "offer", "Explain"],
                                 ["SYN_A#2", "Information retrieval", "concept", None, " "],
                                 [None, None, None, None, None]])
        items = L.read_dev_review_xlsx(p)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].labels[TOPIC], ["code", "concept"])
        self.assertEqual(items[1].masked_axes(), [AIDEL, AIRESP])
        self.assertEqual(items[1].mask_reason(AIDEL), L.DEV_REVIEW_BLANK)
        self.assertEqual(items[0].record_kind, L.DEV_REVIEW_KIND)

    def test_dev_review_export_refusals(self):
        good = ["SYN_A#1", "Task automation", "code", "offer", "Explain"]
        cases = {
            "bad code": [["SYN_A#1", "Task automation", "cod", "offer", "Explain"]],
            "neg + other": [["SYN_A#1", "Task automation", "code", "ND; offer", "Explain"]],
            "single with two": [["SYN_A#1", "Task automation; Guided generation", "code", "offer", "Explain"]],
            "bad key": [["SYN_A-1", "Task automation", "code", "offer", "Explain"]],
            "zero segment": [["SYN_A#0", "Task automation", "code", "offer", "Explain"]],
            "duplicate": [good, good],
        }
        for name, rows in cases.items():
            with self.assertRaises(ValueError, msg=name):
                L.read_dev_review_xlsx(self.xlsx(name + ".xlsx", rows))
        with self.assertRaises(ValueError):
            L.read_dev_review_xlsx(self.xlsx("h.xlsx", [good], header=["Interaction", "type", "topic", "a", "b"]))
        with self.assertRaises(ValueError):
            L.read_dev_review_xlsx(self.xlsx("s.xlsx", [good], sheets=2))


class Assembly(Tmp):
    def items(self):
        a = L.read_direct(self.jsonl("a.jsonl", [human("SYN_A", 1, coder="A")]))
        b = L.read_direct(self.jsonl("b.jsonl", [human("SYN_A", 1, coder="B")]))
        return a, b

    def test_two_records_for_one_interaction_refused(self):
        a, b = self.items()
        with self.assertRaises(ValueError):
            L.assemble(a + b, {"human_independent"})

    def test_kinds_must_be_permitted_and_predictions_are_not_labels(self):
        a, _ = self.items()
        with self.assertRaises(ValueError):
            L.assemble(a, {L.DEV_REVIEW_KIND})
        with self.assertRaises(ValueError):
            L.assemble(a, {L.PREDICTION_KIND})
        self.assertEqual(len(L.assemble(a, {"human_independent"})), 1)

    def test_manifest_names_sources_and_masks(self):
        a, _ = self.items()
        m = L.manifest(a)
        self.assertEqual(m["sources"][0]["sha256"], a[0].source_sha256)
        self.assertEqual(m["vocabulary_sha256"], L.vocabulary_sha256())
        self.assertEqual(m["per_axis"][AIDEL], {"labelled": 1, "masked": 0, "mask_reasons": {}})


class Join(Tmp):
    def test_join_context_stops_at_gap_and_missing_keys_reported(self):
        df = corpus([("S1", "SYN_A", 1, "p1", "r1"), ("S1", "SYN_A", 2, "p2", "r2"),
                     ("S1", "SYN_A", 4, "p4", "r4"), ("S2", "SYN_B", 3, "q3", "s3")])
        items = L.read_direct(self.jsonl("a.jsonl", [human("SYN_A", 2), human("SYN_A", 4),
                                                     human("SYN_B", 3), human("SYN_C", 1)]))
        ex, missing = L.attach(items, df)
        self.assertEqual(missing, [("SYN_C", 1)])
        by = {(e["conversation_id"], e["segment_order"]): e for e in ex}
        self.assertEqual([c["segment_order"] for c in by[("SYN_A", 2)]["context"]], [1])
        self.assertEqual(by[("SYN_A", 4)]["context"], [])
        self.assertEqual(by[("SYN_A", 4)]["context_short_because"], "gap")
        self.assertEqual(by[("SYN_B", 3)]["context_short_because"], "record starts late")
        self.assertIn("[earlier turns not available]", L.render_input(by[("SYN_A", 4)]))
        self.assertTrue(L.render_input(by[("SYN_A", 2)]).endswith("Assistant: r2"))

    def test_duplicate_corpus_row_refused(self):
        df = corpus([("S1", "SYN_A", 1, "p1", "r1"), ("S1", "SYN_A", 1, "p1b", "r1b")])
        with self.assertRaises(ValueError):
            L.attach(L.read_direct(self.jsonl("a.jsonl", [human("SYN_A", 1)])), df)


class Predictions(Tmp):
    RUN = {"run_id": "syn", "model_route": "synthetic", "config_sha256": "c", "dataset_manifest_sha256": "m"}

    def test_prediction_cannot_pass_as_human_record(self):
        r = L.prediction_record("SYN_A", 1, lab(), self.RUN)
        self.assertTrue(L.direct_coding.validate(r))
        self.assertEqual(r["record_kind"], L.PREDICTION_KIND)

    def test_prediction_requires_run_info_and_reasons_for_nulls(self):
        with self.assertRaises(ValueError):
            L.prediction_record("SYN_A", 1, lab(), {**self.RUN, "run_id": ""})
        with self.assertRaises(ValueError):
            L.prediction_record("SYN_A", 1, lab(**{AIDEL: None}), self.RUN)
        with self.assertRaises(ValueError):
            L.prediction_record("SYN_A", 1, lab(), self.RUN, {AIDEL: "why"})
        ok = L.prediction_record("SYN_A", 1, lab(**{AIDEL: None}), self.RUN, {AIDEL: "truncated capture"})
        self.assertIsNone(ok["labels"][AIDEL])

    def test_write_read_round_trip_and_refusals(self):
        recs = [L.prediction_record("SYN_A", n, lab(), self.RUN) for n in (1, 2)]
        p = L.write_predictions(self.d / "pred.jsonl", recs)
        self.assertEqual(L.read_predictions(p), recs)
        with self.assertRaises(FileExistsError):
            L.write_predictions(p, recs)
        self.assertRaises(ValueError, L.read_predictions, self.jsonl("h.jsonl", [human("SYN_A", 1)]))
        stale = dict(recs[0], vocabulary_sha256="0" * 64)
        self.assertRaises(ValueError, L.read_predictions, self.jsonl("v.jsonl", [stale]))
        self.assertRaises(ValueError, L.read_predictions, self.jsonl("d.jsonl", [recs[0], recs[0]]))


class ReviewRegressions(Tmp):
    """Three holes found in review on 30 September 2026, each reproduced before it was fixed."""
    RUN = Predictions.RUN

    def test_prediction_with_human_fields_is_not_read_as_human(self):
        r = L.prediction_record("SYN_A", 1, lab(), self.RUN)
        r.update(coder_id="X", stage="independent", flags=[], notes="", unavailable={})
        with self.assertRaises(ValueError):
            L.read_direct(self.jsonl("mixed.jsonl", [r]))
        for field in L.NOT_HUMAN_FIELDS:                     # any one of them is enough
            h = human("SYN_A", 1)
            h[field] = "x"
            with self.assertRaises(ValueError, msg=field):
                L.read_direct(self.jsonl(f"one_{field}.jsonl", [h]))
        self.assertRaises(ValueError, L.read_predictions, self.jsonl("pred_mixed.jsonl", [r]))

    def test_non_finite_scores_and_bad_threshold_refused(self):
        base = Decoding.scores(Decoding())
        nan, inf = float("nan"), float("inf")
        with self.assertRaises(ValueError):
            L.decode_scores({a: [nan] * len(L.DIRECT_CODES[a]) for a in AX})
        for axis in AX:
            for bad in (nan, inf, -inf):
                s = copy.deepcopy(base)
                s[axis][0] = bad
                with self.assertRaises(ValueError, msg=(axis, bad)):
                    L.decode_scores(s)
        for bad_threshold in (nan, inf, 0, -0.1, 1.5, True, "0.5", None):
            with self.assertRaises(ValueError, msg=repr(bad_threshold)):
                L.decode_scores(base, threshold=bad_threshold)
        logits = copy.deepcopy(base)
        logits[TOPIC][0] = 1.7                               # a logit where a probability belongs
        self.assertRaises(ValueError, L.decode_scores, logits)
        text = copy.deepcopy(base)
        text[AIDEL][0] = "high"
        self.assertRaises(ValueError, L.decode_scores, text)
        self.assertEqual(L.check_labels(L.decode_scores(base, threshold=1), allow_null=False), [])

    def test_read_back_applies_the_full_checks(self):
        good = L.prediction_record("SYN_A", 1, lab(**{AIDEL: None}), self.RUN, {AIDEL: "truncated capture"})
        self.assertEqual(L.read_predictions(self.jsonl("good.jsonl", [good])), [good])
        cases = {
            "no conversation_id": {k: v for k, v in good.items() if k != "conversation_id"},
            "empty conversation_id": dict(good, conversation_id=" "),
            "segment 0": dict(good, segment_order=0),
            "segment as text": dict(good, segment_order="1"),
            "segment as bool": dict(good, segment_order=True),
            "numeric reason": dict(good, not_predicted={AIDEL: 5}),
            "blank reason": dict(good, not_predicted={AIDEL: " "}),
            "reason for a labelled axis": dict(good, not_predicted={AIDEL: "x", TOPIC: "y"}),
            "reason missing": dict(good, not_predicted={}),
            "unknown axis in reasons": dict(good, not_predicted={AIDEL: "x", "mood": "y"}),
            "not_predicted absent": {k: v for k, v in good.items() if k != "not_predicted"},
            "not_predicted a list": dict(good, not_predicted=[AIDEL]),
            "labels absent": {k: v for k, v in good.items() if k != "labels"},
            "run field blank": dict(good, model_route=""),
            "coder field added": dict(good, coder_id="X"),
            "wrong kind": dict(good, record_kind="human_independent"),
            "not an object": [good],
            "fully labelled, not_predicted absent": {k: v for k, v in L.prediction_record("SYN_A", 2, lab(), self.RUN).items()
                                                     if k != "not_predicted"},
            "fully labelled, not_predicted a list": dict(L.prediction_record("SYN_A", 2, lab(), self.RUN),
                                                         not_predicted=[]),
        }
        for name, rec in cases.items():
            with self.assertRaises(ValueError, msg=name):
                L.read_predictions(self.jsonl(name.replace(" ", "_") + ".jsonl", [rec]))
        for bad_key in ({"conversation_id": ""}, {"segment_order": 0}, {"segment_order": "1"}):
            args = {"conversation_id": "SYN_A", "segment_order": 1, **bad_key}
            with self.assertRaises(ValueError, msg=bad_key):
                L.prediction_record(args["conversation_id"], args["segment_order"], lab(), self.RUN)
        with self.assertRaises(ValueError):
            L.prediction_record("SYN_A", 1, lab(**{AIDEL: None}), self.RUN, {AIDEL: 5})


def reviewed(cid, n, note):
    """A synthetic D-105-format review record (as in test_prepare_data.py) with a free-text note."""
    r = human(cid, n)
    r.update(record_kind=L.DEV_REVIEW_KIND, coder_id="Synthetic AI-assisted review", notes=note,
             stage_semantics="source allocation phase only; not independent coding",
             review_provenance={"decision": "D-105", "independent_coding": False, "joint_adjudication": False,
                                "original_source_sha256": "a" * 64, "base_path": "data/synthetic-original.jsonl",
                                "revision": "synthetic-v1", "instrument_release": "direct-20261016"},
             review_history=[{"source_coder_id": "X", "source_stage": "independent", "original_notes": note,
                              "original_flags": [], "original_labels": lab()}])
    return r


class LineBoundaries(Tmp):
    """JSON may carry U+0085, U+2028 and U+2029 unescaped inside a string, and json.dumps with
    ensure_ascii=False writes them so. str.splitlines() ended lines there and cut such a record in
    two (training-code audit, 19 October 2026); reproduced before it was fixed."""
    RUN = Predictions.RUN
    SEPARATORS = ("\u0085", "\u2028", "\u2029")

    def raw(self, name, rows, end="\n"):
        p = self.d / name
        p.write_bytes("".join(json.dumps(r, ensure_ascii=False) + end for r in rows).encode("utf-8"))
        return p

    def test_every_jsonl_reader_keeps_such_a_record_whole(self):
        for sep in self.SEPARATORS:
            for end in ("\n", "\r\n", "\r"):                 # LF, CRLF and a lone CR all end a line
                note, tag = f"before{sep}after", f"{ord(sep):x}-{len(end)}"
                h = human("SYN_A", 1)
                h["notes"] = note
                path = self.raw(f"human-{tag}.jsonl", [h, human("SYN_A", 2)], end)
                self.assertIn(sep, path.read_text(encoding="utf-8"))
                self.assertEqual(L.direct_coding.read_records(path)[0]["notes"], note)
                self.assertEqual([i.source_record["notes"] for i in L.read_direct(path)], [note, ""])
                items = L.read_reviewed_direct_jsonl(self.raw(f"review-{tag}.jsonl", [reviewed("SYN_A", 1, note)], end))
                self.assertEqual(items[0].source_record["review_history"][0]["original_notes"], note)
                pred = L.prediction_record("SYN_A", 1, lab(**{AIDEL: None}), self.RUN, {AIDEL: note})
                self.assertEqual(L.read_predictions(self.raw(f"pred-{tag}.jsonl", [pred], end)), [pred])

    def test_a_byte_order_mark_is_skipped_by_every_reader(self):
        # Found in review 4 of the fix: nothing tested that a BOM-prefixed file still reads.
        bom = b"\xef\xbb\xbf"
        human_path = self.d / "bom-human.jsonl"
        human_path.write_bytes(bom + json.dumps(human("SYN_A", 1)).encode() + b"\n")
        self.assertEqual(len(L.direct_coding.read_records(human_path)), 1)
        self.assertEqual(len(L.read_direct(human_path)), 1)
        review_path = self.d / "bom-review.jsonl"
        review_path.write_bytes(bom + json.dumps(reviewed("SYN_A", 1, "x")).encode() + b"\n")
        self.assertEqual(len(L.read_reviewed_direct_jsonl(review_path)), 1)
        pred = L.prediction_record("SYN_A", 1, lab(), self.RUN)
        pred_path = self.d / "bom-pred.jsonl"
        pred_path.write_bytes(bom + json.dumps(pred).encode() + b"\n")
        self.assertEqual(L.read_predictions(pred_path), [pred])
        self.assertEqual(L.jsonl_lines(pred_path), [(1, json.dumps(pred))])

    def test_refusals_still_name_the_physical_line(self):
        for sep in self.SEPARATORS:
            h = human("SYN_A", 1)
            h["notes"] = f"a{sep}b"
            path = self.raw("human.jsonl", [h, human("SYN_A", 2, **{AIDEL: None})])   # line 2: unfilled judgement
            with self.assertRaisesRegex(ValueError, r"^line 2: "):
                L.direct_coding.read_records(path)
            with self.assertRaisesRegex(ValueError, r"^line 2: "):
                L.read_direct(path)
            bad = reviewed("SYN_A", 2, "x")
            del bad["stage_semantics"]
            with self.assertRaisesRegex(ValueError, r"review\.jsonl line 2: "):
                L.read_reviewed_direct_jsonl(self.raw("review.jsonl", [reviewed("SYN_A", 1, f"a{sep}b"), bad]))
            first = L.prediction_record("SYN_A", 1, lab(**{AIDEL: None}), self.RUN, {AIDEL: f"a{sep}b"})
            with self.assertRaisesRegex(ValueError, r"^line 2: "):
                L.read_predictions(self.raw("pred.jsonl", [first, dict(first, segment_order=2, model_route="")]))

    def test_outside_a_string_they_are_not_json_and_the_refusal_names_the_file_and_line(self):
        # JSON whitespace is only space, tab, LF and CR; the old split dropped such a trailing
        # character silently. Found in review of the fix: the refusal then pointed at "line 1".
        pred = L.prediction_record("SYN_A", 2, lab(), self.RUN)
        for sep in self.SEPARATORS:
            def write(name, good, bad):
                p = self.d / name
                p.write_bytes((json.dumps(good, ensure_ascii=False) + "\n"
                               + json.dumps(bad, ensure_ascii=False) + sep + "\n").encode("utf-8"))
                return p
            path = write("human.jsonl", human("SYN_A", 2), human("SYN_A", 1))
            with self.assertRaisesRegex(ValueError, r"^line 2: not valid JSON \(Extra data"):
                L.direct_coding.read_records(path)
            with self.assertRaisesRegex(ValueError, r"^human\.jsonl line 2: not valid JSON \(Extra data"):
                L.read_direct(path)
            with self.assertRaisesRegex(ValueError, r"^review\.jsonl line 2: not valid JSON"):
                L.read_reviewed_direct_jsonl(write("review.jsonl", reviewed("SYN_A", 2, "x"), reviewed("SYN_A", 1, "y")))
            with self.assertRaisesRegex(ValueError, r"^pred\.jsonl line 2: not valid JSON"):
                L.read_predictions(write("pred.jsonl", pred, dict(pred, segment_order=1)))

    def test_json_that_is_not_an_object_is_refused_by_line_not_crashed_on(self):
        # Found in review 2 of the fix (older than it): direct_coding raised AttributeError.
        for bad in (None, [1, 2], "text", 5):
            path = self.d / "objects.jsonl"
            path.write_text(json.dumps(human("SYN_A", 1)) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"^line 2: record must be an object", msg=repr(bad)):
                L.direct_coding.read_records(path)
            with self.assertRaisesRegex(ValueError, r"^line 2: record must be an object", msg=repr(bad)):
                L.read_direct(path)

    def test_undecodable_text_and_unconvertible_numbers_name_the_file(self):
        latin = self.d / "latin1.jsonl"
        latin.write_bytes(json.dumps(human("SYN_A", 1)).encode() + b"\n" + '"caf\xe9"'.encode("latin-1") + b"\n")
        for reader in (L.read_direct, L.read_reviewed_direct_jsonl, L.read_predictions):
            with self.assertRaisesRegex(ValueError, r"^latin1\.jsonl: not UTF-8 text", msg=reader.__name__):
                reader(latin)
        first = json.dumps(human("SYN_A", 1)).encode() + b"\n"
        for bom in (b"", b"\xef\xbb\xbf"):                 # the offset counts the file's bytes, BOM included
            p = self.d / "offset.jsonl"
            p.write_bytes(bom + first + b'"caf\xe9"\n')
            with self.assertRaisesRegex(ValueError, rf"not UTF-8 text \(byte {len(bom) + len(first) + 4} of the file\)"):
                L.read_predictions(p)
        too_long = '{"segment_order": 1' + "0" * 5000 + "}"                   # beyond int's 4300-digit limit
        p = self.d / "long.jsonl"
        p.write_text(json.dumps(L.prediction_record("SYN_A", 1, lab(), self.RUN)) + "\n" + too_long + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"^long\.jsonl line 2: not valid JSON \(Exceeds the limit"):
            L.read_predictions(p)
        h = self.d / "long-human.jsonl"
        h.write_text(json.dumps(human("SYN_A", 1)) + "\n" + too_long + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"^line 2: not valid JSON \(Exceeds the limit"):
            L.direct_coding.read_records(h)


class MaskedTrainingContract(unittest.TestCase):
    """Synthetic data only. A linear multi-head model trained by gradient descent, to check that
    the targets and masks from encode_targets behave as a loss needs them to."""

    def setUp(self):
        rng = np.random.default_rng(0)
        n, d = 120, 6
        self.X = rng.normal(size=(n, d))
        self.W_true = {a: rng.normal(size=(d, len(L.DIRECT_CODES[a]))) for a in AX}
        self.labels = []
        for i in range(n):
            l = {}
            for a in AX:
                z = self.X[i] @ self.W_true[a]
                codes = L.DIRECT_CODES[a]
                if a in L.SINGLE:
                    l[a] = codes[int(np.argmax(z))]
                else:
                    neg = L.NEGATIVES.get(a)
                    pos = [c for c, v in zip(codes, z) if v > 0.5 and c != neg]
                    l[a] = pos or [neg or codes[int(np.argmax(z))]]
                if a.startswith("AI ") and i % 5 == 0:
                    l[a] = None                        # masked, like a truncated capture
            self.labels.append(l)
        self.T = [L.encode_targets(l) for l in self.labels]

    def loss_and_grad(self, W, rows):
        total, grads = 0.0, {a: np.zeros_like(W[a]) for a in AX}
        for i in rows:
            x = self.X[i]
            for a in AX:
                t = self.T[i][a]
                if not t["mask"]:
                    continue
                z = x @ W[a]
                if a in L.SINGLE:
                    p = np.exp(z - z.max()); p /= p.sum()
                    total -= np.log(p[t["index"]] + 1e-12)
                    y = np.zeros_like(p); y[t["index"]] = 1
                    grads[a] += np.outer(x, p - y)
                else:
                    p = 1 / (1 + np.exp(-z)); y = np.array(t["multi_hot"], dtype=float)
                    total -= float(np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
                    grads[a] += np.outer(x, p - y)
        return total, grads

    def test_loss_falls_masked_rows_give_no_gradient_and_outputs_decode(self):
        W = {a: np.zeros_like(self.W_true[a]) for a in AX}
        rows = range(len(self.X))
        first, _ = self.loss_and_grad(W, rows)
        for _ in range(300):
            _, g = self.loss_and_grad(W, rows)
            for a in AX:
                W[a] -= 0.02 * g[a]
        last, _ = self.loss_and_grad(W, rows)
        self.assertLess(last, 0.5 * first)
        masked_only = [i for i in rows if self.T[i][L.AXES[2]]["mask"] == 0]
        self.assertTrue(masked_only)
        _, g = self.loss_and_grad(W, masked_only)
        for a in (AIDEL, AIRESP):
            self.assertTrue(np.all(g[a] == 0), a)
        for i in rows:
            z = {a: self.X[i] @ W[a] for a in AX}
            s = {a: (z[a] if a in L.SINGLE else 1 / (1 + np.exp(-z[a]))) for a in AX}
            self.assertEqual(L.check_labels(L.decode_scores(s), allow_null=False), [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
