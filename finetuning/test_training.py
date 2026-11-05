"""Synthetic-only checks for the multi-head trainer. No private data and no download: a tiny
randomly initialised Qwen3 on CPU with a byte-level stand-in tokenizer.

    python -m unittest finetuning/test_training.py
"""
import contextlib
import copy
import dataclasses
import io
import json
import random
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation as EV  # noqa: E402
import grouped_split as GS  # noqa: E402
import labels as L  # noqa: E402
import model_inputs as MI  # noqa: E402

try:
    import peft  # noqa: F401
    import torch
    import transformers  # noqa: F401
    import multihead as M
    import train_multihead as TM
    HAVE_TORCH = True
except ImportError:                     # the data-preparation environment has no model stack
    HAVE_TORCH = False

STUDENT, TOPIC, AIDEL, AIRESP = L.AXES
POLICY = MI.InputPolicy(max_tokens=600, context_interactions=3, prompt_cap=160, response_cap=200,
                        min_context_fragment=64)
PILOT_CONFIG = Path(__file__).resolve().parent / "configs" / "pilot-20261017.json"
ROOT_DIR = Path(__file__).resolve().parents[1]
IDENTITY = {"id": "tiny-random-qwen3", "revision": "synthetic-test", "files_sha256": {}}


def encode(text):
    """Byte-level stand-in tokenizer: ids 0..255."""
    return list(text.encode("utf-8"))


def decode(ids):
    return bytes(ids).decode("utf-8", "replace")


def example(seg=4, prompt="p", response="r", context=(), short=None, **extra):
    ctx = [{"segment_order": seg - len(context) + i, "student_prompt": p, "llm_response": r}
           for i, (p, r) in enumerate(context)]
    return {"student_id": "SYN_S", "conversation_id": "SYN_A", "segment_order": seg, "student_prompt": prompt,
            "llm_response": response, "context": ctx, "context_available": len(ctx),
            "context_short_because": short, **extra}


def labels(**over):
    base = {STUDENT: "Task automation", TOPIC: ["code"], AIDEL: ["offer"], AIRESP: ["Solve-Provide", "Explain"]}
    base.update(over)
    return base


def tiny_backbone():
    from transformers import Qwen3Config, Qwen3Model
    return Qwen3Model(Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                  max_position_embeddings=1024, tie_word_embeddings=False))


def tiny_config(**over):
    cfg = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    cfg.update(device="cpu", seed=11)
    cfg["input"].update(max_tokens=600, prompt_cap=160, response_cap=200, min_context_fragment=64)
    cfg["lora"].update(r=4, alpha=8)
    cfg["model"].update(gradient_checkpointing=False)
    cfg["schedule"].update(epochs=2, grad_accumulation=3)
    cfg["checkpoint"].update(every_steps=2, keep_last=2)
    cfg.update(over)
    return cfg


def review_reference(directory, decision="D-105", name="review.jsonl"):
    """A reference whose source is a one-record synthetic AI-assisted review file (D-105 or D-144)."""
    import test_prepare_data as TP
    record = {"D-105": TP.reviewed, "D-144": TP.reviewed_v9}[decision]()
    path = Path(directory) / name
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return {"sources": [{"path": str(path), "sha256": L.sha256_file(path), "record_kind": L.DEV_REVIEW_KIND,
                         "records": 1}]}


def prepared_dir(root, rows):
    """A synthetic prepared directory below root/data; returns (directory, manifest SHA-256)."""
    prepared = root / "data" / "prepared"
    prepared.mkdir(parents=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    with (prepared / "examples.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    keys = sorted([r["conversation_id"], r["segment_order"]] for r in rows)
    (prepared / "manifest.json").write_text(json.dumps({
        "prepared_file": {"sha256": L.sha256_bytes(text.encode())}, "keys_sha256": L.sha256_bytes(json.dumps(keys).encode()),
        "vocabulary_sha256": L.vocabulary_sha256(), "sources": [{"record_kind": "synthetic"}]}), encoding="utf-8")
    return prepared, L.sha256_file(prepared / "manifest.json")


class Inputs(unittest.TestCase):
    def test_short_input_rendering_is_exact(self):
        b = MI.build_input(example(seg=4, context=[("c1", "a1")]), encode, POLICY)
        self.assertEqual(decode(b["input_ids"]),
                         "[earlier interaction 3]\nStudent: c1\nAssistant: a1\n[interaction to code]\nStudent: p"
                         "\n[end of student prompt]\nAssistant: r\n[end of assistant response]")
        self.assertTrue(decode(b["input_ids"][:b["student_pool"] + 1]).endswith("[end of student prompt]"))
        self.assertEqual(b["ai_pool"], len(b["input_ids"]) - 1)
        self.assertEqual(b["report"]["context_included"], 1)

    def test_annotations_never_enter_the_input(self):
        plain = example(context=[("c1", "a1"), ("c2", "a2")])
        annotated = dict(copy.deepcopy(plain), labels=labels(), targets=L.encode_targets(labels()),
                         notes="CANARY-NOTE", flags=["needs discussion"], masked_axes=[],
                         source_record={"notes": "CANARY-SOURCE", "review_history": [{"original_notes": "CANARY-H"}],
                                        "labels": labels()})
        self.assertEqual(MI.build_input(plain, encode, POLICY), MI.build_input(annotated, encode, POLICY))
        self.assertNotIn("CANARY", decode(MI.build_input(annotated, encode, POLICY)["input_ids"]))

    def test_budget_priority_head_tail_and_markers(self):
        ex = example(prompt="HEAD-" + "P" * 1000 + "-TAIL", response="START-" + "R" * 1000 + "-END",
                     context=[("old " * 60, "x" * 200), ("mid " * 60, "y" * 200), ("new " * 30, "z" * 100)])
        b = MI.build_input(ex, encode, POLICY)
        text, r = decode(b["input_ids"]), b["report"]
        self.assertLessEqual(len(b["input_ids"]), POLICY.max_tokens)
        self.assertGreater(r["prompt_tokens_omitted"], 0)
        self.assertGreater(r["response_tokens_omitted"], 0)
        self.assertTrue(text.split("[interaction to code]")[1].startswith("\nStudent: HEAD-"))
        self.assertIn("-TAIL\n[end of student prompt]", text)
        self.assertIn("START-", text)
        self.assertTrue(text.endswith("-END\n[end of assistant response]"))
        self.assertIn("tokens omitted", text)
        self.assertGreaterEqual(r["context_dropped"], 1)
        self.assertIn(f"[{r['context_dropped']} earlier interaction(s) omitted for length]", text)
        if not r["context_clipped"]:                    # every omitted context token is counted exactly
            dropped = ex["context"][:r["context_dropped"]]
            self.assertEqual(r["context_tokens_omitted"],
                             sum(len(encode(c["student_prompt"])) + len(encode(c["llm_response"])) for c in dropped))
        if r["context_included"]:                       # what is kept is the most recent
            self.assertIn("[earlier interaction 3]", text)
            self.assertNotIn("[earlier interaction 1]", text)

    def test_missing_context_is_marked_only_when_turns_existed(self):
        gap = MI.build_input(example(context=[("c", "a")], short="gap"), encode, POLICY)
        opens = MI.build_input(example(seg=2, context=[("c", "a")], short="conversation opens"), encode, POLICY)
        self.assertTrue(decode(gap["input_ids"]).startswith("[earlier turns not available]\n"))
        self.assertNotIn("not available", decode(opens["input_ids"]))

    def test_context_must_be_the_adjacent_earlier_run(self):
        later = example(seg=4)
        later["context"] = [{"segment_order": 5, "student_prompt": "future", "llm_response": "future"}]
        skipped = example(seg=4)
        skipped["context"] = [{"segment_order": 2, "student_prompt": "x", "llm_response": "y"}]
        for bad in (later, skipped):
            with self.assertRaises(ValueError):
                MI.build_input(bad, encode, POLICY)

    def test_clip_keeps_head_and_tail_within_cap(self):
        ids, omitted = MI.clip(list(range(200)), 60, encode)
        self.assertLessEqual(len(ids), 60)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[-1], 199)
        self.assertEqual(omitted, 200 - (len(ids) - len(encode(MI.T_ELISION.format(n=omitted)))))
        self.assertEqual(MI.clip([1, 2], 5, encode), ([1, 2], 0))

    def test_every_available_earlier_interaction_up_to_the_limit_is_kept(self):
        for n in range(0, 5):                     # fewer, as many as and more than the limit of 3
            ex = example(seg=6, context=[(f"c{i}", f"a{i}") for i in range(n)])
            b = MI.build_input(ex, encode, POLICY)
            r = b["report"]
            self.assertEqual((r["context_prepared"], r["context_within_limit"], r["context_included"]),
                             (n, min(n, 3), min(n, 3)), n)
            text = decode(b["input_ids"])
            kept = [f"[earlier interaction {6 - n + i}]" for i in range(n)][-3:]
            self.assertTrue(all(h in text for h in kept), (n, text))

    def test_context_limit_zero_uses_no_context(self):
        b = MI.build_input(example(context=[("c1", "a1"), ("c2", "a2")]), encode, MI.InputPolicy(600, 0, 160, 200, 64))
        self.assertEqual((b["report"]["context_within_limit"], b["report"]["context_included"]), (0, 0))
        self.assertNotIn("earlier interaction", decode(b["input_ids"]))

    def test_student_side_never_depends_on_the_current_response(self):
        ctx = [("old " * 40, "x" * 180), ("mid " * 40, "y" * 180), ("new " * 40, "z" * 180)]
        builds = [MI.build_input(example(prompt="same prompt", response=resp, context=ctx), encode, POLICY)
                  for resp in ("r", "R" * 150, "S" * 199, "T" * 5000)]
        prefixes = {tuple(b["input_ids"][:b["student_pool"] + 1]) for b in builds}
        self.assertEqual(len(prefixes), 1)                     # identical student-side tokens
        self.assertTrue(any(b["report"]["context_tokens_omitted"] for b in builds))   # the budget binds
        self.assertGreater(builds[-1]["report"]["response_tokens_omitted"], 0)
        for b in builds:
            self.assertLessEqual(len(b["input_ids"]), POLICY.max_tokens)

    def test_policy_is_checked_against_the_tokenizer(self):
        costs = POLICY.check_with(encode)
        self.assertEqual(costs["student_side_budget"], POLICY.max_tokens - costs["response_reservation"])
        for bad in (MI.InputPolicy(600, 3, 20, 200, 64),       # cap cannot hold an elision marker
                    MI.InputPolicy(400, 3, 160, 200, 64)):      # fixed costs exceed max_tokens
            with self.assertRaises(ValueError):
                bad.check_with(encode)
        for args in ((600, -1, 160, 200, 64), (600, 3, 0, 200, 64), (600, 3, 300, 300, 64), (600, True, 160, 200, 64),
                     (600.0, 3, 160, 200, 64)):
            with self.assertRaises(ValueError, msg=args):      # refused however the policy is built
                MI.InputPolicy(*args)


class Evaluation(unittest.TestCase):
    def test_metrics_on_a_hand_checked_case(self):
        k = [("SYN_E", i) for i in range(1, 5)]
        refs = {k[0]: labels(), k[1]: labels(**{AIDEL: ["ND"]}),
                k[2]: labels(**{STUDENT: "Information retrieval", TOPIC: ["code", "concept"]}),
                k[3]: labels(**{STUDENT: "Guided generation", AIDEL: None, AIRESP: None})}
        preds = {k[0]: labels(), k[1]: labels(**{STUDENT: "Information retrieval", AIDEL: ["offer"]}),
                 k[2]: labels(**{STUDENT: "Information retrieval", TOPIC: ["code"]}),
                 k[3]: labels(**{STUDENT: "Task automation"})}
        r = EV.evaluate(refs, preds)
        s = r["axes"][STUDENT]
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["row_accuracy"], 0.5)
        self.assertAlmostEqual(s["per_category"]["Task automation"]["f1"], 0.5)
        self.assertAlmostEqual(s["per_category"]["Information retrieval"]["f1"], 2 / 3)
        self.assertEqual(s["per_category"]["Guided generation"]["f1"], 0.0)
        self.assertIsNone(s["per_category"]["Guided generation"]["precision"])
        self.assertNotIn("Providing material / responding", s["macro_f1_categories"])
        self.assertAlmostEqual(s["macro_f1"], (0.5 + 2 / 3 + 0.0) / 3)
        self.assertEqual(s["confusion_matrix"]["counts"][2][1], 1)    # TA reference predicted IR
        d = r["axes"][AIDEL]
        self.assertEqual((d["n"], d["excluded"]["reference_null"]), (3, 1))   # null excluded, not negative
        self.assertAlmostEqual(d["row_accuracy"], 2 / 3)
        self.assertEqual(d["per_category"]["ND"], {**d["per_category"]["ND"], "tp": 0, "fn": 1, "fp": 0})
        self.assertIn("hand-back: Socratic", d["zero_support_categories"])
        self.assertIn({"reference": "ND", "prediction": "offer", "count": 1}, d["joint_set_distribution"])
        t = r["axes"][TOPIC]
        self.assertEqual(t["per_category"]["concept"]["fn"], 1)
        self.assertIn("conversation management", t["rare_categories"])
        self.assertEqual(r["all_axis_exact_match"], {"n": 3, "matches": 1, "rows_with_null_prediction": 0, "rate": 1 / 3})
        self.assertIn("Development", EV.markdown({"title": "Development check", "statement": "s", **r}))

    def test_set_order_null_predictions_and_union_macro(self):
        k1, k2 = ("SYN_E", 1), ("SYN_E", 2)
        refs = {k1: labels(**{TOPIC: ["code", "concept"]}), k2: labels()}
        preds = {k1: labels(**{TOPIC: ["concept", "code"]}), k2: labels(**{AIDEL: None})}
        r = EV.evaluate(refs, preds)
        self.assertIn({"reference": "code + concept", "prediction": "code + concept", "count": 1},
                      r["axes"][TOPIC]["joint_set_distribution"])
        self.assertNotIn("concept + code", json.dumps(r["axes"][TOPIC]))
        self.assertEqual(r["all_axis_exact_match"], {"n": 2, "matches": 1, "rows_with_null_prediction": 1, "rate": 0.5})
        self.assertEqual(r["axes"][AIDEL]["excluded"]["prediction_null"], 1)
        only_predicted = EV.evaluate({k1: labels()}, {k1: labels(**{AIDEL: ["offer", "hand-back: Socratic"]})})
        d = only_predicted["axes"][AIDEL]
        self.assertEqual(d["macro_f1"], 1.0)                       # over reference-present codes
        self.assertEqual(d["macro_f1_union"], 0.5)                 # Socratic predicted only: F1 0
        self.assertEqual(len(r["evaluation_module_sha256"]), 64)

    def test_missing_or_invalid_predictions_refused(self):
        refs = {("SYN_E", 1): labels()}
        with self.assertRaises(ValueError):
            EV.evaluate(refs, {})
        with self.assertRaises(ValueError):
            EV.evaluate(refs, {("SYN_E", 1): labels(**{AIDEL: ["ND", "offer"]})})


@unittest.skipUnless(HAVE_TORCH, "model stack not installed")
class ModelContract(unittest.TestCase):
    def model(self):
        torch.manual_seed(0)
        m = M.MultiAxisClassifier(M.attach_lora(tiny_backbone(), tiny_config()["lora"]), 32, 0.1, 0.02)
        with torch.no_grad():                          # make the adapter matter
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.normal_(std=0.05)
        return m

    def forward(self, m, ex):
        b = MI.build_input(ex, encode, POLICY)
        return m(torch.tensor([b["input_ids"]]), b["student_pool"], b["ai_pool"])

    def test_student_heads_cannot_see_the_current_response(self):
        m = self.model().eval()
        ctx = [("earlier", "answer")]
        a = self.forward(m, example(prompt="same prompt", response="first answer!", context=ctx))
        b = self.forward(m, example(prompt="same prompt", response="other reply??", context=ctx))
        c = self.forward(m, example(prompt="same prompt", response="a much longer and different reply", context=ctx))
        for axis in M.STUDENT_AXES:
            self.assertTrue(torch.equal(a[axis], b[axis]), axis)
            self.assertTrue(torch.allclose(a[axis], c[axis], atol=1e-6), axis)
        self.assertTrue(any(not torch.equal(a[x], b[x]) for x in M.AI_AXES))
        d = self.forward(m, example(prompt="another prompt", response="first answer!", context=ctx))
        self.assertTrue(any(not torch.equal(a[x], d[x]) for x in M.STUDENT_AXES))

    def test_null_labels_contribute_no_loss_and_no_gradient(self):
        m = self.model().eval()
        logits = self.forward(m, example(context=[("c", "a")]))
        masked = L.encode_targets(labels(**{AIDEL: None, AIRESP: None}))
        loss, parts = M.example_loss(logits, masked)
        self.assertIsNone(parts[AIDEL])
        self.assertIsNone(parts[AIRESP])
        poisoned = copy.deepcopy(masked)              # whatever sits in a masked target is unread
        poisoned[AIDEL]["multi_hot"] = [1] * 5
        poisoned[AIRESP].update(multi_hot=[0, 0, 0, 0, 1])
        self.assertTrue(torch.equal(loss, M.example_loss(logits, poisoned)[0]))
        loss.backward()
        for axis in M.AI_AXES:
            g = m.heads[M.SLUG[axis]].weight.grad
            self.assertTrue(g is None or not g.any(), axis)
        for axis in M.STUDENT_AXES:
            self.assertTrue(m.heads[M.SLUG[axis]].weight.grad.any(), axis)
        observed_nd = L.encode_targets(labels(**{AIDEL: ["ND"], AIRESP: None}))
        self.assertIsNotNone(M.example_loss(logits, observed_nd)[1][AIDEL])   # ND is a real label
        all_null = L.encode_targets({a: None for a in L.AXES})
        self.assertIsNone(M.example_loss(logits, all_null)[0])

    def test_config_guards(self):
        TM.check_config(json.loads(PILOT_CONFIG.read_text(encoding="utf-8")))
        for change in ({"device": "mps", "lora": {**tiny_config()["lora"], "dropout": 0.05},
                        "model": {**tiny_config()["model"], "gradient_checkpointing": True}},
                       {"validation": {**tiny_config()["validation"], "used_for_selection": True}},
                       {"schedule": {**tiny_config()["schedule"], "micro_batch": 2}},
                       {"optim": {**tiny_config()["optim"], "name": "SGD"}},
                       {"loss": {**tiny_config()["loss"], "class_weights": [5, 1, 1, 1]}},
                       {"model": {**tiny_config()["model"], "dtype": "bfloat16"}},
                       {"validation": {**tiny_config()["validation"], "evaluated_state": "best validation loss"}},
                       {"decode": {**tiny_config()["decode"], "single_select": "top-2"}}):
            with self.assertRaises(ValueError, msg=str(change)):
                TM.check_config(tiny_config(**change))

    def test_report_statements_follow_the_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = TM.statements(review_reference(tmp))
        self.assertIn("基于待复核研究者标注的开发验证", reviewed["validation"])
        self.assertIn("非独立编码", reviewed["validation"])
        self.assertIn("not joint adjudication", reviewed["validation"])
        original = TM.statements({"sources": [{"record_kind": "human_independent"}]})
        self.assertNotIn("D-105", original["validation"])
        with self.assertRaises(ValueError):
            TM.statements({"sources": [{"record_kind": "human_adjudicated"}]})
        with self.assertRaises(ValueError):
            TM.statements({"sources": []})

    def test_review_statement_names_the_decision_read_from_the_review_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d105, d144 = TM.statements(review_reference(tmp, "D-105", "a.jsonl")), \
                TM.statements(review_reference(tmp, "D-144", "b.jsonl"))
            for part in ("validation", "train", "prior", "all"):
                self.assertIn("D-105 AI辅助复核版（待复核；非独立编码；非共同裁定）", d105[part])
                self.assertIn("the D-105 AI-assisted review", d105[part])
                self.assertNotIn("D-144", d105[part])
                self.assertIn("D-144 AI辅助复核版（待复核；非独立编码；非共同裁定）", d144[part])
                self.assertIn("the D-144 AI-assisted review", d144[part])
                self.assertNotIn("D-105", d144[part])
            root = Path(tmp).resolve()
            relative = review_reference(root, "D-144", "c.jsonl")
            relative["sources"][0]["path"] = "c.jsonl"              # manifests store repository-relative paths
            with patch.object(L, "ROOT", root):
                self.assertEqual(EV.review_decisions(relative), "D-144")
            changed = review_reference(tmp, "D-105", "d.jsonl")
            Path(changed["sources"][0]["path"]).write_text(
                json.dumps(dict(json.loads(Path(changed["sources"][0]["path"]).read_text(encoding="utf-8")), notes="edited")) + "\n", encoding="utf-8")
            missing = review_reference(tmp, "D-105", "e.jsonl")
            Path(missing["sources"][0]["path"]).unlink()
            unnamed = {"sources": [{"record_kind": L.DEV_REVIEW_KIND}]}
            empty = review_reference(tmp, "D-105", "f.jsonl")
            Path(empty["sources"][0]["path"]).write_text("", encoding="utf-8")
            empty["sources"][0]["sha256"] = L.sha256_file(empty["sources"][0]["path"])     # no record, no decision
            for bad in (changed, missing, unnamed, empty):
                with self.assertRaises(ValueError, msg=str(bad)):
                    TM.statements(bad)

    @unittest.skipUnless(HAVE_TORCH and torch.backends.mps.is_available(), "MPS not available")
    def test_mps_checkpointed_gradients_match_without_adapter_dropout(self):
        grads = []
        for checkpointing in (False, True):
            torch.manual_seed(0)
            backbone = tiny_backbone()
            if checkpointing:
                backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m = M.MultiAxisClassifier(M.attach_lora(backbone, tiny_config()["lora"]), 32, 0.0, 0.02).to("mps").train()
            b = MI.build_input(example(context=[("c", "a")]), encode, POLICY)
            logits = m(torch.tensor([b["input_ids"]], device="mps"), b["student_pool"], b["ai_pool"])
            M.example_loss(logits, L.encode_targets(labels()))[0].backward()
            grads.append({n: p.grad.cpu() for n, p in m.named_parameters() if p.requires_grad})
        for n in grads[0]:
            self.assertTrue(torch.equal(grads[0][n], grads[1][n]), n)


@unittest.skipUnless(HAVE_TORCH, "model stack not installed")
class TrainingRuns(unittest.TestCase):
    """Uninterrupted, stopped-and-resumed and crash-and-resumed runs must end in the same state."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        examples = TM.synthetic_examples(seed=3, sizes=(5, 3, 2, 2, 1, 1, 1, 1))
        for e in examples:            # SYN_C00 has 5 rows > validation_max 4, so it always trains
            if (e["conversation_id"], e["segment_order"]) == ("SYN_C00", 3):
                e.update(labels={a: None for a in L.AXES}, masked_axes=list(L.AXES),
                         mask_reasons={a: "synthetic all-null row" for a in L.AXES})
                e["targets"] = L.encode_targets(e["labels"])
        prepared, split = TM.write_synthetic_inputs(cls.d / "inputs", examples, 11, 3, 4)
        cls.data = TM.load_data(prepared, split)
        cls.cfg = tiny_config()
        cls.cpu = torch.device("cpu")
        cls.full = cls.train_run("full")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tearDown(self):
        TM._Stop.signal = None                         # a stop request never leaks into the next test

    @classmethod
    def train_run(cls, name, resume=False, stop=None, cfg=None, enc=encode):
        return TM.run_training(cls.d / name, cfg or cls.cfg, cls.data, IDENTITY, tiny_backbone, enc, cls.cpu,
                               resume=resume, stop_after_steps=stop, synthetic=True, command=["unittest"])

    def final_state(self, name):
        from safetensors.torch import load_file
        ckpt = TM.checkpoint_path(self.d / name, "final")
        progress = json.loads((ckpt / "progress.json").read_text(encoding="utf-8"))
        optimizer = torch.load(ckpt / "optimizer.pt", weights_only=True)
        return load_file(ckpt / "trainable.safetensors"), progress, optimizer

    def assert_same_final_state(self, name):
        a_state, a_prog, a_opt = self.final_state("full")
        b_state, b_prog, b_opt = self.final_state(name)
        self.assertEqual(set(a_state), set(b_state))
        for n in a_state:
            self.assertTrue(torch.equal(a_state[n], b_state[n]), n)
        for field in ("loss_mean", "lr", "grad_norm", "axis_loss_mean"):
            self.assertEqual([s[field] for s in a_prog["steps"]], [s[field] for s in b_prog["steps"]], field)
        self.assertEqual(a_prog["epochs"], b_prog["epochs"])
        for pid, st in a_opt["state"].items():
            for key, value in st.items():
                self.assertTrue(torch.equal(value, b_opt["state"][pid][key]), (pid, key))
        pa = [json.loads(x) for x in (self.d / "full" / "outputs" / "predictions-validation.jsonl").read_text(encoding="utf-8").splitlines()]
        pb = [json.loads(x) for x in (self.d / name / "outputs" / "predictions-validation.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["labels"] for r in pa], [r["labels"] for r in pb])

    def scramble_global_rng(self):                     # a new process starts from unrelated RNG state
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)

    def test_full_run_trains_predicts_and_evaluates(self):
        self.assertEqual(self.full["outcome"], "complete")
        run = self.d / "full"
        prog = self.final_state("full")[1]
        self.assertEqual(prog["global_step"], prog["plan"]["total_steps"])
        self.assertTrue(prog["complete"])
        for name in ("predictions-validation.jsonl", "scores-validation.jsonl", "evaluation-validation.json",
                     "evaluation-validation.md", "evaluation-train-fit.json",
                     "evaluation-validation-prior-reference.json", "summary.json"):
            self.assertTrue((run / "outputs" / name).exists(), name)
        records = L.read_predictions(run / "outputs" / "predictions-validation.jsonl")
        self.assertTrue(all(r["record_kind"] == L.PREDICTION_KIND for r in records))
        with self.assertRaises(ValueError):            # never readable as a human record
            L.read_direct(run / "outputs" / "predictions-validation.jsonl")
        report = json.loads((run / "outputs" / "evaluation-validation.json").read_text(encoding="utf-8"))
        self.assertEqual(report["rows"], len(self.data.keys("validation")))
        self.assertIn("Synthetic", report["statement"])
        record = json.loads((run / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["fingerprint_sha256"], TM.canonical_sha256(record["fingerprint"]))
        inputs = json.loads((run / "inputs-report.json").read_text(encoding="utf-8"))
        self.assertTrue(any(r["prompt_tokens_omitted"] for r in inputs["per_interaction"].values()))

    def test_stop_then_resume_matches_uninterrupted_run(self):
        first = self.train_run("stopped", stop=3)
        self.assertEqual(first["outcome"], "interrupted")
        self.scramble_global_rng()
        self.assertEqual(self.train_run("stopped", resume=True)["outcome"], "complete")
        self.assert_same_final_state("stopped")

    def test_hard_kill_then_resume_matches_uninterrupted_run(self):
        self.train_run("killed", stop=7)
        root = self.d / "killed" / "checkpoints"
        saved = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("s"))
        # As if killed after step 7's directory was renamed into place but before latest.json moved,
        # with a half-written directory left beside it.
        (root / "latest.json").write_text(json.dumps({"tag": saved[-2], "global_step": 6, "complete": False}), encoding="utf-8")
        (root / ".s000099-e09-g009.partial").mkdir()
        self.scramble_global_rng()
        self.train_run("killed", resume=True)
        self.assertFalse((root / ".s000099-e09-g009.partial").exists())
        self.assert_same_final_state("killed")
        log = [json.loads(x) for x in (self.d / "killed" / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(e["event"] == "removed_unreferenced_checkpoint" for e in log))

    def test_checkpoint_tags_of_long_runs_remain_addressable(self):
        for progress in ({"global_step": 1650, "epoch": 1, "group": 1050}, {"global_step": 144000, "epoch": 120, "group": 0},
                         {"global_step": 3, "epoch": 0, "group": 3}):
            self.assertTrue(TM.TAG.match(TM.checkpoint_tag(progress)), progress)
        for bad in ("../s000001-e00-g001", "/abs/s000001-e00-g001", "s1-e0-g0", "final.json"):
            self.assertIsNone(TM.TAG.match(bad), bad)

    def test_checkpoint_is_complete_and_tamper_evident(self):
        self.train_run("inspect", stop=2)
        ckpt = TM.checkpoint_path(self.d / "inspect", "latest")
        for name in ("trainable.safetensors", "heads.safetensors", "adapter/adapter_model.safetensors",
                     "adapter/adapter_config.json", "optimizer.pt", "scheduler.pt", "rng.json", "rng.safetensors",
                     "progress.json", "COMPLETE.json"):
            self.assertTrue((ckpt / name).exists(), name)
        progress = json.loads((ckpt / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual((progress["global_step"], progress["epoch"], progress["group"]), (2, 0, 2))
        self.assertEqual(progress["order"], TM.epoch_order(self.cfg["seed"], 0, self.data.keys("train")))
        (ckpt / "optimizer.pt").rename(ckpt / "moved.pt")    # an adapter alone is not a resumable state
        with self.assertRaises(ValueError):
            TM.verify_checkpoint(ckpt)

    def test_reloaded_final_state_predicts_identically_under_any_file_name(self):
        out = self.d / "reloaded.jsonl"
        result = TM.predict_from_checkpoint(self.d / "full", self.data, IDENTITY, tiny_backbone, encode, self.cpu,
                                            "validation", out)
        self.assertEqual(Path(result["scores"]).name, "reloaded.scores.jsonl")
        fresh = {(r["conversation_id"], r["segment_order"]): r for r in L.read_predictions(out)}
        made = {(r["conversation_id"], r["segment_order"]): r
                for r in L.read_predictions(self.d / "full" / "outputs" / "predictions-validation.jsonl")}
        self.assertEqual({k: v["labels"] for k, v in fresh.items()}, {k: v["labels"] for k, v in made.items()})
        self.assertTrue(all(v["model_route"].endswith("#" + result["checkpoint"]) for v in fresh.values()))
        scores = [json.loads(x) for x in Path(result["scores"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(scores), len(fresh))
        for row in scores:
            record = fresh[(row["conversation_id"], row["segment_order"])]
            self.assertEqual((row["record_kind"], row["schema_version"], row["vocabulary_sha256"]),
                             ("model_scores", L.DIRECT_VERSION, L.vocabulary_sha256()))
            self.assertEqual({f: row[f] for f in L.RUN_FIELDS}, {f: record[f] for f in L.RUN_FIELDS})
            self.assertEqual(row["codes"], {a: L.DIRECT_CODES[a] for a in L.AXES})
            self.assertEqual({a: len(v) for a, v in row["probabilities"].items()}, {a: len(L.DIRECT_CODES[a]) for a in L.AXES})
        with self.assertRaises(FileExistsError):
            TM.predict_from_checkpoint(self.d / "full", self.data, IDENTITY, tiny_backbone, encode, self.cpu,
                                       "validation", out)
        summary = json.loads((self.d / "full" / "outputs" / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("Synthetic", summary["evaluation-validation"]["statement"])
        report = json.loads((self.d / "full" / "outputs" / "evaluation-validation.json").read_text(encoding="utf-8"))
        self.assertTrue((ROOT_DIR / report["predictions"]["path"]).exists() or Path(report["predictions"]["path"]).exists())

    def test_resume_refuses_changed_config_or_inputs(self):
        self.train_run("guarded", stop=2)
        changed = copy.deepcopy(self.cfg)
        changed["optim"]["lr"] = 0.01
        with self.assertRaisesRegex(ValueError, "config_sha256"):
            self.train_run("guarded", resume=True, cfg=changed)
        with self.assertRaisesRegex(ValueError, "inputs_sha256"):
            self.train_run("guarded", resume=True, enc=lambda text: encode(text.upper()))
        with self.assertRaises(FileExistsError):
            self.train_run("guarded")

    # Refusals, bookkeeping and recovery paths found in review 2.

    def test_step_objective_and_epoch_aggregates(self):
        prog = self.final_state("full")[1]
        for st in prog["steps"]:
            self.assertEqual(st["loss_mean"], st["loss_sum"] / st["examples"])
        self.assertTrue(any(st["examples_with_labels"] < st["examples"] for st in prog["steps"]))   # all-null row seen
        for ep in prog["epochs"]:
            mine = [st for st in prog["steps"] if st["epoch"] == ep["epoch"]]
            self.assertAlmostEqual(ep["train_during_epoch"]["loss_mean"],
                                   sum(x["loss_sum"] for x in mine) / sum(x["examples"] for x in mine), places=12)
        # Replay step 1 independently: its gradient is the MEAN loss over the whole group.
        model, _, _, _ = TM.build_training_objects(self.cfg, tiny_backbone, self.cpu, len(self.data.keys("train")))
        model.train()
        policy = MI.InputPolicy.from_config(self.cfg["input"])
        group = TM.epoch_order(self.cfg["seed"], 0, self.data.keys("train"))[:self.cfg["schedule"]["grad_accumulation"]]
        for text in group:
            key = L.parse_key(text)
            b = MI.build_input(self.data.examples[key], encode, policy)
            loss, _ = M.example_loss(model(torch.tensor([b["input_ids"]]), b["student_pool"], b["ai_pool"]),
                                     self.data.targets[key])
            if loss is not None:
                (loss / len(group)).backward()
        norm = float(torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], float("inf")))
        self.assertAlmostEqual(norm, prog["steps"][0]["grad_norm"], places=5)

    def test_start_refusals_leave_no_run_directory(self):
        prepared = self.data.fingerprint["prepared"]
        support = copy.deepcopy(self.data.summary)
        support["train"]["label_support"][AIDEL]["labelled"] = 0
        small_cap = copy.deepcopy(self.cfg)
        small_cap["input"]["prompt_cap"] = 30
        cases = {
            "unhonourable policy": (small_cap, self.data, True),
            "window too small": (self.cfg, dataclasses.replace(self.data, fingerprint={
                **self.data.fingerprint, "prepared": {**prepared, "context_interactions": 2}}), True),
            "window unknown": (self.cfg, dataclasses.replace(self.data, fingerprint={
                **self.data.fingerprint, "prepared": {**prepared, "context_interactions": None}}), True),
            "no validation": (self.cfg, dataclasses.replace(self.data, part_of={k: "train" for k in self.data.part_of}), True),
            "no training": (self.cfg, dataclasses.replace(self.data, part_of={k: "validation" for k in self.data.part_of}), True),
            "purpose differs": (self.cfg, self.data, False),
            "undescribed reference": (self.cfg, dataclasses.replace(self.data, fingerprint={
                **self.data.fingerprint, "prepared": {**prepared, "sources": [{"record_kind": "human_adjudicated"}]}}), True),
            "no reference kinds": (self.cfg, dataclasses.replace(self.data, fingerprint={
                **self.data.fingerprint, "prepared": {**prepared, "sources": []}}), True),
            "axis never labelled in training": (self.cfg, dataclasses.replace(self.data, summary=support), True),
        }
        for i, (name, (cfg, data, synthetic)) in enumerate(cases.items()):
            target = self.d / f"refused-{i}"
            with self.assertRaises(ValueError, msg=name):
                TM.run_training(target, cfg, data, IDENTITY, tiny_backbone, encode, self.cpu, synthetic=synthetic,
                                command=["unittest"])
            self.assertFalse(target.exists(), name)

    def test_predict_refuses_foreign_checkpoints_changed_code_and_taken_targets(self):
        other = copy.deepcopy(self.cfg)
        other["optim"]["lr"] = 0.01
        self.train_run("other", cfg=other)
        foreign = TM.checkpoint_path(self.d / "other", "final")
        args = (self.data, IDENTITY, tiny_backbone, encode, self.cpu, "validation")
        for i, which in enumerate((str(foreign.resolve()), "../../other/checkpoints/" + foreign.name, "s1", "nothing")):
            with self.assertRaises(ValueError, msg=which):
                TM.predict_from_checkpoint(self.d / "full", *args, self.d / f"foreign-{i}.jsonl", which)
        planted = self.d / "full" / "checkpoints" / "s000099-e09-g009"
        shutil.copytree(foreign, planted)
        try:
            with self.assertRaisesRegex(ValueError, "records the tag"):
                TM.predict_from_checkpoint(self.d / "full", *args, self.d / "planted.jsonl", planted.name)
        finally:
            shutil.rmtree(planted)
        own = TM.checkpoint_path(self.d / "full", "final")
        outside = self.d / "elsewhere" / own.name
        shutil.copytree(own, outside)                 # same run, same fingerprint, but not inside the run
        for which in (str(outside.resolve()), "s000999-e09-g009"):
            with self.assertRaisesRegex(ValueError, "is not a checkpoint of", msg=which):
                TM.predict_from_checkpoint(self.d / "full", *args, self.d / "outside.jsonl", which)
        self.train_run("twin")                        # identical config, data and code: same fingerprint
        swapped = self.d / "full-swapped"
        shutil.copytree(self.d / "full", swapped)
        shutil.rmtree(swapped / "checkpoints" / own.name)
        shutil.copytree(TM.checkpoint_path(self.d / "twin", "final"), swapped / "checkpoints" / own.name)
        with self.assertRaisesRegex(ValueError, "does not belong"):
            TM.predict_from_checkpoint(swapped, *args, self.d / "swapped.jsonl")
        tampered = self.d / "full-tampered"
        shutil.copytree(self.d / "full", tampered)
        record = json.loads((tampered / "run.json").read_text(encoding="utf-8"))
        record["fingerprint"]["code_sha256"]["finetuning/evaluation.py"] = "0" * 64
        (tampered / "run.json").write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "code_sha256"):
            TM.predict_from_checkpoint(tampered, *args, self.d / "tampered.jsonl")
        out = self.d / "predictions-taken.jsonl"
        TM.scores_path(out).write_text("already here", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            TM.predict_from_checkpoint(self.d / "full", *args, out)
        self.assertFalse(out.exists())

    def test_cli_keeps_private_predictions_in_data_and_explains_missing_runs(self):
        private = self.d / "full-private"
        shutil.copytree(self.d / "full", private)
        record = json.loads((private / "run.json").read_text(encoding="utf-8"))
        record["synthetic"] = False
        (private / "run.json").write_text(json.dumps(record), encoding="utf-8")
        out = self.d / "outside.jsonl"
        saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
        cases = ((["predict", "--run-dir", str(private), "--out", str(out)], "below repository data/"),
                 (["predict", "--run-dir", str(self.d / "never-started"), "--out", str(out)], "no run.json"),
                 (["resume", "--run-dir", str(self.d / "never-started")], "no run.json"),
                 (["train", "--config", str(PILOT_CONFIG), "--run-dir", str(self.d / "outside-data")],
                  "below repository data/"))
        try:
            with patch.object(TM, "_real_inputs", side_effect=AssertionError("no data may be read before the guard")):
                for argv, reason in cases:
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop:
                        TM.main(argv)
                    self.assertEqual(stop.exception.code, 2, argv)
                    self.assertIn(reason, err.getvalue(), argv)
        finally:
            signal.signal(signal.SIGINT, saved[0])
            signal.signal(signal.SIGTERM, saved[1])
        self.assertFalse(out.exists())
        self.assertFalse((self.d / "outside-data").exists())

    def test_finder_metadata_does_not_block_resume_but_other_files_do(self):
        self.train_run("finder", stop=2)
        ckpt = TM.checkpoint_path(self.d / "finder", "latest")
        (ckpt / ".DS_Store").write_bytes(b"\0")
        (ckpt / "adapter" / "._adapter_model.safetensors").write_bytes(b"\0")
        TM.verify_checkpoint(ckpt)
        self.assertEqual(self.train_run("finder", resume=True)["outcome"], "complete")
        final = TM.checkpoint_path(self.d / "finder", "final")
        for name in ("unexpected.bin", ".hidden"):
            (final / name).write_bytes(b"\0")
            with self.assertRaises(ValueError, msg=name):
                TM.verify_checkpoint(final)
            (final / name).unlink()

    def test_unreadable_final_pointer_is_repaired_on_resume(self):
        repaired = self.d / "final-repair"
        shutil.copytree(self.d / "full", repaired)
        (repaired / "checkpoints" / "final.json").write_text("", encoding="utf-8")
        self.assertEqual(self.train_run("final-repair", resume=True)["outcome"], "complete")
        self.assertEqual(json.loads((repaired / "checkpoints" / "final.json").read_text(encoding="utf-8"))["tag"],
                         TM.checkpoint_path(self.d / "full", "final").name)

    def test_torn_pointer_writes_leave_the_previous_pointer_usable(self):
        real = TM.write_json
        for name in ("latest.json", "final.json"):
            run = f"torn-{name}"
            self.train_run(run, stop=2)

            def torn(path, obj, exclusive=True, _target=f".{name}.partial"):
                if Path(path).name == _target:
                    Path(path).write_text('{"tag": "', encoding="utf-8")
                    raise OSError("simulated crash in the middle of a pointer write")
                return real(path, obj, exclusive)
            with patch.object(TM, "write_json", torn), self.assertRaises(OSError):
                self.train_run(run, resume=True)
            latest = json.loads((self.d / run / "checkpoints" / "latest.json").read_text(encoding="utf-8"))   # still valid JSON
            self.assertGreaterEqual(latest["global_step"], 2)
            self.assertEqual(self.train_run(run, resume=True)["outcome"], "complete")
            self.assert_same_final_state(run)

    def test_a_start_killed_after_its_first_checkpoint_starts_again_cleanly(self):
        real = TM.write_json_atomic

        def no_run_json(path, obj):
            if Path(path).name == "run.json":
                raise RuntimeError("killed just before run.json")
            return real(path, obj)
        with patch.object(TM, "write_json_atomic", no_run_json), self.assertRaises(RuntimeError):
            self.train_run("aborted-late")
        target = self.d / "aborted-late"
        self.assertTrue((target / "checkpoints").exists() and (target / "train_log.jsonl").exists())
        self.assertEqual(self.train_run("aborted-late")["outcome"], "complete")
        aside = [p for p in target.iterdir() if p.name.startswith(".aborted-start-")]
        self.assertEqual(len(aside), 1)
        self.assertTrue((aside[0] / "checkpoints").exists() and (aside[0] / "train_log.jsonl").exists())
        log = (target / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(sum('"event": "start"' in x for x in log), 1)
        self.assert_same_final_state("aborted-late")

    def test_an_epochs_smaller_last_group_is_averaged_over_its_own_size(self):
        keys = self.data.keys("train")
        a = self.cfg["schedule"]["grad_accumulation"]
        order = TM.epoch_order(self.cfg["seed"], 0, keys)
        groups = [order[i:i + a] for i in range(0, len(order), a)]
        self.assertLess(len(groups[-1]), a)           # the data are built so the last group is smaller
        self.train_run("last-group", stop=len(groups) - 1)
        model, optimizer, scheduler, _ = TM.build_training_objects(self.cfg, tiny_backbone, self.cpu, len(keys))
        TM.restore_checkpoint(TM.checkpoint_path(self.d / "last-group", "latest"), model, optimizer, scheduler, self.cpu)
        model.train()
        policy = MI.InputPolicy.from_config(self.cfg["input"])
        for text in groups[-1]:
            key = L.parse_key(text)
            b = MI.build_input(self.data.examples[key], encode, policy)
            loss, _ = M.example_loss(model(torch.tensor([b["input_ids"]]), b["student_pool"], b["ai_pool"]),
                                     self.data.targets[key])
            if loss is not None:
                (loss / len(groups[-1])).backward()
        norm = float(torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], float("inf")))
        self.assertAlmostEqual(norm, self.final_state("full")[1]["steps"][len(groups) - 1]["grad_norm"], places=5)

    def test_fingerprint_covers_every_repository_module_the_trainer_loads(self):
        record = json.loads((self.d / "full" / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(set(record["fingerprint"]["code_sha256"]), set(TM.TRAINING_CODE))
        here = str(Path(__file__).resolve().parent)
        code = (f"import sys, json; sys.path.insert(0, {here!r}); import train_multihead; "
                "print(json.dumps(sorted({m.__file__ for m in list(sys.modules.values()) if getattr(m, '__file__', None)})))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, encoding="utf-8")
        loaded = json.loads(out.stdout.strip().splitlines()[-1])
        root = ROOT_DIR.resolve()
        real = [Path(f).resolve() for f in loaded if os.path.isabs(f) and Path(f).is_file()]   # some are bare names
        repo = {f.relative_to(root).as_posix() for f in real
                if f.is_relative_to(root) and ".venv" not in f.relative_to(root).parts}
        self.assertTrue(repo, out.stderr[-500:])
        self.assertLessEqual(repo, set(TM.TRAINING_CODE))
        self.assertIn("codebook_schema.json", TM.TRAINING_CODE)

    def test_synthetic_inputs_are_reused_only_when_identical(self):
        examples = TM.synthetic_examples(seed=3, sizes=(5, 3, 2, 2, 1, 1, 1, 1))
        where = self.d / "synthetic-reuse"
        prepared, split = TM.write_synthetic_inputs(where, examples, 11, 3, 4)
        stamp = (prepared / "examples.jsonl").stat().st_mtime_ns
        TM.write_synthetic_inputs(where, examples, 11, 3, 4)
        self.assertEqual((prepared / "examples.jsonl").stat().st_mtime_ns, stamp)            # reused
        TM.write_synthetic_inputs(where, examples, 12, 3, 4)
        self.assertEqual(json.loads(split.read_text(encoding="utf-8"))["seed"], 12)                           # regenerated
        split.unlink()                                                                         # half-removed set
        TM.write_synthetic_inputs(where, examples, 12, 3, 4)
        GS.load_split(split, prepared)

    def test_a_start_that_died_before_run_json_can_be_started_again(self):
        target = self.d / "aborted"

        def crash(*_args, **_kwargs):
            raise RuntimeError("simulated crash while writing the first checkpoint")
        with patch.object(TM, "save_checkpoint", crash), self.assertRaises(RuntimeError):
            self.train_run("aborted")
        self.assertFalse((target / "run.json").exists())
        self.assertTrue((target / "inputs-report.json").exists())
        self.assertEqual(self.train_run("aborted")["outcome"], "complete")
        self.assertTrue(any(p.name.startswith(".aborted-start-") for p in target.iterdir()))
        self.assert_same_final_state("aborted")

    def test_a_closed_console_keeps_the_exit_status_and_the_handler_quiet(self):
        here = str(Path(__file__).resolve().parent)
        code = (f"import sys; sys.path.insert(0, {here!r}); import train_multihead as TM; "
                "[TM.say(f'progress line {i}') for i in range(3)]; sys.exit(3)")   # short: stays buffered
        proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.stdout.close()                           # the console reader is gone
        proc.stdout = None
        _, err = proc.communicate(timeout=300)
        self.assertEqual(proc.returncode, 3, err[-800:])

        class Closed(io.TextIOBase):
            def write(self, _text):
                raise BrokenPipeError
        try:
            with contextlib.redirect_stdout(Closed()), contextlib.redirect_stderr(Closed()):
                TM._on_signal(signal.SIGTERM, None)
            self.assertEqual(TM._Stop.signal, signal.SIGTERM)
        finally:
            TM._Stop.signal = None
        closed = open(os.devnull, "w")
        closed.close()
        TM.say("ignored", closed)                     # ValueError on a closed file is swallowed too

    def test_a_closed_console_cannot_prevent_the_stop_checkpoint(self):
        class Closed(io.TextIOBase):
            def write(self, _text):
                raise BrokenPipeError
        with contextlib.redirect_stdout(Closed()):
            TM.say("ignored")
            result = self.train_run("piped", stop=2)
        self.assertEqual(result["outcome"], "interrupted")
        self.assertEqual(json.loads((self.d / "piped" / "checkpoints" / "latest.json").read_text(encoding="utf-8"))["global_step"], 2)

    # Findings of the training-code audit of 19 October 2026, each reproduced before it was fixed.

    def log(self, name):
        return [json.loads(x) for x in (self.d / name / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]

    def latest(self, name):
        return json.loads((self.d / name / "checkpoints" / "latest.json").read_text(encoding="utf-8"))

    def cli(self, argv):
        """main() on these synthetic inputs, without reading a model snapshot."""
        saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
        try:
            with patch.object(TM, "_real_inputs", return_value=(self.data, IDENTITY, tiny_backbone, encode)), \
                    contextlib.redirect_stdout(io.StringIO()):
                return TM.main(argv)
        finally:
            signal.signal(signal.SIGINT, saved[0])
            signal.signal(signal.SIGTERM, saved[1])

    def poison(self, kind, position=9):
        """Corrupt one labelled training interaction, picked by its place in epoch 0's order, each
        time it is trained: 'loss' makes its loss NaN; 'gradient' keeps the loss finite and makes
        its gradient NaN. Returns the patch, the interaction and the optimizer step it falls in."""
        order = TM.epoch_order(self.cfg["seed"], 0, self.data.keys("train"))
        text = next(t for t in order[position:] if self.data.targets[L.parse_key(t)][STUDENT]["mask"] == 1)
        target, real = self.data.targets[L.parse_key(text)], M.example_loss

        class NaNGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad):
                return grad * float("nan")

        def wrapped(logits, targets):
            loss, parts = real(logits, targets)
            if targets is target and loss is not None and torch.is_grad_enabled():
                loss = NaNGradient.apply(loss) if kind == "gradient" else loss * float("nan")
            return loss, parts
        return patch.object(M, "example_loss", wrapped), text, order.index(text) // self.cfg["schedule"]["grad_accumulation"] + 1

    def test_a_non_finite_loss_or_gradient_stops_before_the_update_and_keeps_the_last_valid_checkpoint(self):
        from safetensors.torch import load_file
        full_steps = self.final_state("full")[1]["steps"]
        for kind in ("gradient", "loss"):
            name = f"nonfinite-{kind}"
            patcher, text, step = self.poison(kind)
            with patcher:
                result = self.train_run(name)
            self.assertEqual(result["outcome"], "failed", kind)
            failure = json.loads((self.d / name / result["failure"]).read_text(encoding="utf-8"))
            self.assertEqual((failure["kind"], failure["optimizer_step"]), (kind, step))
            if kind == "gradient":
                self.assertEqual(failure["grad_norm"], "nan")
                self.assertTrue(failure["nonfinite_gradient_elements"])
            else:
                self.assertEqual((failure["loss"], failure["interaction"]), ("nan", text))
            log = self.log(name)
            self.assertEqual(log[-1]["event"], "failed")
            taken = [e for e in log if e["event"] == "step"]
            self.assertEqual([e["step"] for e in taken], list(range(1, step)))        # the step was never taken
            self.assertEqual([e["grad_norm"] for e in taken], [s["grad_norm"] for s in full_steps[:step - 1]])
            latest = self.latest(name)
            self.assertEqual(latest["tag"], result["latest_checkpoint"])
            self.assertLess(latest["global_step"], step)
            self.assertFalse(latest["complete"])
            dirs = [p for p in (self.d / name / "checkpoints").iterdir() if p.is_dir()]
            self.assertTrue(all(TM.TAG.match(p.name) and p.name <= latest["tag"] for p in dirs), dirs)
            for p in dirs:
                TM.verify_checkpoint(p)
                self.assertTrue(all(bool(torch.isfinite(t).all()) for t in load_file(p / "trainable.safetensors").values()))
            self.assertFalse((self.d / name / "checkpoints" / "final.json").exists())
            self.assertFalse((self.d / name / "outputs").exists() or (self.d / name / ".outputs.partial").exists())
            with patcher:                             # resume replays the same data, so the fault recurs there
                self.assertEqual(self.cli(["resume", "--run-dir", str(self.d / name)]), TM.EXIT_FAILED)
            self.assertEqual(self.latest(name), latest)
            self.assertEqual(sorted(p.name for p in (self.d / name / "failures").iterdir()),
                             [f"001-step{step:06d}.json", f"002-step{step:06d}.json"])
            # The kept checkpoint is a valid state: without the fault the run ends exactly where an
            # uninterrupted run does.
            self.assertEqual(self.train_run(name, resume=True)["outcome"], "complete")
            self.assert_same_final_state(name)

    def test_a_checkpoint_is_never_written_from_a_non_finite_state(self):
        self.train_run("nonfinite-state", stop=2)
        root = self.d / "nonfinite-state" / "checkpoints"
        before = {p.name: p.stat().st_mtime_ns for p in root.iterdir()}
        pointer = (root / "latest.json").read_text(encoding="utf-8")
        for where in ("parameter", "optimizer moment"):
            model, optimizer, scheduler, _ = TM.build_training_objects(self.cfg, tiny_backbone, self.cpu,
                                                                       len(self.data.keys("train")))
            progress = TM.restore_checkpoint(root / json.loads(pointer)["tag"], model, optimizer, scheduler, self.cpu)
            progress["global_step"] += 1
            with torch.no_grad():
                if where == "parameter":
                    next(p for p in model.parameters() if p.requires_grad).view(-1)[0] = float("inf")
                else:
                    next(iter(optimizer.state.values()))["exp_avg_sq"].view(-1)[0] = float("nan")
            with self.assertRaisesRegex(TM.NumericalFailure, "non-finite state", msg=where):
                TM.save_checkpoint(self.d / "nonfinite-state", model, optimizer, scheduler, progress, self.cpu, 2)
            self.assertEqual({p.name: p.stat().st_mtime_ns for p in root.iterdir()}, before, where)
            self.assertEqual((root / "latest.json").read_text(encoding="utf-8"), pointer, where)

    def signal_during(self, what, at):
        """SIGTERM through the real handler at the `at`-th validation loss ('validation') or score
        ('scores') computation; returns the patch and a count of such computations after it."""
        count = {"calls": 0, "after": 0}
        real_loss, real_scores = M.example_loss, M.scores

        def tick():
            count["calls"] += 1
            if count["calls"] == at:
                TM._on_signal(signal.SIGTERM, None)
            elif count["calls"] > at:
                count["after"] += 1

        def loss(logits, targets):
            if not torch.is_grad_enabled():
                tick()
            return real_loss(logits, targets)

        def scores(logits):
            tick()
            return real_scores(logits)
        return patch.object(M, "example_loss", loss) if what == "validation" else patch.object(M, "scores", scores), count

    def test_a_stop_during_validation_takes_no_further_step_and_resumes_exactly(self):
        n_val, per_epoch = len(self.data.keys("validation")), self.final_state("full")[1]["plan"]["steps_per_epoch"]
        self.assertGreaterEqual(n_val, 3)
        cases = {"first epoch, first interaction": (1, per_epoch, "validation"),
                 "last epoch, middle interaction": (n_val + 2, 2 * per_epoch, "validation"),
                 "last epoch, last interaction": (2 * n_val, 2 * per_epoch, "after epoch end")}
        for i, (case, (at, steps, phase)) in enumerate(cases.items()):
            name = f"stop-validation-{i}"
            patcher, count = self.signal_during("validation", at)
            with patcher:
                result = self.train_run(name)
            TM._Stop.signal = None
            self.assertEqual(result["outcome"], "interrupted", case)
            self.assertEqual(count["after"], 0, case)             # no validation interaction after the request
            self.assertEqual(self.latest(name)["global_step"], steps, case)     # and no optimizer step
            self.assertEqual((self.log(name)[-1]["event"], self.log(name)[-1]["phase"]), ("stopped", phase), case)
            self.assertFalse((self.d / name / "outputs").exists(), case)
            self.assertFalse((self.d / name / "checkpoints" / "final.json").exists(), case)
            self.assertEqual(self.train_run(name, resume=True)["outcome"], "complete", case)
            self.assert_same_final_state(name)                    # validation losses included

    def test_a_stop_during_prediction_publishes_nothing_and_a_later_run_writes_identical_outputs(self):
        patcher, count = self.signal_during("scores", 1)
        with patcher:
            result = self.train_run("stop-outputs")
        TM._Stop.signal = None
        run = self.d / "stop-outputs"
        self.assertEqual((result["outcome"], count["after"]), ("interrupted", 0))
        self.assertFalse((run / "outputs").exists() or (run / ".outputs.partial").exists())
        self.assertTrue(self.latest("stop-outputs")["complete"])          # training itself had finished
        self.assertEqual(self.train_run("stop-outputs", resume=True)["outcome"], "complete")
        self.assert_same_final_state("stop-outputs")
        out = self.d / "stop-predict.jsonl"
        patcher, count = self.signal_during("scores", 1)
        with patcher:
            self.assertEqual(self.cli(["predict", "--run-dir", str(self.d / "full"), "--out", str(out)]),
                             TM.EXIT_INTERRUPTED)
        TM._Stop.signal = None                         # the next command is a new process
        self.assertEqual(count["after"], 0)
        self.assertFalse(out.exists() or TM.scores_path(out).exists())
        self.assertEqual(self.cli(["predict", "--run-dir", str(self.d / "full"), "--out", str(out)]), 0)
        made = L.read_predictions(self.d / "full" / "outputs" / "predictions-validation.jsonl")
        self.assertEqual([r["labels"] for r in L.read_predictions(out)], [r["labels"] for r in made])

    def test_a_stop_before_a_step_or_while_a_checkpoint_is_written_takes_no_further_step(self):
        TM._on_signal(signal.SIGTERM, None)           # e.g. while the model was still loading
        self.assertEqual(self.train_run("stop-at-start")["outcome"], "interrupted")
        TM._Stop.signal = None
        self.assertEqual(self.latest("stop-at-start")["global_step"], 0)
        self.assertFalse(any(e["event"] == "step" for e in self.log("stop-at-start")))
        real = TM.save_checkpoint

        def signalling(*args, **kwargs):
            tag = real(*args, **kwargs)
            if args[4]["global_step"] == 2:
                TM._on_signal(signal.SIGTERM, None)
            return tag
        with patch.object(TM, "save_checkpoint", signalling):
            self.assertEqual(self.train_run("stop-at-checkpoint")["outcome"], "interrupted")
        TM._Stop.signal = None
        self.assertEqual(self.latest("stop-at-checkpoint")["global_step"], 2)
        self.assertEqual(max(e["step"] for e in self.log("stop-at-checkpoint") if e["event"] == "step"), 2)
        for name in ("stop-at-start", "stop-at-checkpoint"):
            self.assertEqual(self.train_run(name, resume=True)["outcome"], "complete", name)
            self.assert_same_final_state(name)

    def test_predictions_keep_the_recorded_run_id_when_the_run_directory_is_renamed(self):
        renamed = self.d / "full-renamed"
        shutil.copytree(self.d / "full", renamed)
        out = self.d / "renamed.jsonl"
        result = TM.predict_from_checkpoint(renamed, self.data, IDENTITY, tiny_backbone, encode, self.cpu,
                                            "validation", out)
        self.assertEqual({r["run_id"] for r in L.read_predictions(out)}, {"full"})
        self.assertEqual((result["run_id"], result["run_dir_name"]), ("full", "full-renamed"))
        self.train_run("before-rename", stop=3)
        (self.d / "before-rename").rename(self.d / "after-rename")
        self.assertEqual(self.train_run("after-rename", resume=True)["outcome"], "complete")
        outputs = self.d / "after-rename" / "outputs"
        self.assertEqual({r["run_id"] for r in L.read_predictions(outputs / "predictions-validation.jsonl")},
                         {"before-rename"})
        self.assertEqual(json.loads((outputs / "evaluation-validation.json").read_text(encoding="utf-8"))["predictions"]["run_id"],
                         "before-rename")
        self.assertIn({"event": "run_dir_renamed", "run_id": "before-rename", "directory": "after-rename"},
                      [{k: v for k, v in e.items() if k != "utc"} for e in self.log("after-rename")])

    def test_the_run_id_is_the_directory_name_also_for_run_dir_dot_and_never_blank(self):
        # Found in review of the audit fixes: `--run-dir .` recorded run_id '' for good.
        args = (self.cfg, self.data, IDENTITY, tiny_backbone, encode, self.cpu)
        target, here = self.d / "dot-run", os.getcwd()
        target.mkdir()
        try:
            os.chdir(target)
            self.assertEqual(TM.run_training(Path("."), *args, stop_after_steps=2, synthetic=True,
                                             command=["unittest"])["outcome"], "interrupted")
            self.assertEqual(json.loads(Path("run.json").read_text(encoding="utf-8"))["run_id"], "dot-run")
            self.assertEqual(TM.run_training(Path("."), *args, resume=True, synthetic=True,
                                             command=["unittest"])["outcome"], "complete")
        finally:
            os.chdir(here)
        self.assertFalse(any(e["event"] == "run_dir_renamed" for e in self.log("dot-run")))
        self.assertEqual({r["run_id"] for r in L.read_predictions(target / "outputs" / "predictions-validation.jsonl")},
                         {"dot-run"})
        self.assert_same_final_state("dot-run")
        # The file-system root has no name anywhere; a name of spaces exists only off Windows.
        for blank in [Path(os.path.abspath(os.sep))] + ([self.d / "  "] if os.name != "nt" else []):
            existed = blank.exists()
            with self.assertRaisesRegex(ValueError, "must not be blank", msg=str(blank)):
                TM.run_training(blank, *args, synthetic=True, command=["unittest"])
            self.assertEqual(blank.exists(), existed, str(blank))
        unnamed = self.d / "unnamed"                   # a run.json from elsewhere with no usable run_id
        shutil.copytree(self.d / "full", unnamed)
        record = json.loads((unnamed / "run.json").read_text(encoding="utf-8"))
        (unnamed / "run.json").write_text(json.dumps(dict(record, run_id=" ")), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no usable run_id"):
            TM.predict_from_checkpoint(unnamed, self.data, IDENTITY, tiny_backbone, encode, self.cpu, "validation",
                                       self.d / "unnamed.jsonl")
        with self.assertRaisesRegex(ValueError, "no usable run_id"):
            self.train_run("unnamed", resume=True)
        self.assertFalse((self.d / "unnamed.jsonl").exists())

    # Found in review 2 of the audit fixes, each reproduced before it was fixed.

    def test_an_update_leaving_a_non_finite_value_is_not_kept_and_failure_records_never_collide(self):
        # With AdamW eps 0, LoRA B's zero start gives LoRA A a zero gradient and the update 0/0:
        # the loss and the gradient are finite, the updated parameters are not.
        cfg = copy.deepcopy(self.cfg)
        cfg["optim"]["eps"] = 0.0
        name = "nonfinite-update"
        result = self.train_run(name, cfg=cfg)
        self.assertEqual(result["outcome"], "failed")
        failure = json.loads((self.d / name / result["failure"]).read_text(encoding="utf-8"))
        self.assertEqual((failure["kind"], failure["optimizer_step"], failure["checkpoint_written"]), ("update", 1, False))
        self.assertTrue(failure["nonfinite_tensors"])
        self.assertTrue(any(".lora_A." in n for n in failure["nonfinite_tensors"]), failure["nonfinite_tensors"])
        self.assertFalse(any(e["event"] == "step" for e in self.log(name)))         # never counted or logged
        self.assertEqual(self.latest(name)["tag"], "s000000-e00-g000")
        for _ in range(2):                                    # deterministic: it recurs at the same step
            self.assertEqual(self.train_run(name, resume=True, cfg=cfg)["outcome"], "failed")
        (self.d / name / "failures" / "001-step000001.json").unlink()                # an earlier record removed
        self.assertEqual(self.train_run(name, resume=True, cfg=cfg)["outcome"], "failed")
        self.assertEqual(sorted(p.name for p in (self.d / name / "failures").iterdir()),
                         ["002-step000001.json", "003-step000001.json", "004-step000001.json"])
        self.assertEqual(self.log(name)[-1]["event"], "failed")
        self.assertEqual(self.latest(name)["tag"], "s000000-e00-g000")
        # Found in review 3: a deleted newest record's number was given to the next record, a name
        # taken meanwhile collided, and an unusable failures/ lost the failure itself.
        (self.d / name / "failures" / "004-step000001.json").unlink()
        real, calls = TM.write_json, []

        def taken_once(path, obj, exclusive=True):
            calls.append(Path(path).name)
            if len(calls) == 1:
                raise FileExistsError(path)               # as if another writer had just taken that name
            return real(path, obj, exclusive)
        with patch.object(TM, "write_json", taken_once):
            result = self.train_run(name, resume=True, cfg=cfg)
        self.assertEqual(calls, ["005-step000001.json", "006-step000001.json"])
        self.assertEqual(result["failure"], "failures/006-step000001.json")
        shutil.rmtree(self.d / name / "failures")
        (self.d / name / "failures").write_text("not a directory", encoding="utf-8")
        result = self.train_run(name, resume=True, cfg=cfg)
        self.assertEqual((result["outcome"], result["failure"]), ("failed", None))
        self.assertEqual((self.log(name)[-1]["event"], self.log(name)[-1]["details"]), ("failed", None))
        self.assertEqual(self.latest(name)["tag"], "s000000-e00-g000")
        # Found in review 4: a log holding an undecodable byte lost the failure (exit 1), a torn last
        # line swallowed the next event, record names in other letter case were not counted, and a
        # failures/ that could be written but not listed gave up.
        where = self.d / name / "failures"
        where.unlink()
        where.mkdir()
        (where / "009-STEP000001.json").write_text("{}", encoding="utf-8")
        with (self.d / name / "train_log.jsonl").open("ab") as f:
            f.write(b'{"event": "note", "dir": "caf\xc3')     # an append torn inside a character
        result = self.train_run(name, resume=True, cfg=cfg)
        self.assertEqual((result["outcome"], result["failure"]), ("failed", "failures/010-step000001.json"))
        lines = (self.d / name / "train_log.jsonl").read_bytes().decode("utf-8", "replace").split("\n")
        self.assertTrue(lines[-4].startswith('{"event": "note"'))           # the torn line, left as it was
        self.assertEqual([json.loads(x)["event"] for x in lines[-3:-1]], ["resume", "failed"])   # each on its own line
        self.assertEqual(json.loads(lines[-2])["details"], "failures/010-step000001.json")
        where.chmod(0o333)                                    # writable, not listable
        try:
            result = self.train_run(name, resume=True, cfg=cfg)
        finally:
            where.chmod(0o755)
        self.assertEqual(result["failure"], "failures/011-step000001.json")
        self.assertEqual(self.latest(name)["tag"], "s000000-e00-g000")

    def test_a_stop_between_output_phases_publishes_nothing(self):
        real = TM._evaluation
        cases = {"evaluation-validation": "train predictions", "evaluation-train-fit": "label-prior reading aid",
                 "evaluation-validation-prior-reference": "publishing the outputs"}
        for i, (report, phase) in enumerate(cases.items()):
            def signalling(out_dir, final_dir, name, *args, _report=report):
                written = real(out_dir, final_dir, name, *args)
                if name == _report:
                    TM._on_signal(signal.SIGTERM, None)
                return written
            run = f"stop-phase-{i}"
            with patch.object(TM, "_evaluation", signalling):
                self.assertEqual(self.train_run(run)["outcome"], "interrupted", report)
            TM._Stop.signal = None
            self.assertFalse((self.d / run / "outputs").exists() or (self.d / run / ".outputs.partial").exists(), report)
            self.assertEqual((self.log(run)[-1]["event"], self.log(run)[-1]["phase"]), ("stopped", f"outputs ({phase})"))
            self.assertEqual(self.train_run(run, resume=True)["outcome"], "complete", report)
            self.assert_same_final_state(run)
        # A request during a pass's last interaction is honoured before that pass is evaluated.
        n_val, n_train = len(self.data.keys("validation")), len(self.data.keys("train"))
        for i, (at, phase) in enumerate(((n_val, "validation evaluation"), (n_val + n_train, "train evaluation"))):
            run = f"stop-last-interaction-{i}"
            patcher, count = self.signal_during("scores", at)
            with patcher:
                self.assertEqual(self.train_run(run)["outcome"], "interrupted", phase)
            TM._Stop.signal = None
            self.assertEqual(count["after"], 0, phase)
            self.assertFalse((self.d / run / "outputs").exists() or (self.d / run / ".outputs.partial").exists(), phase)
            self.assertEqual(self.log(run)[-1]["phase"], f"outputs ({phase})")
            self.assertEqual(self.train_run(run, resume=True)["outcome"], "complete", phase)
            self.assert_same_final_state(run)

    def test_each_update_uses_the_gradient_clipped_to_the_configured_norm(self):
        from torch.optim.optimizer import register_optimizer_step_pre_hook
        seen = []

        def record(optimizer, _args, _kwargs):
            grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            seen.append(float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))))
        handle = register_optimizer_step_pre_hook(record)
        try:
            self.train_run("clipped", stop=4)
        finally:
            handle.remove()
        logged = [e["grad_norm"] for e in self.log("clipped") if e["event"] == "step"]
        limit = self.cfg["optim"]["grad_clip_norm"]
        self.assertEqual(len(seen), len(logged))
        self.assertTrue(any(n > limit for n in logged))        # clipping is active in this run
        for before, after in zip(logged, seen):
            self.assertAlmostEqual(after, before * min(1.0, limit / (before + 1e-6)), places=5)

    def test_cli_refuses_a_blank_run_name_and_data_itself_before_anything_is_written(self):
        root = self.d / "fake-root"
        (root / "data").mkdir(parents=True)
        cases = ((["train", "--synthetic", "--config", str(PILOT_CONFIG), "--run-dir", os.path.abspath(os.sep)],
                  "must not be blank"),
                 (["train", "--config", str(PILOT_CONFIG), "--run-dir", str(root / "data")], "below repository data/"),
                 (["train", "--config", str(PILOT_CONFIG), "--run-dir", str(root / "data" / "runs" / "..")],
                  "below repository data/"))
        saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
        try:
            with patch.object(TM, "ROOT", root), \
                    patch.object(TM, "_real_inputs", side_effect=AssertionError("nothing may be written before the guard")):
                for argv, reason in cases:
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop:
                        TM.main(argv)
                    self.assertEqual(stop.exception.code, 2, argv)
                    self.assertIn(reason, err.getvalue(), argv)
        finally:
            signal.signal(signal.SIGINT, saved[0])
            signal.signal(signal.SIGTERM, saved[1])
        self.assertEqual(list((root / "data").iterdir()), [])

    def test_a_symlinked_or_differently_cased_path_to_a_run_is_not_a_rename(self):
        self.train_run("alias-real", stop=2)
        paths, link = [], self.d / "alias-link"
        try:
            link.symlink_to(self.d / "alias-real", target_is_directory=True)
            paths.append(link)
        except OSError:                                # e.g. Windows without the symlink privilege
            pass
        if (self.d / "ALIAS-REAL").exists():           # a case-insensitive file system (the macOS default)
            paths.append(self.d / "ALIAS-REAL")
        self.assertTrue(paths)
        # Found in review 3: another Unicode normalisation of the same name (macOS file systems
        # treat NFC and NFD spellings as one name).
        nfc, nfd = self.d / "caf\u00e9-run", self.d / "cafe\u0301-run"
        self.train_run(nfc.name, stop=1)
        if nfd.exists():
            out = self.d / "alias-nfd.jsonl"
            result = TM.predict_from_checkpoint(nfd, self.data, IDENTITY, tiny_backbone, encode, self.cpu,
                                                "validation", out, "latest")
            self.assertNotIn("run_dir_name", result)
            self.assertEqual({r["run_id"] for r in L.read_predictions(out)}, {nfc.name})
        self.assertEqual(TM.run_training(paths[0], self.cfg, self.data, IDENTITY, tiny_backbone, encode, self.cpu,
                                         resume=True, synthetic=True, command=["unittest"])["outcome"], "complete")
        self.assertFalse(any(e["event"] == "run_dir_renamed" for e in self.log("alias-real")))
        for i, path in enumerate(paths):
            out = self.d / f"alias-{i}.jsonl"
            result = TM.predict_from_checkpoint(path, self.data, IDENTITY, tiny_backbone, encode, self.cpu,
                                                "validation", out)
            self.assertNotIn("run_dir_name", result, path)
            self.assertEqual({r["run_id"] for r in L.read_predictions(out)}, {"alias-real"})

    def test_failure_records_are_numbered_from_any_log_and_never_lost(self):
        # Found in review 5: these defensive branches had no test.
        run = self.d / "record-only"
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints" / "latest.json").write_text(json.dumps({"tag": "s000000-e00-g000", "global_step": 0}), encoding="utf-8")
        failure = TM.NumericalFailure({"what": "synthetic", "kind": "loss", "optimizer_step": 1})
        (run / "train_log.jsonl").write_bytes(b'{"event": "failed", "details": "failures/007-step000001.json"}\r[]\r')
        self.assertEqual(TM._record_failure(run, failure)["failure"], "failures/008-step000001.json")   # lone CR ends lines
        (run / "train_log.jsonl").unlink()                    # no log at all: the directory's numbers still apply
        self.assertEqual(TM._record_failure(run, failure)["failure"], "failures/009-step000001.json")
        self.assertEqual(json.loads((run / "train_log.jsonl").read_text(encoding="utf-8"))["details"], "failures/009-step000001.json")

    def test_the_run_name_survives_an_unlistable_parent_and_an_entry_that_cannot_be_examined(self):
        # Found in review 5: these defensive branches had no test.
        parent = self.d / "guarded-parent"
        (parent / "run-x").mkdir(parents=True)
        (parent / "other").mkdir()
        real, calls = os.path.samefile, []

        def vanishing_first(a, b):
            calls.append(a)
            if len(calls) == 1:
                raise FileNotFoundError(a)                # as if that entry vanished while being examined
            return real(a, b)
        with patch.object(TM.os.path, "samefile", vanishing_first):
            self.assertEqual(TM._dir_name(parent / "run-x"), "run-x")
        parent.chmod(0o311)                                  # searchable, not listable
        try:
            self.assertEqual(TM._dir_name(parent / "run-x"), "run-x")
        finally:
            parent.chmod(0o755)

    def test_a_symlink_beside_the_run_never_names_it(self):
        # Found in review 4: the alias test passed only because its directory is listed first.
        parent = self.d / "linked-parent"
        parent.mkdir()
        pairs = (("latest", "run-7"), ("a-link", "z-real"))
        try:
            for link, target in pairs:                     # made before the directories they point to
                (parent / link).symlink_to(parent / target, target_is_directory=True)
        except OSError:
            self.skipTest("no symbolic links here")
        for _, target in pairs:
            (parent / target).mkdir()
        order = os.listdir(parent)
        if not any(order.index(link) < order.index(target) for link, target in pairs):
            self.skipTest("this file system lists each link after its directory")
        for link, target in pairs:
            self.assertEqual((TM._dir_name(parent / target), TM._dir_name(parent / link)), (target, target))

    def test_a_signal_that_interrupts_a_console_write_cannot_raise(self):
        class Reentrant(io.TextIOBase):
            def write(self, _text):
                raise RuntimeError("reentrant call inside <_io.BufferedWriter name='<stderr>'>")
        with contextlib.redirect_stderr(Reentrant()):
            TM._on_signal(signal.SIGTERM, None)         # the handler writes to stderr
        self.assertEqual(TM._Stop.signal, signal.SIGTERM)


@unittest.skipUnless(HAVE_TORCH, "model stack not installed")
class ClassWeightsAndHeadLr(unittest.TestCase):
    """Optional class weights (from the training part only) and a separate head learning rate."""

    @staticmethod
    def run_data(train_rows, validation_rows=()):
        rows = list(train_rows) + list(validation_rows)
        keys = [("SYN_W", i + 1) for i in range(len(rows))]
        return TM.RunData({k: {"labels": r} for k, r in zip(keys, rows)},
                          {k: "train" if i < len(train_rows) else "validation" for i, k in enumerate(keys)}, {}, {}, {})

    def test_class_weights_follow_the_training_part_only(self):
        cfg = tiny_config(loss={**tiny_config()["loss"], "class_weights": TM.CLASS_WEIGHT_SCHEME})
        train = [labels(), labels(), labels(**{TOPIC: ["code", "concept"]}), labels(**{STUDENT: "Guided generation"}),
                 labels(**{STUDENT: None})]
        other = [labels(**{STUDENT: "Information retrieval", TOPIC: ["concept"]})] * 3
        weights = TM.class_weights(cfg, self.run_data(train, other))
        self.assertEqual(weights, TM.class_weights(cfg, self.run_data(train)))       # validation labels unused
        codes = L.DIRECT_CODES[STUDENT]
        expected = [1.0] * len(codes)        # 4 labelled rows: Task automation 3, Guided generation 1
        expected[codes.index("Task automation")], expected[codes.index("Guided generation")] = 4 / (2 * 3), 4 / (2 * 1)
        self.assertEqual(weights[STUDENT], expected)
        self.assertEqual(weights[TOPIC], [1.0, 4.0, 1.0, 1.0])     # code: 5 of 5 -> floor 1; concept: 1 of 5 -> 4
        self.assertEqual(weights[AIDEL], [1.0] * 5)
        self.assertEqual(weights[AIRESP], [1.0] * 5)
        rare = TM.class_weights(cfg, self.run_data([labels()] * 11 + [labels(**{TOPIC: ["concept"]})]))
        self.assertEqual(rare[TOPIC][1], TM.CLASS_WEIGHT_MAX)       # 11 / 1 capped
        self.assertIsNone(TM.class_weights(tiny_config(), self.run_data(train)))

    def test_weighted_loss_matches_its_definition(self):
        torch.manual_seed(0)
        logits = {a: torch.randn(len(L.DIRECT_CODES[a])) for a in L.AXES}
        targets = L.encode_targets(labels())
        weights = {a: [0.5 + i for i in range(len(L.DIRECT_CODES[a]))] for a in L.AXES}
        i = targets[STUDENT]["index"]
        expected = weights[STUDENT][i] * torch.nn.functional.cross_entropy(logits[STUDENT].unsqueeze(0), torch.tensor([i]))
        for a in (TOPIC, AIDEL, AIRESP):
            y, z, w = torch.tensor(targets[a]["multi_hot"], dtype=torch.float32), logits[a], torch.tensor(weights[a])
            F = torch.nn.functional
            expected = expected - (w * y * F.logsigmoid(z) + (1 - y) * F.logsigmoid(-z)).mean()
        self.assertTrue(torch.allclose(M.example_loss(logits, targets, weights)[0], expected))
        ones = {a: [1.0] * len(L.DIRECT_CODES[a]) for a in L.AXES}
        self.assertTrue(torch.allclose(M.example_loss(logits, targets, ones)[0], M.example_loss(logits, targets)[0]))
        masked = L.encode_targets(labels(**{STUDENT: None}))
        self.assertIsNone(M.example_loss(logits, masked, weights)[1][STUDENT])

    def test_config_accepts_only_the_implemented_options(self):
        TM.check_config(tiny_config(loss={**tiny_config()["loss"], "class_weights": TM.CLASS_WEIGHT_SCHEME},
                                    optim={**tiny_config()["optim"], "head_lr": 1e-3}))
        TM.check_config(tiny_config(optim={**tiny_config()["optim"], "head_lr": None}))
        for bad in ({"loss": {**tiny_config()["loss"], "class_weights": "balanced"}},
                    {"optim": {**tiny_config()["optim"], "head_lr": 0}},
                    {"optim": {**tiny_config()["optim"], "head_lr": -1e-3}},
                    {"optim": {**tiny_config()["optim"], "head_lr": True}},
                    {"optim": {**tiny_config()["optim"], "head_lr": "1e-3"}},
                    {"optim": {**tiny_config()["optim"], "head_lr": float("nan")}}):
            with self.assertRaises(ValueError, msg=str(bad)):
                TM.check_config(tiny_config(**bad))

    def test_head_lr_puts_the_four_heads_in_their_own_group(self):
        cpu = torch.device("cpu")
        model, optimizer, _, _ = TM.build_training_objects(tiny_config(), tiny_backbone, cpu, 6)
        self.assertEqual(len(optimizer.param_groups), 1)
        cfg = tiny_config(optim={**tiny_config()["optim"], "head_lr": 1e-3})
        model, optimizer, _, _ = TM.build_training_objects(cfg, tiny_backbone, cpu, 6)
        adapter, heads = optimizer.param_groups
        self.assertEqual((adapter["initial_lr"], heads["initial_lr"]), (cfg["optim"]["lr"], 1e-3))
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.assertEqual({id(p) for p in heads["params"]}, {id(p) for n, p in named if n.startswith("heads.")})
        self.assertEqual({id(p) for p in adapter["params"]}, {id(p) for n, p in named if not n.startswith("heads.")})
        self.assertTrue(heads["params"] and adapter["params"])

    def test_a_run_with_both_options_records_them_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, cpu = Path(tmp), torch.device("cpu")
            prepared, split = TM.write_synthetic_inputs(d / "inputs", TM.synthetic_examples(seed=3, sizes=(5, 3, 2, 2, 1, 1, 1, 1)),
                                                        11, 3, 4)
            data = TM.load_data(prepared, split)
            cfg = tiny_config(loss={**tiny_config()["loss"], "class_weights": TM.CLASS_WEIGHT_SCHEME},
                              optim={**tiny_config()["optim"], "head_lr": 1e-3})

            def run(name, **kw):
                return TM.run_training(d / name, cfg, data, IDENTITY, tiny_backbone, encode, cpu, synthetic=True,
                                       command=["unittest"], **kw)
            self.assertEqual(run("full")["outcome"], "complete")
            self.assertEqual(run("stopped", stop_after_steps=3)["outcome"], "interrupted")
            random.seed(12345)
            torch.manual_seed(12345)
            self.assertEqual(run("stopped", resume=True)["outcome"], "complete")
            record = json.loads((d / "full" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["plan"]["class_weights"], TM.class_weights(cfg, data))
            states = []
            for name in ("full", "stopped"):
                ckpt = TM.checkpoint_path(d / name, "final")
                from safetensors.torch import load_file
                states.append((load_file(ckpt / "trainable.safetensors"), json.loads((ckpt / "progress.json").read_text(encoding="utf-8")),
                               torch.load(ckpt / "optimizer.pt", weights_only=True)))
            (a, pa, oa), (b, pb, ob) = states
            for n in a:
                self.assertTrue(torch.equal(a[n], b[n]), n)
            for field in ("loss_mean", "lr", "head_lr", "grad_norm"):
                self.assertEqual([s[field] for s in pa["steps"]], [s[field] for s in pb["steps"]], field)
            self.assertEqual([g["lr"] for g in oa["param_groups"]], [g["lr"] for g in ob["param_groups"]])
            for s in pa["steps"]:                       # one schedule scales both groups
                self.assertAlmostEqual(s["head_lr"] * cfg["optim"]["lr"], s["lr"] * 1e-3, places=15)

    def test_training_loss_uses_the_weights_and_validation_loss_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            prepared, split = TM.write_synthetic_inputs(d / "inputs", TM.synthetic_examples(seed=3, sizes=(5, 3, 2, 2, 1, 1, 1, 1)),
                                                        11, 3, 4)
            data = TM.load_data(prepared, split)
            cfg = tiny_config(loss={**tiny_config()["loss"], "class_weights": TM.CLASS_WEIGHT_SCHEME})
            cfg["schedule"]["epochs"] = 1
            seen, real = [], M.example_loss

            def wrapped(logits, targets, weights=None):
                seen.append((torch.is_grad_enabled(), weights))
                return real(logits, targets) if weights is None else real(logits, targets, weights)
            with patch.object(M, "example_loss", wrapped):
                TM.run_training(d / "run", cfg, data, IDENTITY, tiny_backbone, encode, torch.device("cpu"),
                                synthetic=True, command=["unittest"])
            expected = TM.class_weights(cfg, data)
            self.assertTrue(any(g for g, _ in seen) and any(not g for g, _ in seen))
            self.assertTrue(all(w == expected for g, w in seen if g))      # every training interaction
            self.assertTrue(all(w is None for g, w in seen if not g))      # validation loss: unweighted


@unittest.skipUnless(HAVE_TORCH, "model stack not installed")
class MemoryRelease(unittest.TestCase):
    """The optional return of cached accelerator memory after optimizer steps (memory section of the config)."""

    @staticmethod
    def fixture(d):
        prepared, split = TM.write_synthetic_inputs(d / "inputs", TM.synthetic_examples(seed=3, sizes=(5, 3, 2, 2, 1, 1, 1, 1)),
                                                    11, 3, 4)
        return TM.load_data(prepared, split)

    def test_config_accepts_only_a_positive_integer_or_null(self):
        for good in ({"mps_empty_cache_every_steps": 3}, {"mps_empty_cache_every_steps": None}, {}):
            TM.check_config(tiny_config(memory=good))
        TM.check_config(tiny_config())
        for bad in ({"mps_empty_cache_every_steps": 0}, {"mps_empty_cache_every_steps": -2}, {"mps_empty_cache_every_steps": True},
                    {"mps_empty_cache_every_steps": 1.5}, {"mps_empty_cache_every_steps": "2"},
                    {"mps_empty_cache_every_steps": 2, "other": 1}, 5, "every step"):
            with self.assertRaises(ValueError, msg=str(bad)):
                TM.check_config(tiny_config(memory=bad))

    def test_release_cache_acts_on_mps_only(self):
        with patch.object(torch.mps, "empty_cache") as empty:
            TM.release_cache(torch.device("cpu"))
            empty.assert_not_called()
            TM.release_cache(type("Device", (), {"type": "mps"})())
            empty.assert_called_once_with()

    def test_release_runs_every_n_steps_and_after_each_epoch_and_nothing_else_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, cpu = Path(tmp), torch.device("cpu")
            data = self.fixture(d)

            def run(name, **over):
                cfg = tiny_config(**over)
                return cfg, TM.run_training(d / name, cfg, data, IDENTITY, tiny_backbone, encode, cpu, synthetic=True,
                                            command=["unittest"])
            calls = []
            with patch.object(TM, "release_cache", lambda device: calls.append(device.type)):
                run("plain")
                self.assertEqual(calls, [])                                    # off unless the config asks
                cfg, _ = run("every3", memory={"mps_empty_cache_every_steps": 3})
            total = json.loads((d / "every3" / "run.json").read_text(encoding="utf-8"))["plan"]["total_steps"]
            self.assertEqual(len(calls), total // 3 + cfg["schedule"]["epochs"])
            self.assertEqual(json.loads((d / "every3" / "run.json").read_text(encoding="utf-8"))["config"]["memory"],
                             {"mps_empty_cache_every_steps": 3})            # the run record states it
            from safetensors.torch import load_file
            states = [load_file(TM.checkpoint_path(d / n, "final") / "trainable.safetensors") for n in ("plain", "every3")]
            self.assertTrue(all(torch.equal(states[0][n], states[1][n]) for n in states[0]))
            steps = [json.loads((TM.checkpoint_path(d / n, "final") / "progress.json").read_text(encoding="utf-8"))["steps"]
                     for n in ("plain", "every3")]
            self.assertEqual([[s[f] for s in steps[0]] for f in ("loss_mean", "grad_norm")],
                             [[s[f] for s in steps[1]] for f in ("loss_mean", "grad_norm")])


class Provenance(unittest.TestCase):
    RUN = {"run_id": "syn-run", "model_route": "multihead-lora/tiny@abc#s000010-e01-g000",
           "config_sha256": "c" * 64, "dataset_manifest_sha256": "d" * 64}

    def records(self, change_at=None, change=None):
        out = []
        for i in (1, 2, 3):
            run = dict(self.RUN, **(change if i == change_at else {}))
            out.append(L.prediction_record("SYN_P", i, labels(), run))
        return out

    def test_mixed_checkpoint_config_dataset_or_run_is_refused_with_line_and_field(self):
        cases = {"checkpoint": {"model_route": "multihead-lora/tiny@abc#s000020-e02-g000"},
                 "config": {"config_sha256": "e" * 64}, "dataset": {"dataset_manifest_sha256": "f" * 64},
                 "run": {"run_id": "another-run"}}
        with tempfile.TemporaryDirectory() as tmp:
            for name, change in cases.items():
                path = L.write_predictions(Path(tmp) / f"{name}.jsonl", self.records(3, change))
                field = next(iter(change))
                with self.assertRaisesRegex(ValueError, rf"line 3: provenance differs from line 1 in {field}", msg=name):
                    EV.predictions_from_file(path)
            # line numbers are file lines, blank lines included
            spaced = Path(tmp) / "spaced.jsonl"
            rows = [json.dumps(r) for r in self.records(3, cases["checkpoint"])]
            spaced.write_text(rows[0] + "\n\n" + rows[1] + "\n" + rows[2] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"line 4: provenance differs from line 1 in model_route"):
                EV.predictions_from_file(spaced)

    def test_line_numbers_stay_physical_after_a_reason_holding_a_unicode_line_separator(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = L.prediction_record("SYN_P", 1, labels(**{AIDEL: None}), self.RUN, {AIDEL: "cut\u2028here"})
            rows = [first] + self.records(2, {"model_route": "multihead-lora/tiny@abc#s000020-e02-g000"})[1:]
            path = Path(tmp) / "separator.jsonl"
            # The blank line matters: without one, numbering the str.splitlines() pieces happens to
            # give the record after the separator its true number.
            path.write_text("\n\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:2]) + "\n"
                            + json.dumps(rows[2], ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertIn("\u2028", path.read_text(encoding="utf-8"))               # written unescaped
            with self.assertRaisesRegex(ValueError, r"line 3: provenance differs from line 1 in model_route"):
                EV.predictions_from_file(path)

    def test_consistent_records_pass_including_legitimately_unpredicted_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self.records()
            records.append(L.prediction_record("SYN_P", 4, labels(**{AIDEL: None}), self.RUN,
                                               {AIDEL: "source capture truncated"}))
            preds, info = EV.predictions_from_file(L.write_predictions(Path(tmp) / "ok.jsonl", records))
            self.assertEqual({f: info[f] for f in L.RUN_FIELDS}, self.RUN)
            self.assertEqual(info["provenance_consistent_records"], 4)
            self.assertIsNone(preds[("SYN_P", 4)][AIDEL])

    def test_declared_dataset_must_be_the_reference_and_cover_only_its_interactions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(L, "ROOT", root):
                rows = [{"student_id": "S", "conversation_id": "SYN_R", "segment_order": i, "labels": labels()} for i in (1, 2)]
                prepared, manifest_sha = prepared_dir(root, rows)
                run = dict(self.RUN, dataset_manifest_sha256=manifest_sha)
                good = L.write_predictions(root / "data" / "good.jsonl",
                                           [L.prediction_record("SYN_R", i, labels(), run) for i in (1, 2)])
                report = EV.run(prepared, good, None, "provenance check")
                self.assertTrue(report["predictions"]["dataset_manifest_matches_reference"])
                other = L.write_predictions(root / "data" / "other.jsonl",
                                            [L.prediction_record("SYN_R", i, labels(), self.RUN) for i in (1, 2)])
                with self.assertRaisesRegex(ValueError, "not made for this evaluation input"):
                    EV.run(prepared, other, None, "provenance check")
                stray = L.write_predictions(root / "data" / "stray.jsonl",
                                            [L.prediction_record("SYN_R", 1, labels(), run),
                                             L.prediction_record("SYN_R", 2, labels(), run),
                                             L.prediction_record("SYN_X", 9, labels(), run)])
                with self.assertRaisesRegex(ValueError, "absent from the reference"):
                    EV.run(prepared, stray, None, "provenance check")
                refs, fingerprint = EV.references_from_prepared(prepared)
                for args in ({"reference": fingerprint}, {"reference_keys": set(refs)}):   # one without the other
                    with self.assertRaisesRegex(ValueError, "go together"):
                        EV.predictions_from_file(good, **args)
                self.assertEqual(len(EV.predictions_from_file(good, reference=fingerprint, reference_keys=set(refs))[0]), 2)


class EntryWithSplit(unittest.TestCase):
    RUN = Provenance.RUN

    def test_part_evaluation_accepts_part_or_wider_predictions_but_not_foreign_interactions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(L, "ROOT", root):
                rows = [{"student_id": f"S{s}", "conversation_id": f"SYN_C{s}", "segment_order": i, "labels": labels()}
                        for s, n in ((1, 3), (2, 1), (3, 1), (4, 1)) for i in range(1, n + 1)]
                prepared, manifest_sha = prepared_dir(root, rows)
                split = root / "data" / "split.json"
                GS.write_split(prepared, split, 1, 2, 3, "check purpose", "provisional")
                _, part_of = GS.load_split(split, prepared)
                val = sorted(k for k, part in part_of.items() if part == "validation")
                run = dict(self.RUN, dataset_manifest_sha256=manifest_sha)
                write = lambda name, keys: L.write_predictions(root / "data" / name,
                                                              [L.prediction_record(c, s, labels(), run) for c, s in keys])
                for name, keys in (("part.jsonl", val), ("all.jsonl", sorted(part_of))):
                    report = EV.run(prepared, write(name, keys), None, "split check", split, "validation")
                    self.assertEqual((report["rows"], report["joint_distribution"]["included"]), (len(val), len(val)), name)
                    self.assertEqual(report["selection"]["part"], "validation")
                with self.assertRaisesRegex(ValueError, "absent from the reference"):
                    EV.run(prepared, write("foreign.jsonl", sorted(part_of) + [("SYN_X", 1)]), None, "split check",
                           split, "validation")
                missing = [k for k in sorted(part_of) if k != val[0]]          # the part is not fully covered
                with self.assertRaisesRegex(ValueError, "have no prediction"):
                    EV.run(prepared, write("missing.jsonl", missing), None, "split check", split, "validation")


class JointDistribution(unittest.TestCase):
    def test_multi_select_order_does_not_split_a_combination(self):
        k = [("SYN_J", 1), ("SYN_J", 2)]
        refs = {k[0]: labels(**{TOPIC: ["code", "concept"]}), k[1]: labels(**{TOPIC: ["concept", "code"]})}
        preds = {k[0]: labels(**{TOPIC: ["concept", "code"], AIRESP: ["Explain", "Solve-Provide"]}),
                 k[1]: labels(**{TOPIC: ["code", "concept"]})}
        j = EV.evaluate(refs, preds)["joint_distribution"]
        self.assertEqual(j["distinct"], {"reference": 1, "model": 1, "union": 1, "both": 1, "reference_only": 0, "model_only": 0})
        row = j["combinations"][0]
        self.assertEqual((row["reference"], row["model"], row["same_interaction_matches"]), (2, 2, 2))
        self.assertEqual((row["labels"][TOPIC], row["labels"][AIRESP]), (["code", "concept"], ["Solve-Provide", "Explain"]))

    def test_reference_only_and_model_only_combinations_are_listed(self):
        a, b = ("SYN_J", 1), ("SYN_J", 2)
        refs = {a: labels(), b: labels(**{STUDENT: "Guided generation"})}
        preds = {a: labels(), b: labels(**{STUDENT: "Information retrieval"})}
        j = EV.evaluate(refs, preds)["joint_distribution"]
        seen = {row["labels"][STUDENT]: (row["seen_in"], row["reference"], row["model"]) for row in j["combinations"]}
        self.assertEqual(seen, {"Task automation": ("both", 1, 1), "Guided generation": ("reference only", 1, 0),
                                "Information retrieval": ("model only", 0, 1)})
        self.assertEqual(j["reference_share_on_combinations_the_model_produced"], 0.5)

    def test_missing_reference_axes_are_excluded_and_missing_predictions_counted_not_filled(self):
        k = [("SYN_M", i) for i in (1, 2, 3, 4)]
        refs = {k[0]: labels(), k[1]: labels(), k[2]: labels(**{AIDEL: None}), k[3]: labels(**{STUDENT: "Guided generation"})}
        preds = {k[0]: labels(), k[1]: labels(**{AIRESP: None}), k[2]: labels(), k[3]: labels()}
        r = EV.evaluate(refs, preds)
        j = r["joint_distribution"]
        self.assertEqual((j["rows_considered"], j["included"], j["excluded_reference_incomplete"], j["prediction_incomplete"],
                          j["denominator"]), (4, 3, 1, 1, 3))
        self.assertAlmostEqual(j["prediction_incomplete_share"], 1 / 3)
        self.assertEqual(sum(row["reference"] for row in j["combinations"]), 3)
        self.assertEqual(sum(row["model"] for row in j["combinations"]) + j["prediction_incomplete"], 3)
        self.assertTrue(all(j["checks"].values()))
        for row in j["combinations"]:
            self.assertNotIn(None, row["labels"].values())      # an incomplete prediction is never a combination
            self.assertAlmostEqual(row["reference_share"], row["reference"] / 3)
            self.assertAlmostEqual(row["model_share"], row["model"] / 3)
        self.assertEqual(sum(row["same_interaction_matches"] for row in j["combinations"]),
                         r["all_axis_exact_match"]["matches"])
        self.assertEqual(r["all_axis_exact_match"], {"n": 3, "matches": 1, "rows_with_null_prediction": 1, "rate": 1 / 3})

    def test_same_per_axis_distributions_but_different_joint_combinations(self):
        k = [("SYN_D", 1), ("SYN_D", 2)]
        refs = {k[0]: {STUDENT: "Task automation", TOPIC: ["code"], AIDEL: ["offer"], AIRESP: ["Solve-Provide"]},
                k[1]: {STUDENT: "Information retrieval", TOPIC: ["concept"], AIDEL: ["ND"], AIRESP: ["Explain"]}}
        preds = {k[0]: {STUDENT: "Task automation", TOPIC: ["concept"], AIDEL: ["offer"], AIRESP: ["Solve-Provide"]},
                 k[1]: {STUDENT: "Information retrieval", TOPIC: ["code"], AIDEL: ["ND"], AIRESP: ["Explain"]}}
        r = EV.evaluate(refs, preds)
        for axis in (TOPIC, AIDEL, AIRESP):                 # every axis: same reference and model distribution
            self.assertEqual(r["axes"][axis]["reference_set_distribution"], r["axes"][axis]["prediction_set_distribution"])
        cm = r["axes"][STUDENT]["confusion_matrix"]["counts"]
        self.assertEqual([sum(row) for row in cm], [sum(col) for col in zip(*cm)])
        j = r["joint_distribution"]                         # ...yet no combination is shared
        self.assertEqual((j["distinct"]["both"], j["distinct"]["reference_only"], j["distinct"]["model_only"]), (0, 2, 2))
        self.assertEqual(j["reference_share_on_combinations_the_model_produced"], 0.0)
        self.assertEqual(r["all_axis_exact_match"]["matches"], 0)

    def test_row_order_and_side_specific_shares(self):
        k = [("SYN_O", i) for i in (1, 2, 3)]
        refs = {k[0]: labels(), k[1]: labels(**{STUDENT: "Guided generation"}), k[2]: labels(**{STUDENT: "Information retrieval"})}
        preds = {k[0]: labels(), k[1]: labels(), k[2]: labels(**{STUDENT: "Providing material / responding"})}
        j = EV.evaluate(refs, preds)["joint_distribution"]
        # ordered by reference count, then model count (both descending), then combination text
        self.assertEqual([r["labels"][STUDENT] for r in j["combinations"]],
                         ["Task automation", "Guided generation", "Information retrieval", "Providing material / responding"])
        self.assertAlmostEqual(j["reference_share_on_combinations_the_model_produced"], 1 / 3)
        self.assertAlmostEqual(j["model_share_on_combinations_in_the_reference"], 2 / 3)

    def test_cross_check_refuses_a_combination_that_disagrees_with_the_axis_comparison(self):
        k = ("SYN_C", 1)
        refs, preds = {k: labels(**{TOPIC: ["code", "concept"]})}, {k: labels(**{TOPIC: ["concept", "code"]})}
        self.assertEqual(EV.evaluate(refs, preds)["all_axis_exact_match"]["matches"], 1)
        broken = lambda lab: tuple(lab[a] if a in L.SINGLE else tuple(lab[a]) for a in L.AXES)   # ignores schema order
        with patch.object(EV, "_combination", broken), self.assertRaises(AssertionError):
            EV.evaluate(refs, preds)

    def test_markdown_shows_the_joint_table_and_its_denominator(self):
        k = [("SYN_M", i) for i in (1, 2, 3)]
        refs = {k[0]: labels(), k[1]: labels(**{AIDEL: None}), k[2]: labels(**{TOPIC: ["concept", "code"]})}
        preds = {k[0]: labels(), k[1]: labels(), k[2]: labels(**{TOPIC: ["code"]})}
        md = EV.markdown({"title": "Joint check", "statement": "s", **EV.evaluate(refs, preds)})
        self.assertIn("## Cross-axis joint distribution (four-axis label combinations)", md)
        self.assertIn("Included: 2 of 3 interactions", md)
        self.assertIn("denominator 2", md)
        self.assertIn("| Task automation | code + concept | offer | Solve-Provide + Explain | 1 (0.500) | 0 (0.000) |", md)
        self.assertIn("| " + " | ".join(L.AXES) + " | reference n (share) | model n (share) |", md)


class EvaluationCli(unittest.TestCase):
    RUN = {"run_id": "syn", "model_route": "synthetic", "config_sha256": "c", "dataset_manifest_sha256": "m"}

    def test_cli_refuses_unsafe_outputs_and_writes_a_private_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(L, "ROOT", root):
                rows = [{"student_id": "S", "conversation_id": "SYN_E", "segment_order": i, "labels": labels()}
                        for i in (1, 2)]
                prepared, manifest_sha = prepared_dir(root, rows)
                run = dict(self.RUN, dataset_manifest_sha256=manifest_sha)
                predictions = L.write_predictions(root / "data" / "predictions.jsonl",
                                                  [L.prediction_record("SYN_E", i, labels(), run) for i in (1, 2)])
                foreign = L.write_predictions(root / "data" / "predictions-other-data.jsonl",
                                              [L.prediction_record("SYN_E", i, labels(), self.RUN) for i in (1, 2)])
                (root / "data" / "taken.json").write_text("{}", encoding="utf-8")
                (root / "data" / "companion.md").write_text("", encoding="utf-8")

                def run(out, err, source=predictions):
                    argv = ["--prepared", str(prepared), "--predictions", str(source), "--note", "context",
                            "--out", str(out)]
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                        return EV.main(argv)
                err = io.StringIO()
                with self.assertRaises(SystemExit) as stop:      # predictions made for another dataset
                    run(root / "data" / "foreign.json", err, foreign)
                self.assertEqual(stop.exception.code, 2)
                self.assertIn("not made for this evaluation input", err.getvalue())
                self.assertFalse((root / "data" / "foreign.json").exists())
                for bad, reason in ((root / "data" / "report.md", "must be a new .json path"),
                                    (root / "data" / "taken.json", "must be a new .json path"),
                                    (root / "data" / "companion.json", "must be a new .json path"),
                                    (root / "outside.json", "below repository data/")):
                    err = io.StringIO()
                    with self.assertRaises(SystemExit, msg=str(bad)) as stop:
                        run(bad, err)
                    self.assertEqual(stop.exception.code, 2)
                    self.assertIn(reason, err.getvalue(), str(bad))
                self.assertFalse((root / "outside.json").exists())
                self.assertEqual(run(root / "data" / "report.json", io.StringIO()), 0)
                report = json.loads((root / "data" / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["all_axis_exact_match"]["rate"], 1.0)
                self.assertEqual((report["statement"], report["note"]), (EV.SYNTHETIC_STATEMENT, "context"))
                self.assertTrue(report["predictions"]["dataset_manifest_matches_reference"])
                self.assertEqual(report["joint_distribution"]["included"], 2)
                self.assertIn("Cross-axis joint distribution", (root / "data" / "report.md").read_text(encoding="utf-8"))
                self.assertTrue((root / "data" / "report.md").exists())
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = EV.statements(review_reference(tmp))
        self.assertIn("not final RQ3 performance", reviewed["all"])
        self.assertIn("可能包含训练用互动", reviewed["all"])


class Splits(unittest.TestCase):
    def examples(self):
        sizes = {"SYN_BIG": 9, "SYN_B": 3, "SYN_C": 2, "SYN_D": 2, "SYN_E": 1, "SYN_F": 1, "SYN_G": 1}
        return [{"student_id": s, "conversation_id": "C_" + s, "segment_order": i + 1,
                 "labels": labels(**({AIDEL: None, AIRESP: None} if (s, i) == ("SYN_C", 0) else {}))}
                for s, n in sizes.items() for i in range(n)]

    def test_students_are_never_divided_and_the_large_one_trains(self):
        for seed in range(20):
            split = GS.make_split(self.examples(), seed, validation_target=4, validation_max=5)
            self.assertIn("SYN_BIG", split["parts"]["train"]["students"])
            self.assertEqual(split["checks"]["student_overlap"], 0)
            self.assertLessEqual(split["parts"]["validation"]["counts"]["interactions"], 5)
            self.assertTrue(split["checks"]["every_interaction_once"])
        self.assertEqual(GS.make_split(self.examples(), 7, 4, 5), GS.make_split(self.examples(), 7, 4, 5))

    def test_empty_validation_is_refused(self):
        big = [{"student_id": s, "conversation_id": "C_" + s, "segment_order": i + 1, "labels": labels()}
               for s, n in (("SYN_A", 5), ("SYN_B", 4), ("SYN_C", 3)) for i in range(n)]
        with self.assertRaises(ValueError):
            GS.make_split(big, 1, validation_target=1, validation_max=2)

    def test_roster_is_private_written_once_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(L, "ROOT", Path(tmp)):
            root = Path(tmp)
            prepared = root / "data" / "prepared"
            prepared.mkdir(parents=True)
            rows = self.examples()
            rows[0]["note_free_text"] = "line\u2028separator\u2029inside\u0085a field"   # JSON may carry these
            text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in rows)
            with (prepared / "examples.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            keys = sorted((e["conversation_id"], e["segment_order"]) for e in self.examples())
            (prepared / "manifest.json").write_text(json.dumps({
                "prepared_file": {"sha256": L.sha256_bytes(text.encode())}, "sources": [],
                "keys_sha256": L.sha256_bytes(json.dumps(keys).encode()), "vocabulary_sha256": L.vocabulary_sha256()}), encoding="utf-8")
            self.assertEqual(len(GS.load_examples(prepared)), len(rows))
            with patch.object(L, "vocabulary_sha256", lambda: "0" * 64):   # schema changed since preparation
                with self.assertRaises(ValueError):
                    GS.prepared_fingerprint(prepared)
            with self.assertRaises(ValueError):
                GS.write_split(prepared, root / "public.json", 1, 4, 5, "purpose", "status")
            out = root / "data" / "splits" / "s.json"
            record = GS.write_split(prepared, out, 1, 4, 5, "check purpose", "provisional")
            self.assertEqual(record["validation_purpose"], "check purpose")
            with self.assertRaises(FileExistsError):
                GS.write_split(prepared, out, 1, 4, 5, "check purpose", "provisional")
            _, part_of = GS.load_split(out, prepared)
            self.assertEqual(len(part_of), len(self.examples()))
            with (prepared / "examples.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                f.write(text.replace("SYN_G", "SYN_H"))
            with self.assertRaises(ValueError):
                GS.load_split(out, prepared)


if __name__ == "__main__":
    unittest.main()
