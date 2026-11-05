"""Tests for comparison/ on synthetic records only (no data files, no network)."""
import json
import re
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "finetuning")]
import deidentify as D  # noqa: E402
import labels as L  # noqa: E402
import llm_compare as LC  # noqa: E402
import score as S  # noqa: E402

SYNTH = """jdoe@Janes-MacBook-Pro project % make
(base) jdoe@ubuntu-box:~/a2$ ./client
PS C:\\Users\\JaneDoe\\Desktop> gcc x.c
open /Users/jdoe/Desktop/a2/server.c and /home/jdoe7/a2/log.txt
-rw-r--r--@ 1 jdoe  staff  1234 Mar  3 10:00 server.c
contact jane.doe@uni.example.edu or 0412 345 678; SID 512345678; unikey abcd1234
see https://github.com/jdoe/comp2017-a2/blob/main/x.c and https://man7.org/linux/man-pages/man2/fork.2.html.
git clone git@github.com:jdoe/comp2017-a2.git
ssh -p 22 jdoe@10.0.0.7
host DESKTOP-AB12CDE ready
Hi jdoe! You've successfully authenticated, but GitHub does not provide shell access.
shared https://chatgpt.com/share/abc-123
"""


def rule(pattern, repl="[user]"):
    return {"kind": "personal name or username", "pattern": re.compile(pattern, re.I), "replacement": repl}


def test_redaction_removes_every_synthetic_identifier():
    rules = [rule(r"jane\s*doe")]
    out, counts = D.redact(SYNTH, D.replacements(rules))
    for token in ("jdoe", "JaneDoe", "jane.doe", "0412", "512345678", "abcd1234", "Janes-MacBook", "DESKTOP-AB12CDE",
                  "github.com/jdoe", "chatgpt.com/share"):
        assert token.lower() not in out.lower(), token
    assert "https://man7.org/linux/man-pages/man2/fork.2.html." in out      # public documentation kept
    assert D.residual(out, rules) == {}
    assert counts["web address"] == 1 and counts["e-mail address"] == 1


def test_residual_finds_what_was_not_redacted():
    hits = D.residual("mail a@b.com from /Users/jdoe/x and https://example.com/p", [])
    assert hits["email address"] == 1 and hits["user folder name"] == 1 and hits["other web address"] == 1


def test_redaction_is_idempotent():
    passes = D.replacements([])
    once, _ = D.redact(SYNTH, passes)
    twice, counts = D.redact(once, passes)
    assert twice == once and not counts


def test_schema_matches_vocabulary_and_is_strict():
    full, student = LC.output_schema(False), LC.output_schema(True)
    assert list(full["properties"]) == list(LC.OUT_KEYS) and list(student["properties"]) == list(LC.OUT_KEYS)[:2]
    for key, axis in LC.OUT_KEYS.items():
        prop = full["properties"][key]
        assert prop["additionalProperties"] is False and set(prop["required"]) == set(prop["properties"])
        enum = prop["properties"]["code"]["enum"] if axis in L.SINGLE else prop["properties"]["codes"]["items"]["enum"]
        assert enum == list(L.DIRECT_CODES[axis])


def test_prompts_carry_the_guide_unchanged_and_every_code():
    guide = "GUIDE TEXT\n**Rule 1.** x"
    system = LC.system_prompt(guide)
    assert system.endswith(guide)
    for axis in L.AXES:
        for code in L.DIRECT_CODES[axis]:
            assert code in system
    ex = {"context_short_because": "gap", "context": [{"segment_order": 4, "student_prompt": "s4", "llm_response": "r4"}],
          "student_prompt": "s5", "llm_response": "r5"}
    user = LC.user_prompt(ex, truncated=True)
    assert "[earlier turns not available]" in user and "[interaction to code]" in user and user.endswith(LC.TRUNCATED_NOTE)


def test_request_body():
    a = LC.request_body(LC.SETTINGS, "SYS", "USER", False)
    b = LC.request_body(LC.SETTINGS, "SYS", "USER", True)
    assert a["store"] is False and "temperature" not in a and a["reasoning"] == {"effort": LC.SETTINGS["reasoning_effort"]}
    assert [m["content"] for m in a["input"]] == ["SYS", "USER"]
    assert [m["content"] for m in b["input"]] == ["SYS", "USER"]
    assert list(b["text"]["format"]["schema"]["properties"]) == ["student_interaction_type", "topic"]


def response(obj, status="completed"):
    return {"status": status, "model": "gpt-test", "usage": {"input_tokens": 1000, "output_tokens": 100,
                                                            "input_tokens_details": {"cached_tokens": 800}},
            "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": json.dumps(obj)}]}]}


GOOD = {"student_interaction_type": {"code": "Guided generation", "reason": "r"},
        "topic": {"codes": ["concept", "code"], "reason": "r"},
        "ai_delegation": {"codes": ["offer"], "reason": "r"},
        "ai_response_type": {"codes": ["Evaluate", "Solve-Provide"], "reason": "r"}}


def test_parse_output():
    labels, reasons, errors = LC.parse_output(response(GOOD), False)
    assert errors == [] and labels["topic"] == ["code", "concept"]
    assert labels["AI response type"] == ["Solve-Provide", "Evaluate"]
    bad = json.loads(json.dumps(GOOD)); bad["ai_delegation"]["codes"] = ["ND", "offer"]
    assert LC.parse_output(response(bad), False)[0] is None
    empty = json.loads(json.dumps(GOOD)); empty["topic"]["codes"] = []
    assert LC.parse_output(response(empty), False)[0] is None
    assert LC.parse_output(response(GOOD, "incomplete"), False)[2][0].startswith("response status incomplete")
    student = {k: GOOD[k] for k in ("student_interaction_type", "topic")}
    labels, _, errors = LC.parse_output(response(student), True)
    assert errors == [] and labels["AI delegation"] is None and labels["AI response type"] is None
    assert LC.parse_output(response(GOOD), True)[0] is None          # AI fields when only student ones were asked


ROW = {"conversation_id": "P001_P2_01", "segment_order": 3, "assignment": "P2", "truncated": False, "user_text": "U"}


def fake_post(sequence):
    calls = []

    def post(url, body, key, timeout):
        calls.append(body)
        item = sequence[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item
    return post, calls


def test_call_one_retries_invalid_once_then_accepts():
    post, calls = fake_post([(200, {}, response({"bad": 1})), (200, {}, response(GOOD))])
    logged = []
    recs = LC.call_one(ROW, "A", "SYS", "k", logged.append, post, sleep=lambda s: None)
    assert [r["outcome"] for r in recs] == ["invalid", "accepted"] and recs[-1]["final"]
    assert calls[0]["input"][1]["content"] == "U" and len(logged) == 2
    assert abs(recs[0]["cost_usd"] - (200 * 1.25 + 800 * 0.05 + 100 * 5) / 1e6) < 1e-12


def test_call_one_fails_after_one_retry_and_handles_transient_errors():
    post, _ = fake_post([(200, {}, response({"bad": 1})), (200, {}, response({"bad": 2}))])
    recs = LC.call_one(ROW, "A", "SYS", "k", lambda r: None, post, sleep=lambda s: None)
    assert [r["outcome"] for r in recs] == ["invalid", "failed"]
    post, _ = fake_post([urllib.error.URLError("x"), (429, {"retry-after": "1"}, {}), (200, {}, response(GOOD))])
    recs = LC.call_one(ROW, "A", "SYS", "k", lambda r: None, post, sleep=lambda s: None)
    assert recs[-1]["outcome"] == "accepted" and recs[-1]["transient_retries"] == 2
    post, _ = fake_post([(400, {}, {"error": {"type": "invalid_request_error", "message": "bad param"}})])
    with pytest.raises(LC.Fatal):
        LC.call_one(ROW, "A", "SYS", "k", lambda r: None, post, sleep=lambda s: None)


def test_pilot_keys_whole_conversations():
    rows = [{"conversation_id": f"P{c:03d}_P2_01", "segment_order": s, "part": "150" if c < 8 else "135"}
            for c in range(10) for s in range(1, 2 + c % 4)]
    k1 = LC.pilot_keys(rows, 20261028, 10, "150")
    assert k1 == LC.pilot_keys(rows, 20261028, 10, "150") and len(k1) <= 10
    convs = {c for c, _ in k1}
    assert all(int(c[1:4]) < 8 for c in convs)
    for c in convs:
        assert {s for cc, s in k1 if cc == c} == {r["segment_order"] for r in rows if r["conversation_id"] == c}


def test_prediction_records_round_trip(tmp_path):
    rows = [dict(ROW, segment_order=1), dict(ROW, segment_order=2, truncated=True), dict(ROW, segment_order=3)]
    good = LC.parse_output(response(GOOD), False)[0]
    student = LC.parse_output(response({k: GOOD[k] for k in ("student_interaction_type", "topic")}), True)[0]
    final = {(ROW["conversation_id"], 1): {"outcome": "accepted", "labels": good},
             (ROW["conversation_id"], 2): {"outcome": "accepted", "labels": student},
             (ROW["conversation_id"], 3): {"outcome": "failed", "errors": ["not JSON: x"]}}
    run = {"run_id": "t", "model_route": "openai/gpt-test", "config_sha256": "c", "dataset_manifest_sha256": "d"}
    recs = LC.prediction_records(rows, final, run)
    path = L.write_predictions(tmp_path / "p.jsonl", recs)
    back = L.read_predictions(path)
    assert back[1]["not_predicted"] == {"AI delegation": LC.TRUNCATED_REASON, "AI response type": LC.TRUNCATED_REASON}
    assert set(back[2]["not_predicted"]) == set(L.AXES) and back[2]["labels"]["topic"] is None


def test_per_code_agreement_and_absent_side():
    fs = frozenset
    ref, a, b = {}, {}, {}
    for i, (r, p1, p2) in enumerate([("SE", "S", "SE"), ("S", "S", "S"), ("E", "SE", "E"), ("S", "S", "SE")]):
        k = (f"P00{i % 2}_P2_01", i + 1)

        def lab(x):
            codes = [c for c, ch in (("Solve-Provide", "S"), ("Explain", "E")) if ch in x]
            return {"student interaction type": fs(["Task automation"]), "topic": fs(["code"]),
                    "AI delegation": fs(["ND"]), "AI response type": fs(codes)}
        ref[k], a[k], b[k] = lab(r), lab(p1), lab(p2)
    out = S.per_code(ref, {"A": a, "B": b}, sorted(ref), [("B", "A")], 200, 1)
    sp = out["AI response type"]["codes"]["Solve-Provide"]
    assert sp["reference"] == 3 and sp["models"]["A"]["B"] == 4 and sp["models"]["A"]["C"] == 3
    assert sp["models"]["A"]["agreement"] == 0.75
    ex = out["AI response type"]["codes"]["Explain (absent)"]          # reference lacks Explain in rows 2 and 4
    # A lacks it in rows 1, 2, 4 (C = 2, 2/3); B only in row 2 (C = 1, 1/2)
    assert ex["reference"] == 2 and ex["models"]["A"]["agreement"] == 2 / 3 and ex["models"]["B"]["agreement"] == 0.5
    assert abs(ex["comparisons"]["B - A"]["difference"] - (0.5 - 2 / 3)) < 1e-12
    assert out["topic"]["codes"]["concept"]["models"]["A"]["agreement"] is None


def test_run_calls_resumes_and_respects_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(LC, "RESULTS", tmp_path)
    rows = [dict(ROW, segment_order=s) for s in (1, 2, 3)]
    post = lambda url, body, key, timeout: (200, {}, response(GOOD))      # noqa: E731
    rep = LC.run_calls(rows[:2], "A", "pilot", "SYS", "k", 1, post, sleep=lambda s: None)
    assert rep["final"] == 2 and rep["outcomes"] == {"accepted": 2}
    rep = LC.run_calls(rows, "A", "pilot", "SYS", "k", 1, post, sleep=lambda s: None)
    assert rep["already_done"] == 2 and rep["final"] == 3
    monkeypatch.setitem(LC.SETTINGS, "budget_usd", LC.spent_so_far() + 0.0001)
    rep = LC.run_calls(rows, "A", "all", "SYS", "k", 1, post, sleep=lambda s: None)
    assert rep["stopped_for_budget"] and rep["final"] == 0


def test_conversation_scope_replaces_tokens_and_hosts_after_user():
    passes = D.replacements([])
    a = "(base) jdoe@ubuntu-box:~/a2$ ./client\nThe key fingerprint is: SHA256:abc [user]@jdoes-laptop\n"
    b = "jdoe asked again; jdoes-laptop is the machine; src, staff and a Mac stay"
    toks = D.conversation_tokens([a, b], passes)
    assert toks == {"jdoe", "jdoes-laptop", "ubuntu-box"}
    ra, ca = D.redact_scoped(a, passes, toks)
    rb, cb = D.redact_scoped(b, passes, toks)
    for t in (ra, rb):
        assert "jdoe" not in t.lower() and D.scoped_residual(t, toks) == 0
    assert "[user]@[host]" in ra and ca["host after replaced user"] == 1
    assert cb["conversation token"] == 2 and "src, staff and a Mac stay" in rb


def test_conversation_scope_leaves_generic_and_confirmed_tokens():
    passes = D.replacements([])
    t = "ubuntu@primary:~$ ls\nconnected as [user]@localhost"
    toks = D.conversation_tokens([t], passes)
    assert toks == set()
    out, counts = D.redact_scoped(t, passes, toks)
    assert "[user]@localhost" in out and "host after replaced user" not in counts
