#!/usr/bin/env python3
"""
comparison/llm_compare.py - prompted LLM comparison with C1 (D-192; docs/planning/llm-comparison-spec-20261028.md).

    python comparison/llm_compare.py build            # inputs, de-identification, prompt files, pilot ids; no network
    python comparison/llm_compare.py estimate         # projected tokens and cost from characters; no network
    python comparison/llm_compare.py call --set pilot                          # 10 interactions
    python comparison/llm_compare.py call --set all --variant A --pilot-approved
    python comparison/llm_compare.py predictions --set all                     # model_prediction files
    python comparison/score.py ...                                             # scoring (see that file)

Reruns (2 November 2026): set LLM_RUN_TAG=r2 (r3, ...) before every command. The run then reads and writes its own
inputs, de-identification report and results folder (data/results/llm-terra-20261028-<tag>), uses the same settings,
needs the original pilot summary instead of a new pilot, and the US$20 budget counts the calls of every run.

`call` needs OPENAI_API_KEY in the environment; the key is never written anywhere. Requests are stateless
(`store: false`), one per interaction. The prompt gives the model the 27 October coding guide; runs and files carry
the label A. Nothing here prints interaction text: only ids, counts, tokens and costs. Every attempt is
appended to a private log under data/results/, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, ROOT / "coding", ROOT / "finetuning", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import labels as L  # noqa: E402
import deidentify as D  # noqa: E402

TASK = ROOT / "data/maintenance/20261028-llm-terra"
PREPARED = TASK / "prepared"
RUN_TAG = os.environ.get("LLM_RUN_TAG", "").strip()
if RUN_TAG and not re.fullmatch(r"r[2-9]", RUN_TAG):
    sys.exit("REFUSED: LLM_RUN_TAG must be r2 to r9")
BASE_RESULTS = ROOT / "data/results/llm-terra-20261028"
INPUTS = TASK / (f"inputs-redacted-{RUN_TAG}.jsonl" if RUN_TAG else "inputs-redacted.jsonl")
DEID_REPORT = TASK / (f"deidentification-report-{RUN_TAG}.json" if RUN_TAG else "deidentification-report.json")
RESULTS = ROOT / f"data/results/llm-terra-20261028-{RUN_TAG}" if RUN_TAG else BASE_RESULTS
GUIDE = ROOT / "docs/codebook/guide-short-20261027.md"
LABELS = ROOT / "data/labels/dw485-reviewed-20261028d.jsonl"
PARTS = (("150", ROOT / "data/labels/dw150-reviewed-v15.jsonl"),
         ("135", ROOT / "data/coding/direct-20260924/researcher-135-20261022/reviewed/dw135-reviewed-v4.jsonl"),
         ("200", ROOT / "data/coding/direct-20260924/researcher-200-20261027/reviewed/dw200-reviewed-v1.jsonl"))
VARIANTS = ("A",)

SETTINGS = {
    "model": "gpt-6.1-terra",
    "endpoint": "https://api.openai.com/v1/responses",
    "reasoning_effort": "medium",
    "temperature": None,
    "temperature_note": "not sent: the model page lists reasoning effort settings and no temperature parameter",
    "max_output_tokens": 16000,
    "store": False,
    "context_interactions": 3,
    "pilot": {"seed": 20261028, "size": 10, "part": "150", "unit": "whole conversations"},
    "budget_usd": 20.0,
    "prices_usd_per_million": {"input": 1.00, "cached_input": 0.05, "output": 5.00, "cache_write": 1.25},
    "cost_rule": "uncached input counted at the cache-write price (1.25), so the budget check errs high",
    "workers": 3,
    "invalid_output_retries": 1,
    "transient_http_retries": 6,
    "timeout_seconds": 300,
}

# Output keys of the structured response, and the axes they fill.
OUT_KEYS = {"student_interaction_type": "student interaction type", "topic": "topic",
            "ai_delegation": "AI delegation", "ai_response_type": "AI response type"}
STUDENT_AXES = ("student interaction type", "topic")
TRUNCATED_REASON = "truncated capture: AI dimensions not requested from the model"

SYSTEM_HEAD = """You code interactions between university students and an AI assistant. The students were taking a \
second-year systems programming course taught in C (COMP2017, 2025) and used the assistant while working on an \
assignment. Each request gives one target interaction (a student prompt and the assistant response that follows it) \
and up to three earlier interactions from the same conversation as context. Code the target interaction on four \
dimensions, following the coding guide below.

# Output

Return JSON with one entry per dimension:
- student_interaction_type: exactly one code.
- topic: one or more codes.
- ai_delegation: one or more codes; ND only on its own.
- ai_response_type: one or more codes; "none of the four applies" only on its own.
Each entry also has a reason of at most 25 words naming the evidence that decided it. Do not copy long passages.
If the request says that the target response is truncated, return only the two student dimensions.

Codes, to be written exactly as here:
{codes}

# How to apply the guide in this task

- Code only the interaction marked [interaction to code]. The [earlier interaction N] blocks are context; do not code \
them.
- "[earlier turns not available]" means that earlier turns of the conversation exist but are not shown: the first \
block after it follows a gap (the guide's "turn after a gap").
- Code what the turn does in its context, not how it is worded (Rule 1). Code the two AI dimensions from the current \
response only (Rule 2).
- The guide's flags and notes have no place in this output. Where it says to flag or mark a code as provisional, \
choose the better-supported code (Rule 17) and name the competing reading in the reason. Never select two codes to \
express doubt.

# Coding guide

"""

USER_HEAD = "Code the interaction marked [interaction to code].\n\n"
TRUNCATED_NOTE = ("\n\nThe assistant response of the target interaction is truncated in the capture. Return only "
                  "student_interaction_type and topic.")


# ----------------------------------------------------------------------------- small helpers

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def key_text(cid: str, seg: int) -> str:
    return f"{cid}#{seg}"


def assignment_of(cid: str) -> str:
    m = re.search(r"_(P[12])_", cid)
    if not m:
        raise ValueError(f"{cid}: no _P1_ or _P2_ in the conversation id")
    return m.group(1)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ----------------------------------------------------------------------------- prompts and schema

def codes_block() -> str:
    hints = {}
    for f in L.DIRECT_INSTRUMENT["fields"]:
        hints.update(f.get("hints", {}))
    lines = []
    for key, axis in OUT_KEYS.items():
        sel = "select one" if axis in L.SINGLE else "select every code that applies"
        lines.append(f"- {key} ({axis}; {sel}):")
        for c in L.DIRECT_CODES[axis]:
            lines.append(f"  - {c}" + (f": {hints[c]}" if c in hints else ""))
    return "\n".join(lines)


def system_prompt(guide_text: str) -> str:
    return SYSTEM_HEAD.format(codes=codes_block()) + guide_text


def user_prompt(example: dict, truncated: bool) -> str:
    return USER_HEAD + L.render_input(example) + (TRUNCATED_NOTE if truncated else "")


def output_schema(student_only: bool = False) -> dict:
    props, required = {}, []
    for key, axis in OUT_KEYS.items():
        if student_only and axis not in STUDENT_AXES:
            continue
        codes = list(L.DIRECT_CODES[axis])
        if axis in L.SINGLE:
            value = {"code": {"type": "string", "enum": codes}}
        else:
            value = {"codes": {"type": "array", "items": {"type": "string", "enum": codes}}}
        value["reason"] = {"type": "string"}
        props[key] = {"type": "object", "properties": value, "required": list(value), "additionalProperties": False}
        required.append(key)
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def request_body(settings: dict, system: str, user: str, student_only: bool) -> dict:
    messages = [{"role": "developer", "content": system}, {"role": "user", "content": user}]
    body = {"model": settings["model"], "input": messages, "store": settings["store"],
            "max_output_tokens": settings["max_output_tokens"],
            "text": {"format": {"type": "json_schema", "name": "interaction_codes", "strict": True,
                                "schema": output_schema(student_only)}}}
    if settings.get("reasoning_effort"):
        body["reasoning"] = {"effort": settings["reasoning_effort"]}
    if settings.get("temperature") is not None:
        body["temperature"] = settings["temperature"]
    return body


# ----------------------------------------------------------------------------- build

def part_map() -> dict:
    out, conv_part = {}, {}
    for part, path in PARTS:
        for r in read_jsonl(path):
            k = (r["conversation_id"], r["segment_order"])
            if k in out:
                raise ValueError(f"{key_text(*k)} is in two parts")
            if conv_part.setdefault(k[0], part) != part:
                raise ValueError(f"{k[0]} has interactions in two parts")
            out[k] = part
    return out


def pilot_keys(rows: list[dict], seed: int, size: int, part: str) -> list[tuple[str, int]]:
    """Whole conversations of one part, in a seeded order, added while the total stays within `size`."""
    by_conv = defaultdict(list)
    for r in rows:
        if r["part"] == part:
            by_conv[r["conversation_id"]].append(r["segment_order"])
    convs = sorted(by_conv)
    random.Random(seed).shuffle(convs)
    chosen = []
    for c in convs:
        if len(chosen) + len(by_conv[c]) <= size:
            chosen += [(c, s) for s in sorted(by_conv[c])]
        if len(chosen) == size:
            break
    return sorted(chosen)


def redact_examples(examples: list[dict], parts: dict, rules: list[dict]):
    """Redact every distinct corpus text once. Returns (redacted examples, counts, residual hits)."""
    passes = D.replacements(rules)
    texts = {}
    for ex in examples:
        cid = ex["conversation_id"]
        for seg, sp, lr in [(ex["segment_order"], ex["student_prompt"], ex["llm_response"])] + \
                           [(c["segment_order"], c["student_prompt"], c["llm_response"]) for c in ex["context"]]:
            for field, text in (("student_prompt", sp), ("llm_response", lr)):
                k = (cid, seg, field)
                if k in texts and texts[k] != text:
                    raise ValueError(f"{key_text(cid, seg)} {field}: two different texts")
                texts[k] = text
    part_of_conv = {cid: p for (cid, _), p in parts.items()}
    by_conv = defaultdict(list)
    for (cid, _, _), text in texts.items():
        by_conv[cid].append(text)
    tokens = {cid: D.conversation_tokens(ts, passes) for cid, ts in by_conv.items()}
    red, counts, affected, residual = {}, defaultdict(Counter), defaultdict(lambda: defaultdict(set)), []
    for (cid, seg, field), text in texts.items():
        new, c = D.redact_scoped(text, passes, tokens[cid])
        red[(cid, seg, field)] = new
        part = part_of_conv[cid]
        for name, n in c.items():
            counts[part][name] += n
            affected[part][name].add((cid, seg))
        for kind, n in D.residual(new, rules).items():
            residual.append({"conversation_id": cid, "segment_order": seg, "field": field, "kind": kind, "hits": n})
        left = D.scoped_residual(new, tokens[cid])
        if left:
            residual.append({"conversation_id": cid, "segment_order": seg, "field": field,
                             "kind": "conversation token or host after replaced user", "hits": left})
    out = []
    for ex in examples:
        cid = ex["conversation_id"]
        out.append({**ex,
                    "student_prompt": red[(cid, ex["segment_order"], "student_prompt")],
                    "llm_response": red[(cid, ex["segment_order"], "llm_response")],
                    "context": [{"segment_order": c["segment_order"],
                                 "student_prompt": red[(cid, c["segment_order"], "student_prompt")],
                                 "llm_response": red[(cid, c["segment_order"], "llm_response")]}
                                for c in ex["context"]]})
    report = {part: {name: {"replacements": n, "rows_affected": len(affected[part][name])}
                     for name, n in sorted(counts[part].items())} for part in sorted(counts)}
    return out, report, residual, len(texts)


def build(args) -> int:
    manifest_path = PREPARED / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"REFUSED: {manifest_path} missing; run finetuning/prepare_data.py first (spec section 3)")
    for p in (INPUTS, DEID_REPORT, RESULTS / "config.json"):
        if p.exists():
            sys.exit(f"REFUSED: {p.relative_to(ROOT)} exists; outputs are never overwritten")
    examples = read_jsonl(PREPARED / "examples.jsonl")
    parts = part_map()
    keys = {(e["conversation_id"], e["segment_order"]) for e in examples}
    if keys != set(parts):
        sys.exit("REFUSED: the prepared examples and the three label parts do not hold the same interactions")
    rules = D.load_adopted_rules()
    redacted, report, residual, n_texts = redact_examples(examples, parts, rules)
    deid = {"created_utc": now(), "distinct_texts": n_texts, "adopted_rules_file": L.rel(D.ADOPTED_RULES),
            "adopted_rules_sha256": L.sha256_file(D.ADOPTED_RULES), "replacements_by_part": report,
            "residual_hits": len(residual), "residual": residual,
            "deidentify_py_sha256": L.sha256_file(HERE / "deidentify.py"), "run_tag": RUN_TAG or None,
            "note": "Counts of distinct corpus texts (a text shown as context and as a target is counted once). "
                    "Names written in free text are not screened."}
    write_new(DEID_REPORT, json.dumps(deid, ensure_ascii=False, indent=1) + "\n")
    if residual:
        print(f"REFUSED: {len(residual)} residual screen hits; ids in {DEID_REPORT.relative_to(ROOT)}")
        return 1

    rows = []
    for ex in redacted:
        truncated = bool(ex["masked_axes"])
        if truncated and set(ex["masked_axes"]) != {"AI delegation", "AI response type"}:
            sys.exit(f"REFUSED: {key_text(ex['conversation_id'], ex['segment_order'])} masks an unexpected axis")
        rows.append({"conversation_id": ex["conversation_id"], "segment_order": ex["segment_order"],
                     "student": ex["conversation_id"].split("_", 1)[0],
                     "part": parts[(ex["conversation_id"], ex["segment_order"])],
                     "assignment": assignment_of(ex["conversation_id"]), "truncated": truncated,
                     "mask_reasons": ex["mask_reasons"], "context_available": ex["context_available"],
                     "user_text": user_prompt(ex, truncated)})
    rows.sort(key=lambda r: (r["conversation_id"], r["segment_order"]))
    inputs_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    write_new(INPUTS, inputs_text)

    files = prompt_files()
    system = files["system.txt"]
    for name, text in files.items():
        write_new(RESULTS / "prompts" / name, text)
    pk = pilot_keys(rows, SETTINGS["pilot"]["seed"], SETTINGS["pilot"]["size"], SETTINGS["pilot"]["part"])
    write_new(RESULTS / "pilot-keys.json", json.dumps([key_text(*k) for k in pk], indent=1) + "\n")
    write_new(RESULTS / "config.json", json.dumps(make_config(), ensure_ascii=False, indent=1) + "\n")

    print(f"inputs: {len(rows)} interactions, parts {dict(Counter(r['part'] for r in rows))}, "
          f"assignments {dict(Counter(r['assignment'] for r in rows))}, truncated {sum(r['truncated'] for r in rows)}")
    print(f"de-identification: {n_texts} distinct texts; residual hits 0")
    for part, kinds in report.items():
        print(f"  part {part}: " + "; ".join(f"{k} {v['replacements']} ({v['rows_affected']} rows)" for k, v in kinds.items()))
    print(f"pilot: {len(pk)} interactions from {len({k[0] for k in pk})} conversations of part {SETTINGS['pilot']['part']}")
    print(f"system prompt {len(system)} characters")
    return 0


def prompt_files() -> dict:
    return {"system.txt": system_prompt(GUIDE.read_text(encoding="utf-8")), "user-head.txt": USER_HEAD, "truncated-note.txt": TRUNCATED_NOTE,
            "output-schema.json": json.dumps(output_schema(False), indent=1) + "\n",
            "output-schema-student-only.json": json.dumps(output_schema(True), indent=1) + "\n"}


def make_config(history: list | None = None) -> dict:
    """Settings and the SHA-256 of every fixed input, computed from the files on disk."""
    files = prompt_files()
    for name, text in files.items():
        if (RESULTS / "prompts" / name).read_text(encoding="utf-8") != text:
            raise ValueError(f"prompt file {name} on disk differs from the code's prompt")
    manifest_path = PREPARED / "manifest.json"
    prompt_sha = {n: sha256_text(t) for n, t in files.items()}
    config = {"created_utc": now(), "run_tag": RUN_TAG or None, "settings": SETTINGS, "variants": {"A": "guide only"},
              "guide": {"path": L.rel(GUIDE), "sha256": L.sha256_file(GUIDE)},
              "prompt_files_sha256": prompt_sha,
              "inputs": {"path": L.rel(INPUTS), "sha256": L.sha256_file(INPUTS), "rows": len(read_jsonl(INPUTS))},
              "dataset_manifest": {"path": L.rel(manifest_path), "sha256": L.sha256_file(manifest_path)},
              "labels": {"path": L.rel(LABELS), "sha256": L.sha256_file(LABELS)},
              "deidentification_report_sha256": L.sha256_file(DEID_REPORT)}
    if history:
        config["history"] = history
    config["config_sha256"] = {v: sha256_text(canon({"settings": SETTINGS, "variant": v, "prompts": prompt_sha,
                                                     "guide": config["guide"]["sha256"]}))
                               for v in VARIANTS}
    return config


def reconfigure(args) -> int:
    """A new config.json after a setting changed; inputs, prompts and pilot ids stay as built."""
    path = RESULTS / "config.json"
    if path.exists():
        sys.exit("REFUSED: archive the current config.json and its raw logs first")
    if any((RESULTS / "raw").glob("*.jsonl")):
        sys.exit("REFUSED: raw logs made under the old settings are still in raw/; archive them first")
    write_new(path, json.dumps(make_config([args.note]), ensure_ascii=False, indent=1) + "\n")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    print("config.json written; config_sha256 " + json.dumps(cfg["config_sha256"]))
    return 0


# ----------------------------------------------------------------------------- estimate

def load_state():
    config = json.loads((RESULTS / "config.json").read_text(encoding="utf-8"))
    if config["settings"] != json.loads(json.dumps(SETTINGS)):
        sys.exit("REFUSED: SETTINGS differ from those recorded in config.json; a changed setting is a new build")
    if L.sha256_file(INPUTS) != config["inputs"]["sha256"]:
        sys.exit("REFUSED: the redacted inputs differ from those recorded in config.json")
    if L.sha256_file(GUIDE) != config["guide"]["sha256"]:
        sys.exit("REFUSED: the guide differs from the one recorded in config.json")
    system = (RESULTS / "prompts/system.txt").read_text(encoding="utf-8")
    if sha256_text(system) != config["prompt_files_sha256"]["system.txt"] or system != system_prompt(GUIDE.read_text(encoding="utf-8")):
        sys.exit("REFUSED: the system prompt differs from the recorded one")
    rows = read_jsonl(INPUTS)
    return config, system, rows


def estimate(args) -> int:
    config, system, rows = load_state()
    chars_per_token = 3.5
    prices = SETTINGS["prices_usd_per_million"]
    out_tokens = 1500
    for v in VARIANTS:
        fixed = [len(system) for r in rows]
        var = [len(r["user_text"]) for r in rows]
        tin = (sum(fixed) + sum(var)) / chars_per_token
        tfixed = sum(fixed) / chars_per_token
        upper = tin * prices["cache_write"] / 1e6 + len(rows) * out_tokens * prices["output"] / 1e6
        cached = ((tin - tfixed) * prices["cache_write"] + tfixed * prices["cached_input"]) / 1e6 \
            + len(rows) * out_tokens * prices["output"] / 1e6
        print(f"variant {v}: about {tin / 1e6:.2f}M input tokens ({tfixed / 1e6:.2f}M repeated prefix); "
              f"cost if nothing is cached ${upper:.2f}; if the prefix is cached ${cached:.2f} "
              f"(assumes {out_tokens} output tokens per call and {chars_per_token} characters per token)")
    return 0


# ----------------------------------------------------------------------------- calling

class Fatal(Exception):
    pass


def post_json(url: str, body: dict, api_key: str, timeout: float) -> tuple[int, dict, dict]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8"))
        except Exception:
            payload = {}
        return e.code, dict(e.headers or {}), payload


def cost_of(usage: dict | None) -> float:
    if not usage:
        return 0.0
    p = SETTINGS["prices_usd_per_million"]
    tin = usage.get("input_tokens", 0) or 0
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    tout = usage.get("output_tokens", 0) or 0
    return ((tin - cached) * p["cache_write"] + cached * p["cached_input"] + tout * p["output"]) / 1e6


def parse_output(response: dict, student_only: bool) -> tuple[dict | None, dict, list[str]]:
    """Labels for the four axes (AI axes None when student_only), reasons, and errors. Never repairs."""
    if response.get("status") != "completed":
        reason = (response.get("incomplete_details") or {}).get("reason")
        return None, {}, [f"response status {response.get('status')}" + (f" ({reason})" if reason else "")]
    texts, refusals = [], 0
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                texts.append(c.get("text", ""))
            elif c.get("type") == "refusal":
                refusals += 1
    if refusals:
        return None, {}, ["refusal"]
    if len(texts) != 1:
        return None, {}, [f"expected one output text, found {len(texts)}"]
    try:
        obj = json.loads(texts[0])
    except json.JSONDecodeError as e:
        return None, {}, [f"not JSON: {e.msg}"]
    wanted = [k for k, a in OUT_KEYS.items() if not student_only or a in STUDENT_AXES]
    if not isinstance(obj, dict) or sorted(obj) != sorted(wanted):
        return None, {}, ["unexpected keys"]
    labels, reasons, errors = {}, {}, []
    for key, axis in OUT_KEYS.items():
        if key not in wanted:
            labels[axis] = None
            continue
        entry = obj[key]
        if not isinstance(entry, dict):
            return None, {}, [f"{key}: not an object"]
        labels[axis] = entry.get("code") if axis in L.SINGLE else entry.get("codes")
        reasons[axis] = entry.get("reason")
    errors += L.check_labels(labels)
    if errors:
        return None, reasons, errors
    return {a: L.canonical(a, labels[a]) for a in L.AXES}, reasons, []


def raw_path(set_name: str, variant: str) -> Path:
    return RESULTS / "raw" / f"{set_name}-{variant}.jsonl"


def spent_so_far() -> float:
    """All calls so far in every run (the original and the reruns), including raw logs archived after a settings
    change (any raw/ folder below a results folder)."""
    total = 0.0
    dirs = {RESULTS}
    if RESULTS.parent == BASE_RESULTS.parent and RESULTS.name.startswith(BASE_RESULTS.name):
        dirs |= {d for d in BASE_RESULTS.parent.glob(BASE_RESULTS.name + "*") if d.is_dir()}
    paths = [p for d in sorted(dirs) for p in d.rglob("*.jsonl")]
    for p in paths:
        if p.parent.name != "raw":
            continue
        for r in read_jsonl(p):
            total += r.get("cost_usd") or 0.0
    return total


def final_records(path: Path) -> dict:
    out = {}
    if path.exists():
        for r in read_jsonl(path):
            if r.get("final"):
                out[(r["conversation_id"], r["segment_order"])] = r
    return out


def call_one(row: dict, variant: str, system: str, api_key: str, log, post=post_json,
             sleep=time.sleep) -> list[dict]:
    """All attempts for one interaction; returns the attempt records (the last one is final)."""
    student_only = row["truncated"]
    body = request_body(SETTINGS, system, row["user_text"], student_only)
    request_sha = sha256_text(canon(body))
    records, invalid = [], 0
    attempt = 0
    while True:
        attempt += 1
        transient = 0
        while True:
            t0 = time.monotonic()
            try:
                status, headers, payload = post(SETTINGS["endpoint"], body, api_key, SETTINGS["timeout_seconds"])
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                status, headers, payload = None, {}, {"error": {"type": type(e).__name__}}
            elapsed = time.monotonic() - t0
            if status == 200:
                break
            if status in (None, 408, 409, 429) or (status or 0) >= 500:
                transient += 1
                if transient > SETTINGS["transient_http_retries"]:
                    break
                wait = min(60.0, 2.0 ** transient) + random.random()
                ra = {k.lower(): v for k, v in headers.items()}
                if "retry-after-ms" in ra:
                    wait = max(wait, float(ra["retry-after-ms"]) / 1000)
                elif "retry-after" in ra:
                    try:
                        wait = max(wait, float(ra["retry-after"]))
                    except ValueError:
                        pass
                sleep(wait)
                continue
            err = payload.get("error") or {}
            raise Fatal(f"HTTP {status}: {err.get('type')} {err.get('code')} {str(err.get('message'))[:200]}")
        rec = {"conversation_id": row["conversation_id"], "segment_order": row["segment_order"], "variant": variant,
               "attempt": attempt, "http_status": status, "transient_retries": transient,
               "elapsed_seconds": round(elapsed, 2), "request_sha256": request_sha, "utc": now()}
        if status != 200:
            rec.update(outcome="failed", final=True, errors=[f"HTTP {status} after {transient - 1} retries"],
                       cost_usd=0.0)
            log(rec)
            records.append(rec)
            return records
        usage = payload.get("usage")
        labels, reasons, errors = parse_output(payload, student_only)
        rec.update(model=payload.get("model"), status=payload.get("status"), usage=usage, cost_usd=cost_of(usage),
                   labels=labels, reasons=reasons, errors=errors)
        if labels is not None:
            rec.update(outcome="accepted", final=True)
        elif invalid < SETTINGS["invalid_output_retries"]:
            invalid += 1
            rec.update(outcome="invalid", final=False)
        else:
            rec.update(outcome="failed", final=True)
        log(rec)
        records.append(rec)
        if rec["final"]:
            return records


def run_calls(rows: list[dict], variant: str, set_name: str, system: str, api_key: str,
              workers: int, post=post_json, sleep=time.sleep) -> dict:
    path = raw_path(set_name, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = final_records(path)
    todo = [r for r in rows if (r["conversation_id"], r["segment_order"]) not in done]
    lock = threading.Lock()
    state = {"spent": spent_so_far(), "costs": [], "fatal": None, "stopped_for_budget": False, "n": 0}
    budget = SETTINGS["budget_usd"]

    def log(rec):
        with lock:
            with path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["spent"] += rec.get("cost_usd") or 0.0
            if rec.get("cost_usd"):
                state["costs"].append(rec["cost_usd"])

    def guarded(row):
        with lock:
            if state["fatal"] or state["stopped_for_budget"]:
                return None
            per_call = (sum(state["costs"]) / len(state["costs"])) if state["costs"] else 0.05
            if state["spent"] + per_call * (workers + 1) > budget:
                state["stopped_for_budget"] = True
                return None
        try:
            recs = call_one(row, variant, system, api_key, log, post, sleep)
        except Fatal as e:
            with lock:
                state["fatal"] = str(e)
            return None
        with lock:
            state["n"] += 1
            if state["n"] % 25 == 0:
                print(f"  {variant}: {state['n']}/{len(todo)} done; spent ${state['spent']:.2f}", flush=True)
        return recs

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(guarded, r) for r in todo]
        for _ in as_completed(futures):
            pass
    final = final_records(path)
    outcome = Counter(r["outcome"] for r in final.values())
    return {"variant": variant, "set": set_name, "requested": len(rows), "already_done": len(done),
            "final": len(final), "outcomes": dict(outcome), "fatal": state["fatal"],
            "stopped_for_budget": state["stopped_for_budget"], "spent_total_usd": round(state["spent"], 4)}


def usage_summary(path: Path) -> dict:
    recs = read_jsonl(path) if path.exists() else []
    ok = [r for r in recs if r.get("usage")]

    def stat(vals):
        vals = sorted(vals)
        return {"mean": round(sum(vals) / len(vals), 1), "max": vals[-1]} if vals else None
    return {"attempts": len(recs), "outcomes_final": dict(Counter(r["outcome"] for r in recs if r.get("final"))),
            "invalid_attempts": sum(r["outcome"] == "invalid" for r in recs),
            "models": sorted({r.get("model") for r in ok}),
            "input_tokens": stat([r["usage"].get("input_tokens", 0) for r in ok]),
            "cached_tokens": stat([(r["usage"].get("input_tokens_details") or {}).get("cached_tokens", 0) for r in ok]),
            "output_tokens": stat([r["usage"].get("output_tokens", 0) for r in ok]),
            "reasoning_tokens": stat([(r["usage"].get("output_tokens_details") or {}).get("reasoning_tokens", 0) for r in ok]),
            "seconds": stat([r["elapsed_seconds"] for r in recs]),
            "cost_usd": round(sum(r.get("cost_usd") or 0 for r in recs), 4),
            "errors": dict(Counter(e for r in recs for e in (r.get("errors") or [])))}


def call(args) -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        sys.exit("REFUSED: set OPENAI_API_KEY in the environment (never in a file)")
    config, system, rows = load_state()
    if args.set == "pilot":
        keys = set(json.loads((RESULTS / "pilot-keys.json").read_text(encoding="utf-8")))
        rows = [r for r in rows if key_text(r["conversation_id"], r["segment_order"]) in keys]
        variants = args.variant or list(VARIANTS)
    else:
        if not args.pilot_approved or not (BASE_RESULTS / "pilot-summary.json").exists():
            sys.exit("REFUSED: the full runs need the pilot summary and DW's approval (--pilot-approved)")
        variants = args.variant or list(VARIANTS)
    workers = args.workers or SETTINGS["workers"]
    reports = []
    for v in variants:
        print(f"variant {v}, set {args.set}: {len(rows)} interactions", flush=True)
        rep = run_calls(rows, v, args.set, system, api_key, workers)
        reports.append(rep)
        print(json.dumps(rep), flush=True)
        if rep["fatal"] or rep["stopped_for_budget"]:
            print("STOPPED: " + (rep["fatal"] or f"budget of ${SETTINGS['budget_usd']:.0f} would be passed"))
            return 1
    if args.set == "pilot":
        summary = {"created_utc": now(), "interactions": len(rows), "settings": SETTINGS,
                   "variants": {v: usage_summary(raw_path("pilot", v)) for v in variants}}
        for v in variants:
            s = summary["variants"][v]
            per_interaction = s["cost_usd"] / max(1, len(rows))
            s["projected_full_run_usd"] = round(per_interaction * 485, 2)
        summary["projected_both_full_runs_usd"] = round(sum(summary["variants"][v]["projected_full_run_usd"] for v in variants), 2)
        summary["spent_so_far_usd"] = round(spent_so_far(), 4)
        path = RESULTS / "pilot-summary.json"
        if path.exists():
            path = RESULTS / f"pilot-summary-{datetime.now().strftime('%H%M%S')}.json"
        write_new(path, json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
        for v in variants:
            s = summary["variants"][v]
            print(f"pilot {v}: final {s['outcomes_final']}, invalid attempts {s['invalid_attempts']}, model {s['models']}, "
                  f"input {s['input_tokens']}, cached {s['cached_tokens']}, output {s['output_tokens']}, "
                  f"reasoning {s['reasoning_tokens']}, seconds {s['seconds']}, cost ${s['cost_usd']}, "
                  f"projected 485 ${s['projected_full_run_usd']}")
        print(f"projected both full runs ${summary['projected_both_full_runs_usd']}; spent so far ${summary['spent_so_far_usd']}; "
              f"budget ${SETTINGS['budget_usd']:.0f}. Summary: {path.relative_to(ROOT)}")
    return 0


# ----------------------------------------------------------------------------- predictions

def prediction_records(rows: list[dict], final: dict, run: dict) -> list[dict]:
    """model_prediction records: accepted labels, or nulls with the reason (truncated capture, failed call)."""
    recs = []
    for r in rows:
        f = final[(r["conversation_id"], r["segment_order"])]
        if f["outcome"] == "accepted":
            labels = f["labels"]
            not_pred = {a: TRUNCATED_REASON for a in L.AXES if labels[a] is None}
        else:
            labels = {a: None for a in L.AXES}
            reason = "model call failed after one retry: " + "; ".join(f.get("errors") or ["unknown"])
            not_pred = {a: reason for a in L.AXES}
            if r["truncated"]:
                not_pred.update({a: TRUNCATED_REASON for a in ("AI delegation", "AI response type")})
        recs.append(L.prediction_record(r["conversation_id"], r["segment_order"], labels, run, not_pred))
    return recs


def predictions(args) -> int:
    config, system, rows = load_state()
    manifest_sha = config["dataset_manifest"]["sha256"]
    if args.set == "pilot":
        keys = set(json.loads((RESULTS / "pilot-keys.json").read_text(encoding="utf-8")))
        rows = [r for r in rows if key_text(r["conversation_id"], r["segment_order"]) in keys]
    out_dir = RESULTS / "predictions"
    written = {}
    for v in VARIANTS:
        final = final_records(raw_path(args.set, v))
        if not final:
            continue
        missing = [r for r in rows if (r["conversation_id"], r["segment_order"]) not in final]
        if missing:
            sys.exit(f"REFUSED: variant {v}: {len(missing)} interactions have no final attempt; resume the run first")
        models = sorted({r.get("model") for r in final.values() if r.get("model")})
        if len(models) != 1:
            sys.exit(f"REFUSED: variant {v}: expected one model id in the responses, found {models}")
        run = {"run_id": f"llm-terra-{v}-{args.set}-20261028" + (f"-{RUN_TAG}" if RUN_TAG else ""),
               "model_route": "openai/" + models[0],
               "config_sha256": config["config_sha256"][v], "dataset_manifest_sha256": manifest_sha}
        recs = prediction_records(rows, final, run)
        path = out_dir / f"predictions-LLM-{v}-{args.set}.jsonl"
        out_dir.mkdir(parents=True, exist_ok=True)
        L.write_predictions(path, recs)
        back = L.read_predictions(path)
        written[v] = {"path": L.rel(path), "sha256": L.sha256_file(path), "records": len(back), "run": run,
                      "failed": sum(f["outcome"] != "accepted" for f in final.values())}
        print(f"variant {v}: {len(back)} predictions, {written[v]['failed']} failed; {L.rel(path)}")
    write_new(out_dir / f"predictions-{args.set}-manifest.json", json.dumps(written, indent=1) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("estimate")
    r = sub.add_parser("reconfigure")
    r.add_argument("--note", required=True)
    c = sub.add_parser("call")
    c.add_argument("--set", choices=("pilot", "all"), required=True)
    c.add_argument("--variant", choices=VARIANTS, action="append")
    c.add_argument("--pilot-approved", action="store_true")
    c.add_argument("--workers", type=int)
    p = sub.add_parser("predictions")
    p.add_argument("--set", choices=("pilot", "all"), required=True)
    a = ap.parse_args(argv)
    return {"build": build, "estimate": estimate, "reconfigure": reconfigure, "call": call,
            "predictions": predictions}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
