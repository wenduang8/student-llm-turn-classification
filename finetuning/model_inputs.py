#!/usr/bin/env python3
"""Token-level model input for the four-axis multi-head classifier.

    build_input(example, encode, policy) -> {"input_ids", "student_pool", "ai_pool", "report"}

One causal sequence per interaction, oldest text first:

    [earlier turns not available]                  only if the record starts late or follows a gap
    [k earlier interaction(s) omitted for length]  only if budget forced context out
    [earlier interaction <n>] Student: ... Assistant: ...         at most policy.context_interactions
    [interaction to code] Student: <prompt> [end of student prompt]      <- student-axis heads
    Assistant: <response> [end of assistant response]                     <- AI-axis heads

The student-axis heads read the last token of "[end of student prompt]". Everything up to that
position is built without looking at the current response: the response always gets a FIXED
reservation of response_cap tokens plus its markers, so even its length cannot change which
context the student heads see. In a causal decoder the pool position cannot attend to a later
token, so the current assistant response is unavailable to the student labels (CLAUDE.md). The
AI-axis heads read the final token.

Only these fields are read: each context item's segment_order, student_prompt and llm_response;
the current segment_order, student_prompt and llm_response; and context_short_because. Labels,
targets, notes, flags, review history and source records are never read, so no annotation can
enter the input. Context comes from finetuning/data.py, which stops at a conversation start or a
Segment Order gap and never reads a later interaction; this module re-checks that order.

Truncation is explicit and recorded per example; the policy values belong to the run config:
  1. the current prompt and response are each capped, keeping head and tail around a marker;
  2. earlier interactions fill the student-side budget (max_tokens minus the response
     reservation) from the most recent backwards. The first one that does not fit whole is
     clipped if at least `min_context_fragment` tokens remain; it is otherwise dropped, with every
     older one, and a marker states how many were omitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

TEMPLATE_VERSION = "causal-two-pool-v2"
T_CONTEXT_MISSING = "[earlier turns not available]\n"
T_CONTEXT_OMITTED = "[{n} earlier interaction(s) omitted for length]\n"
T_CONTEXT_HEAD = "[earlier interaction {segment}]\nStudent: "
T_CONTEXT_MID = "\nAssistant: "
T_CONTEXT_END = "\n"
T_CURRENT_HEAD = "[interaction to code]\nStudent: "
T_STUDENT_END = "\n[end of student prompt]"
T_RESPONSE_HEAD = "\nAssistant: "
T_RESPONSE_END = "\n[end of assistant response]"
T_ELISION = "\n[... {n} tokens omitted ...]\n"
# Reasons from finetuning/data.py meaning earlier turns existed but are not in the record.
SHORT_CONTEXT_MARKED = ("record starts late", "gap")
# Large enough to bound any omitted count the elision marker will ever show.
_LARGEST_COUNT = 10 ** 9


@dataclass(frozen=True)
class InputPolicy:
    max_tokens: int
    context_interactions: int
    prompt_cap: int
    response_cap: int
    min_context_fragment: int

    def __post_init__(self):
        # Checked on every construction path, not only from a config.
        for name, value in asdict(self).items():
            if type(value) is not int or value < 0 or (value == 0 and name != "context_interactions"):
                kind = "a non-negative" if name == "context_interactions" else "a positive"
                raise ValueError(f"input policy {name} must be {kind} integer")
        if self.prompt_cap + self.response_cap >= self.max_tokens:
            raise ValueError("prompt_cap + response_cap must leave room inside max_tokens")

    @classmethod
    def from_config(cls, cfg: dict) -> "InputPolicy":
        if cfg.get("template_version") != TEMPLATE_VERSION:
            raise ValueError(f"input template_version must be {TEMPLATE_VERSION!r}")
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__})

    def response_reservation(self, encode) -> int:
        """Tokens always set aside after the student pool, whatever the response's length."""
        return len(encode(T_RESPONSE_HEAD)) + self.response_cap + len(encode(T_RESPONSE_END))

    def check_with(self, encode) -> dict:
        """Refuse a policy this tokenizer cannot honour; returns the fixed token costs."""
        marker = len(encode(T_ELISION.format(n=_LARGEST_COUNT)))
        for name in ("prompt_cap", "response_cap"):
            if getattr(self, name) < marker + 2:
                raise ValueError(f"{name} {getattr(self, name)} cannot hold an elision marker ({marker} tokens) "
                                 "and two kept tokens")
        fixed = {"current_head": len(encode(T_CURRENT_HEAD)), "student_end": len(encode(T_STUDENT_END)),
                 "context_missing": len(encode(T_CONTEXT_MISSING)),
                 "context_omitted": len(encode(T_CONTEXT_OMITTED.format(n=max(self.context_interactions, 1)))),
                 "response_reservation": self.response_reservation(encode), "elision_marker_max": marker}
        worst = (fixed["context_missing"] + fixed["context_omitted"] + fixed["current_head"] + self.prompt_cap
                 + fixed["student_end"] + fixed["response_reservation"])
        if worst > self.max_tokens:
            raise ValueError(f"policy needs up to {worst} tokens before any context but max_tokens is {self.max_tokens}")
        fixed["student_side_budget"] = self.max_tokens - fixed["response_reservation"]
        return fixed


def clip(ids: list, cap: int, encode) -> tuple[list, int]:
    """Keep at most `cap` tokens: head and tail around an elision marker. Returns (ids, omitted)."""
    ids = list(ids)
    if len(ids) <= cap:
        return ids, 0
    keep = cap - len(encode(T_ELISION.format(n=len(ids))))
    while keep >= 2:
        head = (keep + 1) // 2
        tail = keep - head
        omitted = len(ids) - head - tail
        marker = encode(T_ELISION.format(n=omitted))
        if head + len(marker) + tail <= cap:
            return ids[:head] + list(marker) + ids[len(ids) - tail:], omitted
        keep -= 1
    raise ValueError(f"a cap of {cap} tokens cannot hold an elision marker")


def _checked_context(example: dict) -> list:
    seg = example["segment_order"]
    context = list(example["context"])
    orders = [c["segment_order"] for c in context]
    if orders and orders != list(range(seg - len(orders), seg)):
        raise ValueError(f"{example['conversation_id']}#{seg}: context is not the adjacent run "
                         "of earlier interactions immediately before the target")
    return context


def build_input(example: dict, encode, policy: InputPolicy) -> dict:
    """Token ids, the two pooling positions and a truncation report for one interaction."""
    prepared = _checked_context(example)
    limit = policy.context_interactions
    context = prepared[-limit:] if limit else []                 # the most recent `limit`, all if fewer
    prompt = encode(example["student_prompt"])
    response = encode(example["llm_response"])
    prompt_ids, prompt_cut = clip(prompt, policy.prompt_cap, encode)
    response_ids, response_cut = clip(response, policy.response_cap, encode)
    current = list(encode(T_CURRENT_HEAD)) + prompt_ids + list(encode(T_STUDENT_END))
    answer = list(encode(T_RESPONSE_HEAD)) + response_ids + list(encode(T_RESPONSE_END))
    reservation = policy.response_reservation(encode)
    if len(answer) > reservation:                                # unreachable: the response was capped
        raise AssertionError("response exceeds its reservation")
    marked = len(prepared) < limit and example.get("context_short_because") in SHORT_CONTEXT_MARKED
    lead = list(encode(T_CONTEXT_MISSING)) if marked else []

    # The student side never depends on the response: its budget subtracts the fixed reservation.
    remaining = policy.max_tokens - reservation - len(lead) - len(current)
    if context:      # reserve the largest "omitted" marker that could be needed
        remaining -= len(encode(T_CONTEXT_OMITTED.format(n=len(context))))
    if remaining < 0:
        raise ValueError("the policy leaves no room for the current interaction")
    pieces, clipped, clipped_cut = [], False, 0
    for item in reversed(context):                              # most recent first
        head = list(encode(T_CONTEXT_HEAD.format(segment=item["segment_order"])))
        mid, end = list(encode(T_CONTEXT_MID)), list(encode(T_CONTEXT_END))
        c_prompt, c_response = encode(item["student_prompt"]), encode(item["llm_response"])
        fixed = len(head) + len(mid) + len(end)
        if fixed + len(c_prompt) + len(c_response) <= remaining:
            pieces.append(head + list(c_prompt) + mid + list(c_response) + end)
            remaining -= fixed + len(c_prompt) + len(c_response)
            continue
        room = remaining - fixed
        if room >= policy.min_context_fragment:
            # Split the room evenly; a short side passes its unused share to the other.
            p_cap = min(len(c_prompt), max(room // 2, room - len(c_response)))
            try:
                p_ids, p_cut = clip(c_prompt, p_cap, encode)
                r_ids, r_cut = clip(c_response, room - len(p_ids), encode)
            except ValueError:                                  # too little room for a marker
                break
            pieces.append(head + p_ids + mid + r_ids + end)
            clipped, clipped_cut = True, p_cut + r_cut
        break
    dropped_items = context[:len(context) - len(pieces)]
    dropped_tokens = sum(len(encode(c["student_prompt"])) + len(encode(c["llm_response"])) for c in dropped_items)
    omitted = list(encode(T_CONTEXT_OMITTED.format(n=len(dropped_items)))) if dropped_items else []
    prefix = lead + omitted + [t for piece in reversed(pieces) for t in piece]
    ids = prefix + current + answer
    if len(ids) > policy.max_tokens:                            # unreachable unless the budget logic is wrong
        raise AssertionError(f"built {len(ids)} tokens for a budget of {policy.max_tokens}")
    return {
        "input_ids": ids,
        "student_pool": len(prefix) + len(current) - 1,
        "ai_pool": len(ids) - 1,
        "report": {"tokens": len(ids), "prompt_tokens": len(prompt), "prompt_tokens_omitted": prompt_cut,
                   "response_tokens": len(response), "response_tokens_omitted": response_cut,
                   "context_prepared": len(prepared), "context_within_limit": len(context),
                   "context_included": len(pieces), "context_dropped": len(dropped_items),
                   "context_clipped": clipped, "context_tokens_omitted": clipped_cut + dropped_tokens,
                   "context_missing_marker": marked},
    }


def summarise(reports: list[dict]) -> dict:
    """Aggregate truncation counts for a run record; no text."""
    n = len(reports)
    tokens = sorted(r["tokens"] for r in reports)
    return {
        "examples": n,
        "tokens": {"min": tokens[0], "median": tokens[n // 2], "max": tokens[-1]} if n else None,
        "prompt_clipped": sum(r["prompt_tokens_omitted"] > 0 for r in reports),
        "response_clipped": sum(r["response_tokens_omitted"] > 0 for r in reports),
        "context_beyond_limit": sum(r["context_prepared"] - r["context_within_limit"] for r in reports),
        "context_clipped": sum(r["context_clipped"] for r in reports),
        "with_context_dropped": sum(r["context_dropped"] > 0 for r in reports),
        "context_interactions_dropped": sum(r["context_dropped"] for r in reports),
        "context_tokens_omitted": sum(r["context_tokens_omitted"] for r in reports),
        "context_missing_marker": sum(r["context_missing_marker"] for r in reports),
    }
