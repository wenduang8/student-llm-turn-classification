#!/usr/bin/env python3
"""Four-axis multi-head classifier on a causal open-weight backbone with a LoRA adapter.

Heads are built from codebook_schema.json `direct_instrument` through finetuning/labels.py: a
softmax head for the single-select student interaction type and one sigmoid unit per code for
each multi-select axis, the exclusive negative codes ("ND", "none of the four applies") being
ordinary observed codes. Axes whose names do not start "AI " read the student pool position; the
AI axes read the final position (finetuning/model_inputs.py explains why the student heads cannot
see the current assistant response).

example_loss removes an axis before its loss is computed when the axis mask is 0, i.e. the label
is null (unavailable, truncated capture). A null contributes no negative code, no all-zero target
and no gradient: it contributes nothing. Decoding to label sets is labels.decode_scores.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L  # noqa: E402

STUDENT_AXES = tuple(a for a in L.AXES if not a.startswith("AI "))
AI_AXES = tuple(a for a in L.AXES if a.startswith("AI "))
SLUG = {a: re.sub(r"[^a-z0-9]+", "_", a.lower()).strip("_") for a in L.AXES}


class MultiAxisClassifier(nn.Module):
    """backbone -> hidden states; student/AI pooled vectors -> one linear head per axis."""

    def __init__(self, backbone: nn.Module, hidden_size: int, head_dropout: float, init_std: float):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(head_dropout)
        self.heads = nn.ModuleDict({SLUG[a]: nn.Linear(hidden_size, len(L.DIRECT_CODES[a])) for a in L.AXES})
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=init_std)
            nn.init.zeros_(head.bias)

    def forward(self, input_ids: torch.Tensor, student_pool: int, ai_pool: int) -> dict:
        """One interaction per call (shape [1, T], no padding). Returns per-axis logit vectors."""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("one unpadded interaction per forward pass")
        if not 0 <= student_pool < ai_pool == input_ids.shape[1] - 1:
            raise ValueError("pool positions must satisfy student_pool < ai_pool = last token")
        hidden = self.backbone(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        pooled = {"student": hidden[student_pool], "ai": hidden[ai_pool]}
        return {a: self.heads[SLUG[a]](self.dropout(pooled["student" if a in STUDENT_AXES else "ai"]))
                for a in L.AXES}


def example_loss(logits: dict, targets: dict, weights: dict | None = None):
    """(summed loss over labelled axes or None, {axis: detached loss or None if masked}).

    `weights` (train_multihead.class_weights) scales a single-select loss by the weight of the
    observed code, and gives each multi-select code its BCE pos_weight; None leaves the loss as is."""
    terms, parts = [], {}
    for axis in L.AXES:
        t = targets[axis]
        if t["mask"] != 1:                      # null label: excluded before any loss is formed
            parts[axis] = None
            continue
        z = logits[axis]
        if axis in L.SINGLE:
            loss = F.cross_entropy(z.unsqueeze(0), torch.tensor([t["index"]], device=z.device))
            if weights is not None:             # one interaction: a weighted mean would cancel the weight
                loss = loss * weights[axis][t["index"]]
        else:
            y = torch.tensor(t["multi_hot"], dtype=z.dtype, device=z.device)
            pos_weight = None if weights is None else torch.tensor(weights[axis], dtype=z.dtype, device=z.device)
            loss = F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight)
        terms.append(loss)
        parts[axis] = loss.detach()
    return (torch.stack(terms).sum() if terms else None), parts


def scores(logits: dict) -> dict:
    """Probabilities per axis: softmax for single-select, independent sigmoids for multi-select."""
    return {a: (torch.softmax(z.float(), -1) if a in L.SINGLE else torch.sigmoid(z.float())).tolist()
            for a, z in logits.items()}


def load_backbone(model_dir: Path, dtype: torch.dtype, attn_implementation: str, gradient_checkpointing: bool):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_dir, dtype=dtype, attn_implementation=attn_implementation,
                                      local_files_only=True)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def attach_lora(backbone: nn.Module, cfg: dict):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(backbone, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                                               target_modules=list(cfg["target_modules"]), bias=cfg["bias"]))


def trainable_state(model: nn.Module) -> dict:
    """Every trainable tensor under its exact parameter name (LoRA adapter and heads)."""
    return {n: p.detach().to("cpu").contiguous().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model: nn.Module, state: dict) -> None:
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(state):
        missing, extra = sorted(set(params) - set(state)), sorted(set(state) - set(params))
        raise ValueError(f"trainable state does not match the model: missing {missing[:3]}, unexpected {extra[:3]}")
    with torch.no_grad():
        for name, p in params.items():
            if p.shape != state[name].shape:
                raise ValueError(f"{name}: shape {tuple(state[name].shape)} != {tuple(p.shape)}")
            p.copy_(state[name].to(device=p.device, dtype=p.dtype))
