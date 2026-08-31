"""Pooled classification head.

HF's default sequence-classification head reads a single [CLS] vector. Concatenating
[CLS] with the attention-masked mean of the token states gives the linear layer both
the sentence-level summary and the averaged evidence, which matters here because the
decisive phrase can sit anywhere inside the evidence block.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class Out:
    logits: torch.Tensor


class PooledClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, config=None, backbone=None):
        super().__init__()
        self.model_name = model_name
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * h, 1)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def gradient_checkpointing_enable(self, **kw):
        self.backbone.gradient_checkpointing_enable(**kw)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        hs = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hs[:, 0]
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        mean = (hs * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return Out(self.classifier(self.dropout(torch.cat([cls, mean], -1))).squeeze(-1))

    # ---- persistence: backbone in its own dir, head alongside it
    def save_pretrained(self, path, safe_serialization=True):
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path, safe_serialization=safe_serialization)
        torch.save(self.classifier.state_dict(), os.path.join(path, "classifier.pt"))
        json.dump({"head": "cls_mean", "hidden": self.backbone.config.hidden_size,
                   "model_name": self.model_name},
                  open(os.path.join(path, "head_config.json"), "w"))

    def half(self):
        self.backbone.half()
        return self

    @classmethod
    def from_saved(cls, path, dtype=None):
        cfg = json.load(open(os.path.join(path, "head_config.json")))
        backbone = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=dtype)
        m = cls(cfg["model_name"], backbone=backbone)
        m.classifier.load_state_dict(torch.load(os.path.join(path, "classifier.pt"),
                                                map_location="cpu"))
        if dtype is not None:
            m.classifier.to(dtype)
        return m


def build_model(model_name: str, head: str = "default", num_labels: int = 1):
    if head == "cls_mean":
        return PooledClassifier(model_name)
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, trust_remote_code=True)
