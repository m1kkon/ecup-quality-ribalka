"""Точное совпадение карточки с обучающей.

Если товар из теста побайтово совпадает по (name, description) с обучающим,
его метка известна — угадывать нечего. Ключ считается по сырым строкам без
нормализации, поэтому ложных срабатываний быть не может.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def key(name: str, desc: str) -> str:
    payload = f"{name}\x00{desc}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build(data_csv: str, out_json: str, category: str | None = None) -> dict:
    df = pd.read_csv(data_csv)
    if category:
        df = df[df["category"] == category]
    df = df.copy()
    df["name"] = df["name"].fillna("")
    df["description"] = df["description"].fillna("")
    table: dict[str, int] = {}
    conflicts = 0
    for n, d, y in zip(df["name"], df["description"], df["label"]):
        k = key(n, d)
        if k in table and table[k] != int(y):
            conflicts += 1
            table.pop(k)          # противоречивая разметка — не угадываем
            continue
        table[k] = int(y)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(out_json, "w"))
    print(f"lookup: {len(table)} записей из {len(df)} строк, отброшено по конфликту {conflicts}")
    return table


class ExactLookup:
    def __init__(self, path):
        p = Path(path)
        self.table = json.load(open(p)) if p.exists() else {}

    def apply(self, df: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, int]:
        if not self.table:
            return pred, 0
        out = pred.copy()
        hit = 0
        for i, (n, d) in enumerate(zip(df["name"].fillna(""), df["description"].fillna(""))):
            v = self.table.get(key(n, d))
            if v is not None:
                out[i] = v
                hit += 1
        return out, hit
