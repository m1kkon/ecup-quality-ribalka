"""Shared CV protocol for the БАД category.

Protocol (fixed, per team decision):
  1. keep only category == "БАД"
  2. drop_duplicates on (name, description)  -- BEFORE any split
  3. StratifiedGroupKFold(n_splits=5), y = label, groups = description
     => identical descriptions never straddle a fold boundary
  4. metric = binary f1_score(y_true, y_pred), pos_label=1  (matches organizer code)
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_SPLITS = 5
CATEGORY = "БАД"


def load_bad(path: str = "data/data.csv", dedup: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["category"] == CATEGORY].copy()
    df["description"] = df["description"].fillna("")
    df["name"] = df["name"].fillna("")
    if dedup:
        df = df.drop_duplicates(subset=["name", "description"], keep="first")
    return df.reset_index(drop=True)


def desc_group(df: pd.DataFrame) -> np.ndarray:
    """Group id = hash of the description string (only the description)."""
    return df["description"].map(
        lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()
    ).values


def folds(df: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = SEED):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    g = desc_group(df)
    y = df["label"].values
    return list(sgkf.split(df, y, groups=g)), y, g


def assign_fold(df: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = SEED) -> np.ndarray:
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, va) in enumerate(folds(df, n_splits, seed)[0]):
        fold[va] = k
    assert (fold >= 0).all()
    return fold
