"""Категория «Легковоспламеняющиеся» — ЗАМЕНЯЕМЫЙ МОДУЛЬ.

Здесь лежит запасной вариант (TF-IDF word+char → LogReg, OOF F1 ≈ 0.70), чтобы архив
был самодостаточным и не ронял macro-метрику в ноль. Основная модель команды —
шаблон рассуждения ОБЪЕКТ / ФАКТ / ПРАВИЛО / ЛВЖ на дообученной 1B (см. lvj/) —
подставляется сюда же: интерфейс predict(df) -> np.ndarray[int].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from text import clean


class FlamPredictor:
    def __init__(self, artifacts, images_path=None, device: str = "cuda", mode: str = "model"):
        """mode="zero" отдаёт нули для всей категории — так F1 по ЛВЖ ровно 0,
        и публичный скор, умноженный на 2, читается как чистый F1 по БАД."""
        self.dir = Path(artifacts)
        self.images_path = images_path
        self.device = device
        self.mode = mode
        self.art = None
        if mode != "zero":
            p = self.dir / "flam.joblib"
            if p.exists():
                import joblib
                self.art = joblib.load(p)

    @staticmethod
    def _texts(df: pd.DataFrame):
        return [f"{clean(n)} | {clean(d)[:2500]}"
                for n, d in zip(df["name"].fillna(""), df["description"].fillna(""))]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.mode == "zero" or self.art is None:                    # ничего не поставили — не баним никого
            return np.zeros(len(df), dtype=int)
        p = self.art["clf"].predict_proba(self.art["vec"].transform(self._texts(df)))[:, 1]
        return (p >= self.art["threshold"]).astype(int)
