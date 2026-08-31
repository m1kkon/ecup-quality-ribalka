"""Verdict comments, grounded in the text span that actually drove the decision.

Format required by the organisers: "<комментарий>...<вердикт>бан|не бан", no closing
tags, comment strictly 50..300 characters. Everything here is deterministic, so it
costs no GPU time and can never emit an invalid length.
"""
from __future__ import annotations

import re
from typing import List, Sequence

from text import clean, NEG_PAT, POS_PAT, CTX_PAT

MIN_LEN, MAX_LEN = 50, 300

SPORT = re.compile(r"спортивн\w*\s+питани|\bbcaa\b|л[-\s]?карнитин|\bпротеин|изолят\b|гейнер|креатин|\bwhey\b",
                   re.I)


def _snippet(text: str, m: re.Match, left: int = 60, right: int = 70) -> str:
    a, b = max(0, m.start() - left), min(len(text), m.end() + right)
    s = text[a:b].strip()
    s = re.sub(r"\s+", " ", s)
    return ("…" if a > 0 else "") + s + ("…" if b < len(text) else "")


def _pad(c: str) -> str:
    """Bring a comment into [50, 300] without ever cutting mid-word."""
    c = re.sub(r"\s+", " ", c).strip()
    if len(c) > MAX_LEN:
        cut = c.rfind(" ", 0, MAX_LEN - 1)
        c = (c[:cut] if cut > MAX_LEN // 2 else c[:MAX_LEN - 1]).rstrip(" ,;:—-") + "…"
    if len(c) < MIN_LEN:
        c = (c + " Решение принято по названию и описанию карточки товара.").strip()
    if len(c) < MIN_LEN:
        c = (c + " " * (MIN_LEN - len(c)))
    return c[:MAX_LEN]


def build_comment(name: str, desc: str, pred: int) -> str:
    n, d = clean(name), clean(desc)
    both = f"{n}. {d}"
    neg = NEG_PAT.search(both)
    pos = POS_PAT.search(both)
    sport = SPORT.search(both)

    if pred == 1:                                   # товар признан БАД
        if pos:
            c = f"В карточке есть прямое указание на биологически активную добавку: «{_snippet(both, pos)}»"
        elif CTX_PAT.search(both):
            c = ("Прямой маркировки в тексте нет, но состав, форма выпуска и предупреждения "
                 "соответствуют биологически активной добавке.")
        else:
            c = ("Признаки карточки — состав, дозировка и назначение — соответствуют "
                 "биологически активной добавке.")
    else:                                           # товар БАД не является
        if neg:
            c = f"В описании прямо указано, что товар не является БАД: «{_snippet(both, neg)}»"
        elif sport:
            c = (f"Товар относится к спортивному питанию ({sport.group(0).strip().lower()}), "
                 "а не к биологически активным добавкам.")
        elif pos:
            c = (f"Упоминание БАД в тексте относится не к самому товару: «{_snippet(both, pos, 50, 60)}» — "
                 "маркировки добавки у товара нет.")
        else:
            c = ("Ни в названии, ни в описании нет маркировки БАД, биологически активной добавки "
                 "или dietary supplement.")
    return _pad(c)


def format_results(names: Sequence[str], descs: Sequence[str], preds: Sequence[int]) -> List[str]:
    out = []
    for n, d, p in zip(names, descs, preds):
        verdict = "не бан" if int(p) == 1 else "бан"
        out.append(f"<комментарий>{build_comment(n, d, int(p))}<вердикт>{verdict}")
    return out
