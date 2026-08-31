"""Rule/structure features for the БАД classifier — cheap, interpretable, and
diverse from the encoder, so they blend well."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from text import clean, NEG_PAT, POS_PAT

DISCLAIMER = re.compile(r"не\s+явля\w*\s+лекарствен\w*(?:\s+средством)?", re.I)
SGR = re.compile(r"свидетельств\w*\s+о\s+государствен\w*\s+регистрац|\bсгр\b|ru\.\d{2}\.|\bау\.\d", re.I)
SPORT = re.compile(r"спортивн\w*\s+питани|\bbcaa\b|л[-\s]?карнитин|l[-\s]?carnitine|\bпротеин|изолят\b"
                   r"|гейнер|креатин|сывороточн\w*\s+бел|предтрен|\bwhey\b|\beaa\b|казеин", re.I)
FORM = {
    "caps": r"капсул",
    "tabs": r"таблетк|таблет\b",
    "powder": r"порошок|порошк",
    "gummy": r"мармелад|жевательн",
    "stick": r"стик\w*\b|саше",
    "drops": r"капл\w*\b|сироп|раствор",
    "amp": r"ампул",
    "tea": r"\bчай\b|фиточай|сбор\b",
    "cream": r"крем\b|мазь|гель\b|шампун|бальзам",
    "food": r"батончик|печень\w*\b|напиток|коктейл",
    "device": r"таблетниц|органайзер|контейнер|дозатор|шейкер|бутылк",
    "cosm": r"космет|уход\w*\s+за\s+кож|сыворотк\w*\s+для\s+лиц",
}
VIT = re.compile(r"витамин|минерал|омега|коллаген|магни|цинк|железо|кальци|биотин|пробиотик|экстракт", re.I)
MED = re.compile(r"лекарствен\w*\s+препарат|рецептурн|гомеопат|мазь|антибиотик", re.I)
NOT_MED = re.compile(r"не\s+явля\w*\s+лекарств", re.I)


def _pos_stats(txt: str, pat: re.Pattern):
    ms = [m.start() for m in pat.finditer(txt)]
    n = len(ms)
    if not n:
        return 0, -1.0, -1.0
    L = max(len(txt), 1)
    return n, ms[0] / L, ms[0]


def build(df: pd.DataFrame) -> pd.DataFrame:
    name = df["name"].fillna("").map(clean)
    desc = df["description"].fillna("").map(clean)
    both = (name + " \n " + desc)
    f = pd.DataFrame(index=df.index)

    f["len_name"] = name.str.len()
    f["len_desc"] = desc.str.len()
    f["log_desc"] = np.log1p(f["len_desc"])
    f["n_words_desc"] = desc.str.count(r"\s+")

    for tag, pat, src in [("pos", POS_PAT, desc), ("neg", NEG_PAT, desc)]:
        st = src.map(lambda t: _pos_stats(t, pat))
        f[f"n_{tag}_desc"] = [s[0] for s in st]
        f[f"rel_{tag}_desc"] = [s[1] for s in st]
        f[f"abs_{tag}_desc"] = [s[2] for s in st]
    f["n_pos_name"] = name.map(lambda t: len(POS_PAT.findall(t)))
    f["n_neg_name"] = name.map(lambda t: len(NEG_PAT.findall(t)))

    f["disclaimer"] = both.str.contains(DISCLAIMER).astype(int)
    f["sgr"] = both.str.contains(SGR).astype(int)
    f["sport_name"] = name.str.contains(SPORT).astype(int)
    f["sport_desc"] = desc.str.contains(SPORT).astype(int)
    f["n_sport"] = desc.map(lambda t: len(SPORT.findall(t)))
    f["vit"] = both.str.contains(VIT).astype(int)
    f["med"] = both.str.contains(MED).astype(int)
    for k, p in FORM.items():
        f[f"form_{k}_n"] = name.str.contains(p, case=False, regex=True).astype(int)
        f[f"form_{k}_d"] = desc.str.contains(p, case=False, regex=True).astype(int)

    # "positive marker close to a copular construction" -> it is talking about THIS product
    def near_copula(t):
        best = 999
        for m in POS_PAT.finditer(t):
            w = t[max(0, m.start() - 90): m.start()]
            if re.search(r"явля\w*|представля\w*\s+собой|это\b|продукт\b|товар\b|препарат\b", w, re.I):
                best = min(best, 1)
        return 0 if best == 999 else 1
    f["pos_near_copula"] = desc.map(near_copula)
    f["pos_in_first_300"] = desc.str[:300].str.contains(POS_PAT).astype(int)
    f["neg_before_pos"] = (
        (f["abs_neg_desc"] >= 0) & (f["abs_pos_desc"] >= 0)
        & (f["abs_neg_desc"] <= f["abs_pos_desc"] + 40)
    ).astype(int)
    f["html_ratio"] = df["description"].fillna("").str.count(r"<") / (f["len_desc"] + 1)
    return f.astype(np.float32)
