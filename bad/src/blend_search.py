"""Подбор коэффициентов бленда для БАД.

Три семейства, все с фиксированным порогом 0.5 на итоговой вероятности:
  1. classic  — свободные неотрицательные веса по всем членам;
  2. qwen-half — квену ровно w, энкодерам (1-w) поровну; перебор w;
  3. qwen-half-tilt — то же, но внутри энкодеров веса тоже подбираются.

Всё считается вложенно: веса выбираются на train-фолдах, применяются к val.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, ".")
import numpy as np
from sklearn.metrics import f1_score

from src.cv import load_bad, folds


def f1_at(y, p, th=0.5):
    return f1_score(y, (p >= th).astype(int))


def nested_fixed(y, P, fl, w, th=0.5):
    """Веса заданы снаружи, порог фиксирован — честная оценка без подбора на val."""
    return f1_at(y, P @ w, th)


def nested_search(y, P, fl, grid, th=0.5):
    """Для каждого фолда веса выбираются на остальных фолдах."""
    pred = np.zeros(len(y), dtype=int)
    chosen = []
    for tr, va in fl:
        best = (-1, None)
        for w in grid:
            s = f1_at(y[tr], P[tr] @ w, th)
            if s > best[0]:
                best = (s, w)
        chosen.append(best[1])
        pred[va] = (P[va] @ best[1] >= th).astype(int)
    return f1_score(y, pred), np.mean(chosen, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="specs/spec_es5_vote3.json")
    args = parser.parse_args()
    df = load_bad(); y = df.label.values; fl, _, _ = folds(df)
    spec = json.load(open(args.spec))
    names = [member["name"] for member in spec["members"]]
    paths = []
    for name in names:
        if name == "qwen":
            paths.append("runs/qwen_fewshot_oof.npy")
            continue
        local = f"runs/{name}/oof.npy"
        remote = f"runs/remote/{name}/oof.npy"
        paths.append(local if os.path.exists(local) else remote)
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing OOF files: {missing}")
    P = np.column_stack([np.load(path) for path in paths])
    qi = names.index("qwen")
    enc = [i for i in range(len(names)) if i != qi]
    print("члены:", names)
    print(f"{'схема':38s} {'F1@0.5':>8s}")
    print(f"{'равные веса':38s} {f1_at(y, P.mean(1)):8.4f}")

    # 1. классический перебор по симплексу
    rng = np.random.default_rng(0)
    grid1 = list(rng.dirichlet(np.ones(len(names)), size=4000))
    f1, w1 = nested_search(y, P, fl, grid1)
    print(f"{'classic (свободные веса)':38s} {f1:8.4f}  w={np.round(w1,3)}")

    # 2. квену ровно w, энкодерам поровну
    best2 = (-1, None)
    for wq in np.arange(0.0, 0.81, 0.05):
        w = np.zeros(len(names)); w[qi] = wq; w[enc] = (1 - wq) / len(enc)
        s = f1_at(y, P @ w)
        if s > best2[0]:
            best2 = (s, wq)
        print(f"    квен={wq:.2f} энкодеры поровну {(1-wq)/len(enc):.3f}: F1={s:.4f}")
    grid2 = []
    for wq in np.arange(0.0, 0.81, 0.05):
        w = np.zeros(len(names)); w[qi] = wq; w[enc] = (1 - wq) / len(enc); grid2.append(w)
    f2, w2 = nested_search(y, P, fl, grid2)
    print(f"{'qwen-half (вложенно)':38s} {f2:8.4f}  квен={w2[qi]:.2f}")

    # 3. квену w, а внутри энкодеров веса тоже подбираются
    grid3 = []
    for wq in np.arange(0.1, 0.71, 0.1):
        for sub in rng.dirichlet(np.ones(len(enc)), size=400):
            w = np.zeros(len(names)); w[qi] = wq; w[enc] = (1 - wq) * sub; grid3.append(w)
    f3, w3 = nested_search(y, P, fl, grid3)
    print(f"{'qwen-half-tilt (вложенно)':38s} {f3:8.4f}  w={np.round(w3,3)}")

    out = {"members": names,
           "equal": float(f1_at(y, P.mean(1))),
           "classic": {"f1": float(f1), "w": [float(x) for x in w1]},
           "qwen_half": {"f1": float(f2), "w": [float(x) for x in w2]},
           "qwen_half_tilt": {"f1": float(f3), "w": [float(x) for x in w3]}}
    json.dump(out, open("runs/blend_search.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
