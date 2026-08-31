"""Blend OOF prediction files honestly.

Level-2 model and the decision threshold are both fitted **inside** the CV loop:
for fold k they only see folds != k. So the reported F1 has no threshold-tuning
leak, unlike "best F1 over the whole OOF".
"""
from __future__ import annotations

import glob, os, sys, json
sys.path.insert(0, '.')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from src.cv import load_bad, folds

THS = np.linspace(0.05, 0.95, 181)


def best_th(y, p):
    """Exact best F1 over all thresholds in O(n log n).

    Sort by score desc; predicting the top-k as positive gives
    F1(k) = 2*TP(k) / (k + P). Scan k, take the max.
    """
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=np.float64)
    o = np.argsort(-p, kind="mergesort")
    ys, ps = y[o], p[o]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    P = ys.sum()
    f1 = 2.0 * tp / (k + P)
    # only cut between distinct scores
    valid = np.ones(len(ys), dtype=bool)
    valid[:-1] = ps[:-1] > ps[1:]
    f1 = np.where(valid, f1, -1.0)
    i = int(np.argmax(f1))
    if f1[i] <= 0:
        return 0.5, 0.0
    th = ps[i] if i == len(ys) - 1 else (ps[i] + ps[i + 1]) / 2.0
    return float(th), float(f1[i])


def naive(y, p):
    """Optimistic: threshold tuned on the same data."""
    t, f = best_th(y, p)
    return dict(auc=roc_auc_score(y, p), f1_05=f1_score(y, (p >= .5).astype(int)),
                f1_star=f, th=t)


def nested_threshold(y, p, fl):
    """Honest: for each fold pick the threshold on the other folds."""
    pred = np.zeros(len(y), dtype=int); ths = []
    for tr, va in fl:
        t, _ = best_th(y[tr], p[tr]); ths.append(t)
        pred[va] = (p[va] >= t).astype(int)
    return f1_score(y, pred), float(np.mean(ths))


def nested_stack(y, P, fl, C=1.0, logit=True):
    """Honest stacking: level-2 LR + threshold both fitted out-of-fold."""
    X = np.clip(P, 1e-6, 1 - 1e-6)
    X = np.log(X / (1 - X)) if logit else X
    oof2 = np.zeros(len(y)); pred = np.zeros(len(y), dtype=int); ths = []
    for tr, va in fl:
        lr = LogisticRegression(C=C, max_iter=2000)
        lr.fit(X[tr], y[tr])
        ptr = lr.predict_proba(X[tr])[:, 1]
        t, _ = best_th(y[tr], ptr); ths.append(t)
        pv = lr.predict_proba(X[va])[:, 1]
        oof2[va] = pv; pred[va] = (pv >= t).astype(int)
    return oof2, f1_score(y, pred), float(np.mean(ths))


SKIP = ("stack", "best_blend")


def load_all(patterns=('runs/*_oof.npy', 'runs/remote/*/oof.npy'), n=None):
    """Every complete OOF vector on disk, keyed by run name.

    `n` filters by length so OOF files from the other category (ЛВЖ has 3951 rows,
    БАД 5565) never silently join the same blend.
    """
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            name = (os.path.basename(os.path.dirname(f)) if f.endswith('/oof.npy')
                    else os.path.basename(f).replace('_oof.npy', ''))
            if name.startswith('smoke') or name in SKIP:
                continue
            a = np.load(f)
            if np.isnan(a).any() or (n is not None and len(a) != n):
                continue
            out[name] = a
    return out


def greedy_forward(y, cand: dict, fl, max_k=8, verbose=True):
    """Greedy forward selection of ensemble members by honest nested-stack F1."""
    chosen, best = [], -1
    while len(chosen) < max_k:
        gains = []
        for k in cand:
            if k in chosen:
                continue
            names = chosen + [k]
            P = np.column_stack([cand[n] for n in names])
            _, f1, _ = nested_stack(y, P, fl)
            gains.append((f1, k))
        gains.sort(reverse=True)
        if not gains or gains[0][0] <= best + 1e-5:
            break
        best, k = gains[0]
        chosen.append(k)
        if verbose:
            print(f"  + {k:26s} -> nested F1 = {best:.4f}", flush=True)
    return chosen, best


if __name__ == "__main__":
    df = load_bad(); y = df.label.values; fl, _, _ = folds(df)
    cand = load_all(n=len(y))
    print(f"{len(cand)} candidate OOF files\n")
    rows = []
    for k, p in sorted(cand.items()):
        n = naive(y, p); f1n, thn = nested_threshold(y, p, fl)
        rows.append((f1n, k, n))
        print(f"{k:26s} AUC={n['auc']:.4f}  F1@.5={n['f1_05']:.4f}  "
              f"F1*(naive)={n['f1_star']:.4f}  F1(nested-th)={f1n:.4f}@{thn:.2f}")
    print("\n--- greedy forward selection (honest nested stack) ---")
    chosen, best = greedy_forward(y, cand, fl)
    print(f"\nchosen = {chosen}\nnested-CV F1 = {best:.4f}")
    P = np.column_stack([cand[n] for n in chosen])
    oof2, f1, th = nested_stack(y, P, fl)
    np.save('runs/stack_oof.npy', oof2)
    json.dump(dict(members=chosen, f1=f1, th=th), open('runs/stack.json', 'w'), indent=1)
    print(f"stack AUC={roc_auc_score(y, oof2):.4f}  mean th={th:.2f}")


# ---------------------------------------------------------------- extra level-2 heads
def nested_weights(y, P, fl, n_iter=3000, seed=0):
    """Honest non-negative weight search (Dirichlet random search) + nested threshold."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(P.shape[1]), size=n_iter)
    pred = np.zeros(len(y), dtype=int); ws = []
    for tr, va in fl:
        S = P[tr] @ W.T                      # (n_tr, n_iter)
        scores = [best_th(y[tr], S[:, j])[1] for j in range(W.shape[0])]
        w = W[int(np.argmax(scores))]
        t, _ = best_th(y[tr], P[tr] @ w)
        pred[va] = (P[va] @ w >= t).astype(int); ws.append(w)
    return f1_score(y, pred), np.mean(ws, axis=0)


def nested_gbm(y, P, fl, F=None, seed=0):
    """Level-2 HistGBM over member probabilities (+ optional rule features)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.column_stack([P] + ([F.values] if F is not None else []))
    pred = np.zeros(len(y), dtype=int); oof2 = np.zeros(len(y)); ths = []
    for tr, va in fl:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=seed)
        m.fit(X[tr], y[tr])
        ptr = m.predict_proba(X[tr])[:, 1]
        t, _ = best_th(y[tr], ptr); ths.append(t)
        pv = m.predict_proba(X[va])[:, 1]
        oof2[va] = pv; pred[va] = (pv >= t).astype(int)
    return oof2, f1_score(y, pred), float(np.mean(ths))
