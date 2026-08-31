import sys, time
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import f1_score, roc_auc_score
from src.cv import load_bad, folds
from src import text as T

df = load_bad()
y = df.label.values
fl, _, _ = folds(df)

def build(kind):
    if kind == 'head':   return [T.head_only(n, d) for n, d in zip(df.name, df.description)]
    if kind == 'full':   return [T.full(n, d) for n, d in zip(df.name, df.description)]
    if kind == 'ev':     return [T.evidence(n, d) for n, d in zip(df.name, df.description)]
    raise ValueError(kind)

def run(kind):
    X = build(kind)
    oof = np.zeros(len(df))
    for tr, va in fl:
        vec = make_union(
            TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000, sublinear_tf=True),
            TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3, max_features=300000, sublinear_tf=True),
        )
        Xtr = vec.fit_transform([X[i] for i in tr]); Xva = vec.transform([X[i] for i in va])
        clf = LogisticRegression(C=4, max_iter=3000, class_weight='balanced')
        clf.fit(Xtr, y[tr])
        oof[va] = clf.predict_proba(Xva)[:,1]
    ths = np.linspace(0.05,0.95,181)
    f1s = [f1_score(y, (oof>=t).astype(int)) for t in ths]
    bi = int(np.argmax(f1s))
    print(f"{kind:6s} chars={np.mean([len(x) for x in X]):7.0f}  AUC={roc_auc_score(y,oof):.4f}  "
          f"F1@0.5={f1_score(y,(oof>=0.5).astype(int)):.4f}  F1*={f1s[bi]:.4f}@{ths[bi]:.2f}")
    return oof

for k in ['head','full','ev']:
    t=time.time(); run(k); print(f"   ({time.time()-t:.0f}s)")
