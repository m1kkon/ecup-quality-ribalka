import sys, itertools, time, json
sys.path.insert(0,'.')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union
from sklearn.metrics import f1_score, roc_auc_score
from joblib import Parallel, delayed
from src.cv import load_bad, folds
from src import text as T

df = load_bad(); y = df.label.values; fl,_,_ = folds(df)

def score(cfg):
    head, radius, budget, limit = cfg
    X = [T.evidence(n, d, head=head, radius=radius, budget=budget, limit=limit)
         for n, d in zip(df.name, df.description)]
    oof = np.zeros(len(df))
    for tr, va in fl:
        vec = make_union(
            TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000, sublinear_tf=True),
            TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3, max_features=300000, sublinear_tf=True))
        A = vec.fit_transform([X[i] for i in tr]); B = vec.transform([X[i] for i in va])
        clf = LogisticRegression(C=4, max_iter=3000, class_weight='balanced'); clf.fit(A, y[tr])
        oof[va] = clf.predict_proba(B)[:,1]
    ths = np.linspace(.05,.95,181); f1s=[f1_score(y,(oof>=t).astype(int)) for t in ths]; bi=int(np.argmax(f1s))
    return dict(head=head, radius=radius, budget=budget, limit=limit,
                chars=float(np.mean([len(x) for x in X])),
                auc=float(roc_auc_score(y,oof)), f1=float(f1s[bi]), th=float(ths[bi]))

grid = list(itertools.product([0,200,400,700], [100,160,240], [1200,2000,3000], [12]))
t0=time.time()
res = Parallel(n_jobs=6, verbose=0)(delayed(score)(c) for c in grid)
res.sort(key=lambda r: -r['auc'])
for r in res[:14]:
    print(f"head={r['head']:4d} rad={r['radius']:4d} bud={r['budget']:5d} chars={r['chars']:6.0f} AUC={r['auc']:.4f} F1*={r['f1']:.4f}@{r['th']:.2f}")
json.dump(res, open('runs/tune_ev.json','w'), indent=1)
print(f"{time.time()-t0:.0f}s, {len(grid)} cfgs")
