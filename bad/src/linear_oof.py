"""Linear / TF-IDF OOF generators. Each writes runs/<name>_oof.npy."""
import sys, os, json, argparse
sys.path.insert(0, '.')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_union
from sklearn.metrics import f1_score, roc_auc_score
from src.cv import load_bad, folds
from src import text as T


def make_vec(word=(1, 2), char=(3, 5)):
    parts = []
    if word:
        parts.append(TfidfVectorizer(ngram_range=word, min_df=2, max_features=300000, sublinear_tf=True))
    if char:
        parts.append(TfidfVectorizer(analyzer='char_wb', ngram_range=char, min_df=3,
                                     max_features=400000, sublinear_tf=True))
    return make_union(*parts) if len(parts) > 1 else parts[0]


def oof_for(X, y, fl, clf_factory, vec_factory):
    oof = np.zeros(len(y))
    for tr, va in fl:
        vec = vec_factory()
        A = vec.fit_transform([X[i] for i in tr]); B = vec.transform([X[i] for i in va])
        c = clf_factory(); c.fit(A, y[tr])
        oof[va] = c.predict_proba(B)[:, 1] if hasattr(c, "predict_proba") else c.decision_function(B)
    return oof


def report(oof, y, tag):
    ths = np.linspace(.05, .95, 181)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in ths]
    bi = int(np.argmax(f1s))
    print(f"{tag:26s} AUC={roc_auc_score(y, oof):.4f} F1*={f1s[bi]:.4f}@{ths[bi]:.2f}", flush=True)


if __name__ == "__main__":
    df = load_bad(); y = df.label.values; fl, _, _ = folds(df)
    os.makedirs('runs', exist_ok=True)
    variants = {
        'tfidf_ev40':  dict(head=0, radius=40,  budget=2000),
        'tfidf_ev100': dict(head=0, radius=100, budget=2000),
        'tfidf_ev200': dict(head=200, radius=180, budget=2600),
    }
    for tag, kw in variants.items():
        X = [T.evidence(n, d, **kw) for n, d in zip(df.name, df.description)]
        o = oof_for(X, y, fl, lambda: LogisticRegression(C=4, max_iter=4000, class_weight='balanced'), make_vec)
        np.save(f'runs/{tag}_oof.npy', o); report(o, y, tag)
        o2 = oof_for(X, y, fl,
                     lambda: CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced'), cv=3),
                     make_vec)
        np.save(f'runs/{tag}_svm_oof.npy', o2); report(o2, y, tag + '_svm')
    # name only
    Xn = [T.clean(n) for n in df.name]
    o = oof_for(Xn, y, fl, lambda: LogisticRegression(C=4, max_iter=4000, class_weight='balanced'),
                lambda: make_vec(word=(1, 2), char=(3, 5)))
    np.save('runs/tfidf_name_oof.npy', o); report(o, y, 'tfidf_name')
    # full cleaned text
    Xf = [T.full(n, d) for n, d in zip(df.name, df.description)]
    o = oof_for(Xf, y, fl, lambda: LogisticRegression(C=4, max_iter=4000, class_weight='balanced'), make_vec)
    np.save('runs/tfidf_full_oof.npy', o); report(o, y, 'tfidf_full')
