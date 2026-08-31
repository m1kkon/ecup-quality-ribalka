"""Fit the production БАД models on the whole (deduplicated) training set and
dump everything the submission needs into --out.

  python -m src.fit_final --out submit/artifacts --spec spec.json
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cv import load_bad
from src.train_enc import build_texts, DS, make_collate, LengthBucketSampler, set_seed
from src import text as T
from src.feats import build as build_feats
from torch.utils.data import DataLoader
import math


def fit_encoder(spec, df, y, outdir: Path, device="cuda"):
    name = spec["name"]
    set_seed(spec.get("seed", 42))
    tok = AutoTokenizer.from_pretrained(spec["model"])
    texts = build_texts(df, spec.get("text", "ev"), spec.get("budget", 1600),
                        spec.get("radius", 140), spec.get("head", 400))
    ids = tok(texts, truncation=True, max_length=spec["max_len"])["input_ids"]
    from src.heads import build_model
    model = build_model(spec["model"], spec.get("head_type", "default")).to(device)
    pad = tok.pad_token_id or 0
    sampler = LengthBucketSampler([len(s) for s in ids], spec["bs"], True, seed=spec.get("seed", 42))
    dl = DataLoader(DS(ids, y), batch_sampler=sampler, collate_fn=make_collate(pad),
                    num_workers=4, pin_memory=True, persistent_workers=True)
    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ["bias", "LayerNorm.weight", "norm.weight"])]
    nod = [p for n, p in model.named_parameters() if any(k in n for k in ["bias", "LayerNorm.weight", "norm.weight"])]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": nod, "weight_decay": 0.0}],
                            lr=spec["lr"])
    accum = spec.get("accum", 1)
    steps = math.ceil(len(dl) / accum) * spec["epochs"]
    sch = get_cosine_schedule_with_warmup(opt, int(steps * 0.06), steps)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(spec["epochs"]):
        sampler.set_epoch(ep); model.train(); t0 = time.time(); tot = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (enc, yb) in enumerate(dl):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            yb = yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(**enc).logits.squeeze(-1).float(), yb) / accum
            loss.backward(); tot += loss.item() * accum
            if (i + 1) % accum == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); sch.step()
        print(f"  [{name}] ep{ep} loss={tot/len(dl):.4f} {time.time()-t0:.0f}s", flush=True)
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(d, safe_serialization=True)
    tok.save_pretrained(d)
    json.dump(spec, open(d / "spec.json", "w"))
    del model; torch.cuda.empty_cache()
    return name


def fit_tfidf(spec, df, y, outdir: Path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_union
    X = [T.evidence(n, d, head=spec.get("head", 0), radius=spec.get("radius", 40),
                    budget=spec.get("budget", 2000)) for n, d in zip(df.name, df.description)]
    vec = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True),
        TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=400000, sublinear_tf=True))
    A = vec.fit_transform(X)
    if spec.get("clf", "logreg") == "svm":
        clf = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3)
    else:
        clf = LogisticRegression(C=4, max_iter=4000, class_weight="balanced")
    clf.fit(A, y)
    joblib.dump({"vec": vec, "clf": clf, "spec": spec}, outdir / f"{spec['name']}.joblib", compress=3)
    return spec["name"]


def fit_gbm(spec, df, y, outdir: Path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    F = build_feats(df)
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                       l2_regularization=1.0, early_stopping=False, random_state=0)
    m.fit(F, y)
    joblib.dump({"model": m, "cols": list(F.columns), "spec": spec}, outdir / f"{spec['name']}.joblib", compress=3)
    return spec["name"]


def member_complete(spec, outdir: Path) -> bool:
    """Return true only for a fully persisted production member."""
    kind = spec["kind"]
    name = spec["name"]
    if kind == "enc":
        directory = outdir / name
        return all((directory / filename).is_file() for filename in (
            "model.safetensors", "spec.json", "tokenizer_config.json",
        ))
    if kind in {"tfidf", "gbm"}:
        return (outdir / f"{name}.joblib").is_file()
    return kind == "llm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default="submit/artifacts")
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--dedup", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated member names to fit")
    ap.add_argument("--name", default="", help="метка для логов")
    ap.add_argument("--holdout_fold", type=int, default=-1,
                    help="обучить на всех фолдах кроме этого — чтобы проверить путь инференса")
    args = ap.parse_args()
    cfg = json.load(open(args.spec))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_bad(args.data, dedup=bool(args.dedup))
    if args.holdout_fold >= 0:
        from src.cv import folds as _folds
        fl, _, _ = _folds(df)
        tr, va = fl[args.holdout_fold]
        np.save(Path(args.out) / f"holdout{args.holdout_fold}_idx.npy", va)
        df = df.iloc[tr].reset_index(drop=True)
        print(f"holdout {args.holdout_fold}: train {len(df)} rows, val {len(va)}", flush=True)
    y = df.label.values.astype(np.float32)
    print(f"final fit on {len(df)} rows, pos rate {y.mean():.4f}", flush=True)
    only = [x for x in args.only.split(",") if x]
    for s in cfg["members"]:
        if only and s["name"] not in only:
            continue
        k = s["kind"]
        print(f"--- {s['name']} ({k})", flush=True)
        if member_complete(s, out):
            print(f"  reuse completed member {s['name']}", flush=True)
            continue
        partial = out / s["name"]
        if k == "enc" and partial.exists():
            print(f"  remove incomplete member {partial}", flush=True)
            shutil.rmtree(partial)
        if k == "enc":
            fit_encoder(s, df, y, out)
        elif k == "tfidf":
            fit_tfidf(s, df, y, out)
        elif k == "gbm":
            fit_gbm(s, df, y, out)
        elif k == "llm":
            # Frozen inference-only member. Его веса не обучаются и не
            # копируются: production берёт backbone по model id/SHARED_MODELS_PATH.
            print(f"  skip frozen runtime member {s['model']}", flush=True)
        else:
            raise ValueError(k)
    if not only:
        incomplete = [s["name"] for s in cfg["members"] if not member_complete(s, out)]
        if incomplete:
            raise RuntimeError(f"BAD artifacts incomplete: {incomplete}")
        json.dump(cfg, open(out / "ensemble.json", "w"), ensure_ascii=False, indent=1)
    print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
