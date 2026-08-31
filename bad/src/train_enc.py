"""Fine-tune a Russian encoder on the БАД subset under the fixed CV protocol.

Usage:
  python -m src.train_enc --model deepvk/RuModernBERT-base --text ev --max_len 512 \
      --epochs 3 --lr 2e-5 --bs 16 --name rmb-ev512
Writes runs/<name>/{oof.npy,meta.json} and prints CV F1.
"""
from __future__ import annotations

import argparse, json, os, sys, time, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cv import load_bad, folds, SEED
from src import text as T


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_texts(df, kind, budget, radius, head):
    if kind == "head":
        return [T.head_only(n, d, budget) for n, d in zip(df.name, df.description)]
    if kind == "full":
        return [T.full(n, d) for n, d in zip(df.name, df.description)]
    if kind == "ev":
        return [T.evidence(n, d, head=head, radius=radius, budget=budget)
                for n, d in zip(df.name, df.description)]
    if kind == "ev_full":                      # evidence first, then the raw head
        return [T.evidence(n, d, head=head, radius=radius, budget=budget) + " || " + (d or "")[:1200]
                for n, d in zip(df.name, df.description)]
    raise ValueError(kind)


class DS(Dataset):
    """Pre-tokenised; the collate only pads, so epochs after the first are free."""
    def __init__(self, ids, labels):
        self.ids, self.y = ids, labels
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return self.ids[i], float(self.y[i])


def make_collate(pad_id):
    def c(batch):
        seqs = [b[0] for b in batch]
        y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long); att[i, :len(s)] = 1
        return {"input_ids": ids, "attention_mask": att}, y
    return c


class LengthBucketSampler(torch.utils.data.Sampler):
    """Shuffle, then sort inside a mega-batch, then shuffle the batches.
    Keeps randomness while making every batch nearly uniform in length."""
    def __init__(self, lengths, batch_size, shuffle=True, mega=50, seed=0):
        self.l, self.bs, self.shuffle, self.mega, self.seed, self.ep = (
            np.asarray(lengths), batch_size, shuffle, mega, seed, 0)
    def set_epoch(self, e): self.ep = e
    def __len__(self): return math.ceil(len(self.l) / self.bs)
    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.ep)
        idx = rng.permutation(len(self.l)) if self.shuffle else np.arange(len(self.l))
        chunk = self.bs * self.mega
        batches = []
        for i in range(0, len(idx), chunk):
            part = idx[i:i + chunk]
            part = part[np.argsort(self.l[part], kind="stable")]
            batches += [part[j:j + self.bs] for j in range(0, len(part), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter([b.tolist() for b in batches])


def run_fold(args, ids_all, tok, y, tr, va, device, fold):
    from src.heads import build_model
    model = build_model(args.model, args.head_type).to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    coll = make_collate(pad_id)
    ids_tr = [ids_all[i] for i in tr]
    sampler = LengthBucketSampler([len(s) for s in ids_tr], args.bs, True, seed=args.seed + fold)
    dtr = DataLoader(DS(ids_tr, y[tr]), batch_sampler=sampler, collate_fn=coll,
                     num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    order = np.argsort([-len(ids_all[i]) for i in va])
    va_sorted = va[order]
    dva = DataLoader(DS([ids_all[i] for i in va_sorted], y[va_sorted]),
                     batch_size=args.eval_bs, shuffle=False, collate_fn=coll,
                     num_workers=args.workers, pin_memory=True)

    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ["bias", "LayerNorm.weight", "norm.weight"])]
    nodecay = [p for n, p in model.named_parameters() if any(k in n for k in ["bias", "LayerNorm.weight", "norm.weight"])]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": nodecay, "weight_decay": 0.0}], lr=args.lr)
    steps = math.ceil(len(dtr) / args.accum) * args.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    pos_w = torch.tensor([args.pos_weight], device=device) if args.pos_weight > 0 else None
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))
    adt = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": torch.float32}[args.amp]

    def _eval():
        model.eval(); probs = []
        with torch.no_grad():
            for enc, _ in dva:
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.autocast("cuda", dtype=adt, enabled=args.amp != "off"):
                    probs.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy())
        p = np.concatenate(probs)
        out = np.zeros(len(va)); out[order] = p
        return out

    best = (-1.0, -1, None)          # (F1@0.5, эпоха, предсказания)
    hist = []
    step = 0
    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        model.train(); t0 = time.time(); tot = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (enc, yb) in enumerate(dtr):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            yb = yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=adt, enabled=args.amp != "off"):
                out = model(**enc).logits.squeeze(-1)
                loss = lossf(out.float(), yb) / args.accum
            scaler.scale(loss).backward()
            tot += loss.item() * args.accum
            if (i + 1) % args.accum == 0 or i + 1 == len(dtr):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); step += 1
        msg = f"  fold{fold} ep{ep} loss={tot/len(dtr):.4f} {time.time()-t0:.0f}s"
        if args.early_stop:
            cur = _eval()
            f1 = f1_score(y[va], (cur >= 0.5).astype(int))
            hist.append(round(float(f1), 4))
            msg += f" F1@0.5={f1:.4f}"
            if f1 > best[0]:
                best = (f1, ep, cur); msg += " *"
        print(msg, flush=True)

    out = best[2] if (args.early_stop and best[2] is not None) else _eval()
    if args.early_stop:
        print(f"  fold{fold} лучшая эпоха {best[1]} (F1@0.5={best[0]:.4f}), по эпохам: {hist}",
              flush=True)
    del model; torch.cuda.empty_cache()
    return out, (best[1] if args.early_stop else args.epochs - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepvk/RuModernBERT-base")
    ap.add_argument("--text", default="ev", choices=["head", "full", "ev", "ev_full"])
    ap.add_argument("--budget", type=int, default=1600)
    ap.add_argument("--radius", type=int, default=140)
    ap.add_argument("--head", type=int, default=400)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--pos_weight", type=float, default=0.0)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--head_type", default="default", choices=["default", "cls_mean"])
    ap.add_argument("--early_stop", action="store_true",
                    help="оценивать F1@0.5 после каждой эпохи и брать лучшую")
    ap.add_argument("--name", default=None)
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--dedup", type=int, default=1,
                    help="0 = учить и мерить на сыром наборе с дублями (как в тесте)")
    args = ap.parse_args()
    name = args.name or f"{args.model.split('/')[-1]}-{args.text}{args.max_len}-s{args.seed}"
    outdir = Path("runs") / name; outdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    df = load_bad(args.data, dedup=bool(args.dedup))
    y = df.label.values.astype(np.float32)
    texts = build_texts(df, args.text, args.budget, args.radius, args.head)
    tok = AutoTokenizer.from_pretrained(args.model)
    t_tok = time.time()
    ids_all = tok(texts, truncation=True, max_length=args.max_len)["input_ids"]
    lens = np.array([len(x) for x in ids_all])
    print(f"tokenised {len(ids_all)} in {time.time()-t_tok:.0f}s  "
          f"len mean={lens.mean():.0f} p90={np.percentile(lens,90):.0f} "
          f"max={lens.max()} trunc={(lens>=args.max_len).mean()*100:.1f}%", flush=True)
    fl, _, _ = folds(df)
    want = list(range(len(fl))) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    device = "cuda"
    oof = np.full(len(df), np.nan)
    best_epochs = []
    t0 = time.time()
    for k, (tr, va) in enumerate(fl):
        if k not in want: continue
        oof[va], be = run_fold(args, ids_all, tok, y, tr, va, device, k)
        best_epochs.append(be)
        m = ~np.isnan(oof)
        print(f"  [fold{k}] running AUC={roc_auc_score(y[m], oof[m]):.4f} "
              f"F1@.5={f1_score(y[m], (oof[m]>=.5).astype(int)):.4f}", flush=True)
    m = ~np.isnan(oof)
    ths = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y[m], (oof[m] >= t).astype(int)) for t in ths]
    bi = int(np.argmax(f1s))
    if best_epochs:
        import collections
        cnt = collections.Counter(best_epochs)
        mode = cnt.most_common(1)[0][0]
        print(f"лучшие эпохи по фолдам: {best_epochs} | мода {mode} | медиана "
              f"{int(np.median(best_epochs))}  -> столько эпох для фул-трейна", flush=True)
    res = dict(name=name, n=int(m.sum()), auc=float(roc_auc_score(y[m], oof[m])),
               best_epochs=best_epochs,
               f1_05=float(f1_score(y[m], (oof[m] >= .5).astype(int))),
               f1_best=float(f1s[bi]), th_best=float(ths[bi]),
               mins=round((time.time()-t0)/60, 1), args=vars(args))
    np.save(outdir / "oof.npy", oof)
    df[["id"]].assign(oof=oof, label=df.label.values).to_csv(outdir / "oof.csv", index=False)
    json.dump(res, open(outdir / "meta.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, ensure_ascii=False))
    print(f"RESULT {name} AUC={res['auc']:.4f} F1@.5={res['f1_05']:.4f} F1*={res['f1_best']:.4f}@{res['th_best']:.2f} ({res['mins']}m)")


if __name__ == "__main__":
    main()
