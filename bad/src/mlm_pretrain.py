"""Domain-adaptive MLM on the competition corpus (both categories, before dedup).

12 971 marketplace descriptions is small but on-domain; a short MLM pass usually
buys a few tenths of an F1 point on the downstream head.

  python -m src.mlm_pretrain --model deepvk/RuModernBERT-base --out ckpt/rmb-dapt --epochs 3
"""
from __future__ import annotations

import argparse, math, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src import text as T


class Chunks(Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return {"input_ids": self.ids[i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepvk/RuModernBERT-base")
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    texts = [f"{T.clean(n)} | {T.clean(d)}" for n, d in zip(df["name"].fillna(""), df["description"].fillna(""))]
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = tok(texts, truncation=True, max_length=args.max_len)["input_ids"]
    # long docs: also keep a second window so the tail is seen at least once
    extra = []
    for t in texts:
        if len(t) > 2200:
            extra.append(tok(t[1800:], truncation=True, max_length=args.max_len)["input_ids"])
    ids = enc + extra
    print(f"{len(ids)} chunks (of which {len(extra)} tail windows)", flush=True)

    model = AutoModelForMaskedLM.from_pretrained(args.model).cuda()
    coll = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=args.mlm_prob)
    order = np.argsort([len(x) for x in ids])
    ids = [ids[i] for i in order]
    dl = DataLoader(Chunks(ids), batch_size=args.bs, shuffle=True, collate_fn=coll,
                    num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl) * args.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for b in dl:
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); sch.step()
            tot += loss.item()
        print(f"  mlm ep{ep} loss={tot/len(dl):.4f} ppl={math.exp(min(tot/len(dl),20)):.1f} "
              f"{time.time()-t0:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
