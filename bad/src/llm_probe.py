"""Zero-shot вопрос к текстовой LLM: является ли товар БАД.

Скор = P("да") из логитов первого сгенерированного токена. Нужен, чтобы понять,
стоит ли вообще заводить LLM-члена в ансамбль: энкодеры теряют ~0.03 при переходе
на тест, а LLM должна быть устойчивее к незнакомым формулировкам.
"""
from __future__ import annotations

import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cv import load_bad
from src import text as T

RULES = (
    "Правила площадки Ozon:\n"
    "Товар ЯВЛЯЕТСЯ биологически активной добавкой, если в описании или на упаковке "
    "есть прямое указание, что это БАД (БАД, биологически активная добавка, dietary supplement).\n"
    "Товар НЕ ЯВЛЯЕТСЯ БАД, если это спортивное питание (аминокислоты, BCAA, L-карнитин, "
    "протеин), если в описании прямо сказано, что товар не является БАД, "
    "или если маркировки БАД нет."
)
PROMPT = ("{rules}\n\nКарточка товара:\n{txt}\n\n"
          "Вопрос: является ли этот товар БАД по правилам выше? Ответь одним словом: да или нет.")

# few-shot: два позитива и два негатива подбираются из трейна ОДИН раз и фиксируются,
# чтобы промпт не зависел от того, какие строки попали в текущий фолд
FEWSHOT_HINT = (
    "\n\nПримеры разбора:\n"
    "1) Карточка: «Витамин D3 2000 МЕ, 60 капсул. Биологически активная добавка к пище, "
    "не является лекарственным средством.» → да\n"
    "2) Карточка: «BCAA 6400, аминокислоты для спортсменов, 375 таблеток. Спортивное питание "
    "для восстановления после тренировок.» → нет\n"
    "3) Карточка: «Омега-3 рыбий жир, 90 капсул. БАД. Свидетельство о государственной "
    "регистрации.» → да\n"
    "4) Карточка: «Таблетница на 7 дней, органайзер для хранения витаминов и БАД.» → нет\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="runs/qwen_fewshot_oof.npy")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--name", default="")
    ap.add_argument("--text", default="ev", choices=["ev", "head"])
    ap.add_argument("--fewshot", action="store_true",
                    help="добавить 2 позитивных и 2 негативных примера в промпт (как 2p2n у коллег)")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    yes = tok(" да", add_special_tokens=False)["input_ids"][0]
    no = tok(" нет", add_special_tokens=False)["input_ids"][0]
    yes2 = tok("да", add_special_tokens=False)["input_ids"][0]
    no2 = tok("нет", add_special_tokens=False)["input_ids"][0]
    print("token ids:", yes, no, yes2, no2, flush=True)

    df = load_bad()
    si, sn = (int(x) for x in args.shard.split("/"))
    if sn > 1:
        df = df.iloc[si::sn].reset_index(drop=True)
    if args.text == "ev":
        body = [T.evidence(n, d, head=400, radius=140, budget=1600)
                for n, d in zip(df.name, df.description)]
    else:
        body = [T.head_only(n, d, 2000) for n, d in zip(df.name, df.description)]
    prompts = []
    for b in body:
        rules = RULES + (FEWSHOT_HINT if args.fewshot else "")
        msgs = [{"role": "user", "content": PROMPT.format(rules=rules, txt=b[:2500])}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(tok(p, add_special_tokens=False, truncation=True,
                           max_length=args.max_len)["input_ids"])
    out = np.zeros(len(prompts))
    order = np.argsort([-len(p) for p in prompts])
    pad = tok.pad_token_id
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), args.bs):
            sel = order[b:b + args.bs]
            seqs = [prompts[i] for i in sel]
            n = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), n), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), n), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, n - len(s):] = torch.tensor(s); att[j, n - len(s):] = 1
            lg = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, -1, :].float()
            ly = torch.logsumexp(torch.stack([lg[:, yes], lg[:, yes2]], -1), -1)
            ln = torch.logsumexp(torch.stack([lg[:, no], lg[:, no2]], -1), -1)
            out[sel] = torch.sigmoid(ly - ln).cpu().numpy()
            if b % (args.bs * 40) == 0:
                el = time.time() - t0
                print(f"  {b}/{len(order)}  {(b+1)/max(el,1e-9):.1f} it/s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, out)
    np.savez(args.out.replace(".npy", "_meta.npz"),
             p=out, ids=df.id.values, label=df.label.values)
    print("saved", args.out, f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
