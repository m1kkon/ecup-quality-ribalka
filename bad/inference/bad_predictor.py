"""Drop-in БАД classifier for the E-CUP quality submission.

    from bad_predictor import BadPredictor
    bp = BadPredictor("artifacts")            # loads once
    prob, pred = bp.predict(df_bad)           # df with columns: name, description

Everything it needs lives inside the artifacts directory, so the container needs
no network. Encoders are stored fp16; inference runs in bf16 on the GPU with
length-sorted batching, which is what keeps it inside the time budget.
"""
from __future__ import annotations

import json, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from text import evidence, clean                    # noqa: F401  (same dir at runtime)

LLM_RULES = (
    "Правила площадки Ozon:\n"
    "Товар ЯВЛЯЕТСЯ биологически активной добавкой, если в описании или на упаковке "
    "есть прямое указание, что это БАД (БАД, биологически активная добавка, dietary supplement).\n"
    "Товар НЕ ЯВЛЯЕТСЯ БАД, если это спортивное питание (аминокислоты, BCAA, L-карнитин, "
    "протеин), если в описании прямо сказано, что товар не является БАД, "
    "или если маркировки БАД нет."
)
LLM_PROMPT = ("{rules}\n\nКарточка товара:\n{txt}\n\n"
              "Вопрос: является ли этот товар БАД по правилам выше? "
              "Ответь одним словом: да или нет.")
LLM_FEWSHOT = (
    "\n\nПримеры разбора:\n"
    "1) Карточка: «Витамин D3 2000 МЕ, 60 капсул. Биологически активная добавка к пище, "
    "не является лекарственным средством.» → да\n"
    "2) Карточка: «BCAA 6400, аминокислоты для спортсменов, 375 таблеток. Спортивное питание "
    "для восстановления после тренировок.» → нет\n"
    "3) Карточка: «Омега-3 рыбий жир, 90 капсул. БАД. Свидетельство о государственной "
    "регистрации.» → да\n"
    "4) Карточка: «Таблетница на 7 дней, органайзер для хранения витаминов и БАД.» → нет\n"
)
from feats import build as build_feats


class BadPredictor:
    def __init__(self, artifacts: str, device: str = "cuda", batch_size: int = 64):
        self.dir = Path(artifacts)
        self.cfg = json.load(open(self.dir / "ensemble.json"))
        self.device = device
        self.bs = batch_size
        self.threshold = float(self.cfg["threshold"])
        self.weights = np.asarray(self.cfg["weights"], dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        self._enc_cache = {}

    # ------------------------------------------------------------ members
    def _texts(self, df, spec):
        kind = spec.get("text", "ev")
        if kind == "ev":
            return [evidence(n, d, head=spec.get("head", 400), radius=spec.get("radius", 140),
                             budget=spec.get("budget", 1600))
                    for n, d in zip(df["name"].fillna(""), df["description"].fillna(""))]
        if kind == "full":
            return [f"{clean(n)} | {clean(d)}"
                    for n, d in zip(df["name"].fillna(""), df["description"].fillna(""))]
        if kind == "head":
            return [f"{clean(n)} | {clean(d)[:2000]}"
                    for n, d in zip(df["name"].fillna(""), df["description"].fillna(""))]
        raise ValueError(kind)

    def _enc_predict(self, df, spec):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        d = self.dir / spec["name"]
        tok = AutoTokenizer.from_pretrained(d)
        # веса лежат в fp16 — грузим ровно в нём, без конверсии через bf16
        # (у bf16 на 3 бита меньше мантиссы, и пограничные карточки перещёлкивают)
        if (d / "head_config.json").exists():
            from heads import PooledClassifier
            model = PooledClassifier.from_saved(d, dtype=torch.float16).to(self.device).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                d, dtype=torch.float16, trust_remote_code=True).to(self.device).eval()
        ids = tok(self._texts(df, spec), truncation=True, max_length=spec["max_len"])["input_ids"]
        order = np.argsort([-len(x) for x in ids])
        pad = tok.pad_token_id or 0
        out = np.zeros(len(ids), dtype=np.float64)
        with torch.no_grad():
            for b in range(0, len(order), self.bs):
                sel = order[b:b + self.bs]
                seqs = [ids[i] for i in sel]
                n = max(len(s) for s in seqs)
                inp = torch.full((len(seqs), n), pad, dtype=torch.long)
                att = torch.zeros((len(seqs), n), dtype=torch.long)
                for j, s in enumerate(seqs):
                    inp[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
                logits = model(input_ids=inp.to(self.device),
                               attention_mask=att.to(self.device)).logits.squeeze(-1).float()
                out[sel] = torch.sigmoid(logits).cpu().numpy()
        del model
        torch.cuda.empty_cache()
        return out

    def _tfidf_predict(self, df, spec):
        import joblib
        art = joblib.load(self.dir / f"{spec['name']}.joblib")
        s = art["spec"]
        X = [evidence(n, d, head=s.get("head", 0), radius=s.get("radius", 40),
                      budget=s.get("budget", 2000))
             for n, d in zip(df["name"].fillna(""), df["description"].fillna(""))]
        return art["clf"].predict_proba(art["vec"].transform(X))[:, 1]

    def _llm_predict(self, df, spec):
        """Qwen3.5-4B из SHARED_MODELS_PATH — в архиве не лежит, поэтому места не занимает.
        Скор = P("да") по логитам первого сгенерированного токена."""
        import os
        from transformers import AutoTokenizer, AutoModelForCausalLM
        shared = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
        mid = spec.get("model", "Qwen/Qwen3.5-4B")
        path = spec.get("path") or os.path.join(shared, mid)
        if not os.path.isdir(path):
            # В проверочном контейнере сети нет. Пробуем локальный кеш, но если и его
            # нет — это не повод ронять весь прогон: член просто выпадает из голосования.
            path = mid
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(self.device).eval()
        ids_yes = [tok(w, add_special_tokens=False)["input_ids"][0] for w in (" да", "да")]
        ids_no = [tok(w, add_special_tokens=False)["input_ids"][0] for w in (" нет", "нет")]
        body = self._texts(df, {**spec, "text": spec.get("text", "ev")})
        rules = LLM_RULES + (LLM_FEWSHOT if spec.get("fewshot") else "")
        prompts = []
        for b in body:
            msgs = [{"role": "user", "content": LLM_PROMPT.format(rules=rules, txt=b[:2500])}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(tok(t, add_special_tokens=False, truncation=True,
                               max_length=spec.get("max_len", 1024))["input_ids"])
        out = np.zeros(len(prompts))
        order = np.argsort([-len(p) for p in prompts])
        pad = tok.pad_token_id
        bs = spec.get("bs", 16)
        with torch.no_grad():
            for b in range(0, len(order), bs):
                sel = order[b:b + bs]
                seqs = [prompts[i] for i in sel]
                n = max(len(x) for x in seqs)
                ids = torch.full((len(seqs), n), pad, dtype=torch.long)
                att = torch.zeros((len(seqs), n), dtype=torch.long)
                for j, x in enumerate(seqs):          # left pad для генерации
                    ids[j, n - len(x):] = torch.tensor(x); att[j, n - len(x):] = 1
                lg = model(input_ids=ids.to(self.device),
                           attention_mask=att.to(self.device)).logits[:, -1, :].float()
                ly = torch.logsumexp(torch.stack([lg[:, i] for i in ids_yes], -1), -1)
                ln = torch.logsumexp(torch.stack([lg[:, i] for i in ids_no], -1), -1)
                out[sel] = torch.sigmoid(ly - ln).cpu().numpy()
        del model
        torch.cuda.empty_cache()
        return out

    def _gbm_predict(self, df, spec):
        import joblib
        art = joblib.load(self.dir / f"{spec['name']}.joblib")
        F = build_feats(df)[art["cols"]]
        return art["model"].predict_proba(F)[:, 1]

    # ------------------------------------------------------------ public
    def _member_probs(self, df: pd.DataFrame):
        """Возвращает (матрица вероятностей, список выживших спеков).

        Член, который не смог загрузиться, выбрасывается, а не роняет весь прогон:
        потерять один голос из пяти лучше, чем получить ноль за весь сабмит.
        """
        cols, kept = [], []
        for spec in self.cfg["members"]:
            k = spec["kind"]
            try:
                p = (self._enc_predict(df, spec) if k == "enc" else
                     self._tfidf_predict(df, spec) if k == "tfidf" else
                     self._llm_predict(df, spec) if k == "llm" else
                     self._gbm_predict(df, spec))
            except Exception as e:                                  # noqa: BLE001
                print(f"[bad] член {spec['name']} не загрузился ({type(e).__name__}: "
                      f"{str(e)[:120]}), пропускаю", flush=True)
                continue
            cols.append(np.asarray(p, dtype=np.float64)); kept.append(spec)
        if not cols:
            raise RuntimeError("ни один член ансамбля не загрузился")
        return np.column_stack(cols), kept

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        P, kept = self._member_probs(df.reset_index(drop=True))
        w = np.ones(len(kept)) / len(kept)
        return P @ w

    def predict(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        P, kept = self._member_probs(df)
        w = np.ones(len(kept)) / len(kept)
        if self.cfg.get("combine") == "majority":
            ths = np.array([m.get("threshold", 0.5) for m in kept])
            need = int(self.cfg.get("votes_needed", len(kept) // 2 + 1))
            need = min(need, len(kept))          # если члены выпали — порог голосов тоже
            votes = (P >= ths).sum(1)
            return P @ w, (votes >= need).astype(int)
        p = P @ w
        return p, (p >= self.threshold).astype(int)
