# LVJ

`train_dora.py` — исходный full-train DoRA experiment:

- Qwen3.5-4B;
- official 128 target modules, rank 16, DoRA;
- restricted two-token CE по `Нет/Да`;
- positive weight ×6;
- checkpoints 0.5, 1.0, 1.5, 2.0, 2.5 и 3.0;
- zero few-shot, data-agreed system rules.

Production routing не зашит в trainer. Его собирает `../solution.py build`:
exact train lookup, затем точные order-aware rules из `best.zip`, затем adapter
checkpoint как fallback.
