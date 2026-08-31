"""End-to-end check of saved BAD artifacts through production inference code."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.cv import load_bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--data", default="data/data.csv")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "inference"))
    from bad_predictor import BadPredictor

    frame = load_bad(args.data)
    if args.limit > 0:
        frame = frame.head(args.limit).copy()
    predictor = BadPredictor(args.artifacts)
    probability, prediction = predictor.predict(frame)
    target = frame["label"].to_numpy(dtype=np.int8)

    assert len(probability) == len(prediction) == len(frame)
    assert np.isfinite(probability).all()
    assert set(np.unique(prediction)).issubset({0, 1})
    print(
        f"INFERENCE_OK rows={len(frame)} "
        f"F1={f1_score(target, prediction):.6f} "
        f"AUC={roc_auc_score(target, probability):.6f}"
    )


if __name__ == "__main__":
    main()
