"""Batch prediction over a split CSV, and stored-prediction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from confusion_bert.data.dataset import load_split
from confusion_bert.evaluation.metrics import classification_report
from confusion_bert.inference import ConfusionPredictor


def predict_split(
    run_dir: str | Path, csv_path: str | Path, split: str, device: str = "auto"
) -> dict:
    """Predict a split, write ``predictions_<split>.csv`` + ``metrics_<split>.json`` in the run dir."""
    import json

    run_dir = Path(run_dir)
    df = load_split(csv_path)
    predictor = ConfusionPredictor.from_run_dir(run_dir, device)
    probs = predictor.predict_proba(df["sentence"].tolist())
    out = pd.DataFrame({"y_true": df["label"].to_numpy(), "y_pred": probs.argmax(axis=1)})
    for c in range(probs.shape[1]):
        out[f"p{c}"] = probs[:, c]
    out.to_csv(run_dir / f"predictions_{split}.csv", index=False)
    report = classification_report(
        out["y_true"].to_numpy(), out["y_pred"].to_numpy(), probs.shape[1]
    )
    (run_dir / f"metrics_{split}.json").write_text(json.dumps(report, indent=2))
    return report


def load_predictions(run_dir: str | Path, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(Path(run_dir) / f"predictions_{split}.csv")
    return df["y_true"].to_numpy(), df["y_pred"].to_numpy()
