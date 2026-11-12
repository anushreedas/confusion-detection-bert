"""Classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from confusion_bert.constants import NUM_CLASSES


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    flat = np.bincount(y_true * num_classes + y_pred, minlength=num_classes * num_classes)
    return flat.reshape(num_classes, num_classes)


def _prf(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    pred = cm.sum(axis=0).astype(float)
    true = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    """Macro F1 over classes (classes absent from truth and prediction count as 0)."""
    return float(_prf(confusion_matrix(y_true, y_pred, num_classes))[2].mean())


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    p, r, f = _prf(cm)
    return {
        "n": int(cm.sum()),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": float(f.mean()),
        "per_class": {
            str(c): {
                "precision": float(p[c]),
                "recall": float(r[c]),
                "f1": float(f[c]),
                "support": int(cm[c].sum()),
            }
            for c in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
    }
