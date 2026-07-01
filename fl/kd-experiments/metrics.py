from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_auc(metric_fn, y_true: np.ndarray, y_score: np.ndarray | None) -> float:
    if y_score is None or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(metric_fn(y_true, y_score))


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    miss_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": specificity,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(roc_auc_score, y_true, y_score),
        "pr_auc": _safe_auc(average_precision_score, y_true, y_score),
        "far": far,
        "miss_rate": miss_rate,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
