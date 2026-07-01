from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WINDOWS_PATH = REPO_ROOT / "data" / "preprocessing_results" / "simple" / "baseline_zscore" / "windows.csv"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
ID_COLUMNS = ["dataset", "client", "trial_id", "label", "variant", "feature_set"]
WINDOW_METADATA_COLUMNS = ["window_start_sec", "window_end_sec", "window_seconds"]
CLIENT_SPLIT_ASSIGNMENTS = {
    "KFall": {
        "train": [
            "SA06", "SA07", "SA08", "SA09", "SA10", "SA12", "SA13", "SA14", "SA15", "SA18",
            "SA19", "SA20", "SA22", "SA23", "SA24", "SA25", "SA26", "SA27", "SA28", "SA29",
            "SA30", "SA33", "SA35", "SA36", "SA37", "SA38",
        ],
        "dev": ["SA11", "SA16", "SA32"],
        "test": ["SA17", "SA21", "SA31"],
    },
    "SisFall": {
        "train": [
            "SA01", "SA02", "SA03", "SA04", "SA05", "SA06", "SA07", "SA08", "SA09", "SA10",
            "SA11", "SA13", "SA14", "SA17", "SA18", "SA19", "SA20", "SA21", "SA22", "SA23",
            "SE01", "SE02", "SE04", "SE07", "SE09", "SE11", "SE12", "SE13", "SE14", "SE15",
        ],
        "dev": ["SA15", "SE05", "SE06", "SE10"],
        "test": ["SA12", "SA16", "SE03", "SE08"],
    },
    "UpFall": {
        "train": ["S01", "S03", "S04", "S05", "S06", "S08", "S09", "S10", "S11", "S12", "S13", "S16", "S17"],
        "dev": ["S14", "S15"],
        "test": ["S02", "S07"],
    },
}


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    random_state: int = 42


def load_baseline_windows(path: Path = DEFAULT_WINDOWS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="all").copy()
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLUMNS + WINDOW_METADATA_COLUMNS)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded]


def build_trial_table(df: pd.DataFrame) -> pd.DataFrame:
    trial_table = (
        df.groupby(["dataset", "trial_id"], as_index=False)
        .agg(label=("label", "max"), windows=("label", "size"), client=("client", "first"))
        .copy()
    )
    return trial_table


def build_client_split_reference(assignments: dict[str, dict[str, list[str]]] = CLIENT_SPLIT_ASSIGNMENTS) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for dataset_name, split_map in assignments.items():
        for split_name, clients in split_map.items():
            for client_name in clients:
                rows.append({"dataset": dataset_name, "client": client_name, "split": split_name})
    return pd.DataFrame(rows).sort_values(["dataset", "split", "client"]).reset_index(drop=True)


def split_trials_by_client(
    trial_table: pd.DataFrame,
    assignments: dict[str, dict[str, list[str]]] = CLIENT_SPLIT_ASSIGNMENTS,
) -> pd.DataFrame:
    client_split_df = build_client_split_reference(assignments)
    merged = trial_table.merge(client_split_df, on=["dataset", "client"], how="left", validate="many_to_one")
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), ["dataset", "client"]].drop_duplicates()
        raise ValueError(f"Some dataset/client pairs were not assigned to a split:\n{missing.to_string(index=False)}")
    return merged.sort_values(["dataset", "split", "client", "trial_id"]).reset_index(drop=True)


def attach_splits(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(split_df[["dataset", "trial_id", "split"]], on=["dataset", "trial_id"], how="left", validate="many_to_one")
    if merged["split"].isna().any():
        raise ValueError("Some rows were not assigned to a split.")
    return merged


def build_matrices(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    matrices: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for split_name in ["train", "dev", "test"]:
        split_frame = df[df["split"] == split_name].copy()
        x_split = split_frame[feature_columns].copy()
        y_split = split_frame["label"].astype(int).copy()
        matrices[split_name] = (x_split, y_split)
    return matrices


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


def evaluate_predictions(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray | None,
    model_name: str,
    split_name: str,
) -> tuple[dict, list[dict]]:
    global_metrics = compute_binary_metrics(frame["label"].to_numpy(), predictions, scores)
    global_row = {"model": model_name, "split": split_name, "dataset": "ALL", **global_metrics}

    dataset_rows: list[dict] = []
    for dataset_name, dataset_frame in frame.groupby("dataset", sort=True):
        dataset_predictions = predictions[dataset_frame.index.to_numpy()]
        dataset_scores = None if scores is None else scores[dataset_frame.index.to_numpy()]
        dataset_metrics = compute_binary_metrics(dataset_frame["label"].to_numpy(), dataset_predictions, dataset_scores)
        dataset_rows.append({"model": model_name, "split": split_name, "dataset": dataset_name, **dataset_metrics})
    return global_row, dataset_rows


def build_split_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        split_df.groupby(["dataset", "split"], as_index=False)
        .agg(
            clients=("client", "nunique"),
            trials=("trial_id", "nunique"),
            positive_trials=("label", "sum"),
            windows=("windows", "sum"),
        )
        .sort_values(["dataset", "split"])
        .reset_index(drop=True)
    )
    summary["positive_ratio"] = summary["positive_trials"] / summary["trials"]
    return summary


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def dataclass_to_dict(instance) -> dict:
    return asdict(instance)
