from __future__ import annotations

import json
import random
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


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.8
    dev_ratio: float = 0.2
    test_ratio: float = 0.0
    random_state: int = 42
    n_splits: int = 5


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


def build_client_fold_reference(
    trial_table: pd.DataFrame,
    config: SplitConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, dataset_trials in trial_table.groupby("dataset", sort=True):
        clients = sorted(dataset_trials["client"].unique().tolist())
        shuffled_clients = clients.copy()
        random.Random(config.random_state).shuffle(shuffled_clients)
        folds = np.array_split(shuffled_clients, config.n_splits)
        for fold_index, fold_clients in enumerate(folds):
            for client_name in fold_clients.tolist():
                rows.append({"dataset": dataset_name, "client": client_name, "fold": int(fold_index)})
    return pd.DataFrame(rows).sort_values(["dataset", "fold", "client"]).reset_index(drop=True)


def build_fold_split_reference(
    trial_table: pd.DataFrame,
    config: SplitConfig,
) -> pd.DataFrame:
    fold_reference = build_client_fold_reference(trial_table, config)
    rows: list[dict[str, object]] = []

    for dataset_name, dataset_frame in fold_reference.groupby("dataset", sort=True):
        clients = sorted(dataset_frame["client"].tolist())
        for fold_index in range(config.n_splits):
            test_clients = sorted(dataset_frame[dataset_frame["fold"] == fold_index]["client"].tolist())
            remaining_clients = [client for client in clients if client not in test_clients]
            shuffled_remaining = remaining_clients.copy()
            random.Random(config.random_state + fold_index).shuffle(shuffled_remaining)
            dev_count = max(1, int(round(len(remaining_clients) * config.dev_ratio)))
            dev_clients = sorted(shuffled_remaining[:dev_count])
            train_clients = sorted([client for client in remaining_clients if client not in dev_clients])

            for split_name, split_clients in [("train", train_clients), ("dev", dev_clients), ("test", test_clients)]:
                for client_name in split_clients:
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "client": client_name,
                            "fold": int(fold_index),
                            "split": split_name,
                        }
                    )

    return pd.DataFrame(rows).sort_values(["fold", "dataset", "split", "client"]).reset_index(drop=True)


def split_trials_kfold_by_client(
    trial_table: pd.DataFrame,
    config: SplitConfig,
) -> pd.DataFrame:
    split_reference = build_fold_split_reference(trial_table, config)
    merged = trial_table.merge(split_reference, on=["dataset", "client"], how="left", validate="many_to_many")
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), ["dataset", "client"]].drop_duplicates()
        raise ValueError(f"Some dataset/client pairs were not assigned to a fold split:\n{missing.to_string(index=False)}")
    return merged.sort_values(["fold", "dataset", "split", "client", "trial_id"]).reset_index(drop=True)


def attach_splits(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        split_df[["dataset", "trial_id", "split", "fold"]],
        on=["dataset", "trial_id"],
        how="left",
        validate="many_to_many",
    )
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
    fold_index = int(frame["fold"].iloc[0]) if "fold" in frame.columns else None
    global_row = {"model": model_name, "split": split_name, "dataset": "ALL", "fold": fold_index, **global_metrics}

    dataset_rows: list[dict] = []
    for dataset_name, dataset_frame in frame.groupby("dataset", sort=True):
        dataset_predictions = predictions[dataset_frame.index.to_numpy()]
        dataset_scores = None if scores is None else scores[dataset_frame.index.to_numpy()]
        dataset_metrics = compute_binary_metrics(dataset_frame["label"].to_numpy(), dataset_predictions, dataset_scores)
        dataset_rows.append({"model": model_name, "split": split_name, "dataset": dataset_name, "fold": fold_index, **dataset_metrics})
    return global_row, dataset_rows


def build_split_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        split_df.groupby(["fold", "dataset", "split"], as_index=False)
        .agg(
            clients=("client", "nunique"),
            trials=("trial_id", "nunique"),
            positive_trials=("label", "sum"),
            windows=("windows", "sum"),
        )
        .sort_values(["fold", "dataset", "split"])
        .reset_index(drop=True)
    )
    summary["positive_ratio"] = summary["positive_trials"] / summary["trials"]
    return summary


def aggregate_metrics(rows: list[dict], group_columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "specificity",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "far",
        "miss_rate",
    ]
    count_columns = ["tn", "fp", "fn", "tp"]

    agg_dict = {column: "mean" for column in metric_columns}
    agg_dict.update({column: "sum" for column in count_columns})

    if "selected_candidate" in df.columns and "selected_candidate" not in group_columns:
        agg_dict["selected_candidate"] = "first"

    aggregated = df.groupby(group_columns, as_index=False).agg(agg_dict)
    return aggregated


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def dataclass_to_dict(instance) -> dict:
    return asdict(instance)
