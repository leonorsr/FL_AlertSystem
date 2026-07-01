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
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WINDOWS_PATH = REPO_ROOT / "data" / "preprocessing_results" / "simple" / "magnitude_only" / "windows.csv"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
ID_COLUMNS = ["dataset", "client", "trial_id", "label", "variant", "feature_set"]
WINDOW_METADATA_COLUMNS = ["window_start_sec", "window_end_sec", "window_seconds"]


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.9
    dev_ratio: float = 0.1
    test_ratio: float = 0.0
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


def _stratify_or_none(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) < 2 or (counts < 2).any():
        return None
    return labels


def get_holdout_datasets(trial_table: pd.DataFrame) -> list[str]:
    return sorted(trial_table["dataset"].unique().tolist())


def split_trials_leave_one_dataset_out(
    trial_table: pd.DataFrame,
    holdout_dataset: str,
    config: SplitConfig,
) -> pd.DataFrame:
    split_frames: list[pd.DataFrame] = []

    for dataset_name, dataset_trials in trial_table.groupby("dataset", sort=True):
        dataset_trials = dataset_trials.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)

        if dataset_name == holdout_dataset:
            split_frames.append(dataset_trials.assign(split="test", held_out_dataset=holdout_dataset))
            continue

        stratify = _stratify_or_none(dataset_trials["label"])
        train_trials, dev_trials = train_test_split(
            dataset_trials,
            train_size=config.train_ratio,
            random_state=config.random_state,
            stratify=stratify,
        )
        split_frames.append(train_trials.assign(split="train", held_out_dataset=holdout_dataset))
        split_frames.append(dev_trials.assign(split="dev", held_out_dataset=holdout_dataset))

    split_df = pd.concat(split_frames, ignore_index=True)
    return split_df.sort_values(["held_out_dataset", "dataset", "split", "trial_id"]).reset_index(drop=True)


def attach_splits(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        split_df[["dataset", "trial_id", "split", "held_out_dataset"]],
        on=["dataset", "trial_id"],
        how="left",
        validate="many_to_one",
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
    holdout_dataset = frame["held_out_dataset"].iloc[0] if "held_out_dataset" in frame.columns else None
    global_row = {
        "model": model_name,
        "split": split_name,
        "dataset": "ALL",
        "held_out_dataset": holdout_dataset,
        **global_metrics,
    }

    dataset_rows: list[dict] = []
    for dataset_name, dataset_frame in frame.groupby("dataset", sort=True):
        dataset_predictions = predictions[dataset_frame.index.to_numpy()]
        dataset_scores = None if scores is None else scores[dataset_frame.index.to_numpy()]
        dataset_metrics = compute_binary_metrics(dataset_frame["label"].to_numpy(), dataset_predictions, dataset_scores)
        dataset_rows.append(
            {
                "model": model_name,
                "split": split_name,
                "dataset": dataset_name,
                "held_out_dataset": holdout_dataset,
                **dataset_metrics,
            }
        )
    return global_row, dataset_rows


def build_split_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        split_df.groupby(["held_out_dataset", "dataset", "split"], as_index=False)
        .agg(
            clients=("client", "nunique"),
            trials=("trial_id", "nunique"),
            positive_trials=("label", "sum"),
            windows=("windows", "sum"),
        )
        .sort_values(["held_out_dataset", "dataset", "split"])
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

    constant_columns = ["selected_candidate"]
    for column in constant_columns:
        if column in df.columns and column not in group_columns:
            agg_dict[column] = "first"

    aggregated = df.groupby(group_columns, as_index=False).agg(agg_dict)
    return aggregated


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def dataclass_to_dict(instance) -> dict:
    return asdict(instance)
