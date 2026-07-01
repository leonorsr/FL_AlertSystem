from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from config import EXPERIMENT_CATALOG

DISPLAY_METRICS = ["accuracy", "pr_auc", "miss_rate", "far"]
EXPERIMENT_ORDER = list(EXPERIMENT_CATALOG.keys())
GLOBAL_SORT_COLUMNS = ["test_pr_auc_mean", "test_miss_rate_mean", "test_far_mean"]
GLOBAL_SORT_ASCENDING = [False, True, True]
LOCAL_SORT_COLUMNS = ["client_id", "pr_auc"]
LOCAL_SORT_ASCENDING = [True, False]
SUMMARY_TXT_PATH = Path("kdexperiments_summary.txt")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _safe_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_run_level_results(kd_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load run-level results from KD experiment folders."""
    rows: list[dict] = []
    local_by_client_frames: list[pd.DataFrame] = []
    local_round_frames: list[pd.DataFrame] = []

    for experiment_id in EXPERIMENT_ORDER:
        results_root = kd_root / experiment_id / "results"
        run_dirs = sorted([path for path in results_root.glob("run_*") if path.is_dir()]) if results_root.exists() else []

        if not run_dirs:
            rows.append({"experiment_id": experiment_id, "comparison_id": experiment_id})
            continue

        for run_dir in run_dirs:
            config_payload = _read_json(run_dir / "config.json")
            run_summary = _read_json(run_dir / "run_summary.json")

            row = {
                "experiment_id": experiment_id,
                "comparison_id": experiment_id,
                "run_path": str(run_dir),
            }

            # Load config parameters
            for key, value in config_payload.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        row[f"model_{subkey}"] = _safe_float(subvalue)
                else:
                    row[key] = _safe_float(value)

            # Load metrics from run_summary
            for prefix in ["dev_metrics", "test_metrics", "fine_tune_summary"]:
                payload = run_summary.get(prefix, {})
                prefix_name = prefix.replace("_metrics", "").replace("_summary", "")
                for metric_name, metric_value in payload.items():
                    row[f"{prefix_name}_{metric_name}"] = _safe_float(metric_value)

            # Load local metrics by client
            local_by_client_path = run_dir / "local_metrics_by_client.csv"
            if local_by_client_path.exists():
                local_df = pd.read_csv(local_by_client_path)
                local_df["comparison_id"] = experiment_id
                local_df["experiment_id"] = experiment_id
                local_df["run_name"] = run_dir.name
                local_by_client_frames.append(local_df)

            # Load local round metrics
            local_round_path = run_dir / "local_round_metrics.csv"
            if local_round_path.exists():
                local_round_df = pd.read_csv(local_round_path)
                local_round_df["comparison_id"] = experiment_id
                local_round_df["experiment_id"] = experiment_id
                local_round_df["run_name"] = run_dir.name
                local_round_frames.append(local_round_df)

            rows.append(row)

    results_df = pd.DataFrame(rows)
    local_by_client_df = pd.concat(local_by_client_frames, ignore_index=True) if local_by_client_frames else pd.DataFrame()
    local_round_df = pd.concat(local_round_frames, ignore_index=True) if local_round_frames else pd.DataFrame()
    return results_df, local_by_client_df, local_round_df


def aggregate_runs(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    numeric_columns = results_df.select_dtypes(include=["number"]).columns.tolist()
    metadata_columns = [
        "comparison_id",
        "experiment_id",
        "scenario",
        "num_rounds",
        "fraction_fit",
        "min_fit_clients",
        "min_available_clients",
        "local_epochs",
        "weighted_aggregation",
        "local_model_selection",
        "clustered_aggregation",
        "num_similarity_groups",
        "personalized_head",
        "final_local_epochs",
        "final_local_finetune_epochs",
        "random_seed",
        "teacher_student_distillation",
        "distillation_temperature",
        "distillation_alpha",
        "model_hidden_layers",
        "model_dropout",
        "model_learning_rate",
        "model_weight_decay",
        "model_batch_size",
        "model_use_batchnorm",
    ]
    metadata_columns = [column for column in metadata_columns if column in results_df.columns]

    aggregated_rows: list[dict] = []
    for comparison_id, group in results_df.groupby("comparison_id", sort=False):
        row = {"comparison_id": comparison_id, "n_runs": int(len(group))}
        for column in metadata_columns:
            if column in group.columns:
                row[column] = group.iloc[0][column]

        for column in numeric_columns:
            series = pd.to_numeric(group[column], errors="coerce")
            if series.notna().any():
                row[f"{column}_mean"] = float(series.mean())
                row[f"{column}_std"] = float(series.std(ddof=1)) if len(series.dropna()) > 1 else 0.0

        aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows)


def aggregate_local_by_client(local_by_client_df: pd.DataFrame) -> pd.DataFrame:
    if local_by_client_df.empty:
        return pd.DataFrame()

    numeric_columns = local_by_client_df.select_dtypes(include=["number"]).columns.tolist()
    key_columns = ["comparison_id", "experiment_id", "dataset", "client", "cluster_id", "evaluation_split", "model_source"]
    key_columns = [column for column in key_columns if column in local_by_client_df.columns]

    if not key_columns or not numeric_columns:
        return pd.DataFrame()

    agg_spec = {column: ["mean", "std"] for column in numeric_columns}
    aggregated = local_by_client_df.groupby(key_columns, dropna=False).agg(agg_spec)
    aggregated.columns = [f"{column}_{stat}" for column, stat in aggregated.columns]
    aggregated = aggregated.reset_index()
    return aggregated


def aggregate_local_rounds(local_round_df: pd.DataFrame) -> pd.DataFrame:
    if local_round_df.empty:
        return pd.DataFrame()

    pivoted = local_round_df.pivot_table(
        index=["comparison_id", "experiment_id", "run_name", "round"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()

    numeric_columns = pivoted.select_dtypes(include=["number"]).columns.tolist()
    key_columns = ["comparison_id", "experiment_id", "round"]
    metric_columns = [column for column in numeric_columns if column not in ["round"]]

    aggregated_rows = []
    for (comparison_id, experiment_id, server_round), group in pivoted.groupby(key_columns, sort=False):
        row = {"comparison_id": comparison_id, "experiment_id": experiment_id, "round": int(server_round)}
        for column in metric_columns:
            series = pd.to_numeric(group[column], errors="coerce")
            if series.notna().any():
                row[f"{column}_mean"] = float(series.mean())
                row[f"{column}_std"] = float(series.std(ddof=1)) if len(series.dropna()) > 1 else 0.0
        aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows).sort_values(["comparison_id", "round"]).reset_index(drop=True)


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"
