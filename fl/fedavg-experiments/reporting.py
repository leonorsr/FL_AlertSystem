from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_EXPERIMENTS_ROOT, EXPERIMENT_CATALOG


SUMMARY_TXT_PATH = DEFAULT_EXPERIMENTS_ROOT / "fedavg_experiments_summary.txt"
EXPERIMENT_ORDER = list(EXPERIMENT_CATALOG.keys())
GLOBAL_SORT_COLUMNS = ["test_pr_auc_mean", "test_miss_rate_mean", "test_far_mean", "test_balanced_accuracy_mean", "test_f1_mean"]
GLOBAL_SORT_ASCENDING = [False, True, True, False, False]
LOCAL_SORT_COLUMNS = ["local_pr_auc_mean", "local_miss_rate_mean", "local_far_mean", "local_balanced_accuracy_mean", "local_f1_mean"]
LOCAL_SORT_ASCENDING = [False, True, True, False, False]
DISPLAY_METRICS = [
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _safe_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _comparison_id_and_title(experiment_id: str, config_payload: dict, run_summary: dict) -> tuple[str, str]:
    title = run_summary.get("title", config_payload.get("title", experiment_id))
    if experiment_id != "exp3_local_epochs":
        return experiment_id, title

    local_epochs = config_payload.get("local_epochs", "unknown")
    return f"exp3_local_epochs_{local_epochs}", f"{title} ({local_epochs} local epochs)"


def load_run_level_results(fedavg_root: Path = DEFAULT_EXPERIMENTS_ROOT):
    rows: list[dict] = []
    local_by_client_frames: list[pd.DataFrame] = []
    local_round_frames: list[pd.DataFrame] = []

    for experiment_id in EXPERIMENT_ORDER:
        results_root = fedavg_root / experiment_id / "results"
        run_dirs = sorted([path for path in results_root.glob("run_*") if path.is_dir()]) if results_root.exists() else []

        if not run_dirs:
            rows.append({"experiment_id": experiment_id, "comparison_id": experiment_id, "available": False, "latest_run": None})
            continue

        for run_dir in run_dirs:
            config_payload = _read_json(run_dir / "config.json")
            run_summary = _read_json(run_dir / "run_summary.json")
            comparison_id, comparison_title = _comparison_id_and_title(experiment_id, config_payload, run_summary)

            row = {
                "experiment_id": experiment_id,
                "comparison_id": comparison_id,
                "title": comparison_title,
                "available": True,
                "latest_run": run_dir.name,
                "run_path": str(run_dir),
            }

            for key, value in config_payload.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        row[f"model_{subkey}"] = _safe_float(subvalue)
                else:
                    row[key] = _safe_float(value)

            for prefix in ["dev_metrics", "test_metrics", "local_metrics_summary", "fine_tune_summary"]:
                payload = run_summary.get(prefix, {})
                prefix_name = prefix.replace("_metrics", "").replace("_summary", "")
                for metric_name, metric_value in payload.items():
                    row[f"{prefix_name}_{metric_name}"] = _safe_float(metric_value)

            local_by_client_path = run_dir / "local_metrics_by_client.csv"
            if local_by_client_path.exists():
                local_df = pd.read_csv(local_by_client_path)
                local_df["comparison_id"] = comparison_id
                local_df["experiment_id"] = experiment_id
                local_df["run_name"] = run_dir.name
                local_by_client_frames.append(local_df)

            local_round_path = run_dir / "local_round_metrics.csv"
            if local_round_path.exists():
                local_round_df = pd.read_csv(local_round_path)
                local_round_df["comparison_id"] = comparison_id
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

    available = results_df[results_df["available"]].copy()
    if available.empty:
        return pd.DataFrame()

    numeric_columns = available.select_dtypes(include=["number"]).columns.tolist()
    metadata_columns = [
        "comparison_id",
        "experiment_id",
        "title",
        "description",
        "scenario",
        "holdout_dataset",
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
        "final_local_finetune_epochs",
        "random_seed",
        "model_hidden_layers",
        "model_dropout",
        "model_learning_rate",
        "model_weight_decay",
        "model_batch_size",
        "model_use_batchnorm",
    ]
    metadata_columns = [column for column in metadata_columns if column in available.columns]

    aggregated_rows: list[dict] = []
    for comparison_id, group in available.groupby("comparison_id", sort=False):
        row = {"comparison_id": comparison_id, "n_runs": int(len(group))}
        for column in metadata_columns:
            row[column] = group.iloc[0][column]

        for column in numeric_columns:
            series = pd.to_numeric(group[column], errors="coerce")
            if series.notna().any():
                row[f"{column}_mean"] = float(series.mean())
                row[f"{column}_std"] = float(series.std(ddof=1)) if len(series.dropna()) > 1 else 0.0

        aggregated_rows.append(row)

    aggregated_df = pd.DataFrame(aggregated_rows)
    if all(column in aggregated_df.columns for column in GLOBAL_SORT_COLUMNS):
        aggregated_df = aggregated_df.sort_values(GLOBAL_SORT_COLUMNS, ascending=GLOBAL_SORT_ASCENDING).reset_index(drop=True)
    return aggregated_df


def aggregate_local_by_client(local_by_client_df: pd.DataFrame) -> pd.DataFrame:
    if local_by_client_df.empty:
        return pd.DataFrame()

    numeric_columns = local_by_client_df.select_dtypes(include=["number"]).columns.tolist()
    key_columns = ["comparison_id", "experiment_id", "dataset", "client", "cluster_id", "evaluation_split", "model_source"]
    key_columns = [column for column in key_columns if column in local_by_client_df.columns]

    agg_spec = {}
    for column in numeric_columns:
        agg_spec[column] = ["mean", "std"]

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


def format_mean_std(mean_value, std_value, decimals: int = 4) -> str:
    if pd.isna(mean_value):
        return "n/a"
    if pd.isna(std_value):
        return f"{mean_value:.{decimals}f}"
    return f"{mean_value:.{decimals}f} +/- {std_value:.{decimals}f}"


def build_summary_text(aggregated_df: pd.DataFrame) -> str:
    lines = [
        "FedAvg Experiments Summary",
        "==========================",
        "",
        "Results are aggregated across all available runs for each experiment/variant.",
        "Uncertainty is reported as sample standard deviation across runs (mean +/- std).",
        "",
    ]

    if aggregated_df.empty:
        lines.extend(
            [
                "No runs are currently available.",
                "",
                "Run the experiments first, then regenerate this summary.",
            ]
        )
        return "\n".join(lines)

    for _, row in aggregated_df.iterrows():
        lines.extend(
            [
                f"{row['comparison_id']}",
                "-" * len(str(row["comparison_id"])),
                f"Title: {row.get('title', row['comparison_id'])}",
                f"Description: {row.get('description', 'n/a')}",
                f"Runs aggregated: {int(row.get('n_runs', 0))}",
                f"Scenario: {row.get('scenario', 'n/a')}",
                f"Global rounds: {int(row.get('num_rounds', 0)) if not pd.isna(row.get('num_rounds')) else 'n/a'}",
                f"Local epochs: {int(row.get('local_epochs', 0)) if not pd.isna(row.get('local_epochs')) else 'n/a'}",
                f"Fraction fit: {row.get('fraction_fit', 'n/a')}",
                f"Weighted aggregation: {row.get('weighted_aggregation', 'n/a')}",
                f"Local model selection: {row.get('local_model_selection', 'n/a')}",
                f"Clustered aggregation: {row.get('clustered_aggregation', 'n/a')}",
                f"Personalized head: {row.get('personalized_head', 'n/a')}",
                f"Final local fine-tuning epochs: {int(row.get('final_local_finetune_epochs', 0)) if not pd.isna(row.get('final_local_finetune_epochs')) else 'n/a'}",
                f"Random seed in config: {row.get('random_seed', 'n/a')}",
                f"Model hidden layers: {row.get('model_hidden_layers', 'n/a')}",
                f"Model dropout: {row.get('model_dropout', 'n/a')}",
                f"Model learning rate: {row.get('model_learning_rate', 'n/a')}",
                f"Model weight decay: {row.get('model_weight_decay', 'n/a')}",
                f"Model batch size: {row.get('model_batch_size', 'n/a')}",
                "",
                "Final global dev metrics:",
            ]
        )
        for metric in DISPLAY_METRICS:
            lines.append(f"  {metric}: {format_mean_std(row.get(f'dev_{metric}_mean'), row.get(f'dev_{metric}_std'))}")

        lines.append("")
        lines.append("Final global test metrics:")
        for metric in DISPLAY_METRICS:
            lines.append(f"  {metric}: {format_mean_std(row.get(f'test_{metric}_mean'), row.get(f'test_{metric}_std'))}")

        if any(f"local_{metric}_mean" in row.index for metric in DISPLAY_METRICS):
            lines.append("")
            lines.append("Final local summary metrics:")
            for metric in DISPLAY_METRICS:
                lines.append(f"  {metric}: {format_mean_std(row.get(f'local_{metric}_mean'), row.get(f'local_{metric}_std'))}")

        if any(f"fine_tune_{metric}_mean" in row.index for metric in DISPLAY_METRICS):
            lines.append("")
            lines.append("Final fine-tuned test-client metrics:")
            for metric in DISPLAY_METRICS:
                lines.append(f"  {metric}: {format_mean_std(row.get(f'fine_tune_{metric}_mean'), row.get(f'fine_tune_{metric}_std'))}")

        lines.extend(["", ""])

    return "\n".join(lines).rstrip() + "\n"


def export_summary_txt(fedavg_root: Path = DEFAULT_EXPERIMENTS_ROOT, output_path: Path = SUMMARY_TXT_PATH) -> Path:
    results_df, _, _ = load_run_level_results(fedavg_root)
    aggregated_df = aggregate_runs(results_df)
    output_path.write_text(build_summary_text(aggregated_df), encoding="utf-8")
    return output_path

