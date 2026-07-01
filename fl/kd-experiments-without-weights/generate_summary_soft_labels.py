from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kd-experiments"))
from config import EXPERIMENT_CATALOG

EXPERIMENT_IDS = [
    "exp1_kkd_base",
    "exp2_fraction_clients",
    "exp3_local_epochs",
    "exp4_unweighted_aggregation",
    "exp5_cross_dataset",
    "exp6_keep_best_local_model",
    "exp7_clustered_aggregation",
    "exp8_personalized_fedavg",
    "exp9_final_local_finetuning",
    "exp10_clustered_keep_best_local",
    "exp11_baseline_final",
    "exp12_final_unweighted",
    "exp13_final_keep_best_local",
    "exp14_final_clustered",
]

EXPERIMENT_METADATA: dict[str, dict[str, str]] = {
    "exp1_kkd_base": {
        "title": "Experiment 1 - KD Base",
        "description": "Baseline KD variant using soft labels without weights.",
    },
    "exp2_fraction_clients": {
        "title": "Experiment 2 - Fraction Of Clients",
        "description": "KD soft labels with partial client participation (50%).",
    },
    "exp3_local_epochs": {
        "title": "Experiment 3 - Local Epochs Comparison",
        "description": "KD soft labels comparing different local epoch settings.",
    },
    "exp4_unweighted_aggregation": {
        "title": "Experiment 4 - Unweighted Aggregation",
        "description": "KD soft labels with unweighted aggregation of client payloads.",
    },
    "exp5_cross_dataset": {
        "title": "Experiment 5 - Cross Dataset KD",
        "description": "KD soft labels across datasets with held-out test split.",
    },
    "exp6_keep_best_local_model": {
        "title": "Experiment 6 - Keep Best Local Model",
        "description": "KD soft labels where each client keeps the better local model.",
    },
    "exp7_clustered_aggregation": {
        "title": "Experiment 7 - Clustered Aggregation",
        "description": "KD soft labels with clustered payload aggregation.",
    },
    "exp8_personalized_fedavg": {
        "title": "Experiment 8 - Personalized KD",
        "description": "KD soft labels with a personalized local head.",
    },
    "exp9_final_local_finetuning": {
        "title": "Experiment 9 - Final Local Fine-Tuning",
        "description": "KD soft labels plus final local fine-tuning.",
    },
    "exp10_clustered_keep_best_local": {
        "title": "Experiment 10 - Clustered Keep-Best Local",
        "description": "KD soft labels combining clustering and local model selection.",
    },
    "exp11_baseline_final": {
        "title": "Experiment 11 - Baseline Final",
        "description": "KD soft labels baseline with final strong settings.",
    },
    "exp12_final_unweighted": {
        "title": "Experiment 12 - Final Unweighted",
        "description": "KD soft labels with final unweighted aggregation.",
    },
    "exp13_final_keep_best_local": {
        "title": "Experiment 13 - Final Keep-Best Local",
        "description": "KD soft labels with final local model selection.",
    },
    "exp14_final_clustered": {
        "title": "Experiment 14 - Final Clustered",
        "description": "KD soft labels with final clustered aggregation.",
    },
}

METRIC_ORDER = [
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flatten_summary(summary: dict[str, Any], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{k}": _safe_float(v) for k, v in summary.items() if _safe_float(v) is not None}


def _aggregate_run_summaries(run_summaries: list[dict[str, Any]]) -> dict[str, float]:
    if not run_summaries:
        return {}
    df = pd.DataFrame(run_summaries)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    agg = {}
    for col in numeric_cols:
        values = df[col].dropna()
        if values.empty:
            continue
        agg[f"{col}_mean"] = float(values.mean())
        agg[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return agg


def load_experiment_runs(experiment_id: str, experiment_dir: Path) -> list[dict[str, Any]]:
    runs = []
    results_dir = experiment_dir / "results"
    if not results_dir.exists():
        return runs

    expected = EXPERIMENT_CATALOG.get(experiment_id)
    for run_path in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        config = _read_json(run_path / "config.json")
        if expected is not None:
            if int(config.get("num_rounds", -1)) != int(expected.num_rounds):
                continue
            if int(config.get("local_epochs", -1)) != int(expected.local_epochs):
                continue
        if config.get("communication_mode") != "soft_labels":
            continue
        summary = _read_json(run_path / "run_summary.json")
        if not summary:
            continue
        if not summary.get("dev_metrics") and not summary.get("test_metrics"):
            continue
        runs.append(summary)
    return runs


def build_experiment_row(experiment_id: str, experiment_dir: Path) -> dict[str, Any]:
    runs = load_experiment_runs(experiment_id, experiment_dir)
    flattened_runs = []
    for run in runs:
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "n_runs": len(runs),
        }
        row.update(_flatten_summary(run.get("dev_metrics", {}), "dev"))
        row.update(_flatten_summary(run.get("test_metrics", {}), "test"))
        row.update(_flatten_summary(run.get("local_metrics_summary", {}), "local"))
        flattened_runs.append(row)

    agg = _aggregate_run_summaries(flattened_runs)
    if not agg:
        agg = {"experiment_id": experiment_id, "n_runs": len(runs)}
    else:
        agg["experiment_id"] = experiment_id
        agg["n_runs"] = len(runs)
    # pass metadata fields if available from first run
    if runs:
        first = runs[0]
        agg["title"] = first.get("title", EXPERIMENT_METADATA.get(experiment_id, {}).get("title", experiment_id))
        agg["description"] = first.get("description", EXPERIMENT_METADATA.get(experiment_id, {}).get("description", ""))
    return agg


def main() -> None:
    root = Path(__file__).resolve().parent
    channel_dir = root / "soft labels"
    rows = []
    for experiment_id in EXPERIMENT_IDS:
        exp_dir = channel_dir / experiment_id
        if not exp_dir.exists():
            continue
        row = build_experiment_row(experiment_id, exp_dir)
        rows.append(row)

    if not rows:
        print("No runs found for soft labels.")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid data to summarize.")
        return

    order_cols = [col for col in df.columns if col.startswith("test_pr_auc") or col.startswith("test_f1")]
    if "test_pr_auc_mean" in df.columns and "test_f1_mean" in df.columns:
        df = df.sort_values(["test_pr_auc_mean", "test_f1_mean"], ascending=[False, False], na_position="last")

    summary_lines = [
        "Soft Labels KD Experiments Summary",
        "=================================",
        "",
        "Results are aggregated across all available runs for each experiment.",
        "Uncertainty is reported as sample standard deviation across runs (mean +/- std).",
        "",
        "Final ranking is by test PR-AUC mean.",
        "",
    ]

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        summary_lines.append(f"{idx}. {row.experiment_id} - test PR-AUC: {getattr(row, 'test_pr_auc_mean', float('nan')):.4f}, test F1: {getattr(row, 'test_f1_mean', float('nan')):.4f}")

    summary_lines.append("")

    for row in df.itertuples(index=False):
        summary_lines.append(str(row.experiment_id))
        summary_lines.append("-" * len(row.experiment_id))
        summary_lines.append(f"Title: {EXPERIMENT_METADATA.get(row.experiment_id, {}).get('title', row.experiment_id)}")
        summary_lines.append(f"Description: {EXPERIMENT_METADATA.get(row.experiment_id, {}).get('description', '')}")
        summary_lines.append(f"Runs aggregated: {int(getattr(row, 'n_runs', 0))}")
        summary_lines.append("")
        summary_lines.append("Final global dev metrics:")
        for metric in ["accuracy", "balanced_accuracy", "specificity", "precision", "recall", "f1", "roc_auc", "pr_auc", "far", "miss_rate"]:
            mean_key = f"dev_{metric}_mean"
            std_key = f"dev_{metric}_std"
            if mean_key in df.columns and std_key in df.columns:
                mean_val = getattr(row, mean_key, None)
                std_val = getattr(row, std_key, None)
                if mean_val is not None and std_val is not None:
                    summary_lines.append(f"  {metric}: {mean_val:.4f} +/- {std_val:.4f}")
        summary_lines.append("")
        summary_lines.append("Final global test metrics:")
        for metric in ["accuracy", "balanced_accuracy", "specificity", "precision", "recall", "f1", "roc_auc", "pr_auc", "far", "miss_rate"]:
            mean_key = f"test_{metric}_mean"
            std_key = f"test_{metric}_std"
            if mean_key in df.columns and std_key in df.columns:
                mean_val = getattr(row, mean_key, None)
                std_val = getattr(row, std_key, None)
                if mean_val is not None and std_val is not None:
                    summary_lines.append(f"  {metric}: {mean_val:.4f} +/- {std_val:.4f}")
        summary_lines.append("")

    summary_path = root / "soft_labels_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
