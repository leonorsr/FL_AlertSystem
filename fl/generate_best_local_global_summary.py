from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "kd-experiments"))

from config import EXPERIMENT_CATALOG  # noqa: E402


STRATEGIES = [
    {
        "name": "FedAvg",
        "paths": [ROOT / "fedavg-experiments-local-metrics", ROOT / "fedavg-experiments"],
        "communication_mode": None,
        "description": "comunica pesos; sem KD",
    },
    {
        "name": "KD com pesos",
        "paths": [ROOT / "kd-experiments-local-metrics", ROOT / "kd-experiments"],
        "communication_mode": None,
        "description": "comunica pesos; KD no treino local",
    },
    {
        "name": "Soft labels",
        "path": ROOT / "kd-experiments-without-weights" / "soft labels",
        "communication_mode": "soft_labels",
        "description": "nao comunica pesos; comunica soft labels",
    },
    {
        "name": "Class prototype",
        "path": ROOT / "kd-experiments-without-weights" / "class prototype",
        "communication_mode": "class_prototype",
        "description": "nao comunica pesos; comunica prototipos por classe",
    },
    {
        "name": "Hidden states",
        "path": ROOT / "kd-experiments-without-weights" / "hidden states",
        "communication_mode": "hidden_states",
        "description": "nao comunica pesos; comunica hidden states",
    },
]

METRICS = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if not math.isnan(result) else None


def is_full_length_run(experiment_id: str, config: dict[str, Any], communication_mode: str | None) -> bool:
    expected = EXPERIMENT_CATALOG.get(experiment_id)
    if expected is not None:
        if int(config.get("num_rounds", -1)) != int(expected.num_rounds):
            return False
        if int(config.get("local_epochs", -1)) != int(expected.local_epochs):
            return False
    if communication_mode is not None and config.get("communication_mode") != communication_mode:
        return False
    return True


def flatten_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    row = {}
    for metric in METRICS:
        value = safe_float(metrics.get(metric))
        if value is not None:
            row[f"{prefix}_{metric}"] = value
    return row


def collect_strategy_rows(strategy: dict[str, Any]) -> pd.DataFrame:
    candidate_paths = strategy.get("paths", [strategy.get("path")])
    for candidate_path in candidate_paths:
        rows = []
        root = Path(candidate_path)
        if not root.exists():
            continue

        for experiment_id in EXPERIMENT_CATALOG:
            results_dir = root / experiment_id / "results"
            if not results_dir.exists():
                continue
            for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
                config = read_json(run_dir / "config.json")
                if not is_full_length_run(experiment_id, config, strategy["communication_mode"]):
                    continue
                summary = read_json(run_dir / "run_summary.json")
                if not summary:
                    continue
                row: dict[str, Any] = {
                    "strategy": strategy["name"],
                    "experiment_id": experiment_id,
                    "run": run_dir.name,
                    "random_seed": config.get("random_seed"),
                    "last_write_time": run_dir.stat().st_mtime,
                    "source_root": str(root.relative_to(ROOT)),
                }
                row.update(flatten_metrics(summary.get("test_metrics", {}), "global"))
                row.update(flatten_metrics(summary.get("local_metrics_summary", {}), "local"))
                row.update(flatten_metrics(summary.get("fine_tune_summary", {}), "fine_tune"))
                rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            if {"experiment_id", "random_seed", "last_write_time"}.issubset(df.columns):
                df = (
                    df.sort_values(["experiment_id", "random_seed", "last_write_time"])
                    .drop_duplicates(subset=["experiment_id", "random_seed"], keep="last")
                    .reset_index(drop=True)
                )
            return df
    return pd.DataFrame()


def aggregate_by_experiment(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    numeric_cols = rows.select_dtypes(include=["number"]).columns.tolist()
    grouped_rows = []
    for experiment_id, group in rows.groupby("experiment_id", sort=True):
        row: dict[str, Any] = {"experiment_id": experiment_id, "n_runs": int(len(group))}
        for col in numeric_cols:
            values = group[col].dropna()
            if values.empty:
                continue
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows)


def select_best(agg: pd.DataFrame, prefix: str) -> pd.Series | None:
    pr_auc = f"{prefix}_pr_auc_mean"
    f1 = f"{prefix}_f1_mean"
    if agg.empty or pr_auc not in agg.columns:
        return None
    candidates = agg.dropna(subset=[pr_auc]).copy()
    if candidates.empty:
        return None
    sort_cols = [pr_auc]
    ascending = [False]
    if f1 in candidates.columns:
        sort_cols.append(f1)
        ascending.append(False)
    return candidates.sort_values(sort_cols, ascending=ascending).iloc[0]


def metric_text(row: pd.Series | None, prefix: str) -> str:
    if row is None:
        return "n/a"
    pr_auc = row.get(f"{prefix}_pr_auc_mean")
    pr_auc_std = row.get(f"{prefix}_pr_auc_std")
    f1 = row.get(f"{prefix}_f1_mean")
    f1_std = row.get(f"{prefix}_f1_std")
    parts = [str(row["experiment_id"]), f"runs={int(row['n_runs'])}"]
    if pr_auc is not None and not pd.isna(pr_auc):
        parts.append(f"PR-AUC={pr_auc:.4f} +/- {pr_auc_std:.4f}")
    if f1 is not None and not pd.isna(f1):
        parts.append(f"F1={f1:.4f} +/- {f1_std:.4f}")
    return ", ".join(parts)


def main() -> None:
    lines = [
        "Best Local vs Global Experiments by Strategy",
        "============================================",
        "",
        "Selection rule: best experiment is selected by mean PR-AUC; F1 is used as a tie-breaker.",
        "Global model uses final global test_metrics.",
        "Local training uses local_metrics_summary when available. fine_tune_summary is reported separately when available.",
        "",
    ]

    all_strategy_summaries = []
    for strategy in STRATEGIES:
        rows = collect_strategy_rows(strategy)
        agg = aggregate_by_experiment(rows)
        if agg.empty:
            lines.extend([strategy["name"], "-" * len(strategy["name"]), f"Description: {strategy['description']}", "No full-length runs found.", ""])
            continue

        best_global = select_best(agg, "global")
        best_local = select_best(agg, "local")
        best_fine_tune = select_best(agg, "fine_tune")

        lines.extend(
            [
                strategy["name"],
                "-" * len(strategy["name"]),
                f"Description: {strategy['description']}",
                f"Experiments with full-length runs: {len(agg)}",
                f"Best global model: {metric_text(best_global, 'global')}",
                f"Best local training: {metric_text(best_local, 'local')}",
                f"Best fine-tuned test-client model: {metric_text(best_fine_tune, 'fine_tune')}",
                "",
                "Per-experiment overview:",
            ]
        )

        display_cols = ["experiment_id", "n_runs", "global_pr_auc_mean", "global_f1_mean", "local_pr_auc_mean", "local_f1_mean"]
        for row in agg.sort_values("experiment_id").itertuples(index=False):
            row_dict = row._asdict()
            values = []
            for col in display_cols:
                value = row_dict.get(col)
                if value is None or pd.isna(value):
                    values.append("n/a")
                elif isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append(
                f"  {values[0]} | runs={values[1]} | global PR-AUC={values[2]} | "
                f"global F1={values[3]} | local PR-AUC={values[4]} | local F1={values[5]}"
            )
        lines.append("")

        all_strategy_summaries.append(
            {
                "strategy": strategy["name"],
                "best_global": metric_text(best_global, "global"),
                "best_local": metric_text(best_local, "local"),
                "best_fine_tune": metric_text(best_fine_tune, "fine_tune"),
            }
        )

    lines.extend(["Compact Summary", "---------------"])
    for item in all_strategy_summaries:
        lines.append(f"{item['strategy']}: global -> {item['best_global']}; local -> {item['best_local']}; fine-tune -> {item['best_fine_tune']}")

    output = ROOT / "best_local_vs_global_by_strategy.txt"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
