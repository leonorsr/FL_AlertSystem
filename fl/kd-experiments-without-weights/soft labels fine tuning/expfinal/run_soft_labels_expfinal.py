from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


EXPFINAL_ROOT = Path(__file__).resolve().parent
NO_WEIGHTS_ROOT = EXPFINAL_ROOT.parents[1]
REPO_ROOT = NO_WEIGHTS_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "kd-experiments"))
sys.path.insert(0, str(NO_WEIGHTS_ROOT))

from config import EXPERIMENT_CATALOG
from run_experiment_no_weights import run_experiment_no_weights


BASE_EXPERIMENT_ID = "exp11_baseline_final"


def _format_metrics(title: str, metrics: dict[str, list[float]]) -> list[str]:
    lines = [title, "-" * len(title)]
    if not metrics:
        return [*lines, "n/a", ""]
    for metric_name, values in metrics.items():
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        lines.append(f"- {metric_name}: {mean:.4f} +/- {std:.4f}")
    lines.append("")
    return lines


def _completed_runs(results_dir: Path) -> list[Path]:
    completed = []
    for path in sorted(results_dir.glob("run_*")):
        if (path / "run_summary.json").exists() and (path / "config.json").exists():
            completed.append(path)
    return completed


def _aggregate(run_dirs: list[Path], section: str) -> dict[str, list[float]]:
    aggregated: dict[str, list[float]] = {}
    for run_dir in run_dirs:
        summary: dict[str, Any] = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        for metric_name, metric_value in summary.get(section, {}).items():
            if isinstance(metric_value, (int, float)):
                aggregated.setdefault(metric_name, []).append(float(metric_value))
    return aggregated


def _write_report(run_dirs: list[Path], config) -> Path:
    latest_summary = json.loads((run_dirs[-1] / "run_summary.json").read_text(encoding="utf-8"))
    report_path = EXPFINAL_ROOT / "expfinal_results.txt"
    lines = [
        "Soft Labels Final Local-First Experiment",
        "========================================",
        "",
        "Selection rationale",
        "-------------------",
        "This final experiment uses the best local-first configuration found during the",
        "lightweight soft-label fine-tuning search. The selected candidate was",
        "c2_local_hard_label_focus from exp11_baseline_final.",
        "",
        "Configuration",
        "-------------",
        f"- base experiment: {BASE_EXPERIMENT_ID}",
        "- communication mode: soft_labels",
        "- communicates model weights: no",
        f"- runs aggregated: {len(run_dirs)}",
        f"- available training clients: {latest_summary.get('num_train_clients', 'n/a')}",
        f"- client participation fraction per round: {config.fraction_fit:.2f}",
        f"- global rounds: {config.num_rounds}",
        f"- local epochs: {config.local_epochs}",
        f"- distillation alpha: {config.distillation_alpha:.2f}",
        f"- distillation temperature: {config.distillation_temperature:.2f}",
        f"- learning rate: {config.model.learning_rate:.6f}",
        f"- dropout: {config.model.dropout:.2f}",
        f"- weighted aggregation: {int(config.weighted_aggregation)}",
        f"- local model selection: {int(config.local_model_selection)}",
        f"- clustered aggregation: {int(config.clustered_aggregation)}",
        f"- base random seed: {config.random_seed}",
        f"- results directory: {EXPFINAL_ROOT / 'results'}",
        "",
    ]
    lines.extend(_format_metrics("Final local metrics", _aggregate(run_dirs, "local_metrics_summary")))
    lines.extend(_format_metrics("Final global dev metrics", _aggregate(run_dirs, "dev_metrics")))
    lines.extend(_format_metrics("Final global test metrics", _aggregate(run_dirs, "test_metrics")))
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selected local-first soft-label final experiment.")
    parser.add_argument("--random-seed", type=int, default=4600)
    parser.add_argument("--num-rounds", type=int, default=25)
    parser.add_argument("--local-epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base_config = EXPERIMENT_CATALOG[BASE_EXPERIMENT_ID]
    model = replace(base_config.model, learning_rate=8e-4, dropout=0.30)
    final_config = replace(
        base_config,
        model=model,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        random_seed=args.random_seed,
        fraction_fit=1.0,
        distillation_alpha=0.75,
        distillation_temperature=2.0,
    )
    results_dir = EXPFINAL_ROOT / "results"
    existing_runs = _completed_runs(results_dir) if args.resume else []
    for run_index in range(len(existing_runs), args.runs):
        run_config = replace(final_config, random_seed=args.random_seed + run_index)
        run_dir = run_experiment_no_weights(
            experiment_id=BASE_EXPERIMENT_ID,
            results_dir=results_dir,
            mode="soft_labels",
            config_override=run_config,
        )
        print(f"Saved final run {run_index + 1}/{args.runs} artifacts to: {run_dir}")
    completed_runs = _completed_runs(results_dir)
    report_path = _write_report(completed_runs, final_config)
    print(f"Wrote final report: {report_path}")


if __name__ == "__main__":
    main()
