from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from gen_syntdata.build_mixed_datasets import RATIOS
from gen_syntdata.smoke_experiments.run_fl_smoke_grid import GENERATOR_NAMES


STRATEGIES = ["fedavg", "soft_labels", "class_prototype"]
EXPECTED_COMBINATIONS = len(GENERATOR_NAMES) * len(RATIOS) * len(STRATEGIES)


def full_training_command(rank: int, row: pd.Series, datasets_dir: Path) -> list[str]:
    strategy, generator, mixture = map(str, (row["model"], row["generator"], row["mixture"]))
    windows_path = datasets_dir / generator / mixture / "mixed_windows.csv"
    results_dir = Path("gen_syntdata/experiments_gans/full_data_full_results") / (
        f"top_{rank:02d}_{strategy}_{generator}_{mixture}"
    )
    if strategy == "fedavg":
        runner = "fl/fedavg-experiments/run_experiment.py"
        experiment = "exp_synthetic_mlp_unweighted"
        mode: list[str] = []
    else:
        runner = "fl/kd-experiments-without-weights/run_experiment_no_weights.py"
        experiment = "exp13_final_keep_best_local"
        mode = [f"    --mode {strategy} \\"]
    return [
        "for seed in {1..10}; do",
        f"  venv/bin/python {runner} \\",
        f"    --experiment {experiment} \\",
        *mode,
        f"    --results-dir {results_dir} \\",
        "    --num-rounds 25 \\",
        "    --local-epochs 100 \\",
        '    --random-seed "$seed" \\',
        f"    --windows-path {windows_path}",
        "done",
    ]


def write_report(output_dir: Path, datasets_dir: Path) -> None:
    summary_path = output_dir / "summary_mean_std.csv"
    if not summary_path.exists():
        return
    try:
        summary = pd.read_csv(summary_path)
    except pd.errors.EmptyDataError:
        return
    score_columns = [c for c in ["test_roc_auc_mean", "test_pr_auc_mean"] if c in summary]
    if summary.empty or not score_columns:
        return
    ranked = summary.copy()
    ranked["selection_score"] = ranked[score_columns].mean(axis=1)
    ranked = ranked.sort_values(
        ["selection_score", "test_roc_auc_mean", "test_pr_auc_mean"], ascending=False
    ).reset_index(drop=True)
    lines = [
        "Top 10 smoke configurations — synthetic falls and non-falls",
        "==========================================================",
        "",
        "Selection score = mean of test ROC-AUC and PR-AUC.",
        "Smoke setup = 1 round, 1 local epoch, seed 1.",
        "",
    ]
    for index, row in ranked.head(10).iterrows():
        lines.append(
            f"{index + 1:2d}. {row['model']} / {row['generator']} / {row['mixture']} / "
            f"score={row['selection_score']:.4f} / ROC-AUC={row['test_roc_auc_mean']:.4f} / "
            f"PR-AUC={row['test_pr_auc_mean']:.4f}"
        )
    lines += ["", "Full training commands for the top 3", "====================================", ""]
    for index, row in ranked.head(3).iterrows():
        lines += [
            f"Top {index + 1}: {row['model']} / {row['generator']} / {row['mixture']}",
            "-" * 80,
            *full_training_command(index + 1, row, datasets_dir),
            "",
        ]
    (output_dir / "best_partitions_by_model.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 60 two-class GAN smoke tests.")
    parser.add_argument("--datasets-dir", type=Path, default=Path("gen_syntdata/experiments_gans/full_data_datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("gen_syntdata/experiments_gans/full_data_smoke_results"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = EXPECTED_COMBINATIONS * len(args.seeds)
    print(f"Grid: 5 GANs x 4 repartitions x 3 strategies x {len(args.seeds)} seed(s) = {total} runs")
    command = [
        sys.executable, "-m", "gen_syntdata.smoke_experiments.run_fl_smoke_grid",
        "--datasets-dir", str(args.datasets_dir), "--output-dir", str(args.output_dir),
        "--generators", *GENERATOR_NAMES, "--mixtures", *RATIOS.keys(),
        "--models", *STRATEGIES, "--seeds", *(str(seed) for seed in args.seeds),
        "--num-rounds", str(args.num_rounds), "--local-epochs", str(args.local_epochs),
    ]
    if args.rerun_complete:
        command.append("--rerun-complete")
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)
    write_report(args.output_dir, args.datasets_dir)


if __name__ == "__main__":
    main()
