from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gen_syntdata.smoke_experiments.run_fl_smoke_grid import run_one, summarize_grid


def select_top_three(summary_path: Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    score_columns = [column for column in ["test_roc_auc_mean", "test_pr_auc_mean"] if column in summary]
    if len(score_columns) != 2:
        raise ValueError(f"Missing smoke selection metrics in {summary_path}")
    summary["selection_score"] = summary[score_columns].mean(axis=1)
    return summary.sort_values(
        ["selection_score", "test_roc_auc_mean", "test_pr_auc_mean"], ascending=False
    ).head(3).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable full training for the smoke top 3.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("gen_syntdata/experiments_gans/full_data_smoke_results/summary_mean_std.csv"),
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("gen_syntdata/experiments_gans/full_data_datasets"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gen_syntdata/experiments_gans/full_data_full_results"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--num-rounds", type=int, default=25)
    parser.add_argument("--local-epochs", type=int, default=100)
    parser.add_argument("--rerun-complete", action="store_true")
    args = parser.parse_args()

    top_three = select_top_three(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for rank, row in top_three.iterrows():
        selected.append(
            {
                "rank": rank + 1,
                "model": row["model"],
                "generator": row["generator"],
                "mixture": row["mixture"],
                "smoke_selection_score": float(row["selection_score"]),
            }
        )
    manifest = {
        "selection_source": str(args.summary),
        "datasets_dir": str(args.datasets_dir),
        "output_dir": str(args.output_dir),
        "seeds": args.seeds,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "selected": selected,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for row in selected:
        windows_path = args.datasets_dir / row["generator"] / row["mixture"] / "mixed_windows.csv"
        if not windows_path.exists():
            raise FileNotFoundError(windows_path)
        for seed in args.seeds:
            run_one(
                model=row["model"],
                generator=row["generator"],
                ratio=row["mixture"],
                seed=seed,
                windows_path=windows_path,
                output_dir=args.output_dir,
                num_rounds=args.num_rounds,
                local_epochs=args.local_epochs,
                rerun_complete=args.rerun_complete,
            )

    summarize_grid(args.output_dir)
    print(f"Full top-3 summary saved to {args.output_dir / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
