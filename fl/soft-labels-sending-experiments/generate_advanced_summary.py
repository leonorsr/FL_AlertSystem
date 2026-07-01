from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from advanced_strategy_configs import DEFAULT_ADVANCED_STRATEGY_ORDER
from run_strategy import DEFAULT_RESULTS_ROOT


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a summary for advanced soft-label sending strategies.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output", default="advanced_probability_strategy_summary.txt")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for strategy_name in DEFAULT_ADVANCED_STRATEGY_ORDER:
        strategy_dir = results_dir / strategy_name
        if not strategy_dir.exists():
            continue
        for summary_path in sorted(strategy_dir.glob("run_*/run_summary.json")):
            summary = _load_json(summary_path)
            test_metrics = summary.get("test_metrics", {})
            local_metrics = summary.get("local_metrics_summary", {})
            rows.append(
                {
                    "strategy": strategy_name,
                    "payload": summary.get("payload_name"),
                    "scope": summary.get("aggregation_scope"),
                    "scalars": summary.get("payload_scalar_count"),
                    "warmup_rounds": summary.get("warmup_rounds"),
                    "uncertain_only": summary.get("uncertain_only"),
                    "run_dir": str(summary_path.parent),
                    "global_pr_auc": test_metrics.get("pr_auc"),
                    "global_f1": test_metrics.get("f1"),
                    "local_pr_auc": local_metrics.get("pr_auc"),
                    "local_f1": local_metrics.get("f1"),
                }
            )

    df = pd.DataFrame(rows)
    lines = [
        "Advanced Soft-label Sending Strategies",
        "======================================",
        "",
    ]
    if df.empty:
        lines.append("No completed runs found.")
    else:
        summary_df = (
            df.groupby(["strategy", "payload", "scope", "scalars", "warmup_rounds", "uncertain_only"], as_index=False)
            .agg(
                runs=("run_dir", "count"),
                global_pr_auc_mean=("global_pr_auc", "mean"),
                global_pr_auc_std=("global_pr_auc", "std"),
                global_f1_mean=("global_f1", "mean"),
                global_f1_std=("global_f1", "std"),
                local_pr_auc_mean=("local_pr_auc", "mean"),
                local_pr_auc_std=("local_pr_auc", "std"),
                local_f1_mean=("local_f1", "mean"),
                local_f1_std=("local_f1", "std"),
            )
            .sort_values(["local_pr_auc_mean", "local_f1_mean", "global_pr_auc_mean"], ascending=[False, False, False])
        )
        lines.append("Ranking by local PR-AUC:")
        for _, row in summary_df.iterrows():
            lines.append(
                f"- {row['strategy']} ({row['payload']}, scope={row['scope']}, runs={int(row['runs'])}): "
                f"local PR-AUC={row['local_pr_auc_mean']:.4f}"
                f"{' +/- ' + format(row['local_pr_auc_std'], '.4f') if pd.notna(row['local_pr_auc_std']) else ''}, "
                f"local F1={row['local_f1_mean']:.4f}, "
                f"global PR-AUC={row['global_pr_auc_mean']:.4f}, global F1={row['global_f1_mean']:.4f}"
            )
        lines.extend(["", "Per-run results:"])
        for _, row in df.sort_values(["strategy", "run_dir"]).iterrows():
            lines.append(
                f"- {row['strategy']} | local PR-AUC={row['local_pr_auc']:.4f} | local F1={row['local_f1']:.4f} | "
                f"global PR-AUC={row['global_pr_auc']:.4f} | global F1={row['global_f1']:.4f} | {row['run_dir']}"
            )

    output_path = results_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summary to: {output_path}")


if __name__ == "__main__":
    main()
