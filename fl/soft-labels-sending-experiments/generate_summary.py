from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from payload_configs import DEFAULT_PAYLOAD_ORDER
from run_experiment import DEFAULT_RESULTS_ROOT


def _load_run_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a summary for probability-payload experiments.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output", default="probability_payload_summary.txt")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for payload_name in DEFAULT_PAYLOAD_ORDER:
        payload_dir = results_dir / payload_name
        if not payload_dir.exists():
            continue
        for summary_path in sorted(payload_dir.glob("run_*/run_summary.json")):
            summary = _load_run_summary(summary_path)
            test_metrics = summary.get("test_metrics", {})
            local_metrics = summary.get("local_metrics_summary", {})
            rows.append(
                {
                    "payload": payload_name,
                    "scalars": summary.get("payload_scalar_count"),
                    "run_dir": str(summary_path.parent),
                    "global_pr_auc": test_metrics.get("pr_auc"),
                    "global_f1": test_metrics.get("f1"),
                    "local_pr_auc": local_metrics.get("pr_auc"),
                    "local_f1": local_metrics.get("f1"),
                }
            )

    df = pd.DataFrame(rows)
    lines = [
        "Soft-label Probability Payload Experiments",
        "==========================================",
        "",
    ]
    if df.empty:
        lines.append("No completed runs found.")
    else:
        summary_df = (
            df.groupby(["payload", "scalars"], as_index=False)
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
            .sort_values(["local_pr_auc_mean", "global_pr_auc_mean"], ascending=[False, False])
        )
        lines.append("Ranking by local PR-AUC:")
        for _, row in summary_df.iterrows():
            lines.append(
                f"- {row['payload']} ({int(row['scalars'])} scalars, runs={int(row['runs'])}): "
                f"local PR-AUC={row['local_pr_auc_mean']:.4f}, local F1={row['local_f1_mean']:.4f}, "
                f"global PR-AUC={row['global_pr_auc_mean']:.4f}, global F1={row['global_f1_mean']:.4f}"
            )
        lines.extend(["", "Per-run results:"])
        for _, row in df.sort_values(["payload", "run_dir"]).iterrows():
            lines.append(
                f"- {row['payload']} | local PR-AUC={row['local_pr_auc']:.4f} | local F1={row['local_f1']:.4f} | "
                f"global PR-AUC={row['global_pr_auc']:.4f} | global F1={row['global_f1']:.4f} | {row['run_dir']}"
            )

    output_path = results_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summary to: {output_path}")


if __name__ == "__main__":
    main()
