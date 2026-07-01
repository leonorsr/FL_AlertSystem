from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any


NO_WEIGHTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = NO_WEIGHTS_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "kd-experiments"))
sys.path.insert(0, str(NO_WEIGHTS_ROOT))

from config import EXPERIMENT_CATALOG
from run_experiment_no_weights import run_experiment_no_weights


EXPERIMENT_IDS = [
    "exp11_baseline_final",
    "exp12_final_unweighted",
    "exp13_final_keep_best_local",
    "exp14_final_clustered",
]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    description: str
    distillation_alpha: float
    distillation_temperature: float
    learning_rate: float
    dropout: float


CANDIDATES = [
    Candidate(
        candidate_id="c1_current_baseline",
        description="Current soft-label configuration used as the reference.",
        distillation_alpha=0.50,
        distillation_temperature=2.0,
        learning_rate=8e-4,
        dropout=0.30,
    ),
    Candidate(
        candidate_id="c2_local_hard_label_focus",
        description="Give more weight to local hard labels while preserving the global soft-label signal.",
        distillation_alpha=0.75,
        distillation_temperature=2.0,
        learning_rate=8e-4,
        dropout=0.30,
    ),
    Candidate(
        candidate_id="c3_local_lower_dropout",
        description="Prioritize local fitting with stronger hard-label weight and lower dropout.",
        distillation_alpha=0.75,
        distillation_temperature=2.0,
        learning_rate=8e-4,
        dropout=0.20,
    ),
    Candidate(
        candidate_id="c4_balanced_lower_lr",
        description="Use a moderate local bias with lower learning rate and a softer temperature.",
        distillation_alpha=0.65,
        distillation_temperature=1.5,
        learning_rate=5e-4,
        dropout=0.20,
    ),
]


def _read_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))


def _metric(summary: dict[str, Any], section: str, metric: str) -> float:
    return float(summary.get(section, {}).get(metric, float("-inf")))


def _score(record: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = record["summary"]
    return (
        _metric(summary, "local_metrics_summary", "pr_auc"),
        _metric(summary, "test_metrics", "pr_auc"),
        _metric(summary, "local_metrics_summary", "f1"),
        _metric(summary, "test_metrics", "f1"),
    )


def _format_metric(value: float) -> str:
    return "n/a" if value == float("-inf") else f"{value:.4f}"


def _write_report(output_root: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    report_path = output_root / "soft_labels_fine_tuning_results.txt"
    lines = [
        "Soft Labels Fine Tuning Results",
        "===============================",
        "",
        "Objective",
        "---------",
        "Improve local training without exchanging model weights. Candidate selection prioritizes",
        "local PR-AUC, then global test PR-AUC, local F1 and global test F1.",
        "",
        "Search setup",
        "------------",
        f"- Experiments: {', '.join(args.experiments)}",
        f"- Search runs per candidate: {args.runs}",
        f"- Search global rounds: {args.num_rounds}",
        f"- Search local epochs: {args.local_epochs}",
        "- Communication mode: soft_labels",
        "- Communicates model weights: no",
        "",
        "Candidates",
        "----------",
    ]
    for candidate in CANDIDATES:
        lines.extend(
            [
                f"- {candidate.candidate_id}: {candidate.description}",
                f"  alpha={candidate.distillation_alpha:.2f}, temperature={candidate.distillation_temperature:.2f}, "
                f"learning_rate={candidate.learning_rate:.6f}, dropout={candidate.dropout:.2f}",
            ]
        )

    lines.extend(["", "Selected configurations", "-----------------------"])
    for experiment_id in args.experiments:
        experiment_records = [record for record in records if record["experiment_id"] == experiment_id]
        best = max(experiment_records, key=_score)
        candidate = best["candidate"]
        summary = best["summary"]
        lines.extend(
            [
                "",
                experiment_id,
                "~" * len(experiment_id),
                f"Selected candidate: {candidate.candidate_id}",
                f"Description: {candidate.description}",
                f"Local PR-AUC: {_format_metric(_metric(summary, 'local_metrics_summary', 'pr_auc'))}",
                f"Local F1: {_format_metric(_metric(summary, 'local_metrics_summary', 'f1'))}",
                f"Global test PR-AUC: {_format_metric(_metric(summary, 'test_metrics', 'pr_auc'))}",
                f"Global test F1: {_format_metric(_metric(summary, 'test_metrics', 'f1'))}",
                "Recommended full-run configuration:",
                f"- global rounds: {EXPERIMENT_CATALOG[experiment_id].num_rounds}",
                f"- local epochs: {EXPERIMENT_CATALOG[experiment_id].local_epochs}",
                f"- distillation alpha: {candidate.distillation_alpha:.2f}",
                f"- distillation temperature: {candidate.distillation_temperature:.2f}",
                f"- learning rate: {candidate.learning_rate:.6f}",
                f"- dropout: {candidate.dropout:.2f}",
                f"- weighted aggregation: {int(EXPERIMENT_CATALOG[experiment_id].weighted_aggregation)}",
                f"- local model selection: {int(EXPERIMENT_CATALOG[experiment_id].local_model_selection)}",
                f"- clustered aggregation: {int(EXPERIMENT_CATALOG[experiment_id].clustered_aggregation)}",
                "",
                "Candidate ranking:",
            ]
        )
        for rank, record in enumerate(sorted(experiment_records, key=_score, reverse=True), start=1):
            candidate = record["candidate"]
            summary = record["summary"]
            lines.append(
                f"{rank}. {candidate.candidate_id}: "
                f"local PR-AUC={_format_metric(_metric(summary, 'local_metrics_summary', 'pr_auc'))}, "
                f"local F1={_format_metric(_metric(summary, 'local_metrics_summary', 'f1'))}, "
                f"global PR-AUC={_format_metric(_metric(summary, 'test_metrics', 'pr_auc'))}, "
                f"global F1={_format_metric(_metric(summary, 'test_metrics', 'f1'))}"
            )

    lines.extend(
        [
            "",
            "Interpretation note",
            "-------------------",
            "These are lightweight search runs. The recommended settings should be validated with",
            "full-length repeated runs before using them as final article results.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight local-first soft-label fine-tuning search.")
    parser.add_argument("--runs", type=int, default=1, help="Runs per candidate and experiment.")
    parser.add_argument("--num-rounds", type=int, default=3, help="Global rounds used during the lightweight search.")
    parser.add_argument("--local-epochs", type=int, default=20, help="Local epochs used during the lightweight search.")
    parser.add_argument("--base-seed", type=int, default=4200)
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENT_IDS, choices=EXPERIMENT_IDS)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for experiment_index, experiment_id in enumerate(args.experiments):
        base_config = EXPERIMENT_CATALOG[experiment_id]
        for candidate_index, candidate in enumerate(CANDIDATES):
            model = replace(
                base_config.model,
                learning_rate=candidate.learning_rate,
                dropout=candidate.dropout,
            )
            tuned_config = replace(
                base_config,
                model=model,
                distillation_alpha=candidate.distillation_alpha,
                distillation_temperature=candidate.distillation_temperature,
            )
            for run_index in range(args.runs):
                # Use matched seeds across candidates so configuration changes are
                # compared under the same random conditions.
                seed = args.base_seed + experiment_index * 100 + run_index
                results_dir = args.root / experiment_id / candidate.candidate_id / "results"
                print(f"Running {experiment_id} / {candidate.candidate_id} / seed {seed}")
                run_dir = run_experiment_no_weights(
                    experiment_id=experiment_id,
                    results_dir=results_dir,
                    mode="soft_labels",
                    num_rounds_override=args.num_rounds,
                    local_epochs_override=args.local_epochs,
                    random_seed_override=seed,
                    config_override=tuned_config,
                )
                summary = _read_summary(run_dir)
                record = {
                    "experiment_id": experiment_id,
                    "candidate": candidate,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "summary": summary,
                }
                records.append(record)
                print(
                    "  local PR-AUC="
                    f"{_format_metric(_metric(summary, 'local_metrics_summary', 'pr_auc'))}, "
                    "global test PR-AUC="
                    f"{_format_metric(_metric(summary, 'test_metrics', 'pr_auc'))}"
                )

    manifest = [
        {
            **{key: value for key, value in record.items() if key not in {"candidate"}},
            "candidate": asdict(record["candidate"]),
        }
        for record in records
    ]
    (args.root / "search_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path = _write_report(args.root, records, args)
    print(f"Wrote fine-tuning report: {report_path}")


if __name__ == "__main__":
    main()
