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
    learning_rate: float
    dropout: float


CANDIDATES = [
    Candidate("c1_current_baseline", "Current class-prototype configuration.", 8e-4, 0.30),
    Candidate("c2_lower_dropout", "Reduce dropout to favor local fitting.", 8e-4, 0.20),
    Candidate("c3_lower_lr_dropout", "Use a lower learning rate with reduced dropout.", 5e-4, 0.20),
    Candidate("c4_higher_lr_dropout", "Use a slightly higher learning rate with reduced dropout.", 1e-3, 0.20),
]


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


def _fmt(value: float) -> str:
    return "n/a" if value == float("-inf") else f"{value:.4f}"


def _write_report(root: Path, records: list[dict[str, Any]], args) -> Path:
    report_path = root / "class_prototype_fine_tuning_results.txt"
    lines = [
        "Class Prototype Fine Tuning Results",
        "===================================",
        "",
        "Objective",
        "---------",
        "Select a local-first class-prototype configuration without exchanging model weights.",
        "Ranking prioritizes local PR-AUC, followed by global test PR-AUC, local F1 and global test F1.",
        "",
        "Search setup",
        "------------",
        f"- experiments: {', '.join(args.experiments)}",
        f"- runs per candidate: {args.runs}",
        f"- global rounds: {args.num_rounds}",
        f"- local epochs: {args.local_epochs}",
        "- matched seeds across candidates: yes",
        "",
        "Candidates",
        "----------",
    ]
    for candidate in CANDIDATES:
        lines.append(
            f"- {candidate.candidate_id}: {candidate.description} "
            f"learning_rate={candidate.learning_rate:.6f}, dropout={candidate.dropout:.2f}"
        )
    lines.extend(["", "Selected configurations", "-----------------------"])
    for experiment_id in args.experiments:
        experiment_records = [record for record in records if record["experiment_id"] == experiment_id]
        best = max(experiment_records, key=_score)
        lines.extend(["", experiment_id, "~" * len(experiment_id)])
        lines.append(f"Selected candidate: {best['candidate'].candidate_id}")
        lines.append("Candidate ranking:")
        for rank, record in enumerate(sorted(experiment_records, key=_score, reverse=True), start=1):
            summary = record["summary"]
            candidate = record["candidate"]
            lines.append(
                f"{rank}. {candidate.candidate_id}: "
                f"local PR-AUC={_fmt(_metric(summary, 'local_metrics_summary', 'pr_auc'))}, "
                f"local F1={_fmt(_metric(summary, 'local_metrics_summary', 'f1'))}, "
                f"global PR-AUC={_fmt(_metric(summary, 'test_metrics', 'pr_auc'))}, "
                f"global F1={_fmt(_metric(summary, 'test_metrics', 'f1'))}"
            )
    lines.extend(["", "Use the highest-ranked experiment and candidate for expfinal.", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight class-prototype local-first fine tuning.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=5200)
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENT_IDS, choices=EXPERIMENT_IDS)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    records = []
    for experiment_index, experiment_id in enumerate(args.experiments):
        base_config = EXPERIMENT_CATALOG[experiment_id]
        for candidate in CANDIDATES:
            tuned_config = replace(
                base_config,
                model=replace(base_config.model, learning_rate=candidate.learning_rate, dropout=candidate.dropout),
            )
            for run_index in range(args.runs):
                seed = args.base_seed + experiment_index * 100 + run_index
                results_dir = args.root / experiment_id / candidate.candidate_id / "results"
                print(f"Running {experiment_id} / {candidate.candidate_id} / seed {seed}")
                run_dir = run_experiment_no_weights(
                    experiment_id=experiment_id,
                    results_dir=results_dir,
                    mode="class_prototype",
                    num_rounds_override=args.num_rounds,
                    local_epochs_override=args.local_epochs,
                    random_seed_override=seed,
                    config_override=tuned_config,
                )
                summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
                records.append({"experiment_id": experiment_id, "candidate": candidate, "seed": seed, "run_dir": str(run_dir), "summary": summary})
                print(f"  local PR-AUC={_fmt(_metric(summary, 'local_metrics_summary', 'pr_auc'))}, global PR-AUC={_fmt(_metric(summary, 'test_metrics', 'pr_auc'))}")

    manifest = [{**record, "candidate": asdict(record["candidate"])} for record in records]
    (args.root / "search_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote fine-tuning report: {_write_report(args.root, records, args)}")


if __name__ == "__main__":
    main()
