from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FL_ROOT = REPO_ROOT / "fl"

for path in [str(FL_ROOT / "kd-experiments"), str(FL_ROOT / "kd-experiments-without-weights")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from config import EXPERIMENT_CATALOG, ExperimentConfig, config_to_dict
from data_utils import CLIENT_SPLIT_ASSIGNMENTS
from run_experiment_no_weights import run_experiment_no_weights

SELECTED_EXPERIMENT_ID = "exp13_final_keep_best_local"
DEFAULT_RESULTS_DIR = REPO_ROOT / "backend" / "results"
DEFAULT_WINDOWS_PATHS = [
    REPO_ROOT / "fl" / "data" / "preprocessing_results" / "simple" / "magnitude_features" / "windows.csv",
    REPO_ROOT / "fl" / "data" / "preprocessing_results" / "simple" / "magnitude_features" / "windows.csv",
    REPO_ROOT / "fl" / "data" / "preprocessing_results" / "magnitude_features" / "windows.csv",
]


def get_selected_experiment_config() -> ExperimentConfig:
    """Return the selected soft-label local-first experiment configuration."""
    return EXPERIMENT_CATALOG[SELECTED_EXPERIMENT_ID]


def build_split_summary() -> dict[str, Any]:
    """Summarize the client-disjoint split used by the alert simulation."""
    summary: dict[str, Any] = {"datasets": {}, "totals": {"train": 0, "dev": 0, "test": 0}}
    for dataset_name, split_map in CLIENT_SPLIT_ASSIGNMENTS.items():
        dataset_summary = {}
        for split_name in ("train", "dev", "test"):
            client_names = split_map.get(split_name, [])
            dataset_summary[split_name] = len(client_names)
            summary["totals"][split_name] += len(client_names)
        summary["datasets"][dataset_name] = dataset_summary
    return summary


def _find_existing_run_artifacts() -> Path | None:
    candidate_roots = [
        REPO_ROOT / "fl" / "kd-experiments-without-weights" / "soft labels" / SELECTED_EXPERIMENT_ID / "results",
        REPO_ROOT / "fl" / "kd-experiments-without-weights" / "soft labels" / "exp13_final_keep_best_local" / "results",
        REPO_ROOT / "fl" / "kd-experiments" / SELECTED_EXPERIMENT_ID / "results",
    ]
    for candidate_root in candidate_roots:
        if not candidate_root.exists():
            continue
        run_dirs = sorted([path for path in candidate_root.glob("run_*") if path.is_dir()], key=lambda path: path.name, reverse=True)
        if run_dirs:
            return run_dirs[0]
    return None


def run_selected_alert_simulation(
    results_dir: str | Path | None = None,
    *,
    num_rounds: int | None = None,
    local_epochs: int | None = None,
    random_seed: int | None = None,
    smoke_test: bool = False,
    windows_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run the full alert-system FL simulation for the selected soft-label strategy."""
    config = get_selected_experiment_config()
    if smoke_test:
        num_rounds = 2 if num_rounds is None else num_rounds
        local_epochs = 1 if local_epochs is None else local_epochs
        random_seed = 42 if random_seed is None else random_seed

    target_dir = Path(results_dir or DEFAULT_RESULTS_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    windows_candidate = Path(windows_path) if windows_path else None
    if windows_candidate is None:
        for candidate in DEFAULT_WINDOWS_PATHS:
            if candidate.exists():
                windows_candidate = candidate
                break

    run_dir: Path
    run_mode = "training"
    if windows_candidate is not None and windows_candidate.exists():
        run_dir = run_experiment_no_weights(
            experiment_id=config.experiment_id,
            results_dir=target_dir,
            mode="soft_labels",
            num_rounds_override=num_rounds,
            local_epochs_override=local_epochs,
            random_seed_override=random_seed,
        )
    else:
        source_run_dir = _find_existing_run_artifacts()
        if source_run_dir is None:
            raise FileNotFoundError(
                "No preprocessing windows.csv found and no prior experiment artifacts are available. "
                "Provide --windows-path or regenerate the preprocessing dataset first."
            )
        run_dir = target_dir / source_run_dir.name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        shutil.copytree(source_run_dir, run_dir)
        run_mode = "existing_artifacts"

    summary = {
        "selected_experiment_id": config.experiment_id,
        "title": config.title,
        "description": config.description,
        "communication_mode": "soft_labels",
        "communicates_model_weights": False,
        "split_summary": build_split_summary(),
        "smoke_test": smoke_test,
        "run_mode": run_mode,
        "run_dir": str(run_dir),
        "config": config_to_dict(config),
    }

    summary_path = target_dir / "selected_strategy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return run_dir, summary
