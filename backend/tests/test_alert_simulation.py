from __future__ import annotations

from backend.alert_simulation import build_split_summary, get_selected_experiment_config


def test_selected_experiment_is_local_first_soft_labels() -> None:
    config = get_selected_experiment_config()
    assert config.experiment_id == "exp13_final_keep_best_local"
    assert config.local_model_selection is True
    assert config.local_epochs == 100


def test_split_summary_matches_requested_counts() -> None:
    summary = build_split_summary()
    assert summary["datasets"]["KFall"] == {"train": 26, "dev": 3, "test": 3}
    assert summary["datasets"]["SisFall"] == {"train": 30, "dev": 4, "test": 4}
    assert summary["datasets"]["UpFall"] == {"train": 13, "dev": 2, "test": 2}
    assert summary["totals"] == {"train": 69, "dev": 9, "test": 9}
