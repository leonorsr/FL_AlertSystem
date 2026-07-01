from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline_utils import (
    DEFAULT_RESULTS_DIR,
    SplitConfig,
    aggregate_metrics,
    attach_splits,
    build_client_fold_reference,
    build_fold_split_reference,
    build_matrices,
    build_split_summary,
    build_trial_table,
    dataclass_to_dict,
    ensure_directory,
    evaluate_predictions,
    get_feature_columns,
    load_baseline_windows,
    split_trials_kfold_by_client,
    write_json,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


FINAL_SORT_COLUMNS = ["pr_auc", "miss_rate", "far", "balanced_accuracy", "f1"]
FINAL_SORT_ASCENDING = [False, True, True, False, False]


class TabularMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            if use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(torch.nn.Linear(current_dim, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


@dataclass(frozen=True)
class NeuralConfig:
    name: str
    hidden_layers: list[int]
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    use_batchnorm: bool = False


NEURAL_CONFIGS = [
    NeuralConfig("mlp_small", [64, 32], 0.20, 1e-3, 1e-4, 256, 35, 6, False),
    NeuralConfig("mlp_medium", [128, 64, 32], 0.30, 1e-3, 1e-4, 256, 40, 7, False),
    NeuralConfig("mlp_wide", [256, 128, 64], 0.30, 8e-4, 1e-4, 256, 45, 8, False),
    NeuralConfig("mlp_regularized", [128, 64], 0.40, 8e-4, 5e-4, 256, 45, 8, True),
]


def build_classical_candidates(seed: int) -> list[tuple[str, str, Pipeline]]:
    candidates: list[tuple[str, str, Pipeline]] = []

    for c_value in [0.01, 0.1, 1.0, 10.0]:
        candidates.append(
            (
                "logistic_regression",
                f"logreg_C{c_value}",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=c_value,
                                max_iter=1000,
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )

    for neighbors in [5, 11, 21]:
        candidates.append(
            (
                "knn",
                f"knn_k{neighbors}",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", KNeighborsClassifier(n_neighbors=neighbors, weights="distance")),
                    ]
                ),
            )
        )

    if XGBClassifier is not None:
        for name, n_estimators, max_depth, learning_rate in [
            ("xgb_small", 200, 4, 0.08),
            ("xgb_medium", 300, 6, 0.05),
            ("xgb_deep", 400, 8, 0.03),
        ]:
            candidates.append(
                (
                    "xgboost",
                    name,
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "model",
                                XGBClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate,
                                    subsample=0.9,
                                    colsample_bytree=0.9,
                                    objective="binary:logistic",
                                    eval_metric="logloss",
                                    random_state=seed,
                                    n_jobs=4,
                                ),
                            ),
                        ]
                    ),
                )
            )

    return candidates


def train_and_predict_classical(model: Pipeline, matrices: dict[str, tuple[pd.DataFrame, pd.Series]]) -> dict[str, dict[str, np.ndarray]]:
    x_train, y_train = matrices["train"]
    x_dev, _ = matrices["dev"]
    x_test, _ = matrices["test"]
    model.fit(x_train, y_train)

    if hasattr(model, "predict_proba"):
        dev_scores = model.predict_proba(x_dev)[:, 1]
        test_scores = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        dev_scores = model.decision_function(x_dev)
        test_scores = model.decision_function(x_test)
    else:
        dev_scores = model.predict(x_dev).astype(float)
        test_scores = model.predict(x_test).astype(float)

    return {
        "dev": {"predictions": model.predict(x_dev).astype(int), "scores": np.asarray(dev_scores, dtype=float)},
        "test": {"predictions": model.predict(x_test).astype(int), "scores": np.asarray(test_scores, dtype=float)},
    }


def train_neural_network(
    config: NeuralConfig,
    matrices: dict[str, tuple[pd.DataFrame, pd.Series]],
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train_df, y_train_series = matrices["train"]
    x_dev_df, y_dev_series = matrices["dev"]
    x_test_df, _ = matrices["test"]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train_np = scaler.fit_transform(imputer.fit_transform(x_train_df)).astype(np.float32)
    x_dev_np = scaler.transform(imputer.transform(x_dev_df)).astype(np.float32)
    x_test_np = scaler.transform(imputer.transform(x_test_df)).astype(np.float32)

    y_train_np = y_train_series.to_numpy(dtype=np.float32)
    y_dev_np = y_dev_series.to_numpy(dtype=np.float32)

    model = TabularMLP(
        input_dim=x_train_np.shape[1],
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
        use_batchnorm=config.use_batchnorm,
    )

    positives = max(float(y_train_np.sum()), 1.0)
    negatives = max(float(len(y_train_np) - y_train_np.sum()), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    x_dev_tensor = torch.tensor(x_dev_np, dtype=torch.float32)
    y_dev_tensor = torch.tensor(y_dev_np, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32)

    best_state = None
    best_dev_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(config.epochs):
        model.train()
        permutation = torch.randperm(x_train_tensor.size(0))
        for start in range(0, x_train_tensor.size(0), config.batch_size):
            batch_indices = permutation[start : start + config.batch_size]
            batch_x = x_train_tensor[batch_indices]
            batch_y = y_train_tensor[batch_indices]
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            dev_logits = model(x_dev_tensor)
            dev_loss = float(criterion(dev_logits, y_dev_tensor).item())

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        dev_probs = torch.sigmoid(model(x_dev_tensor)).cpu().numpy()
        test_probs = torch.sigmoid(model(x_test_tensor)).cpu().numpy()

    return {
        "dev": {"predictions": (dev_probs >= 0.5).astype(int), "scores": dev_probs.astype(float)},
        "test": {"predictions": (test_probs >= 0.5).astype(int), "scores": test_probs.astype(float)},
    }


def build_fold_scenarios(
    windows_df: pd.DataFrame,
    trial_table: pd.DataFrame,
    feature_columns: list[str],
    split_config: SplitConfig,
) -> list[dict]:
    split_df = split_trials_kfold_by_client(trial_table, split_config)
    labeled_df = attach_splits(windows_df, split_df)
    scenarios: list[dict] = []

    for fold_index in range(split_config.n_splits):
        scenario_df = labeled_df[labeled_df["fold"] == fold_index].copy().reset_index(drop=True)
        scenarios.append(
            {
                "fold": fold_index,
                "labeled_df": scenario_df,
                "matrices": build_matrices(scenario_df, feature_columns),
            }
        )
    return split_df, scenarios


def select_best_classical_models(
    scenarios: list[dict],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    dev_selection_rows: list[dict] = []
    fold_test_rows: list[dict] = []
    fold_dataset_rows: list[dict] = []
    chosen_models: list[dict] = []

    grouped_candidates: dict[str, list[tuple[str, Pipeline]]] = {}
    for family, candidate_name, model in build_classical_candidates(seed):
        grouped_candidates.setdefault(family, []).append((candidate_name, model))

    for family, candidates in grouped_candidates.items():
        best_candidate_name = None
        best_candidate_score = -1.0

        for candidate_name, candidate_model in candidates:
            fold_f1s: list[float] = []
            for scenario in scenarios:
                matrices = scenario["matrices"]
                labeled_df = scenario["labeled_df"]
                dev_frame = labeled_df[labeled_df["split"] == "dev"].copy().reset_index(drop=True)
                _, y_dev = matrices["dev"]
                predictions = train_and_predict_classical(copy.deepcopy(candidate_model), matrices)
                dev_metrics = evaluate_predictions(
                    dev_frame,
                    predictions["dev"]["predictions"],
                    predictions["dev"]["scores"],
                    candidate_name,
                    "dev",
                )[0]
                dev_selection_rows.append(
                    {
                        "fold": scenario["fold"],
                        "model_family": family,
                        "candidate": candidate_name,
                        **dev_metrics,
                    }
                )
                fold_f1s.append(float(f1_score(y_dev, predictions["dev"]["predictions"], zero_division=0)))

            mean_f1 = float(np.mean(fold_f1s)) if fold_f1s else -1.0
            if mean_f1 > best_candidate_score:
                best_candidate_score = mean_f1
                best_candidate_name = candidate_name

        if best_candidate_name is None:
            continue

        chosen_models.append(
            {
                "model_family": family,
                "selected_candidate": best_candidate_name,
                "selection_metric": "mean_dev_f1_across_folds",
                "dev_f1": best_candidate_score,
            }
        )

        for scenario in scenarios:
            selected_pipeline = next(model for name, model in grouped_candidates[family] if name == best_candidate_name)
            predictions = train_and_predict_classical(copy.deepcopy(selected_pipeline), scenario["matrices"])
            test_frame = scenario["labeled_df"][scenario["labeled_df"]["split"] == "test"].copy().reset_index(drop=True)
            global_row, dataset_rows = evaluate_predictions(
                test_frame,
                predictions["test"]["predictions"],
                predictions["test"]["scores"],
                family,
                "test",
            )
            global_row["selected_candidate"] = best_candidate_name
            fold_test_rows.append(global_row)
            for row in dataset_rows:
                row["selected_candidate"] = best_candidate_name
                fold_dataset_rows.append(row)

    if XGBClassifier is None:
        chosen_models.append(
            {
                "model_family": "xgboost",
                "selected_candidate": "skipped",
                "selection_metric": "mean_dev_f1_across_folds",
                "dev_f1": np.nan,
            }
        )

    return dev_selection_rows, fold_test_rows, fold_dataset_rows, chosen_models


def select_best_neural_model(
    scenarios: list[dict],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    dev_selection_rows: list[dict] = []
    fold_test_rows: list[dict] = []
    fold_dataset_rows: list[dict] = []

    best_config = None
    best_score = -1.0

    for neural_config in NEURAL_CONFIGS:
        fold_f1s: list[float] = []
        for scenario in scenarios:
            predictions = train_neural_network(neural_config, scenario["matrices"], seed)
            dev_frame = scenario["labeled_df"][scenario["labeled_df"]["split"] == "dev"].copy().reset_index(drop=True)
            _, y_dev = scenario["matrices"]["dev"]
            dev_metrics = evaluate_predictions(
                dev_frame,
                predictions["dev"]["predictions"],
                predictions["dev"]["scores"],
                neural_config.name,
                "dev",
            )[0]
            dev_selection_rows.append(
                {
                    "fold": scenario["fold"],
                    "model_family": "neural_network",
                    "candidate": neural_config.name,
                    **dev_metrics,
                }
            )
            fold_f1s.append(float(f1_score(y_dev, predictions["dev"]["predictions"], zero_division=0)))

        mean_f1 = float(np.mean(fold_f1s)) if fold_f1s else -1.0
        if mean_f1 > best_score:
            best_score = mean_f1
            best_config = neural_config

    if best_config is None:
        return dev_selection_rows, fold_test_rows, fold_dataset_rows, {
            "model_family": "neural_network",
            "selected_candidate": "none",
            "selection_metric": "mean_dev_f1_across_folds",
            "dev_f1": np.nan,
        }

    for scenario in scenarios:
        predictions = train_neural_network(best_config, scenario["matrices"], seed)
        test_frame = scenario["labeled_df"][scenario["labeled_df"]["split"] == "test"].copy().reset_index(drop=True)
        global_row, dataset_rows = evaluate_predictions(
            test_frame,
            predictions["test"]["predictions"],
            predictions["test"]["scores"],
            "neural_network",
            "test",
        )
        global_row["selected_candidate"] = best_config.name
        fold_test_rows.append(global_row)
        for row in dataset_rows:
            row["selected_candidate"] = best_config.name
            fold_dataset_rows.append(row)

    chosen_model = {
        "model_family": "neural_network",
        "selected_candidate": best_config.name,
        "selection_metric": "mean_dev_f1_across_folds",
        "dev_f1": best_score,
    }
    return dev_selection_rows, fold_test_rows, fold_dataset_rows, chosen_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run centralized k-fold-by-client experiments on client_zscore windows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--max-trials-per-dataset",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs before launching the full experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_directory(args.output_dir / f"run_{timestamp}")

    windows_df = load_baseline_windows()
    feature_columns = get_feature_columns(windows_df)
    trial_table = build_trial_table(windows_df)

    if args.max_trials_per_dataset is not None:
        sampled_trials = []
        for _, dataset_trials in trial_table.groupby("dataset", sort=True):
            sampled_trials.append(
                dataset_trials.sample(
                    n=min(args.max_trials_per_dataset, len(dataset_trials)),
                    random_state=args.seed,
                )
            )
        trial_table = pd.concat(sampled_trials, ignore_index=True)
        windows_df = windows_df.merge(trial_table[["dataset", "trial_id"]], on=["dataset", "trial_id"], how="inner")

    split_config = SplitConfig(random_state=args.seed, n_splits=args.n_splits)
    split_df, scenarios = build_fold_scenarios(windows_df, trial_table, feature_columns, split_config)

    build_client_fold_reference(trial_table, split_config).to_csv(run_dir / "client_fold_assignments.csv", index=False)
    build_fold_split_reference(trial_table, split_config).to_csv(run_dir / "client_split_assignments.csv", index=False)
    build_split_summary(split_df).to_csv(run_dir / "split_summary.csv", index=False)

    selection_rows, fold_global_rows, fold_dataset_rows, chosen_rows = select_best_classical_models(scenarios=scenarios, seed=args.seed)
    nn_selection_rows, nn_fold_global_rows, nn_fold_dataset_rows, nn_chosen_row = select_best_neural_model(scenarios=scenarios, seed=args.seed)

    selection_rows.extend(nn_selection_rows)
    fold_global_rows.extend(nn_fold_global_rows)
    fold_dataset_rows.extend(nn_fold_dataset_rows)
    chosen_rows.append(nn_chosen_row)

    selection_df = pd.DataFrame(selection_rows).sort_values(["model_family", "fold", "candidate"]).reset_index(drop=True)
    fold_global_df = pd.DataFrame(fold_global_rows).sort_values(["fold", *FINAL_SORT_COLUMNS], ascending=[True, *FINAL_SORT_ASCENDING]).reset_index(drop=True)
    fold_dataset_df = pd.DataFrame(fold_dataset_rows).sort_values(["fold", "dataset", *FINAL_SORT_COLUMNS], ascending=[True, True, *FINAL_SORT_ASCENDING]).reset_index(drop=True)
    chosen_models_df = pd.DataFrame(chosen_rows).sort_values("model_family").reset_index(drop=True)

    final_global_df = aggregate_metrics(fold_global_rows, ["model", "split", "dataset"]).sort_values(FINAL_SORT_COLUMNS, ascending=FINAL_SORT_ASCENDING).reset_index(drop=True)
    final_dataset_df = aggregate_metrics(fold_dataset_rows, ["model", "split", "dataset"]).sort_values(["dataset", *FINAL_SORT_COLUMNS], ascending=[True, *FINAL_SORT_ASCENDING]).reset_index(drop=True)

    selection_df.to_csv(run_dir / "model_selection_dev.csv", index=False)
    fold_global_df.to_csv(run_dir / "fold_model_metrics.csv", index=False)
    fold_dataset_df.to_csv(run_dir / "fold_model_metrics_by_dataset.csv", index=False)
    final_global_df.to_csv(run_dir / "final_model_metrics.csv", index=False)
    final_dataset_df.to_csv(run_dir / "final_model_metrics_by_dataset.csv", index=False)
    chosen_models_df.to_csv(run_dir / "chosen_models.csv", index=False)
    final_global_df.to_csv(run_dir / "metrics_global.csv", index=False)
    final_dataset_df.to_csv(run_dir / "metrics_by_dataset.csv", index=False)

    config_payload = {
        "seed": args.seed,
        "windows_path": str((Path(__file__).resolve().parents[3] / "data" / "preprocessing_results" / "simple" / "client_zscore" / "windows.csv").resolve()),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "split_config": dataclass_to_dict(split_config),
        "split_policy": "kfold_by_client",
        "n_splits": args.n_splits,
        "neural_configs": [dataclass_to_dict(config) for config in NEURAL_CONFIGS],
        "classical_candidate_families": sorted({family for family, _, _ in build_classical_candidates(args.seed)}),
        "xgboost_available": XGBClassifier is not None,
    }
    write_json(run_dir / "config.json", config_payload)

    summary_payload = {
        "run_dir": str(run_dir.resolve()),
        "ranking_criterion": FINAL_SORT_COLUMNS,
        "best_model_on_test_by_priority": None if final_global_df.empty else final_global_df.iloc[0].to_dict(),
    }
    write_json(run_dir / "run_summary.json", summary_payload)

    print(f"Saved run artifacts to: {run_dir.resolve()}")
    print("Chosen models after dev selection:")
    print(chosen_models_df.to_string(index=False))
    print("")
    print("Final k-fold-by-client comparison:")
    print(final_global_df.to_string(index=False))


if __name__ == "__main__":
    main()
