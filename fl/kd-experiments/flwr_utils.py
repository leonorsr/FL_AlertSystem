from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from config import ExperimentConfig
from data_utils import ClientPartition, FederatedSetup
from metrics import compute_binary_metrics
from modeling import (
    create_model,
    create_teacher_model,
    evaluate_model,
    get_model_parameters,
    predict_probabilities,
    set_model_parameters,
    train_local_model,
)


def require_flwr():
    try:
        import flwr as fl
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    except ImportError as exc:
        raise ImportError(
            "Flower is required for KD experiments. Install it with `pip install flwr`."
        ) from exc
    return fl, ndarrays_to_parameters, parameters_to_ndarrays


def make_evaluate_fn(setup: FederatedSetup, config: ExperimentConfig):
    def evaluate(server_round: int, parameters_ndarrays: list[np.ndarray], split_name: str) -> dict[str, float | int]:
        model = create_model(len(setup.feature_columns), config.model)
        set_model_parameters(model, parameters_ndarrays, keep_local_head=False)
        x_values, y_true = setup.dev_arrays if split_name == "dev" else setup.test_arrays
        if len(x_values) == 0:
            return {"f1": float("nan")}
        probs = predict_probabilities(model, x_values)
        preds = (probs >= 0.5).astype(int)
        metrics = compute_binary_metrics(y_true, preds, probs)
        metrics["round"] = server_round
        metrics["split"] = split_name
        return metrics

    def flower_evaluate(server_round: int, parameters, _config):
        metrics = evaluate(server_round, parameters, "dev")
        loss = 1.0 - float(metrics["pr_auc"]) if not np.isnan(metrics["pr_auc"]) else 1.0
        compact_metrics = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.integer, np.floating))}
        return loss, compact_metrics

    return evaluate, flower_evaluate


def make_client_fn(setup: FederatedSetup, config: ExperimentConfig):
    fl, _, _ = require_flwr()
    client_states: dict[str, dict[str, any]] = {client_id: {} for client_id in setup.train_clients}
    ordered_client_ids = sorted(setup.train_clients.keys())

    class FallNumPyClient(fl.client.NumPyClient):
        def __init__(self, partition: ClientPartition, state: dict[str, any]) -> None:
            self.partition = partition
            self.state = state
            self.model = create_model(len(setup.feature_columns), config.model)
            self.teacher_model = create_teacher_model(len(setup.feature_columns), config.model)

        def get_parameters(self, config_dict):
            return get_model_parameters(self.model)

        def fit(self, parameters, config_dict):
            keep_local_head = bool(config.personalized_head)
            set_model_parameters(self.teacher_model, parameters, keep_local_head=keep_local_head)
            set_model_parameters(self.model, parameters, keep_local_head=keep_local_head)

            if config.local_model_selection:
                candidate_global = create_model(len(setup.feature_columns), config.model)
                set_model_parameters(candidate_global, parameters, keep_local_head=keep_local_head)
                global_f1 = float(evaluate_model(candidate_global, self.partition.x_val, self.partition.y_val)["f1"])

                if self.state.get("best_global_f1", -1.0) <= global_f1:
                    self.state["best_global_f1"] = global_f1
                    self.state["best_global_params"] = copy.deepcopy(parameters)

                if self.state.get("best_global_params") is not None:
                    candidate_previous_global = create_model(len(setup.feature_columns), config.model)
                    set_model_parameters(candidate_previous_global, self.state["best_global_params"], keep_local_head=False)
                    previous_global_f1 = float(evaluate_model(candidate_previous_global, self.partition.x_val, self.partition.y_val)["f1"])
                    if previous_global_f1 > global_f1:
                        self.model = candidate_previous_global

            self.model, val_metrics = train_local_model(
                model=self.model,
                x_train=self.partition.x_train,
                y_train=self.partition.y_train,
                x_val=self.partition.x_val,
                y_val=self.partition.y_val,
                config=config,
                local_epochs=int(config_dict.get("local_epochs", config.local_epochs)),
                seed=config.random_seed + int(config_dict.get("server_round", 0)) + self.partition.cluster_id,
                teacher_model=self.teacher_model if config.teacher_student_distillation else None,
            )

            current_params = get_model_parameters(self.model)
            current_f1 = float(val_metrics["f1"])
            self.state["current_local_params"] = copy.deepcopy(current_params)
            self.state["last_val_metrics"] = copy.deepcopy(val_metrics)

            metrics = {
                "client_id": self.partition.client_id,
                "dataset": self.partition.dataset,
                "client": self.partition.client,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_specificity": float(val_metrics["specificity"]),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_f1": current_f1,
                "val_roc_auc": float(val_metrics["roc_auc"]) if not np.isnan(val_metrics["roc_auc"]) else float("nan"),
                "val_pr_auc": float(val_metrics["pr_auc"]) if not np.isnan(val_metrics["pr_auc"]) else float("nan"),
                "val_far": float(val_metrics["far"]),
                "val_miss_rate": float(val_metrics["miss_rate"]),
                "cluster_id": int(self.partition.cluster_id),
                "num_examples": int(len(self.partition.y_train)),
            }
            return current_params, len(self.partition.y_train), metrics

        def evaluate(self, parameters, config_dict):
            keep_local_head = bool(config.personalized_head)
            set_model_parameters(self.model, parameters, keep_local_head=keep_local_head)
            metrics = evaluate_model(self.model, self.partition.x_val, self.partition.y_val)
            loss = 1.0 - float(metrics["pr_auc"]) if not np.isnan(metrics["pr_auc"]) else 1.0
            return loss, len(self.partition.y_val), {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.integer, np.floating))}

    def client_fn(context):
        partition_id = None
        raw_id = None
        if isinstance(context, str):
            raw_id = context
        else:
            if hasattr(context, "node_config"):
                partition_id = context.node_config.get("partition-id")
            raw_id = getattr(context, "node_id", None)
            if raw_id is None:
                raw_id = getattr(context, "cid", None)

        client_id = None
        if partition_id is not None:
            partition_idx = int(partition_id)
            if 0 <= partition_idx < len(ordered_client_ids):
                client_id = ordered_client_ids[partition_idx]

        if client_id is None:
            raw_id = str(raw_id)
            if raw_id in setup.train_clients:
                client_id = raw_id
            elif raw_id.isdigit() and int(raw_id) < len(ordered_client_ids):
                client_id = ordered_client_ids[int(raw_id)]
            else:
                raise KeyError(
                    f"Could not map Flower client id '{raw_id}' to a known training client."
                )

        partition = setup.train_clients[client_id]
        return FallNumPyClient(partition, client_states[client_id])

    return client_fn, client_states


def create_strategy(setup: FederatedSetup, config: ExperimentConfig, initial_parameters):
    fl, ndarrays_to_parameters, parameters_to_ndarrays = require_flwr()

    def aggregate_local_fit_metrics(metrics_list):
        if not metrics_list:
            return {}
        total_examples = sum(num_examples for num_examples, _ in metrics_list)
        aggregated: dict[str, float] = {}
        metric_names = sorted(
            key
            for _, metrics in metrics_list
            for key in metrics.keys()
            if key.startswith("val_") or key in {"cluster_id", "num_examples"}
        )
        for metric_name in metric_names:
            weighted_sum = 0.0
            weight_sum = 0
            for num_examples, metrics in metrics_list:
                value = metrics.get(metric_name)
                if value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isnan(value):
                    continue
                weighted_sum += num_examples * value
                weight_sum += num_examples
            aggregated[metric_name] = weighted_sum / weight_sum if weight_sum > 0 else float("nan")
        aggregated["participating_clients"] = float(len(metrics_list))
        aggregated["total_examples"] = float(total_examples)
        return aggregated

    class BaseTrackingFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.latest_parameters_ndarrays = copy.deepcopy(initial_parameters)
            self.latest_local_metrics_by_client = pd.DataFrame()
            self.latest_local_metrics_summary: dict[str, float] = {}

        def _capture_local_fit_metrics(self, results) -> None:
            rows: list[dict] = []
            for _, fit_res in results:
                metrics = getattr(fit_res, "metrics", {}) or {}
                row = {
                    "client_id": metrics.get("client_id", ""),
                    "dataset": metrics.get("dataset", ""),
                    "client": metrics.get("client", ""),
                    "cluster_id": int(metrics.get("cluster_id", 0)),
                    "evaluation_split": "local_val",
                    "model_source": "last_round_local_fit",
                    "train_examples": int(getattr(fit_res, "num_examples", metrics.get("num_examples", 0))),
                }
                for key, value in metrics.items():
                    if not key.startswith("val_"):
                        continue
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isnan(value):
                        continue
                    row[key.removeprefix("val_")] = value
                if "pr_auc" in row or "f1" in row:
                    rows.append(row)

            if not rows:
                self.latest_local_metrics_by_client = pd.DataFrame()
                self.latest_local_metrics_summary = {}
                return

            local_df = (
                pd.DataFrame(rows)
                .sort_values(["pr_auc", "miss_rate", "far", "balanced_accuracy", "f1"], ascending=[False, True, True, False, False])
                .reset_index(drop=True)
            )
            self.latest_local_metrics_by_client = local_df
            self.latest_local_metrics_summary = local_df.mean(numeric_only=True).to_dict()

        def aggregate_fit(self, server_round, results, failures):
            self._capture_local_fit_metrics(results)
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated is not None:
                parameters, metrics = aggregated
                if parameters is not None:
                    self.latest_parameters_ndarrays = parameters_to_ndarrays(parameters)
                return parameters, metrics
            return aggregated

    class UnweightedFedAvg(BaseTrackingFedAvg):
        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None
            self._capture_local_fit_metrics(results)
            arrays_per_client = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            averaged = []
            for tensors in zip(*arrays_per_client):
                averaged.append(np.mean(np.stack(tensors, axis=0), axis=0))
            self.latest_parameters_ndarrays = averaged
            return ndarrays_to_parameters(averaged), {}

    class ClusteredFedAvg(BaseTrackingFedAvg):
        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None
            self._capture_local_fit_metrics(results)

            grouped: dict[int, list[tuple[int, list[np.ndarray]]]] = {}
            for _, fit_res in results:
                cluster_id = int(fit_res.metrics.get("cluster_id", 0))
                grouped.setdefault(cluster_id, []).append((int(fit_res.num_examples), parameters_to_ndarrays(fit_res.parameters)))

            subgroup_models: list[tuple[int, list[np.ndarray]]] = []
            for cluster_results in grouped.values():
                total_examples = sum(num_examples for num_examples, _ in cluster_results)
                averaged_cluster = []
                for tensors in zip(*[params for _, params in cluster_results]):
                    weighted = sum(num_examples * tensor for num_examples, tensor in zip([n for n, _ in cluster_results], tensors))
                    averaged_cluster.append(weighted / max(total_examples, 1))
                subgroup_models.append((total_examples, averaged_cluster))

            total_examples = sum(num_examples for num_examples, _ in subgroup_models)
            global_arrays = []
            for tensors in zip(*[params for _, params in subgroup_models]):
                weighted = sum(num_examples * tensor for num_examples, tensor in zip([n for n, _ in subgroup_models], tensors))
                global_arrays.append(weighted / max(total_examples, 1))

            self.latest_parameters_ndarrays = global_arrays
            return ndarrays_to_parameters(global_arrays), {}

    strategy_cls = BaseTrackingFedAvg
    if not config.weighted_aggregation:
        strategy_cls = UnweightedFedAvg
    if config.clustered_aggregation:
        strategy_cls = ClusteredFedAvg

    strategy = strategy_cls(
        fraction_fit=config.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=config.min_fit_clients,
        min_evaluate_clients=0,
        min_available_clients=config.min_available_clients,
        evaluate_fn=make_evaluate_fn(setup, config)[1],
        fit_metrics_aggregation_fn=aggregate_local_fit_metrics,
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "local_epochs": config.local_epochs,
        },
        initial_parameters=ndarrays_to_parameters(initial_parameters),
    )
    return strategy


def build_final_local_metrics_table(setup: FederatedSetup, config: ExperimentConfig, client_states: dict[str, dict[str, any]]):
    rows: list[dict] = []
    for client_id, partition in setup.train_clients.items():
        state = client_states.get(client_id)
        params = None
        source = "current_local_model"
        if state is not None and state.get("current_local_params") is not None:
            params = state.get("current_local_params")
        elif state is not None and state.get("best_global_params") is not None:
            params = state.get("best_global_params")
            source = "best_global_model"

        if params is None:
            continue

        model = create_model(len(setup.feature_columns), config.model)
        set_model_parameters(model, params, keep_local_head=False)
        metrics = evaluate_model(model, partition.x_val, partition.y_val)
        rows.append(
            {
                "client_id": client_id,
                "dataset": partition.dataset,
                "client": partition.client,
                "cluster_id": partition.cluster_id,
                "evaluation_split": "local_val",
                "model_source": source,
                "train_examples": int(len(partition.y_train)),
                "val_examples": int(len(partition.y_val)),
                **metrics,
            }
        )

    if not rows:
        return None, {}

    local_df = (
        pd.DataFrame(rows)
        .sort_values(["pr_auc", "miss_rate", "far", "balanced_accuracy", "f1"], ascending=[False, True, True, False, False])
        .reset_index(drop=True)
    )
    local_summary = local_df.mean(numeric_only=True).to_dict()
    return local_df, local_summary
