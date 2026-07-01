from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np

from advanced_strategy_configs import ADVANCED_STRATEGIES, AdvancedStrategyConfig
from payload_configs import PAYLOAD_CONFIGS
from probability_payloads import (
    _confidence_weight,
    _serialize_payload,
    _teacher_targets,
    aggregate_probability_payloads,
    build_probability_payload,
    require_flwr,
)


def train_local_model_with_advanced_strategy(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config,
    local_epochs: int,
    seed: int,
    global_payload: dict[str, Any] | None,
    strategy_config: AdvancedStrategyConfig,
) -> tuple[Any, dict[str, float | int]]:
    import torch
    from modeling import evaluate_model

    payload_config = PAYLOAD_CONFIGS[strategy_config.payload_name]
    torch.manual_seed(seed)
    np.random.seed(seed)

    positives = max(int(y_train.sum()), 1)
    negatives = max(int((1 - y_train).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)
    batch_size = min(config.model.batch_size, max(len(x_train), 1))
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0

    for _ in range(local_epochs):
        model.train()
        permutation = torch.randperm(x_train_tensor.size(0))
        for start in range(0, x_train_tensor.size(0), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = x_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()
            logits = model(batch_x)
            hard_loss = criterion(logits, batch_y)
            loss = hard_loss

            if global_payload is not None:
                teacher_logits = _teacher_targets(batch_y, global_payload, payload_config).to(logits.device)
                student_logits_two = torch.stack([-logits, logits], dim=1)
                teacher_logits_two = torch.stack([-teacher_logits, teacher_logits], dim=1)
                temperature = float(config.distillation_temperature)
                student_log_probs = torch.log_softmax(student_logits_two / temperature, dim=1)
                teacher_probs = torch.softmax(teacher_logits_two / temperature, dim=1)

                if strategy_config.uncertain_only:
                    local_probs = torch.sigmoid(logits.detach())
                    mask = (local_probs >= strategy_config.uncertainty_low) & (local_probs <= strategy_config.uncertainty_high)
                    if bool(mask.any()):
                        per_sample = torch.nn.functional.kl_div(
                            student_log_probs,
                            teacher_probs,
                            reduction="none",
                        ).sum(dim=1) * (temperature**2)
                        kd_loss = per_sample[mask].mean()
                    else:
                        kd_loss = None
                else:
                    kd_loss = torch.nn.functional.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction="batchmean",
                    ) * (temperature**2)

                if kd_loss is not None:
                    kd_weight = _confidence_weight(global_payload, payload_config)
                    alpha = float(config.distillation_alpha)
                    loss = alpha * hard_loss + (1.0 - alpha) * kd_weight * kd_loss

            loss.backward()
            optimizer.step()

        metrics = evaluate_model(model, x_val, y_val) if len(x_val) > 0 else evaluate_model(model, x_train, y_train)
        if float(metrics["f1"]) >= best_f1:
            best_f1 = float(metrics["f1"])
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final_metrics = evaluate_model(model, x_val, y_val) if len(x_val) > 0 else evaluate_model(model, x_train, y_train)
    return model, final_metrics


def make_advanced_probability_client_fn(setup, config, strategy_name: str, client_states: dict | None = None):
    fl = require_flwr()
    strategy_config = ADVANCED_STRATEGIES[strategy_name]
    payload_config = PAYLOAD_CONFIGS[strategy_config.payload_name]

    class AdvancedProbabilityClient(fl.client.NumPyClient):
        def __init__(self, partition, state: dict[str, Any]):
            self.partition = partition
            self.state = state
            from modeling import create_model, set_model_parameters

            self.model = create_model(len(setup.feature_columns), config.model)
            if self.state.get("current_local_params") is not None:
                set_model_parameters(self.model, self.state["current_local_params"], keep_local_head=bool(config.personalized_head))

        def get_parameters(self, config=None):
            return []

        def fit(self, parameters, config_dict):
            from modeling import evaluate_model, get_model_parameters, predict_probabilities

            server_round = int(config_dict.get("server_round", 0))
            local_epochs = int(config_dict.get("local_epochs", config.local_epochs))
            raw_payload = config_dict.get("probability_payload")
            raw_payload_map = config_dict.get("strategy_payload") or config_dict.get("probability_payload_map")
            global_payload = None
            if raw_payload:
                try:
                    global_payload = json.loads(str(raw_payload))
                except Exception:
                    global_payload = None
            elif raw_payload_map:
                try:
                    payload_map = json.loads(str(raw_payload_map))
                except Exception:
                    payload_map = None
                if isinstance(payload_map, dict):
                    if payload_map.get("scope") == "cluster":
                        clusters = payload_map.get("clusters", {})
                        if isinstance(clusters, dict):
                            global_payload = clusters.get(str(self.partition.cluster_id))
                    elif payload_map.get("scope") == "global":
                        global_payload = payload_map.get("global")

            self.model, val_metrics = train_local_model_with_advanced_strategy(
                model=self.model,
                x_train=self.partition.x_train,
                y_train=self.partition.y_train,
                x_val=self.partition.x_val,
                y_val=self.partition.y_val,
                config=config,
                local_epochs=local_epochs,
                seed=config.random_seed + server_round + self.partition.cluster_id,
                global_payload=global_payload,
                strategy_config=strategy_config,
            )

            probs = predict_probabilities(self.model, self.partition.x_train)
            payload = build_probability_payload(probs, self.partition.y_train, payload_config)
            payload["cluster_id"] = float(self.partition.cluster_id)

            current_params = get_model_parameters(self.model)
            if config.local_model_selection:
                current_f1 = float(val_metrics.get("f1", 0.0))
                if current_f1 >= float(self.state.get("best_local_f1", -1.0)):
                    self.state["best_local_f1"] = current_f1
                    self.state["best_local_params"] = copy.deepcopy(current_params)
                elif self.state.get("best_local_params") is not None:
                    current_params = copy.deepcopy(self.state["best_local_params"])
            self.state["current_local_params"] = current_params
            self.state["last_payload"] = copy.deepcopy(payload)
            self.state["last_val_metrics"] = {
                k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))
            }

            dev_metrics = evaluate_model(self.model, setup.dev_arrays[0], setup.dev_arrays[1]) if len(setup.dev_arrays[0]) else {}
            test_metrics = evaluate_model(self.model, setup.test_arrays[0], setup.test_arrays[1]) if len(setup.test_arrays[0]) else {}

            metrics = {
                "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
                "val_balanced_accuracy": float(val_metrics.get("balanced_accuracy", 0.0)),
                "val_specificity": float(val_metrics.get("specificity", 0.0)),
                "val_precision": float(val_metrics.get("precision", 0.0)),
                "val_recall": float(val_metrics.get("recall", 0.0)),
                "val_f1": float(val_metrics.get("f1", 0.0)),
                "val_roc_auc": float(val_metrics.get("roc_auc", 0.0)),
                "val_pr_auc": float(val_metrics.get("pr_auc", 0.0)),
                "val_far": float(val_metrics.get("far", 0.0)),
                "val_miss_rate": float(val_metrics.get("miss_rate", 0.0)),
                "cluster_id": int(self.partition.cluster_id),
                "num_examples": int(len(self.partition.y_train)),
                "probability_payload": _serialize_payload(payload),
            }
            for metric_name, metric_value in dev_metrics.items():
                if isinstance(metric_value, (int, float, np.integer, np.floating)):
                    metrics[f"dev_{metric_name}"] = float(metric_value)
            for metric_name, metric_value in test_metrics.items():
                if isinstance(metric_value, (int, float, np.integer, np.floating)):
                    metrics[f"test_{metric_name}"] = float(metric_value)
            return [], int(len(self.partition.y_train)), metrics

        def evaluate(self, parameters, config=None):
            from modeling import evaluate_model

            metrics = evaluate_model(self.model, self.partition.x_val, self.partition.y_val)
            loss = 1.0 - float(metrics.get("pr_auc", 0.0)) if metrics.get("pr_auc") is not None else 1.0
            return float(loss), int(len(self.partition.y_val)), {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))
            }

    ordered_client_ids = sorted(setup.train_clients.keys())

    def client_fn(context):
        partition_id = None
        raw_id = None
        if isinstance(context, str):
            raw_id = context
        else:
            if hasattr(context, "node_config"):
                partition_id = context.node_config.get("partition-id")
            raw_id = getattr(context, "node_id", None) or getattr(context, "cid", None)
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
                raise KeyError(f"Unknown client id mapping for '{raw_id}'")

        partition = setup.train_clients[client_id]
        state = client_states.setdefault(client_id, {}) if client_states is not None else {}
        return AdvancedProbabilityClient(partition, state).to_client()

    return client_fn


class AdvancedPayloadAggregator:
    def __init__(self, strategy_name: str):
        self.strategy_config = ADVANCED_STRATEGIES[strategy_name]
        self.payload_config = PAYLOAD_CONFIGS[self.strategy_config.payload_name]
        self.latest_payload: dict[str, Any] | None = None

    def aggregate_payloads(self, results, weighted: bool = True) -> dict[str, Any] | None:
        payloads: list[tuple[int, dict[str, Any]]] = []
        for _, fit_res in results:
            metrics = getattr(fit_res, "metrics", {}) or {}
            raw_payload = metrics.get("probability_payload")
            if raw_payload is None:
                continue
            try:
                payload = json.loads(str(raw_payload))
            except Exception:
                continue
            payloads.append((int(getattr(fit_res, "num_examples", 0)), payload))
        if not payloads:
            self.latest_payload = None
            return None

        if self.strategy_config.aggregation_scope == "cluster":
            grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
            for num_examples, payload in payloads:
                cluster_id = int(payload.get("cluster_id", 0))
                grouped.setdefault(cluster_id, []).append((num_examples, payload))
            cluster_payloads = {}
            for cluster_id, group_payloads in grouped.items():
                cluster_payloads[str(cluster_id)] = aggregate_probability_payloads(
                    group_payloads,
                    self.payload_config,
                    weighted=weighted,
                )
            self.latest_payload = {"scope": "cluster", "clusters": cluster_payloads}
            return self.latest_payload

        aggregate = aggregate_probability_payloads(payloads, self.payload_config, weighted=weighted)
        self.latest_payload = {"scope": "global", "global": aggregate}
        return self.latest_payload
