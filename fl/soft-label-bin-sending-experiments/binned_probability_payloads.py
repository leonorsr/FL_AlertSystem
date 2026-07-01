from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np

from bin_payload_configs import BIN_PAYLOAD_CONFIGS, BinPayloadConfig


def require_flwr():
    try:
        import flwr as fl
    except ImportError as exc:
        raise ImportError("Flower is required for binned soft-label experiments. Install with `pip install flwr`.") from exc
    return fl


def _bin_edges(num_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, num_bins + 1)


def _bin_index_from_probs(probs: np.ndarray, num_bins: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    indices = np.floor(np.clip(probs, 0.0, 1.0 - 1e-12) * num_bins).astype(int)
    return np.clip(indices, 0, num_bins - 1)


def build_binned_probability_payload(probs: np.ndarray, payload_config: BinPayloadConfig) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=float)
    num_bins = payload_config.num_bins
    bin_indices = _bin_index_from_probs(probs, num_bins) if len(probs) else np.asarray([], dtype=int)
    bins: list[dict[str, float]] = []

    fallback = float(np.mean(probs)) if len(probs) else 0.5
    for bin_idx in range(num_bins):
        values = probs[bin_indices == bin_idx]
        if len(values):
            bins.append(
                {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                    "count": float(len(values)),
                }
            )
        else:
            bins.append({"mean": fallback, "std": 0.0, "count": 0.0})

    return {
        "payload_type": payload_config.name,
        "num_bins": int(num_bins),
        "edges": _bin_edges(num_bins).tolist(),
        "bins": bins,
    }


def aggregate_binned_probability_payloads(
    weighted_payloads: list[tuple[int, dict[str, Any]]],
    payload_config: BinPayloadConfig,
) -> dict[str, Any] | None:
    if not weighted_payloads:
        return None

    num_bins = payload_config.num_bins
    aggregate_bins: list[dict[str, float]] = []
    global_values = []
    for _, payload in weighted_payloads:
        for bin_payload in payload.get("bins", []):
            count = float(bin_payload.get("count", 0.0))
            mean = float(bin_payload.get("mean", 0.5))
            if count > 0:
                global_values.append((count, mean))
    fallback = sum(count * mean for count, mean in global_values) / sum(count for count, _ in global_values) if global_values else 0.5

    for bin_idx in range(num_bins):
        total_count = 0.0
        weighted_mean_numerator = 0.0
        second_moment_numerator = 0.0
        for _, payload in weighted_payloads:
            bins = payload.get("bins", [])
            if bin_idx >= len(bins):
                continue
            bin_payload = bins[bin_idx]
            count = max(float(bin_payload.get("count", 0.0)), 0.0)
            if count <= 0:
                continue
            mean = float(bin_payload.get("mean", fallback))
            std = float(bin_payload.get("std", 0.0))
            total_count += count
            weighted_mean_numerator += count * mean
            second_moment_numerator += count * (std**2 + mean**2)

        if total_count > 0:
            mean = weighted_mean_numerator / total_count
            variance = max(second_moment_numerator / total_count - mean**2, 0.0)
            aggregate_bins.append({"mean": float(mean), "std": float(np.sqrt(variance)), "count": float(total_count)})
        else:
            aggregate_bins.append({"mean": float(fallback), "std": 0.0, "count": 0.0})

    return {
        "payload_type": payload_config.name,
        "num_bins": int(num_bins),
        "edges": _bin_edges(num_bins).tolist(),
        "bins": aggregate_bins,
    }


def _probability_to_logit(probability: float) -> float:
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(probability / (1.0 - probability)))


def _teacher_logits_from_bins(student_logits, payload: dict[str, Any], payload_config: BinPayloadConfig):
    import torch

    with torch.no_grad():
        probs = torch.sigmoid(student_logits).detach().cpu().numpy()
        bin_indices = _bin_index_from_probs(probs, payload_config.num_bins)
        bins = payload.get("bins", [])
        targets = []
        for bin_idx in bin_indices:
            if bin_idx < len(bins):
                targets.append(_probability_to_logit(float(bins[bin_idx].get("mean", 0.5))))
            else:
                targets.append(_probability_to_logit(0.5))
    return torch.tensor(targets, dtype=torch.float32, device=student_logits.device)


def _confidence_weight_from_bins(student_logits, payload: dict[str, Any], payload_config: BinPayloadConfig) -> float:
    with np.errstate(all="ignore"):
        probs = 1.0 / (1.0 + np.exp(-student_logits.detach().cpu().numpy()))
    bin_indices = _bin_index_from_probs(probs, payload_config.num_bins)
    bins = payload.get("bins", [])
    std_values = []
    for bin_idx in np.unique(bin_indices):
        if bin_idx < len(bins):
            std_values.append(float(bins[bin_idx].get("std", 0.0)))
    if not std_values:
        return 1.0
    return float(np.clip(1.0 - 2.0 * float(np.mean(std_values)), 0.2, 1.0))


def train_local_model_with_binned_payload(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config,
    local_epochs: int,
    seed: int,
    global_payload: dict[str, Any] | None,
    payload_config: BinPayloadConfig,
) -> tuple[Any, dict[str, float | int]]:
    import torch
    from modeling import evaluate_model

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
                teacher_logits = _teacher_logits_from_bins(logits, global_payload, payload_config)
                student_logits_two = torch.stack([-logits, logits], dim=1)
                teacher_logits_two = torch.stack([-teacher_logits, teacher_logits], dim=1)
                temperature = float(config.distillation_temperature)
                student_log_probs = torch.log_softmax(student_logits_two / temperature, dim=1)
                teacher_probs = torch.softmax(teacher_logits_two / temperature, dim=1)
                kd_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)
                kd_weight = _confidence_weight_from_bins(logits, global_payload, payload_config)
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


def make_binned_payload_client_fn(setup, config, payload_name: str, client_states: dict | None = None):
    fl = require_flwr()
    payload_config = BIN_PAYLOAD_CONFIGS[payload_name]

    class BinnedPayloadClient(fl.client.NumPyClient):
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
            raw_payload = config_dict.get("global_probability_payload")
            global_payload = None
            if raw_payload:
                try:
                    global_payload = json.loads(str(raw_payload))
                except Exception:
                    global_payload = None

            self.model, val_metrics = train_local_model_with_binned_payload(
                model=self.model,
                x_train=self.partition.x_train,
                y_train=self.partition.y_train,
                x_val=self.partition.x_val,
                y_val=self.partition.y_val,
                config=config,
                local_epochs=local_epochs,
                seed=config.random_seed + server_round + self.partition.cluster_id,
                global_payload=global_payload,
                payload_config=payload_config,
            )

            probs = predict_probabilities(self.model, self.partition.x_train)
            payload = build_binned_probability_payload(probs, payload_config)
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
            self.state["last_val_metrics"] = {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}

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
                "probability_payload": json.dumps(payload, ensure_ascii=True),
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
        return BinnedPayloadClient(partition, state).to_client()

    return client_fn


class BinnedProbabilityPayloadStrategy:
    def __init__(self, payload_name: str):
        self.payload_config = BIN_PAYLOAD_CONFIGS[payload_name]
        self.latest_payload: dict[str, Any] | None = None

    def aggregate_payloads(self, results) -> dict[str, Any] | None:
        payloads = []
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
        self.latest_payload = aggregate_binned_probability_payloads(payloads, self.payload_config)
        return self.latest_payload
