from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np

from payload_configs import PAYLOAD_CONFIGS, PayloadConfig


def require_flwr():
    try:
        import flwr as fl
    except ImportError as exc:
        raise ImportError("Flower is required for probability payload experiments. Install with `pip install flwr`.") from exc
    return fl


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "count": float(len(values)),
    }


def build_probability_payload(probs: np.ndarray, labels: np.ndarray, payload_config: PayloadConfig) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    payload: dict[str, float] = {
        "payload_type": payload_config.name,
        "mean": float(np.mean(probs)) if len(probs) else 0.5,
        "count": float(len(probs)),
    }
    if payload_config.uses_std_confidence:
        payload["std"] = float(np.std(probs, ddof=0)) if len(probs) else 0.0

    if payload_config.uses_class_conditioning:
        pos_stats = _safe_stats(probs[labels == 1])
        neg_stats = _safe_stats(probs[labels == 0])
        for prefix, stats in (("pos", pos_stats), ("neg", neg_stats)):
            if not stats:
                continue
            payload[f"{prefix}_mean"] = stats["mean"]
            payload[f"{prefix}_count"] = stats["count"]
            if payload_config.uses_std_confidence:
                payload[f"{prefix}_std"] = stats["std"]
            if payload_config.uses_quantiles:
                payload[f"{prefix}_q25"] = stats["q25"]
                payload[f"{prefix}_q50"] = stats["q50"]
                payload[f"{prefix}_q75"] = stats["q75"]
    return payload


def aggregate_probability_payloads(
    weighted_payloads: list[tuple[int, dict[str, Any]]],
    payload_config: PayloadConfig,
    weighted: bool = True,
) -> dict[str, float] | None:
    if not weighted_payloads:
        return None

    def weight_for(payload: dict[str, Any], prefix: str | None = None) -> float:
        if not weighted:
            return 1.0
        if prefix is None:
            return max(float(payload.get("count", 0.0)), 0.0)
        return max(float(payload.get(f"{prefix}_count", 0.0)), 0.0)

    def weighted_mean(key: str, prefix: str | None = None) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for _, payload in weighted_payloads:
            value = _as_float(payload.get(key))
            if value is None:
                continue
            w = weight_for(payload, prefix)
            if w <= 0:
                continue
            numerator += w * value
            denominator += w
        if denominator <= 0:
            return None
        return numerator / denominator

    def pooled_std(mean_key: str, std_key: str, prefix: str | None = None) -> float | None:
        global_mean = weighted_mean(mean_key, prefix)
        if global_mean is None:
            return None
        second_moment = 0.0
        denominator = 0.0
        for _, payload in weighted_payloads:
            mean_value = _as_float(payload.get(mean_key))
            std_value = _as_float(payload.get(std_key), 0.0)
            if mean_value is None or std_value is None:
                continue
            w = weight_for(payload, prefix)
            if w <= 0:
                continue
            second_moment += w * (std_value**2 + mean_value**2)
            denominator += w
        if denominator <= 0:
            return None
        variance = max(second_moment / denominator - global_mean**2, 0.0)
        return float(np.sqrt(variance))

    aggregate: dict[str, float] = {"payload_type": payload_config.name}
    total_count = sum(max(float(payload.get("count", 0.0)), 0.0) for _, payload in weighted_payloads)
    aggregate["count"] = float(total_count)

    mean_value = weighted_mean("mean")
    if mean_value is not None:
        aggregate["mean"] = mean_value
    if payload_config.uses_std_confidence:
        std_value = pooled_std("mean", "std")
        if std_value is not None:
            aggregate["std"] = std_value

    if payload_config.uses_class_conditioning:
        for prefix in ("pos", "neg"):
            count_value = sum(max(float(payload.get(f"{prefix}_count", 0.0)), 0.0) for _, payload in weighted_payloads)
            aggregate[f"{prefix}_count"] = float(count_value)
            class_mean = weighted_mean(f"{prefix}_mean", prefix)
            if class_mean is not None:
                aggregate[f"{prefix}_mean"] = class_mean
            if payload_config.uses_std_confidence:
                class_std = pooled_std(f"{prefix}_mean", f"{prefix}_std", prefix)
                if class_std is not None:
                    aggregate[f"{prefix}_std"] = class_std
            if payload_config.uses_quantiles:
                for q_key in ("q25", "q50", "q75"):
                    q_value = weighted_mean(f"{prefix}_{q_key}", prefix)
                    if q_value is not None:
                        aggregate[f"{prefix}_{q_key}"] = q_value
    return aggregate


def _probability_to_logit(probability: float) -> float:
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(probability / (1.0 - probability)))


def _teacher_targets(batch_y, payload: dict[str, Any], payload_config: PayloadConfig):
    import torch

    default_mean = _as_float(payload.get("mean"), 0.5) or 0.5
    if not payload_config.uses_class_conditioning:
        probability = _as_float(payload.get("q50"), default_mean) if payload_config.uses_quantiles else default_mean
        return torch.full_like(batch_y, _probability_to_logit(probability), dtype=torch.float32)

    pos_probability = _as_float(payload.get("pos_q50"), None) if payload_config.uses_quantiles else None
    neg_probability = _as_float(payload.get("neg_q50"), None) if payload_config.uses_quantiles else None
    pos_probability = pos_probability if pos_probability is not None else (_as_float(payload.get("pos_mean"), default_mean) or default_mean)
    neg_probability = neg_probability if neg_probability is not None else (_as_float(payload.get("neg_mean"), default_mean) or default_mean)

    pos_logit = _probability_to_logit(pos_probability)
    neg_logit = _probability_to_logit(neg_probability)
    return torch.where(
        batch_y >= 0.5,
        torch.full_like(batch_y, pos_logit, dtype=torch.float32),
        torch.full_like(batch_y, neg_logit, dtype=torch.float32),
    )


def _confidence_weight(payload: dict[str, Any], payload_config: PayloadConfig) -> float:
    if not payload_config.uses_std_confidence:
        return 1.0
    std_values = []
    if payload_config.uses_class_conditioning:
        for key in ("pos_std", "neg_std"):
            value = _as_float(payload.get(key))
            if value is not None:
                std_values.append(value)
    else:
        value = _as_float(payload.get("std"))
        if value is not None:
            std_values.append(value)
    if not std_values:
        return 1.0
    # Probabilities have maximum std close to 0.5. Lower dispersion gives a stronger KD signal.
    avg_std = float(np.mean(std_values))
    return float(np.clip(1.0 - 2.0 * avg_std, 0.2, 1.0))


def train_local_model_with_probability_payload(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config,
    local_epochs: int,
    seed: int,
    global_payload: dict[str, Any] | None,
    payload_config: PayloadConfig,
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
                teacher_logits = _teacher_targets(batch_y, global_payload, payload_config).to(logits.device)
                student_logits_two = torch.stack([-logits, logits], dim=1)
                teacher_logits_two = torch.stack([-teacher_logits, teacher_logits], dim=1)
                temperature = float(config.distillation_temperature)
                student_log_probs = torch.log_softmax(student_logits_two / temperature, dim=1)
                teacher_probs = torch.softmax(teacher_logits_two / temperature, dim=1)
                kd_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)
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


def _serialize_payload(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True)


def make_probability_payload_client_fn(setup, config, payload_name: str, client_states: dict | None = None):
    fl = require_flwr()
    payload_config = PAYLOAD_CONFIGS[payload_name]

    class ProbabilityPayloadClient(fl.client.NumPyClient):
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

            self.model, val_metrics = train_local_model_with_probability_payload(
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
            payload = build_probability_payload(probs, self.partition.y_train, payload_config)

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
        return ProbabilityPayloadClient(partition, state).to_client()

    return client_fn


class ProbabilityPayloadStrategy:
    def __init__(self, payload_name: str):
        self.payload_config = PAYLOAD_CONFIGS[payload_name]
        self.latest_payload: dict[str, float] | None = None

    def aggregate_payloads(self, results, weighted: bool = True) -> dict[str, float] | None:
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
        self.latest_payload = aggregate_probability_payloads(payloads, self.payload_config, weighted=weighted)
        return self.latest_payload
