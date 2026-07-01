from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np

from prototype_strategy_configs import PROTOTYPE_STRATEGIES, PrototypeStrategyConfig


def require_flwr():
    try:
        import flwr as fl
    except ImportError as exc:
        raise ImportError("Flower is required for class-prototype experiments. Install with `pip install flwr`.") from exc
    return fl


def _serialize_payload(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True)


def _extract_hidden_values(model, x_values: np.ndarray) -> np.ndarray:
    import torch

    output_layer = model.network[-1]
    hidden_dim = int(getattr(output_layer, "in_features", 0))
    if len(x_values) == 0:
        return np.empty((0, hidden_dim), dtype=float)

    hidden_layers = torch.nn.Sequential(*list(model.network.children())[:-1])
    hidden_layers.eval()
    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    with torch.no_grad():
        hidden = hidden_layers(x_tensor).cpu().numpy()
    return hidden.astype(float)


def _kmeans_prototypes(values: np.ndarray, k: int) -> tuple[list[list[float]], list[int]]:
    if len(values) == 0:
        return [], []
    if len(values) < k:
        mean = np.mean(values, axis=0).astype(float).tolist()
        return [mean for _ in range(k)], [int(len(values))] + [0 for _ in range(k - 1)]
    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=k, n_init=5, random_state=0)
        labels = kmeans.fit_predict(values)
        centers = kmeans.cluster_centers_.astype(float).tolist()
        counts = [int(np.sum(labels == idx)) for idx in range(k)]
        return centers, counts
    except Exception:
        mean = np.mean(values, axis=0).astype(float).tolist()
        return [mean for _ in range(k)], [int(len(values))] + [0 for _ in range(k - 1)]


def build_prototype_payload(model, x_train: np.ndarray, y_train: np.ndarray, strategy_config: PrototypeStrategyConfig) -> dict[str, Any]:
    hidden_values = _extract_hidden_values(model, x_train)
    pos_values = hidden_values[y_train == 1]
    neg_values = hidden_values[y_train == 0]
    k = strategy_config.num_prototypes_per_class
    pos_protos, pos_counts = _kmeans_prototypes(pos_values, k)
    neg_protos, neg_counts = _kmeans_prototypes(neg_values, k)
    return {
        "strategy": strategy_config.name,
        "proto_dim": int(hidden_values.shape[1]) if hidden_values.ndim == 2 else 0,
        "num_prototypes_per_class": int(k),
        "pos_prototypes": pos_protos,
        "neg_prototypes": neg_protos,
        "pos_counts": pos_counts,
        "neg_counts": neg_counts,
        "count": int(len(y_train)),
    }


def aggregate_prototype_payloads(payloads: list[tuple[int, dict[str, Any]]], strategy_config: PrototypeStrategyConfig) -> dict[str, Any] | None:
    if not payloads:
        return None
    k = strategy_config.num_prototypes_per_class

    def aggregate_class(proto_key: str, count_key: str) -> tuple[list[list[float]], list[int]]:
        slots: list[list[np.ndarray]] = [[] for _ in range(k)]
        weights: list[list[float]] = [[] for _ in range(k)]
        for _, payload in payloads:
            protos = payload.get(proto_key, [])
            counts = payload.get(count_key, [])
            for idx in range(min(k, len(protos))):
                proto = np.asarray(protos[idx], dtype=float)
                if proto.ndim != 1 or proto.size == 0:
                    continue
                count = float(counts[idx]) if idx < len(counts) else 0.0
                if count <= 0:
                    continue
                slots[idx].append(proto)
                weights[idx].append(count)
        result_protos: list[list[float]] = []
        result_counts: list[int] = []
        fallback = None
        for idx in range(k):
            if slots[idx]:
                w = np.asarray(weights[idx], dtype=float)
                stacked = np.vstack(slots[idx])
                proto = np.average(stacked, axis=0, weights=w)
                fallback = proto
                result_protos.append(proto.astype(float).tolist())
                result_counts.append(int(np.sum(w)))
            elif fallback is not None:
                result_protos.append(fallback.astype(float).tolist())
                result_counts.append(0)
            else:
                result_protos.append([])
                result_counts.append(0)
        return result_protos, result_counts

    pos_protos, pos_counts = aggregate_class("pos_prototypes", "pos_counts")
    neg_protos, neg_counts = aggregate_class("neg_prototypes", "neg_counts")
    proto_dim = 0
    for protos in (pos_protos, neg_protos):
        for proto in protos:
            if proto:
                proto_dim = len(proto)
                break
        if proto_dim:
            break
    return {
        "strategy": strategy_config.name,
        "proto_dim": int(proto_dim),
        "num_prototypes_per_class": int(k),
        "pos_prototypes": pos_protos,
        "neg_prototypes": neg_protos,
        "pos_counts": pos_counts,
        "neg_counts": neg_counts,
        "count": int(sum(n for n, _ in payloads)),
    }


def _prototype_weight(server_round: int, strategy_config: PrototypeStrategyConfig) -> float:
    if server_round <= strategy_config.warmup_rounds:
        return 0.0
    if strategy_config.warmup_rounds <= 0:
        return strategy_config.prototype_weight_end
    progress = min(max((server_round - strategy_config.warmup_rounds) / 10.0, 0.0), 1.0)
    return strategy_config.prototype_weight_start + progress * (strategy_config.prototype_weight_end - strategy_config.prototype_weight_start)


def _train_prototype_alignment(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    payload: dict[str, Any],
    config,
    seed: int,
    server_round: int,
    strategy_config: PrototypeStrategyConfig,
) -> None:
    import torch
    import torch.nn.functional as F

    weight = _prototype_weight(server_round, strategy_config)
    if weight <= 0.0 or len(x_train) == 0:
        return

    hidden_layers = torch.nn.Sequential(*list(model.network.children())[:-1])
    output_layer = model.network[-1]
    hidden_dim = int(getattr(output_layer, "in_features", 0))
    if hidden_dim <= 0:
        return

    def load_protos(key: str):
        protos = []
        for proto in payload.get(key, []):
            if isinstance(proto, list) and len(proto) == hidden_dim:
                protos.append(torch.tensor(proto, dtype=torch.float32))
        return protos

    pos_protos = load_protos("pos_prototypes")
    neg_protos = load_protos("neg_prototypes")
    if not pos_protos and not neg_protos:
        return

    torch.manual_seed(seed)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)
    batch_size = min(config.model.batch_size, max(len(x_train), 1))
    model.train()

    pos_stack = torch.stack(pos_protos) if pos_protos else None
    neg_stack = torch.stack(neg_protos) if neg_protos else None

    permutation = torch.randperm(x_tensor.size(0))
    for start in range(0, x_tensor.size(0), batch_size):
        indices = permutation[start : start + batch_size]
        batch_x = x_tensor[indices]
        batch_y = y_tensor[indices]
        optimizer.zero_grad()
        hidden = hidden_layers(batch_x)
        losses = []

        if strategy_config.alignment_loss == "cosine_margin":
            margin = 0.25
            hidden_norm = F.normalize(hidden, dim=1)
            if pos_stack is not None and neg_stack is not None:
                pos_proto = F.normalize(pos_stack[0].to(hidden.device), dim=0)
                neg_proto = F.normalize(neg_stack[0].to(hidden.device), dim=0)
                sim_pos = hidden_norm @ pos_proto
                sim_neg = hidden_norm @ neg_proto
                target_sim = torch.where(batch_y == 1, sim_pos, sim_neg)
                opposite_sim = torch.where(batch_y == 1, sim_neg, sim_pos)
                losses.append(F.relu(margin - target_sim + opposite_sim).mean())
                losses.append((1.0 - target_sim).mean())
        elif strategy_config.alignment_loss == "nearest_mse":
            for class_value, proto_stack in ((1, pos_stack), (0, neg_stack)):
                if proto_stack is None:
                    continue
                mask = batch_y == class_value
                if bool(mask.any()):
                    class_hidden = hidden[mask]
                    proto_stack_device = proto_stack.to(hidden.device)
                    distances = torch.cdist(class_hidden, proto_stack_device)
                    nearest = proto_stack_device[torch.argmin(distances, dim=1)]
                    losses.append(F.mse_loss(class_hidden, nearest))
        else:
            for class_value, proto_stack in ((1, pos_stack), (0, neg_stack)):
                if proto_stack is None:
                    continue
                mask = batch_y == class_value
                if bool(mask.any()):
                    losses.append(F.mse_loss(hidden[mask].mean(dim=0), proto_stack[0].to(hidden.device)))

        if not losses:
            continue
        loss = weight * sum(losses) / len(losses)
        loss.backward()
        optimizer.step()


def make_advanced_prototype_client_fn(setup, config, strategy_name: str, client_states: dict | None = None):
    fl = require_flwr()
    strategy_config = PROTOTYPE_STRATEGIES[strategy_name]

    class AdvancedPrototypeClient(fl.client.NumPyClient):
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
            from modeling import evaluate_model, get_model_parameters, train_local_model

            server_round = int(config_dict.get("server_round", 0))
            local_epochs = int(config_dict.get("local_epochs", config.local_epochs))
            raw_payload = config_dict.get("global_prototype_payload")
            global_payload = None
            if raw_payload:
                try:
                    global_payload = json.loads(str(raw_payload))
                except Exception:
                    global_payload = None

            self.model, val_metrics = train_local_model(
                model=self.model,
                x_train=self.partition.x_train,
                y_train=self.partition.y_train,
                x_val=self.partition.x_val,
                y_val=self.partition.y_val,
                config=config,
                local_epochs=local_epochs,
                seed=config.random_seed + server_round + self.partition.cluster_id,
                teacher_model=None,
            )

            if global_payload is not None:
                _train_prototype_alignment(
                    self.model,
                    self.partition.x_train,
                    self.partition.y_train,
                    global_payload,
                    config,
                    seed=config.random_seed + server_round + self.partition.cluster_id + 30_000,
                    server_round=server_round,
                    strategy_config=strategy_config,
                )
                val_metrics = evaluate_model(self.model, self.partition.x_val, self.partition.y_val)

            payload = build_prototype_payload(self.model, self.partition.x_train, self.partition.y_train, strategy_config)
            current_params = get_model_parameters(self.model)
            if config.local_model_selection:
                current_f1 = float(val_metrics.get("f1", 0.0))
                if current_f1 >= float(self.state.get("best_local_f1", -1.0)):
                    self.state["best_local_f1"] = current_f1
                    self.state["best_local_params"] = copy.deepcopy(current_params)
                elif self.state.get("best_local_params") is not None:
                    current_params = copy.deepcopy(self.state["best_local_params"])
            self.state["current_local_params"] = current_params
            self.state["last_val_metrics"] = {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}
            self.state["last_prototype_payload"] = copy.deepcopy(payload)

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
                "prototype_payload": _serialize_payload(payload),
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
        return AdvancedPrototypeClient(partition, state).to_client()

    return client_fn


class AdvancedPrototypeStrategy:
    def __init__(self, strategy_name: str):
        self.strategy_config = PROTOTYPE_STRATEGIES[strategy_name]
        self.latest_payload: dict[str, Any] | None = None

    def aggregate_payloads(self, results) -> dict[str, Any] | None:
        payloads = []
        for _, fit_res in results:
            metrics = getattr(fit_res, "metrics", {}) or {}
            raw_payload = metrics.get("prototype_payload")
            if raw_payload is None:
                continue
            try:
                payload = json.loads(str(raw_payload))
            except Exception:
                continue
            payloads.append((int(getattr(fit_res, "num_examples", 0)), payload))
        self.latest_payload = aggregate_prototype_payloads(payloads, self.strategy_config)
        return self.latest_payload
