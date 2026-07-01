from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np


def require_flwr():
    try:
        import flwr as fl
    except ImportError as exc:
        raise ImportError("Flower is required for KD-without-weights experiments. Install with `pip install flwr`.") from exc
    return fl


def _serialize_payload(payload: Any) -> str:
    try:
        return json.dumps(payload)
    except Exception:
        # Fallback: convert numpy arrays to lists
        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        return json.dumps(_serialize_numpy(payload, _convert))


def _serialize_numpy(obj, conv):
    if isinstance(obj, dict):
        return {k: _serialize_numpy(v, conv) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_numpy(v, conv) for v in obj]
    try:
        return conv(obj)
    except Exception:
        return str(obj)


def _is_numeric_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, (int, float)) for item in value)


def _extract_hidden_values(model, x_values: np.ndarray) -> np.ndarray:
    import torch

    if len(x_values) == 0:
        hidden_dim = getattr(model.network[-1], "in_features", 0)
        return np.empty((0, hidden_dim), dtype=float)

    hidden_layers = torch.nn.Sequential(*list(model.network.children())[:-1])
    hidden_layers.eval()
    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    with torch.no_grad():
        hidden = hidden_layers(x_tensor).cpu().numpy()
    return hidden.astype(float)


def _train_hidden_alignment(model, x_train: np.ndarray, target_hidden_mean: list[float], config, seed: int) -> None:
    import torch

    if len(x_train) == 0 or not target_hidden_mean:
        return

    hidden_layers = torch.nn.Sequential(*list(model.network.children())[:-1])
    output_layer = model.network[-1]
    if not hasattr(output_layer, "in_features") or int(output_layer.in_features) != len(target_hidden_mean):
        return

    torch.manual_seed(seed)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    target = torch.tensor(target_hidden_mean, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)
    batch_size = min(config.model.batch_size, max(len(x_train), 1))

    model.train()
    permutation = torch.randperm(x_tensor.size(0))
    for start in range(0, x_tensor.size(0), batch_size):
        indices = permutation[start : start + batch_size]
        batch_x = x_tensor[indices]
        optimizer.zero_grad()
        hidden_mean = hidden_layers(batch_x).mean(dim=0)
        loss = torch.nn.functional.mse_loss(hidden_mean, target)
        loss.backward()
        optimizer.step()


def _train_prototype_alignment(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    target_pos_mean: list[float],
    target_neg_mean: list[float],
    config,
    seed: int,
) -> None:
    import torch

    if len(x_train) == 0 or (not target_pos_mean and not target_neg_mean):
        return

    hidden_layers = torch.nn.Sequential(*list(model.network.children())[:-1])
    output_layer = model.network[-1]
    hidden_dim = int(getattr(output_layer, "in_features", 0))
    if hidden_dim <= 0:
        return

    target_pos = None
    target_neg = None
    if target_pos_mean and len(target_pos_mean) == hidden_dim:
        target_pos = torch.tensor(target_pos_mean, dtype=torch.float32)
    if target_neg_mean and len(target_neg_mean) == hidden_dim:
        target_neg = torch.tensor(target_neg_mean, dtype=torch.float32)
    if target_pos is None and target_neg is None:
        return

    torch.manual_seed(seed)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)
    batch_size = min(config.model.batch_size, max(len(x_train), 1))

    model.train()
    permutation = torch.randperm(x_tensor.size(0))
    for start in range(0, x_tensor.size(0), batch_size):
        indices = permutation[start : start + batch_size]
        batch_x = x_tensor[indices]
        batch_y = y_tensor[indices]

        optimizer.zero_grad()
        hidden = hidden_layers(batch_x)
        losses = []
        if target_pos is not None:
            pos_mask = batch_y == 1
            if bool(pos_mask.any()):
                losses.append(torch.nn.functional.mse_loss(hidden[pos_mask].mean(dim=0), target_pos))
        if target_neg is not None:
            neg_mask = batch_y == 0
            if bool(neg_mask.any()):
                losses.append(torch.nn.functional.mse_loss(hidden[neg_mask].mean(dim=0), target_neg))
        if not losses:
            continue
        loss = sum(losses) / len(losses)
        loss.backward()
        optimizer.step()


def make_client_fn_no_weights(setup, config, mode: str = "soft_labels", client_states: dict | None = None):
    fl = require_flwr()

    class ConstantSoftLabelTeacher:
        def __init__(self, positive_probability: float):
            probability = float(np.clip(positive_probability, 1e-6, 1.0 - 1e-6))
            self.logit = float(np.log(probability / (1.0 - probability)))

        def eval(self):
            return self

        def __call__(self, batch_x):
            import torch

            return torch.full((batch_x.shape[0],), self.logit, dtype=batch_x.dtype, device=batch_x.device)

    class KDNumPyClient(fl.client.NumPyClient):
        def __init__(self, partition, state: dict[str, Any]):
            self.partition = partition
            self.state = state
            from modeling import create_model, create_teacher_model, set_model_parameters

            self.model = create_model(len(setup.feature_columns), config.model)
            self.teacher_model = create_teacher_model(len(setup.feature_columns), config.model)
            if self.state.get("current_local_params") is not None:
                set_model_parameters(self.model, self.state["current_local_params"], keep_local_head=bool(config.personalized_head))

        def get_parameters(self, config=None):
            return []

        def fit(self, parameters, config_dict):
            from modeling import train_local_model, predict_probabilities, get_model_parameters, evaluate_model

            def _config_get(obj, key, default=None):
                if obj is None:
                    return default
                if isinstance(obj, dict):
                    return obj.get(key, default)
                if hasattr(obj, "get"):
                    try:
                        return obj.get(key, default)
                    except Exception:
                        pass
                return getattr(obj, key, default)

            local_epochs = int(_config_get(config_dict, "local_epochs", config.local_epochs))
            server_round = int(_config_get(config_dict, "server_round", 0))
            global_soft_pos = _config_get(config_dict, "global_soft_pos", None)
            global_hidden_mean_raw = _config_get(config_dict, "global_hidden_mean", None)
            global_proto_pos_mean_raw = _config_get(config_dict, "global_proto_pos_mean", None)
            global_proto_neg_mean_raw = _config_get(config_dict, "global_proto_neg_mean", None)
            teacher_model = None
            if mode == "soft_labels" and global_soft_pos is not None and config.teacher_student_distillation:
                teacher_model = ConstantSoftLabelTeacher(float(global_soft_pos))
            self.model, val_metrics = train_local_model(
                model=self.model,
                x_train=self.partition.x_train,
                y_train=self.partition.y_train,
                x_val=self.partition.x_val,
                y_val=self.partition.y_val,
                config=config,
                local_epochs=local_epochs,
                seed=config.random_seed + server_round + self.partition.cluster_id,
                teacher_model=teacher_model,
            )

            if mode == "hidden_states" and global_hidden_mean_raw is not None:
                try:
                    global_hidden_mean = json.loads(str(global_hidden_mean_raw))
                except Exception:
                    global_hidden_mean = []
                if _is_numeric_list(global_hidden_mean):
                    _train_hidden_alignment(
                        self.model,
                        self.partition.x_train,
                        global_hidden_mean,
                        config,
                        seed=config.random_seed + server_round + self.partition.cluster_id + 10_000,
                    )
                    val_metrics = evaluate_model(self.model, self.partition.x_val, self.partition.y_val)

            if mode == "class_prototype" and (global_proto_pos_mean_raw is not None or global_proto_neg_mean_raw is not None):
                try:
                    global_proto_pos_mean = json.loads(str(global_proto_pos_mean_raw)) if global_proto_pos_mean_raw is not None else []
                except Exception:
                    global_proto_pos_mean = []
                try:
                    global_proto_neg_mean = json.loads(str(global_proto_neg_mean_raw)) if global_proto_neg_mean_raw is not None else []
                except Exception:
                    global_proto_neg_mean = []
                if _is_numeric_list(global_proto_pos_mean) or _is_numeric_list(global_proto_neg_mean):
                    _train_prototype_alignment(
                        self.model,
                        self.partition.x_train,
                        self.partition.y_train,
                        global_proto_pos_mean if _is_numeric_list(global_proto_pos_mean) else [],
                        global_proto_neg_mean if _is_numeric_list(global_proto_neg_mean) else [],
                        config,
                        seed=config.random_seed + server_round + self.partition.cluster_id + 20_000,
                    )
                    val_metrics = evaluate_model(self.model, self.partition.x_val, self.partition.y_val)

            payload = {}
            if mode == "soft_labels":
                probs = predict_probabilities(self.model, self.partition.x_train)
                payload = {
                    "mean_soft_pos": float(np.mean(probs)),
                    "count": int(len(probs)),
                }
            elif mode == "hidden_states":
                hidden_values = _extract_hidden_values(self.model, self.partition.x_train)
                hidden_mean = np.mean(hidden_values, axis=0).astype(float).tolist() if len(hidden_values) else []
                payload = {
                    "hidden_mean": hidden_mean,
                    "hidden_dim": int(len(hidden_mean)),
                    "count": int(len(self.partition.y_train)),
                }
            elif mode == "class_prototype":
                hidden_values = _extract_hidden_values(self.model, self.partition.x_train)
                pos_mask = self.partition.y_train == 1
                neg_mask = self.partition.y_train == 0
                pos_hidden = hidden_values[pos_mask]
                neg_hidden = hidden_values[neg_mask]
                payload = {
                    "proto_pos_mean": (np.mean(pos_hidden, axis=0).astype(float).tolist() if len(pos_hidden) else []),
                    "proto_neg_mean": (np.mean(neg_hidden, axis=0).astype(float).tolist() if len(neg_hidden) else []),
                    "proto_dim": int(hidden_values.shape[1]) if hidden_values.ndim == 2 else 0,
                    "count": int(len(self.partition.y_train)),
                    "pos_count": int(np.sum(pos_mask)),
                    "neg_count": int(np.sum(neg_mask)),
                }
            else:
                payload = {"note": "unknown_mode"}

            # Store in state for later inspection
            self.state["last_kd_payload"] = copy.deepcopy(payload)
            self.state["last_val_metrics"] = {k: float(v) for k, v in val_metrics.items()}
            current_params = get_model_parameters(self.model)
            current_f1 = float(val_metrics.get("f1", 0.0))
            if config.local_model_selection:
                if current_f1 >= float(self.state.get("best_local_f1", -1.0)):
                    self.state["best_local_f1"] = current_f1
                    self.state["best_local_params"] = copy.deepcopy(current_params)
                elif self.state.get("best_local_params") is not None:
                    current_params = copy.deepcopy(self.state["best_local_params"])
                    self.state["best_local_f1"] = float(self.state.get("best_local_f1", current_f1))
            self.state["current_local_params"] = current_params

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
                "kd_payload": _serialize_payload(payload),
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
            return float(loss), int(len(self.partition.y_val)), {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}

    ordered_client_ids = sorted(setup.train_clients.keys())

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
                raise KeyError(f"Unknown client id mapping for '{raw_id}'")

        partition = setup.train_clients[client_id]
        if client_states is not None:
            state = client_states.setdefault(client_id, {})
        else:
            state = {}
        return KDNumPyClient(partition, state).to_client()

    return client_fn


class KDNoWeightsStrategy:
    """Minimal strategy wrapper that aggregates KD payloads from client metrics.

    This is not a full `fl.server.strategy.Strategy` subclass but provides the
    aggregation logic that can be plugged into a custom Flower strategy or used
    post-hoc from simulation history.
    """

    def __init__(self):
        self.latest_aggregated_payload = None

    def aggregate_payloads(self, results, weighted: bool = True, clustered: bool = False):
        # `results` is list of tuples (client_fit_res, fit_res)
        payloads = []
        for _, fit_res in results:
            metrics = getattr(fit_res, "metrics", {})
            raw = metrics.get("kd_payload")
            if raw is None:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"raw": raw}
            payloads.append((int(getattr(fit_res, "num_examples", 0)), payload))

        if not payloads:
            return None

        def aggregate_group(group_payloads):
            agg: dict[str, Any] = {}
            totals: dict[str, float] = {}
            if weighted:
                weights = [n for n, _ in group_payloads]
            else:
                weights = [1 for _ in group_payloads]
            total = float(sum(weights))
            for weight, (_, p) in zip(weights, group_payloads):
                if not isinstance(p, dict):
                    continue
                for k, v in p.items():
                    if k == "cluster_id":
                        continue
                    key_weight = float(weight)
                    if k == "proto_pos_mean":
                        key_weight = float(p.get("pos_count", key_weight))
                    elif k == "proto_neg_mean":
                        key_weight = float(p.get("neg_count", key_weight))
                    if key_weight <= 0:
                        continue
                    if isinstance(v, (int, float)):
                        if k in {"count", "pos_count", "neg_count"}:
                            agg.setdefault(k, 0.0)
                            agg[k] += float(v)
                            totals[k] = 1.0
                            continue
                        agg.setdefault(k, 0.0)
                        agg[k] += key_weight * float(v)
                        totals[k] = totals.get(k, 0.0) + key_weight
                    elif _is_numeric_list(v):
                        if k not in agg:
                            agg[k] = [0.0 for _ in v]
                        if isinstance(agg[k], list) and len(agg[k]) == len(v):
                            for idx, item in enumerate(v):
                                agg[k][idx] += key_weight * float(item)
                            totals[k] = totals.get(k, 0.0) + key_weight
            if total > 0:
                for k in list(agg.keys()):
                    key_total = totals.get(k, total)
                    if key_total <= 0:
                        continue
                    if isinstance(agg[k], list):
                        agg[k] = [item / key_total for item in agg[k]]
                    else:
                        agg[k] = agg[k] / key_total
            return agg, total

        if not clustered:
            agg, _ = aggregate_group(payloads)
            self.latest_aggregated_payload = agg
            return agg

        grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for _, fit_res in results:
            metrics = getattr(fit_res, "metrics", {})
            raw = metrics.get("kd_payload")
            if raw is None:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"raw": raw}
            grouped.setdefault(int(metrics.get("cluster_id", 0)), []).append((int(getattr(fit_res, "num_examples", 0)), payload))

        subgroup_payloads = []
        for cluster_id, group in grouped.items():
            cluster_payload, group_weight = aggregate_group(group)
            cluster_payload["cluster_id"] = float(cluster_id)
            subgroup_payloads.append((group_weight, cluster_payload))

        agg, _ = aggregate_group(subgroup_payloads)
        self.latest_aggregated_payload = agg
        return agg


def example_usage_stub():
    """Demonstrate how to construct a no-weights client function and aggregate payloads.

    This function is a lightweight example and does not start a Flower simulation.
    Use it as a template for wiring into `run_experiment.py` style runners.
    """
    print("Read this module's docstring for usage. Instantiate KD clients via make_client_fn_no_weights.")
