from __future__ import annotations

import copy

import numpy as np
import torch

from config import ModelConfig
from metrics import compute_binary_metrics


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


def create_model(input_dim: int, config: ModelConfig) -> TabularMLP:
    return TabularMLP(
        input_dim=input_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
        use_batchnorm=config.use_batchnorm,
    )


def create_teacher_model(input_dim: int, config: ModelConfig) -> TabularMLP:
    return create_model(input_dim, config)


def get_model_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy().copy() for tensor in model.state_dict().values()]


def set_model_parameters(
    model: torch.nn.Module,
    parameters: list[np.ndarray],
    keep_local_head: bool = False,
) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    incoming = copy.deepcopy(parameters)
    if keep_local_head and len(keys) >= 2:
        incoming[-2:] = [state_dict[keys[-2]].detach().cpu().numpy(), state_dict[keys[-1]].detach().cpu().numpy()]
    new_state = {}
    for key, array in zip(keys, incoming):
        new_state[key] = torch.tensor(array, dtype=state_dict[key].dtype)
    model.load_state_dict(new_state, strict=True)


def predict_probabilities(model: torch.nn.Module, x_values: np.ndarray) -> np.ndarray:
    model.eval()
    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(x_tensor)).cpu().numpy()
    return probs.astype(float)


def evaluate_model(model: torch.nn.Module, x_values: np.ndarray, y_true: np.ndarray) -> dict[str, float | int]:
    probs = predict_probabilities(model, x_values)
    predictions = (probs >= 0.5).astype(int)
    return compute_binary_metrics(y_true, predictions, probs)


def train_local_model(
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: any,
    local_epochs: int,
    seed: int,
    teacher_model: torch.nn.Module | None = None,
) -> tuple[torch.nn.Module, dict[str, float | int]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    positives = max(int(y_train.sum()), 1)
    negatives = max(int((1 - y_train).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)

    if teacher_model is not None:
        teacher_model.eval()

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0

    batch_size = min(config.model.batch_size, max(len(x_train), 1))
    for epoch in range(local_epochs):
        model.train()
        permutation = torch.randperm(x_train_tensor.size(0))
        for start in range(0, x_train_tensor.size(0), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = x_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(batch_x)
                student_logits_two = torch.stack([-logits, logits], dim=1)
                teacher_logits_two = torch.stack([-teacher_logits, teacher_logits], dim=1)
                temperature = float(config.distillation_temperature)
                student_log_probs = torch.log_softmax(student_logits_two / temperature, dim=1)
                teacher_probs = torch.softmax(teacher_logits_two / temperature, dim=1)
                distillation_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)
                alpha = float(config.distillation_alpha)
                loss = alpha * loss + (1.0 - alpha) * distillation_loss

            loss.backward()
            optimizer.step()

        metrics = evaluate_model(model, x_val, y_val) if len(x_val) > 0 else evaluate_model(model, x_train, y_train)
        if float(metrics["f1"]) >= best_f1:
            best_f1 = float(metrics["f1"])
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final_metrics = evaluate_model(model, x_val, y_val) if len(x_val) > 0 else evaluate_model(model, x_train, y_train)
    return model, final_metrics
