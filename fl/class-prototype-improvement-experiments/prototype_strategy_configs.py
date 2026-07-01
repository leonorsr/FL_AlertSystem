from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrototypeStrategyConfig:
    name: str
    description: str
    num_prototypes_per_class: int
    alignment_loss: str
    warmup_rounds: int = 0
    prototype_weight_start: float = 1.0
    prototype_weight_end: float = 1.0

    @property
    def scalar_count(self) -> int:
        # Each prototype is 64-dimensional in the current MLP representation.
        # Counts are one scalar per class/prototype.
        return 2 * self.num_prototypes_per_class * 64 + 2 * self.num_prototypes_per_class


PROTOTYPE_STRATEGIES: dict[str, PrototypeStrategyConfig] = {
    "multi_proto_k2": PrototypeStrategyConfig(
        name="multi_proto_k2",
        description="Two prototypes per class, with each local embedding aligned to the nearest same-class prototype.",
        num_prototypes_per_class=2,
        alignment_loss="nearest_mse",
    ),
    "cosine_margin_proto": PrototypeStrategyConfig(
        name="cosine_margin_proto",
        description="One prototype per class, L2-normalized embeddings, and cosine margin separation from the opposite class.",
        num_prototypes_per_class=1,
        alignment_loss="cosine_margin",
    ),
    "warmup_proto_schedule": PrototypeStrategyConfig(
        name="warmup_proto_schedule",
        description="One prototype per class, KD disabled for the first 5 rounds and then gradually increased.",
        num_prototypes_per_class=1,
        alignment_loss="mse",
        warmup_rounds=5,
        prototype_weight_start=0.25,
        prototype_weight_end=1.0,
    ),
}


DEFAULT_PROTOTYPE_STRATEGY_ORDER = ["multi_proto_k2", "cosine_margin_proto", "warmup_proto_schedule"]
