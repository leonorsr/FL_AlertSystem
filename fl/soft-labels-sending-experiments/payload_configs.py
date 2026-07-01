from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PayloadConfig:
    name: str
    description: str
    scalar_count: int
    uses_class_conditioning: bool
    uses_std_confidence: bool
    uses_quantiles: bool


PAYLOAD_CONFIGS: dict[str, PayloadConfig] = {
    "global_mean_count": PayloadConfig(
        name="global_mean_count",
        description="Baseline payload: one global mean predicted fall probability plus count.",
        scalar_count=2,
        uses_class_conditioning=False,
        uses_std_confidence=False,
        uses_quantiles=False,
    ),
    "global_mean_std_count": PayloadConfig(
        name="global_mean_std_count",
        description="Global mean predicted fall probability, standard deviation, and count.",
        scalar_count=3,
        uses_class_conditioning=False,
        uses_std_confidence=True,
        uses_quantiles=False,
    ),
    "class_mean_counts": PayloadConfig(
        name="class_mean_counts",
        description="Class-conditional mean predicted fall probabilities plus class counts.",
        scalar_count=4,
        uses_class_conditioning=True,
        uses_std_confidence=False,
        uses_quantiles=False,
    ),
    "class_mean_std_counts": PayloadConfig(
        name="class_mean_std_counts",
        description="Class-conditional means, standard deviations, and counts.",
        scalar_count=6,
        uses_class_conditioning=True,
        uses_std_confidence=True,
        uses_quantiles=False,
    ),
    "class_quantile_stats": PayloadConfig(
        name="class_quantile_stats",
        description="Class-conditional means, standard deviations, q25/q50/q75 quantiles, and counts.",
        scalar_count=12,
        uses_class_conditioning=True,
        uses_std_confidence=True,
        uses_quantiles=True,
    ),
}


DEFAULT_PAYLOAD_ORDER = [
    "global_mean_count",
    "global_mean_std_count",
    "class_mean_counts",
    "class_mean_std_counts",
    "class_quantile_stats",
]
