from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdvancedStrategyConfig:
    name: str
    description: str
    payload_name: str
    aggregation_scope: str = "global"  # global or cluster
    warmup_rounds: int = 0
    uncertain_only: bool = False
    uncertainty_low: float = 0.3
    uncertainty_high: float = 0.7


ADVANCED_STRATEGIES: dict[str, AdvancedStrategyConfig] = {
    "cluster_mean_count": AdvancedStrategyConfig(
        name="cluster_mean_count",
        description="Cluster-specific soft-label mean and count. Each client receives the aggregate from its similarity cluster.",
        payload_name="global_mean_count",
        aggregation_scope="cluster",
    ),
    "cluster_class_mean_counts": AdvancedStrategyConfig(
        name="cluster_class_mean_counts",
        description="Cluster-specific class-conditional soft-label means and counts.",
        payload_name="class_mean_counts",
        aggregation_scope="cluster",
    ),
    "warmup_global_mean_count": AdvancedStrategyConfig(
        name="warmup_global_mean_count",
        description="Global mean/count payload, but KD is disabled for the first 5 rounds.",
        payload_name="global_mean_count",
        aggregation_scope="global",
        warmup_rounds=5,
    ),
    "uncertain_global_mean_count": AdvancedStrategyConfig(
        name="uncertain_global_mean_count",
        description="Global mean/count payload applied only to locally uncertain examples.",
        payload_name="global_mean_count",
        aggregation_scope="global",
        uncertain_only=True,
        uncertainty_low=0.3,
        uncertainty_high=0.7,
    ),
}


DEFAULT_ADVANCED_STRATEGY_ORDER = [
    "cluster_mean_count",
    "cluster_class_mean_counts",
    "warmup_global_mean_count",
    "uncertain_global_mean_count",
]
