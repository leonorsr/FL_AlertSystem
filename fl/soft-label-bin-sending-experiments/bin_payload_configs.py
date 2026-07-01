from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinPayloadConfig:
    name: str
    description: str
    num_bins: int

    @property
    def scalar_count(self) -> int:
        return self.num_bins * 3


BIN_PAYLOAD_CONFIGS: dict[str, BinPayloadConfig] = {
    "prob_bins_3": BinPayloadConfig(
        name="prob_bins_3",
        description="Three probability bins; each bin sends mean, standard deviation, and count.",
        num_bins=3,
    ),
    "prob_bins_5": BinPayloadConfig(
        name="prob_bins_5",
        description="Five probability bins; each bin sends mean, standard deviation, and count.",
        num_bins=5,
    ),
    "prob_bins_10": BinPayloadConfig(
        name="prob_bins_10",
        description="Ten probability bins; each bin sends mean, standard deviation, and count.",
        num_bins=10,
    ),
}


DEFAULT_BIN_PAYLOAD_ORDER = ["prob_bins_3", "prob_bins_5", "prob_bins_10"]
