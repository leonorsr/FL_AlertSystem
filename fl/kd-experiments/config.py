from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINDOWS_PATH = REPO_ROOT / "data" / "preprocessing_results" / "simple" / "magnitude_features" / "windows.csv"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_EXPERIMENTS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelConfig:
    hidden_layers: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.30
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    use_batchnorm: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    title: str
    description: str
    scenario: str = "byclient"
    holdout_dataset: str | None = None
    num_rounds: int = 25
    fraction_fit: float = 1.0
    min_fit_clients: int = 2
    min_available_clients: int = 2
    local_epochs: int = 10
    weighted_aggregation: bool = True
    teacher_student_distillation: bool = True
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5
    local_model_selection: bool = False
    clustered_aggregation: bool = False
    num_similarity_groups: int = 3
    personalized_head: bool = False
    final_local_finetune_epochs: int = 0
    random_seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)


EXPERIMENT_CATALOG: dict[str, ExperimentConfig] = {
    "exp1_kkd_base": ExperimentConfig(
        experiment_id="exp1_kkd_base",
        title="Experiment 1 - KKD Base",
        description="Standard KKD federated training using the same MLP architecture and byclient split.",
    ),
    "exp2_fraction_clients": ExperimentConfig(
        experiment_id="exp2_fraction_clients",
        title="Experiment 2 - Fraction Of Clients",
        description="KD with partial client participation per round.",
        fraction_fit=0.5,
    ),
    "exp3_local_epochs": ExperimentConfig(
        experiment_id="exp3_local_epochs",
        title="Experiment 3 - Local Epochs Comparison",
        description="KD comparison across different numbers of local epochs while keeping the global setup fixed.",
    ),
    "exp4_unweighted_aggregation": ExperimentConfig(
        experiment_id="exp4_unweighted_aggregation",
        title="Experiment 4 - Unweighted Aggregation",
        description="KD variant with unweighted averaging across participating clients.",
        weighted_aggregation=False,
    ),
    "exp5_cross_dataset": ExperimentConfig(
        experiment_id="exp5_cross_dataset",
        title="Experiment 5 - Cross Dataset KD",
        description="Federated KD training on multiple datasets and evaluation on a held-out dataset.",
        scenario="crossdataset",
        holdout_dataset="UpFall",
    ),
    "exp6_keep_best_local_model": ExperimentConfig(
        experiment_id="exp6_keep_best_local_model",
        title="Experiment 6 - Keep Best Local Model",
        description="Each client keeps the better model between the incoming global model and its previous best local model during KD training.",
        local_model_selection=True,
    ),
    "exp7_clustered_aggregation": ExperimentConfig(
        experiment_id="exp7_clustered_aggregation",
        title="Experiment 7 - Clustered Aggregation",
        description="Aggregate similar clients into subgroup models before sending them to the global server in a KD workflow.",
        clustered_aggregation=True,
        num_similarity_groups=3,
    ),
    "exp8_personalized_fedavg": ExperimentConfig(
        experiment_id="exp8_personalized_fedavg",
        title="Experiment 8 - Personalized KD",
        description="KD with a personalized local head kept on each client.",
        personalized_head=True,
    ),
    "exp9_final_local_finetuning": ExperimentConfig(
        experiment_id="exp9_final_local_finetuning",
        title="Experiment 9 - Final Local Fine-Tuning",
        description="Standard KD followed by a short local fine-tuning stage on held-out test clients.",
        final_local_finetune_epochs=3,
    ),
    "exp10_clustered_keep_best_local": ExperimentConfig(
        experiment_id="exp10_clustered_keep_best_local",
        title="Experiment 10 - Clustered Keep-Best Local",
        description="Clustered aggregation combined with local selection between the incoming grouped/global model and the previous best local model in KD training.",
        local_model_selection=True,
        clustered_aggregation=True,
        num_similarity_groups=3,
    ),
    "exp11_baseline_final": ExperimentConfig(
        experiment_id="exp11_baseline_final",
        title="Experiment 11 - Baseline Final",
        description="Final KD baseline using the strongest setting found so far: 100 local epochs with standard weighted aggregation.",
        local_epochs=100,
    ),
    "exp12_final_unweighted": ExperimentConfig(
        experiment_id="exp12_final_unweighted",
        title="Experiment 12 - Final Unweighted",
        description="Final KD baseline with 100 local epochs plus unweighted aggregation across participating clients.",
        local_epochs=100,
        weighted_aggregation=False,
    ),
    "exp13_final_keep_best_local": ExperimentConfig(
        experiment_id="exp13_final_keep_best_local",
        title="Experiment 13 - Final Keep-Best Local",
        description="Final KD baseline with 100 local epochs plus local selection between the incoming global model and the previous best local model.",
        local_epochs=100,
        local_model_selection=True,
    ),
    "exp14_final_clustered": ExperimentConfig(
        experiment_id="exp14_final_clustered",
        title="Experiment 14 - Final Clustered",
        description="Final KD baseline with 100 local epochs plus clustered aggregation over similar clients before the global update.",
        local_epochs=100,
        clustered_aggregation=True,
        num_similarity_groups=3,
    ),
}


def config_to_dict(config: ExperimentConfig) -> dict:
    return asdict(config)


def get_experiment_results_dir(config: ExperimentConfig, experiments_root: Path = DEFAULT_EXPERIMENTS_ROOT) -> Path:
    return experiments_root / config.experiment_id / "results"
