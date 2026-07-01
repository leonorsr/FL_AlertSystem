from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_WINDOWS_PATH, ExperimentConfig


ID_COLUMNS = ["dataset", "client", "trial_id", "label", "variant", "feature_set"]
WINDOW_METADATA_COLUMNS = ["window_start_sec", "window_end_sec", "window_seconds"]
CLIENT_SPLIT_ASSIGNMENTS = {
    "KFall": {
        "train": [
            "SA06", "SA07", "SA08", "SA09", "SA10", "SA12", "SA13", "SA14", "SA15", "SA18",
            "SA19", "SA20", "SA22", "SA23", "SA24", "SA25", "SA26", "SA27", "SA28", "SA29",
            "SA30", "SA33", "SA35", "SA36", "SA37", "SA38",
        ],
        "dev": ["SA11", "SA16", "SA32"],
        "test": ["SA17", "SA21", "SA31"],
    },
    "SisFall": {
        "train": [
            "SA01", "SA02", "SA03", "SA04", "SA05", "SA06", "SA07", "SA08", "SA09", "SA10",
            "SA11", "SA13", "SA14", "SA17", "SA18", "SA19", "SA20", "SA21", "SA22", "SA23",
            "SE01", "SE02", "SE04", "SE07", "SE09", "SE11", "SE12", "SE13", "SE14", "SE15",
        ],
        "dev": ["SA15", "SE05", "SE06", "SE10"],
        "test": ["SA12", "SA16", "SE03", "SE08"],
    },
    "UpFall": {
        "train": ["S01", "S03", "S04", "S05", "S06", "S08", "S09", "S10", "S11", "S12", "S13", "S16", "S17"],
        "dev": ["S14", "S15"],
        "test": ["S02", "S07"],
    },
}


@dataclass
class ClientPartition:
    client_id: str
    dataset: str
    client: str
    cluster_id: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


@dataclass
class FineTunePartition:
    client_id: str
    dataset: str
    client: str
    x_adapt: np.ndarray
    y_adapt: np.ndarray
    x_eval: np.ndarray
    y_eval: np.ndarray


@dataclass
class FederatedSetup:
    feature_columns: list[str]
    train_clients: dict[str, ClientPartition]
    dev_arrays: tuple[np.ndarray, np.ndarray]
    test_arrays: tuple[np.ndarray, np.ndarray]
    fine_tune_clients: dict[str, FineTunePartition]
    client_summary: pd.DataFrame


def load_windows(path: Path = DEFAULT_WINDOWS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna(axis=1, how="all").copy()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLUMNS + WINDOW_METADATA_COLUMNS)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded]


def build_trial_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "trial_id"], as_index=False)
        .agg(label=("label", "max"), windows=("label", "size"), client=("client", "first"))
        .copy()
    )


def build_client_split_reference(assignments: dict[str, dict[str, list[str]]] = CLIENT_SPLIT_ASSIGNMENTS) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for dataset_name, split_map in assignments.items():
        for split_name, clients in split_map.items():
            for client_name in clients:
                rows.append({"dataset": dataset_name, "client": client_name, "split": split_name})
    return pd.DataFrame(rows).sort_values(["dataset", "split", "client"]).reset_index(drop=True)


def split_trials_by_client(trial_table: pd.DataFrame) -> pd.DataFrame:
    client_split_df = build_client_split_reference()
    merged = trial_table.merge(client_split_df, on=["dataset", "client"], how="left", validate="many_to_one")
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), ["dataset", "client"]].drop_duplicates()
        raise ValueError(f"Missing client assignments:\n{missing.to_string(index=False)}")
    return merged


def attach_splits(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(split_df[["dataset", "trial_id", "split"]], on=["dataset", "trial_id"], how="left", validate="many_to_one")
    if merged["split"].isna().any():
        raise ValueError("Some rows were not assigned to a split.")
    return merged


def assign_crossdataset_splits(df: pd.DataFrame, holdout_dataset: str, seed: int) -> pd.DataFrame:
    trial_table = build_trial_table(df)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str | int]] = []
    for dataset_name, dataset_trials in trial_table.groupby("dataset", sort=True):
        if dataset_name == holdout_dataset:
            for _, row in dataset_trials.iterrows():
                rows.append({"dataset": dataset_name, "trial_id": row["trial_id"], "split": "test"})
            continue

        ordered_trials = dataset_trials.sort_values("trial_id").copy()
        shuffled_indices = rng.permutation(len(ordered_trials))
        dev_count = max(1, int(round(0.15 * len(ordered_trials))))
        dev_indices = set(shuffled_indices[:dev_count].tolist())
        for idx, (_, row) in enumerate(ordered_trials.iterrows()):
            split_name = "dev" if idx in dev_indices else "train"
            rows.append({"dataset": dataset_name, "trial_id": row["trial_id"], "split": split_name})
    return pd.DataFrame(rows)


def _split_trials_for_local_use(client_trials: pd.DataFrame, seed: int, adapt_ratio: float) -> tuple[list[str], list[str]]:
    ordered = client_trials.sort_values(["label", "trial_id"]).copy()
    rng = np.random.default_rng(seed)
    unique_trials = ordered["trial_id"].tolist()
    shuffled = rng.permutation(unique_trials).tolist()
    adapt_count = min(max(1, int(round(len(shuffled) * adapt_ratio))), max(len(shuffled) - 1, 1))
    adapt_trials = shuffled[:adapt_count]
    eval_trials = shuffled[adapt_count:]
    if not eval_trials:
        eval_trials = adapt_trials[-1:]
        adapt_trials = adapt_trials[:-1]
    return adapt_trials, eval_trials


def _build_client_clusters(train_frame: pd.DataFrame, feature_columns: list[str], num_clusters: int, seed: int) -> dict[str, int]:
    client_stats = (
        train_frame.groupby(["dataset", "client"])[feature_columns]
        .mean()
        .reset_index()
        .copy()
    )
    client_ids = client_stats["dataset"] + "::" + client_stats["client"]
    n_clusters = max(1, min(num_clusters, len(client_stats)))
    if n_clusters == 1:
        return {client_id: 0 for client_id in client_ids}

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(client_stats[feature_columns].to_numpy())
    return dict(zip(client_ids.tolist(), labels.tolist()))


def build_federated_setup(config: ExperimentConfig) -> FederatedSetup:
    windows_df = load_windows()
    feature_columns = get_feature_columns(windows_df)

    if config.scenario == "byclient":
        split_df = split_trials_by_client(build_trial_table(windows_df))
    elif config.scenario == "crossdataset":
        if not config.holdout_dataset:
            raise ValueError("crossdataset experiments require holdout_dataset.")
        split_df = assign_crossdataset_splits(windows_df, config.holdout_dataset, config.random_seed)
    else:
        raise ValueError(f"Unsupported scenario: {config.scenario}")

    labeled_df = attach_splits(windows_df, split_df)
    train_frame = labeled_df[labeled_df["split"] == "train"].copy().reset_index(drop=True)
    dev_frame = labeled_df[labeled_df["split"] == "dev"].copy().reset_index(drop=True)
    test_frame = labeled_df[labeled_df["split"] == "test"].copy().reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_global = imputer.fit_transform(train_frame[feature_columns])
    x_train_global = scaler.fit_transform(x_train_global)

    def transform(frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.empty((0, len(feature_columns)), dtype=float)
        transformed = imputer.transform(frame[feature_columns])
        transformed = scaler.transform(transformed)
        return transformed.astype(np.float32)

    train_frame = train_frame.assign(client_id=train_frame["dataset"] + "::" + train_frame["client"])
    dev_frame = dev_frame.assign(client_id=dev_frame["dataset"] + "::" + dev_frame["client"])
    test_frame = test_frame.assign(client_id=test_frame["dataset"] + "::" + test_frame["client"])

    cluster_map = _build_client_clusters(train_frame, feature_columns, config.num_similarity_groups, config.random_seed)

    train_clients: dict[str, ClientPartition] = {}
    summary_rows: list[dict[str, str | int]] = []
    for idx, (client_id, client_frame) in enumerate(train_frame.groupby("client_id", sort=True)):
        dataset_name = str(client_frame["dataset"].iloc[0])
        client_name = str(client_frame["client"].iloc[0])
        trial_table = (
            client_frame.groupby("trial_id", as_index=False)
            .agg(label=("label", "max"))
            .copy()
        )
        fit_trials, val_trials = _split_trials_for_local_use(trial_table, config.random_seed + idx, adapt_ratio=0.2)
        fit_frame = client_frame[client_frame["trial_id"].isin(fit_trials)].copy()
        val_frame = client_frame[client_frame["trial_id"].isin(val_trials)].copy()
        if fit_frame.empty or val_frame.empty:
            midpoint = max(1, len(client_frame) // 5)
            val_frame = client_frame.iloc[:midpoint].copy()
            fit_frame = client_frame.iloc[midpoint:].copy()

        train_clients[client_id] = ClientPartition(
            client_id=client_id,
            dataset=dataset_name,
            client=client_name,
            cluster_id=int(cluster_map.get(client_id, 0)),
            x_train=transform(fit_frame),
            y_train=fit_frame["label"].to_numpy(dtype=np.int64),
            x_val=transform(val_frame),
            y_val=val_frame["label"].to_numpy(dtype=np.int64),
        )
        summary_rows.append(
            {
                "client_id": client_id,
                "dataset": dataset_name,
                "client": client_name,
                "cluster_id": int(cluster_map.get(client_id, 0)),
                "train_windows": int(len(fit_frame)),
                "val_windows": int(len(val_frame)),
            }
        )

    fine_tune_clients: dict[str, FineTunePartition] = {}
    for idx, (client_id, client_frame) in enumerate(test_frame.groupby("client_id", sort=True)):
        dataset_name = str(client_frame["dataset"].iloc[0])
        client_name = str(client_frame["client"].iloc[0])
        trial_table = (
            client_frame.groupby("trial_id", as_index=False)
            .agg(label=("label", "max"))
            .copy()
        )
        adapt_trials, eval_trials = _split_trials_for_local_use(trial_table, config.random_seed + 1000 + idx, adapt_ratio=0.4)
        adapt_frame = client_frame[client_frame["trial_id"].isin(adapt_trials)].copy()
        eval_frame = client_frame[client_frame["trial_id"].isin(eval_trials)].copy()
        fine_tune_clients[client_id] = FineTunePartition(
            client_id=client_id,
            dataset=dataset_name,
            client=client_name,
            x_adapt=transform(adapt_frame),
            y_adapt=adapt_frame["label"].to_numpy(dtype=np.int64),
            x_eval=transform(eval_frame),
            y_eval=eval_frame["label"].to_numpy(dtype=np.int64),
        )

    return FederatedSetup(
        feature_columns=feature_columns,
        train_clients=train_clients,
        dev_arrays=(transform(dev_frame), dev_frame["label"].to_numpy(dtype=np.int64)),
        test_arrays=(transform(test_frame), test_frame["label"].to_numpy(dtype=np.int64)),
        fine_tune_clients=fine_tune_clients,
        client_summary=pd.DataFrame(summary_rows).sort_values(["dataset", "client"]).reset_index(drop=True),
    )
