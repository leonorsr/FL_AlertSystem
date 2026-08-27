from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


ID_COLUMNS = ["dataset", "client", "trial_id", "label", "variant", "feature_set", "split"]
WINDOW_METADATA_COLUMNS = ["window_start_sec", "window_end_sec", "window_seconds"]
DEFAULT_SORT_COLUMNS = ["window_start_sec", "window_end_sec"]


@dataclass
class SequencePreprocessor:
    feature_columns: list[str]
    scaler: MinMaxScaler
    imputer: SimpleImputer
    seq_len: int

    def transform_array(self, array: np.ndarray) -> np.ndarray:
        shape = array.shape
        flat = array.reshape(-1, shape[-1])
        flat = self.imputer.transform(flat)
        flat = self.scaler.transform(flat)
        flat = np.clip(flat, 0.0, 1.0)
        return flat.reshape(shape).astype(np.float32)

    def inverse_transform_array(self, array: np.ndarray) -> np.ndarray:
        shape = array.shape
        flat = array.reshape(-1, shape[-1])
        flat = self.scaler.inverse_transform(flat)
        return flat.reshape(shape).astype(np.float32)


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLUMNS + WINDOW_METADATA_COLUMNS)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded]


def _resample_sequence(values: np.ndarray, seq_len: int) -> np.ndarray:
    if values.shape[0] == seq_len:
        return values
    if values.shape[0] == 1:
        return np.repeat(values, seq_len, axis=0)

    old_index = np.linspace(0.0, 1.0, num=values.shape[0])
    new_index = np.linspace(0.0, 1.0, num=seq_len)
    columns = [np.interp(new_index, old_index, values[:, idx]) for idx in range(values.shape[1])]
    return np.stack(columns, axis=1)


def load_npz_sequences(path: Path, seq_len: int | None = None) -> tuple[np.ndarray, list[str], SequencePreprocessor]:
    payload = np.load(path, allow_pickle=True)
    if "sequences" not in payload:
        raise ValueError("NPZ input must contain an array named 'sequences' with shape (n, t, d).")
    sequences = np.asarray(payload["sequences"], dtype=np.float32)
    if sequences.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {sequences.shape}.")
    if seq_len is not None and sequences.shape[1] != seq_len:
        sequences = np.stack([_resample_sequence(sequence, seq_len) for sequence in sequences], axis=0)

    feature_columns = (
        [str(value) for value in payload["feature_columns"].tolist()]
        if "feature_columns" in payload
        else [f"feature_{idx}" for idx in range(sequences.shape[-1])]
    )
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()
    flat = sequences.reshape(-1, sequences.shape[-1])
    flat = imputer.fit_transform(flat)
    scaler.fit(flat)
    preprocessor = SequencePreprocessor(feature_columns, scaler, imputer, sequences.shape[1])
    return preprocessor.transform_array(sequences), feature_columns, preprocessor


def load_csv_sequences(
    path: Path,
    seq_len: int,
    sequence_columns: list[str] | None = None,
    label: int | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, SequencePreprocessor]:
    df = pd.read_csv(path).dropna(axis=1, how="all")
    if label is not None:
        if "label" not in df.columns:
            raise ValueError("--label was provided, but the CSV has no 'label' column.")
        df = df[df["label"] == label].copy()
    if df.empty:
        raise ValueError("No rows left after filtering the CSV input.")

    feature_columns = feature_columns or infer_feature_columns(df)
    if not feature_columns:
        raise ValueError("Could not infer numeric feature columns. Pass --feature-columns explicitly.")

    sequence_columns = sequence_columns or ["dataset", "trial_id"]
    missing_sequence_columns = [column for column in sequence_columns if column not in df.columns]
    if missing_sequence_columns:
        raise ValueError(f"Missing sequence columns: {missing_sequence_columns}")

    sort_columns = [column for column in DEFAULT_SORT_COLUMNS if column in df.columns]
    metadata_rows: list[dict[str, object]] = []
    raw_sequences: list[np.ndarray] = []
    for sequence_key, group in df.groupby(sequence_columns, sort=True):
        ordered = group.sort_values(sort_columns) if sort_columns else group
        values = ordered[feature_columns].to_numpy(dtype=np.float32)
        if values.size == 0:
            continue
        raw_sequences.append(_resample_sequence(values, seq_len))
        if not isinstance(sequence_key, tuple):
            sequence_key = (sequence_key,)
        row = dict(zip(sequence_columns, sequence_key))
        if "label" in ordered.columns:
            row["label"] = int(ordered["label"].max())
        row["original_length"] = int(len(ordered))
        metadata_rows.append(row)

    if not raw_sequences:
        raise ValueError("No valid sequences were built from the input CSV.")

    raw = np.stack(raw_sequences, axis=0).astype(np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()
    flat = imputer.fit_transform(raw.reshape(-1, raw.shape[-1]))
    scaler.fit(flat)
    preprocessor = SequencePreprocessor(feature_columns, scaler, imputer, seq_len)
    return preprocessor.transform_array(raw), pd.DataFrame(metadata_rows), preprocessor


def save_generated_csv(
    sequences: np.ndarray,
    output_path: Path,
    feature_columns: list[str],
    sequence_prefix: str = "synthetic",
    label: int = 1,
) -> None:
    count, seq_len, feature_count = sequences.shape
    if feature_count != len(feature_columns):
        raise ValueError("Generated feature dimension does not match feature_columns.")
    frame = pd.DataFrame(sequences.reshape(-1, feature_count), columns=feature_columns)
    frame.insert(0, "label", int(label))
    frame.insert(0, "timestep", np.tile(np.arange(seq_len), count))
    frame.insert(
        0,
        "trial_id",
        np.repeat([f"{sequence_prefix}_{index:07d}" for index in range(count)], seq_len),
    )
    frame.insert(0, "client", "TimeGAN")
    frame.insert(0, "dataset", "Synthetic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
