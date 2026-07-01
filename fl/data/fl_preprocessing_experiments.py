from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt
except ImportError:  # pragma: no cover - optional dependency in this workspace
    butter = None
    filtfilt = None


ROOT = Path(__file__).resolve().parent
UPFALL_PATH = ROOT / "UpFall" / "CompleteDataSet.csv"
SISFALL_PATH = ROOT / "sisfall"
KFALL_SENSOR_PATH = ROOT / "Kfall" / "sensor_data_new"
TARGET_HZ = 20
DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_WINDOW_OVERLAP = 0.5
COMMON_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
DATASET_LOADERS = {
    "KFall": "load_kfall_trials",
    "SisFall": "load_sisfall_trials",
    "UpFall": "load_upfall_trials",
}


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    smooth_window: int | None = None
    lowpass_cutoff_hz: float | None = 5.0
    lowpass_order: int = 4
    add_magnitude: bool = False
    magnitude_only: bool = False
    scaling: str = "zscore"
    clip_quantile: float | None = None
    gravity_window: int | None = None
    sequence_ready: bool = False
    add_frequency_features: bool = False
    quality_min_std: float | None = None
    balance_per_client: bool = False
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    window_overlap: float = DEFAULT_WINDOW_OVERLAP
    add_article_features: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant(
        name="baseline_zscore",
        description="Filtro low-pass 5 Hz, resample para 20 Hz, janela de 1 s e normalizacao z-score por trial.",
    ),
    Variant(
        name="robust_clip",
        description="Filtro low-pass 5 Hz, resample para 20 Hz, clipping por quantis e escala robusta por trial.",
        scaling="robust",
        clip_quantile=0.01,
    ),
    Variant(
        name="magnitude_features",
        description="Filtro low-pass 5 Hz, mantem eixos, adiciona magnitudes e aplica z-score por trial.",
        add_magnitude=True,
    ),
    Variant(
        name="smoothed_magnitude",
        description="Filtro low-pass 5 Hz, suavizacao leve, clipping, magnitudes e escala robusta por cliente.",
        smooth_window=3,
        add_magnitude=True,
        scaling="client_robust",
        clip_quantile=0.01,
        add_article_features=True,
    ),
    Variant(
        name="magnitude_only",
        description="Filtro low-pass 5 Hz e reducao para acc_mag e gyro_mag para minimizar diferencas de orientacao.",
        add_magnitude=True,
        magnitude_only=True,
        scaling="client_robust",
        clip_quantile=0.01,
    ),
    Variant(
        name="client_zscore",
        description="Filtro low-pass 5 Hz, resample para 20 Hz, janela de 1 s e normalizacao z-score por cliente local.",
        scaling="client_zscore",
        add_article_features=True,
    ),
    Variant(
        name="client_raw_plus_magnitude",
        description="Filtro low-pass 5 Hz, eixos mais magnitudes, janela de 1 s e normalizacao robusta por cliente local.",
        add_magnitude=True,
        scaling="client_robust",
        clip_quantile=0.01,
        add_article_features=True,
    ),
    Variant(
        name="gravity_removed_magnitude",
        description="Filtro low-pass 5 Hz, remove componente lenta da aceleracao, adiciona magnitudes e usa escala robusta por cliente.",
        add_magnitude=True,
        scaling="client_robust",
        clip_quantile=0.01,
        gravity_window=9,
        add_article_features=True,
    ),
    Variant(
        name="sequence_ready_nn",
        description="Pipeline orientado a redes neuronais: low-pass 5 Hz, suavizacao leve, clipping, magnitudes e normalizacao robusta por cliente.",
        smooth_window=3,
        add_magnitude=True,
        scaling="client_robust",
        clip_quantile=0.01,
        sequence_ready=True,
        add_article_features=True,
    ),
    Variant(
        name="frequency_augmented",
        description="Low-pass 5 Hz, eixos e magnitudes com features no dominio da frequencia por janela.",
        add_magnitude=True,
        scaling="client_robust",
        clip_quantile=0.01,
        add_frequency_features=True,
        add_article_features=True,
    ),
    Variant(
        name="quality_filtered_magnitude",
        description="Low-pass 5 Hz, magnitudes com filtragem de janelas de baixa qualidade e normalizacao robusta por cliente.",
        add_magnitude=True,
        scaling="client_robust",
        clip_quantile=0.01,
        quality_min_std=0.05,
        add_article_features=True,
    ),
    Variant(
        name="client_balanced_magnitude",
        description="Low-pass 5 Hz, magnitudes com normalizacao robusta por cliente, janelas de 1 s e balanceamento local Fall/ADL.",
        add_magnitude=True,
        scaling="client_robust",
        clip_quantile=0.01,
        balance_per_client=True,
        add_article_features=True,
    ),
    Variant(
        name="short_window_article",
        description="Low-pass 5 Hz, janelas de 0.5 s e features horizontais/area inspiradas na literatura.",
        add_magnitude=True,
        scaling="client_robust",
        clip_quantile=0.01,
        window_seconds=0.5,
        add_article_features=True,
    ),
    Variant(
        name="article_feature_set",
        description="Low-pass 5 Hz, janelas de 1 s e conjunto de features alinhado com SisFall/KFall para FL.",
        add_magnitude=True,
        scaling="client_robust",
        clip_quantile=0.01,
        add_article_features=True,
    ),
)


def safe_iterdir(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if not p.name.startswith("."))


def balanced_limit(items: list, max_items: int | None, label_getter) -> list:
    if max_items is None or len(items) <= max_items:
        return items
    positives = [item for item in items if label_getter(item)]
    negatives = [item for item in items if not label_getter(item)]
    pos_target = max_items // 2
    neg_target = max_items - pos_target
    selected = positives[:pos_target] + negatives[:neg_target]

    if len(selected) < max_items:
        selected_ids = {id(item) for item in selected}
        remaining = [item for item in items if id(item) not in selected_ids]
        selected.extend(remaining[: max_items - len(selected)])
    return selected


def ensure_numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=list(columns))


def maybe_clip(df: pd.DataFrame, columns: list[str], q: float | None) -> pd.DataFrame:
    if q is None:
        return df
    out = df.copy()
    lower = out[columns].quantile(q)
    upper = out[columns].quantile(1 - q)
    out[columns] = out[columns].clip(lower=lower, upper=upper, axis=1)
    return out


def maybe_smooth(df: pd.DataFrame, columns: list[str], window: int | None) -> pd.DataFrame:
    if not window or window <= 1:
        return df
    out = df.copy()
    out[columns] = out[columns].rolling(window=window, center=True, min_periods=1).mean()
    return out


def maybe_remove_gravity(df: pd.DataFrame, window: int | None) -> pd.DataFrame:
    if not window or window <= 1:
        return df
    out = df.copy()
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    gravity = out[acc_cols].rolling(window=window, center=True, min_periods=1).mean()
    out[acc_cols] = out[acc_cols] - gravity
    return out


def maybe_lowpass_filter(
    df: pd.DataFrame,
    columns: list[str],
    source_hz: float,
    cutoff_hz: float | None,
    order: int = 4,
) -> pd.DataFrame:
    if cutoff_hz is None or cutoff_hz <= 0 or len(df) < 5:
        return df
    nyquist = float(source_hz) / 2.0
    if not np.isfinite(nyquist) or nyquist <= 0:
        return df
    normalized_cutoff = min(float(cutoff_hz) / nyquist, 0.99)
    if normalized_cutoff <= 0:
        return df

    out = df.copy()
    if butter is not None and filtfilt is not None:
        b, a = butter(order, normalized_cutoff, btype="low", analog=False)
        min_samples = max(3 * max(len(a), len(b)), 9)
        if len(out) <= min_samples:
            return out
        for col in columns:
            values = pd.to_numeric(out[col], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
            out[col] = filtfilt(b, a, values)
        return out

    # Fallback: four cascaded exponential smoothing passes approximate a low-pass response.
    alpha = normalized_cutoff / (normalized_cutoff + 1.0)
    alpha = float(np.clip(alpha, 0.01, 0.99))
    for col in columns:
        values = pd.to_numeric(out[col], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
        filtered = values.copy()
        for _ in range(max(1, order)):
            smoothed = np.empty_like(filtered)
            smoothed[0] = filtered[0]
            for idx in range(1, len(filtered)):
                smoothed[idx] = alpha * filtered[idx] + (1.0 - alpha) * smoothed[idx - 1]
            filtered = smoothed
        out[col] = filtered
    return out


def scale_frame(df: pd.DataFrame, columns: list[str], scaling: str) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    params: dict[str, float] = {}
    if scaling == "zscore":
        means = out[columns].mean()
        stds = out[columns].std(ddof=0).replace(0, 1.0).fillna(1.0)
        out[columns] = (out[columns] - means) / stds
        for col in columns:
            params[f"{col}_center"] = float(means[col])
            params[f"{col}_scale"] = float(stds[col])
    elif scaling == "robust":
        medians = out[columns].median()
        q1 = out[columns].quantile(0.25)
        q3 = out[columns].quantile(0.75)
        iqr = (q3 - q1).replace(0, 1.0).fillna(1.0)
        out[columns] = (out[columns] - medians) / iqr
        for col in columns:
            params[f"{col}_center"] = float(medians[col])
            params[f"{col}_scale"] = float(iqr[col])
    else:
        raise ValueError(f"Unsupported scaling: {scaling}")
    return out, params


def apply_scaling_params(df: pd.DataFrame, columns: list[str], stats: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        center = stats.get(f"{col}_center", 0.0)
        scale = stats.get(f"{col}_scale", 1.0) or 1.0
        out[col] = (out[col] - center) / scale
    return out


def add_magnitude_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["acc_mag"] = np.sqrt((out[["acc_x", "acc_y", "acc_z"]] ** 2).sum(axis=1))
    out["gyro_mag"] = np.sqrt((out[["gyro_x", "gyro_y", "gyro_z"]] ** 2).sum(axis=1))
    return out


def transform_without_scaling(
    df: pd.DataFrame,
    variant: Variant,
    source_hz: float = TARGET_HZ,
) -> tuple[pd.DataFrame, list[str]]:
    out = maybe_lowpass_filter(df, COMMON_AXES, source_hz=source_hz, cutoff_hz=variant.lowpass_cutoff_hz, order=variant.lowpass_order)
    out = maybe_smooth(out, COMMON_AXES, variant.smooth_window)
    out = maybe_remove_gravity(out, variant.gravity_window)
    out = maybe_clip(out, COMMON_AXES, variant.clip_quantile)
    if variant.add_magnitude:
        out = add_magnitude_features(out)
    feature_cols = COMMON_AXES.copy()
    if variant.add_magnitude:
        feature_cols.extend(["acc_mag", "gyro_mag"])
    if variant.magnitude_only:
        feature_cols = ["acc_mag", "gyro_mag"]
    return out, feature_cols


def resample_frame(df: pd.DataFrame, source_hz: float, target_hz: int = TARGET_HZ) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy().reset_index(drop=True)
    if "time_sec" in out.columns and out["time_sec"].notna().sum() >= 2:
        time_values = pd.to_numeric(out["time_sec"], errors="coerce").to_numpy(dtype=float)
    else:
        time_values = np.arange(len(out), dtype=float) / float(source_hz)

    if len(time_values) < 2:
        out["time_sec"] = np.arange(len(out), dtype=float) / float(target_hz)
        return out

    features = COMMON_AXES
    new_time = np.arange(time_values[0], time_values[-1] + (1.0 / target_hz), 1.0 / target_hz)
    resampled = {"time_sec": new_time}
    for col in features:
        values = pd.to_numeric(out[col], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
        resampled[col] = np.interp(new_time, time_values, values)
    return pd.DataFrame(resampled)


def preprocess_variant(
    df: pd.DataFrame,
    variant: Variant,
    source_hz: float = TARGET_HZ,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    out, feature_cols = transform_without_scaling(df, variant, source_hz=source_hz)
    out, stats = scale_frame(out, feature_cols, variant.scaling)
    return out, feature_cols, stats


def is_client_scaling(scaling: str) -> bool:
    return scaling in {"client_zscore", "client_robust"}


def base_scaling_name(scaling: str) -> str:
    if scaling == "client_zscore":
        return "zscore"
    if scaling == "client_robust":
        return "robust"
    return scaling


def compute_client_scalers(
    trials: list[dict[str, object]],
    variant: Variant,
) -> dict[tuple[str, str], tuple[list[str], dict[str, float]]]:
    client_frames: dict[tuple[str, str], list[pd.DataFrame]] = {}
    client_feature_cols: dict[tuple[str, str], list[str]] = {}

    for trial in trials:
        key = (str(trial["dataset"]), str(trial["client"]))
        resampled = resample_frame(trial["data"], source_hz=float(trial["source_hz"]), target_hz=TARGET_HZ)
        transformed, feature_cols = transform_without_scaling(resampled, variant, source_hz=TARGET_HZ)
        client_frames.setdefault(key, []).append(transformed[feature_cols].copy())
        client_feature_cols[key] = feature_cols

    scalers: dict[tuple[str, str], tuple[list[str], dict[str, float]]] = {}
    for key, frames in client_frames.items():
        merged = pd.concat(frames, ignore_index=True)
        feature_cols = client_feature_cols[key]
        _, stats = scale_frame(merged, feature_cols, base_scaling_name(variant.scaling))
        scalers[key] = (feature_cols, stats)
    return scalers


def window_geometry(variant: Variant) -> tuple[int, int]:
    window_size = max(1, int(round(TARGET_HZ * variant.window_seconds)))
    step = max(1, int(round(window_size * (1.0 - variant.window_overlap))))
    return window_size, step


def compute_article_window_features(chunk: pd.DataFrame) -> dict[str, float]:
    dt = 1.0 / TARGET_HZ
    acc_x = chunk["acc_x"].to_numpy(dtype=float)
    acc_y = chunk["acc_y"].to_numpy(dtype=float)
    acc_z = chunk["acc_z"].to_numpy(dtype=float)

    horiz_mag = np.sqrt(acc_x**2 + acc_z**2)
    total_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    sigma_x = float(np.std(acc_x, ddof=0))
    sigma_y = float(np.std(acc_y, ddof=0))
    sigma_z = float(np.std(acc_z, ddof=0))

    horiz_integral = np.array([np.sum(np.abs(acc_x)) * dt, np.sum(np.abs(acc_z)) * dt], dtype=float)
    vel_integral = np.array([np.sum(acc_x) * dt, np.sum(acc_z) * dt], dtype=float)

    return {
        "article_c2_horizontal_svm_mean": float(horiz_mag.mean()),
        "article_c8_horizontal_std_mag": float(np.sqrt(sigma_x**2 + sigma_z**2)),
        "article_c9_total_std_mag": float(np.sqrt(sigma_x**2 + sigma_y**2 + sigma_z**2)),
        "article_c10_sma": float((np.sum(np.abs(acc_x)) + np.sum(np.abs(acc_y)) + np.sum(np.abs(acc_z))) * dt / len(chunk)),
        "article_c11_horizontal_sma": float((np.sum(np.abs(acc_x)) + np.sum(np.abs(acc_z))) * dt / len(chunk)),
        "article_c12_activity_sma": float(np.sum(total_mag) * dt),
        "article_c13_horizontal_activity_sma": float(np.sum(horiz_mag) * dt),
        "article_c14_horizontal_velocity_approx": float(np.linalg.norm(vel_integral) / len(chunk)),
        "horizontal_peak_to_peak": float(horiz_mag.max() - horiz_mag.min()),
    }


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    dataset: str,
    client: str,
    trial_id: str,
    label: int,
    variant: Variant,
) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    window_size, window_step = window_geometry(variant)
    if len(df) < window_size:
        return windows

    for start in range(0, len(df) - window_size + 1, window_step):
        end = start + window_size
        chunk = df.iloc[start:end]
        if variant.quality_min_std is not None:
            window_std = float(chunk[feature_cols].std(ddof=0).mean())
            if not np.isfinite(window_std) or window_std < variant.quality_min_std:
                continue
        summary = {
            "dataset": dataset,
            "client": client,
            "trial_id": trial_id,
            "label": label,
            "window_start_sec": float(chunk["time_sec"].iloc[0]),
            "window_end_sec": float(chunk["time_sec"].iloc[-1]),
            "window_seconds": float(variant.window_seconds),
        }
        for col in feature_cols:
            values = chunk[col].to_numpy(dtype=float)
            summary[f"{col}_mean"] = float(values.mean())
            summary[f"{col}_std"] = float(values.std(ddof=0))
            summary[f"{col}_maxabs"] = float(np.abs(values).max())
            summary[f"{col}_energy"] = float(np.mean(values**2))
            if variant.add_frequency_features:
                centered = values - values.mean()
                spectrum = np.fft.rfft(centered)
                freqs = np.fft.rfftfreq(len(centered), d=1.0 / TARGET_HZ)
                power = np.abs(spectrum) ** 2
                if len(power) > 1:
                    power_wo_dc = power[1:]
                    freqs_wo_dc = freqs[1:]
                    dom_idx = int(np.argmax(power_wo_dc))
                    total_power = float(power_wo_dc.sum())
                    norm_power = power_wo_dc / total_power if total_power > 0 else np.zeros_like(power_wo_dc)
                    spectral_entropy = float(
                        -(norm_power[norm_power > 0] * np.log(norm_power[norm_power > 0] + 1e-12)).sum()
                    )
                    summary[f"{col}_dom_freq"] = float(freqs_wo_dc[dom_idx])
                    summary[f"{col}_spec_entropy"] = spectral_entropy
                else:
                    summary[f"{col}_dom_freq"] = 0.0
                    summary[f"{col}_spec_entropy"] = 0.0
        if variant.add_article_features and {"acc_x", "acc_y", "acc_z"}.issubset(chunk.columns):
            summary.update(compute_article_window_features(chunk))
        windows.append(summary)
    return windows


def balance_windows_per_client(windows_df: pd.DataFrame) -> pd.DataFrame:
    if windows_df.empty:
        return windows_df
    balanced_parts: list[pd.DataFrame] = []
    for (_, _), group in windows_df.groupby(["dataset", "client"], sort=False):
        class_counts = group["label"].value_counts()
        if len(class_counts) < 2:
            balanced_parts.append(group)
            continue
        target = int(class_counts.min())
        sampled_groups: list[pd.DataFrame] = []
        for _, class_group in group.groupby("label", sort=False):
            if len(class_group) > target:
                sampled_groups.append(class_group.sample(n=target, random_state=42))
            else:
                sampled_groups.append(class_group)
        sampled = pd.concat(sampled_groups, ignore_index=True)
        balanced_parts.append(sampled)
    return pd.concat(balanced_parts, ignore_index=True)


def load_kfall_trials(max_trials_per_client: int | None = None) -> list[dict[str, object]]:
    trials: list[dict[str, object]] = []
    for subject_dir in safe_iterdir(KFALL_SENSOR_PATH):
        if not subject_dir.is_dir():
            continue
        subject_files = sorted(subject_dir.glob("*.csv"))
        if max_trials_per_client is not None:
            subject_files = balanced_limit(
                subject_files,
                max_trials_per_client,
                lambda path: 20 <= int(re.search(r"T(\d+)", path.stem).group(1)) <= 34,
            )
        for csv_file in subject_files:
            match = re.search(r"S(\d+)T(\d+)R(\d+)", csv_file.stem)
            if not match:
                continue
            task_id = int(match.group(2))
            trial_number = int(match.group(3))
            df = pd.read_csv(csv_file)
            df = df.rename(
                columns={
                    "AccX": "acc_x",
                    "AccY": "acc_y",
                    "AccZ": "acc_z",
                    "GyrX": "gyro_x",
                    "GyrY": "gyro_y",
                    "GyrZ": "gyro_z",
                    "TimeStamp(s)": "time_sec",
                }
            )
            df = ensure_numeric_frame(df, COMMON_AXES)
            if df.empty:
                continue
            if "time_sec" in df.columns:
                time_df = df[["time_sec"] + COMMON_AXES].copy()
                source_hz = 100.0
            else:
                time_df = df[COMMON_AXES].copy()
                source_hz = 100.0
            trials.append(
                {
                    "dataset": "KFall",
                    "client": subject_dir.name,
                    "trial_id": csv_file.stem,
                    "label": int(20 <= task_id <= 34),
                    "source_hz": source_hz,
                    "data": time_df,
                    "task_id": task_id,
                    "trial_number": trial_number,
                }
            )
    return trials


def load_sisfall_trials(max_trials_per_client: int | None = None) -> list[dict[str, object]]:
    trials: list[dict[str, object]] = []
    col_names = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "aux_x",
        "aux_y",
        "aux_z",
    ]
    for subject_dir in safe_iterdir(SISFALL_PATH):
        if not subject_dir.is_dir():
            continue
        txt_files = sorted(subject_dir.glob("*.txt"))
        if max_trials_per_client is not None:
            txt_files = balanced_limit(
                txt_files,
                max_trials_per_client,
                lambda path: path.stem.startswith("F"),
            )
        for txt_file in txt_files:
            parts = txt_file.stem.split("_")
            if len(parts) < 3:
                continue
            activity = parts[0]
            trial_number = parts[2]
            df = pd.read_csv(
                txt_file,
                header=None,
                usecols=list(range(9)),
                names=col_names,
                sep=r"\s*,\s*|\s*;\s*",
                engine="python",
            )
            df = ensure_numeric_frame(df, COMMON_AXES)
            if df.empty:
                continue
            df["time_sec"] = np.arange(len(df), dtype=float) / 200.0
            trials.append(
                {
                    "dataset": "SisFall",
                    "client": subject_dir.name,
                    "trial_id": txt_file.stem,
                    "label": int(activity.startswith("F")),
                    "source_hz": 200.0,
                    "data": df[["time_sec"] + COMMON_AXES].copy(),
                    "activity": activity,
                    "trial_number": trial_number,
                }
            )
    return trials


def load_upfall_trials(max_trials_per_client: int | None = None) -> list[dict[str, object]]:
    raw = pd.read_csv(
        UPFALL_PATH,
        low_memory=False,
        skiprows=1,
        usecols=[0, 15, 16, 17, 18, 19, 20, 43, 44, 45, 46],
    )
    raw.columns = [
        "timestamp",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "subject",
        "activity",
        "trial",
        "tag",
    ]
    raw = ensure_numeric_frame(raw, COMMON_AXES + ["subject", "activity", "trial"])
    raw["subject"] = raw["subject"].astype(int)
    raw["activity"] = raw["activity"].astype(int)
    raw["trial"] = raw["trial"].astype(int)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")

    trials: list[dict[str, object]] = []
    for subject, subject_df in raw.groupby("subject", sort=True):
        grouped = list(subject_df.groupby(["activity", "trial"], sort=True))
        if max_trials_per_client is not None:
            grouped = balanced_limit(
                grouped,
                max_trials_per_client,
                lambda item: item[0][0] <= 5,
            )
        for (activity, trial), trial_df in grouped:
            trial_df = trial_df.sort_index().copy()
            if trial_df["timestamp"].notna().sum() >= 2:
                base_time = trial_df["timestamp"].dropna().iloc[0]
                trial_df["time_sec"] = (trial_df["timestamp"] - base_time).dt.total_seconds()
                source_hz = 1.0 / trial_df["time_sec"].diff().dropna().median()
            else:
                source_hz = 21.0
                trial_df["time_sec"] = np.arange(len(trial_df), dtype=float) / source_hz
            trials.append(
                {
                    "dataset": "UpFall",
                    "client": f"S{subject:02d}",
                    "trial_id": f"S{subject:02d}_A{activity:02d}_T{trial:02d}",
                    "label": int(activity <= 5),
                    "source_hz": float(source_hz) if not math.isnan(source_hz) else 21.0,
                    "data": trial_df[["time_sec"] + COMMON_AXES].copy(),
                    "activity": activity,
                    "trial_number": trial,
                }
            )
    return trials


def normalize_dataset_selection(selected_datasets: Iterable[str] | None) -> list[str]:
    if selected_datasets is None:
        return list(DATASET_LOADERS.keys())
    normalized: list[str] = []
    alias_map = {name.lower(): name for name in DATASET_LOADERS}
    for dataset_name in selected_datasets:
        canonical = alias_map.get(str(dataset_name).strip().lower())
        if canonical is None:
            valid = ", ".join(DATASET_LOADERS)
            raise ValueError(f"Unknown dataset '{dataset_name}'. Valid options: {valid}")
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def normalize_variant_selection(selected_variants: Iterable[str] | None) -> tuple[Variant, ...]:
    if selected_variants is None:
        return VARIANTS
    variants_by_name = {variant.name: variant for variant in VARIANTS}
    normalized: list[Variant] = []
    for variant_name in selected_variants:
        key = str(variant_name).strip()
        if key not in variants_by_name:
            valid = ", ".join(variants_by_name)
            raise ValueError(f"Unknown variant '{variant_name}'. Valid options: {valid}")
        variant = variants_by_name[key]
        if variant not in normalized:
            normalized.append(variant)
    return tuple(normalized)


def load_all_trials(
    max_trials_per_client: int | None = None,
    selected_datasets: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    dataset_names = normalize_dataset_selection(selected_datasets)
    loaders = {
        "KFall": load_kfall_trials,
        "SisFall": load_sisfall_trials,
        "UpFall": load_upfall_trials,
    }
    all_trials: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        all_trials.extend(loaders[dataset_name](max_trials_per_client=max_trials_per_client))
    return all_trials


def summarise_variant(
    windows_df: pd.DataFrame,
    variant: Variant,
    preview_windows_per_variant: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if windows_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary = (
        windows_df.groupby("dataset")
        .agg(
            windows=("trial_id", "size"),
            clients=("client", "nunique"),
            positive_windows=("label", "sum"),
        )
        .reset_index()
    )
    summary["variant"] = variant.name
    summary["positive_ratio"] = summary["positive_windows"] / summary["windows"]
    summary["window_seconds"] = variant.window_seconds
    summary["target_hz"] = TARGET_HZ
    summary["description"] = variant.description

    client_summary = (
        windows_df.groupby(["dataset", "client"])
        .agg(
            windows=("trial_id", "size"),
            positive_windows=("label", "sum"),
        )
        .reset_index()
    )
    client_summary["variant"] = variant.name
    client_summary["positive_ratio"] = client_summary["positive_windows"] / client_summary["windows"]

    preview = windows_df.groupby("dataset", group_keys=False).head(preview_windows_per_variant).copy()
    preview["variant"] = variant.name
    return summary, client_summary, preview


def run_experiments(
    output_dir: str | Path = "preprocessing_outputs",
    max_trials_per_client: int | None = None,
    preview_windows_per_variant: int = 3,
    selected_datasets: Iterable[str] | None = None,
    selected_variants: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    all_trials = load_all_trials(
        max_trials_per_client=max_trials_per_client,
        selected_datasets=selected_datasets,
    )
    variants = normalize_variant_selection(selected_variants)
    all_summaries: list[pd.DataFrame] = []
    all_client_summaries: list[pd.DataFrame] = []
    all_previews: list[pd.DataFrame] = []

    for variant in variants:
        client_scalers = compute_client_scalers(all_trials, variant) if is_client_scaling(variant.scaling) else {}
        all_windows: list[dict[str, object]] = []
        for trial in all_trials:
            resampled = resample_frame(trial["data"], source_hz=float(trial["source_hz"]), target_hz=TARGET_HZ)
            if is_client_scaling(variant.scaling):
                transformed, feature_cols = transform_without_scaling(resampled, variant, source_hz=TARGET_HZ)
                scaler_cols, scaler_stats = client_scalers[(str(trial["dataset"]), str(trial["client"]))]
                processed = apply_scaling_params(transformed, scaler_cols, scaler_stats)
                feature_cols = scaler_cols
            else:
                processed, feature_cols, _ = preprocess_variant(resampled, variant, source_hz=TARGET_HZ)
            windows = make_windows(
                processed,
                feature_cols=feature_cols,
                dataset=str(trial["dataset"]),
                client=str(trial["client"]),
                trial_id=str(trial["trial_id"]),
                label=int(trial["label"]),
                variant=variant,
            )
            all_windows.extend(windows)

        windows_df = pd.DataFrame(all_windows)
        if variant.balance_per_client:
            windows_df = balance_windows_per_client(windows_df)
        summary_df, client_summary_df, preview_df = summarise_variant(
            windows_df, variant, preview_windows_per_variant=preview_windows_per_variant
        )
        all_summaries.append(summary_df)
        all_client_summaries.append(client_summary_df)
        all_previews.append(preview_df)

    summary = pd.concat(all_summaries, ignore_index=True)
    client_summary = pd.concat(all_client_summaries, ignore_index=True)
    preview = pd.concat(all_previews, ignore_index=True)

    summary.to_csv(output_path / "summary.csv", index=False)
    client_summary.to_csv(output_path / "client_summary.csv", index=False)
    preview.to_csv(output_path / "preview_windows.csv", index=False)
    return summary, client_summary, preview


def run_experiments_detailed(
    max_trials_per_client: int | None = None,
    selected_datasets: Iterable[str] | None = None,
    selected_variants: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_trials = load_all_trials(
        max_trials_per_client=max_trials_per_client,
        selected_datasets=selected_datasets,
    )
    variants = normalize_variant_selection(selected_variants)
    all_windows_by_variant: list[pd.DataFrame] = []
    all_trial_records: list[dict[str, object]] = []

    for trial in all_trials:
        all_trial_records.append(
            {
                "dataset": str(trial["dataset"]),
                "client": str(trial["client"]),
                "trial_id": str(trial["trial_id"]),
                "label": int(trial["label"]),
                "source_hz": float(trial["source_hz"]),
            }
        )

    for variant in variants:
        client_scalers = compute_client_scalers(all_trials, variant) if is_client_scaling(variant.scaling) else {}
        variant_windows: list[dict[str, object]] = []
        for trial in all_trials:
            resampled = resample_frame(trial["data"], source_hz=float(trial["source_hz"]), target_hz=TARGET_HZ)
            if is_client_scaling(variant.scaling):
                transformed, feature_cols = transform_without_scaling(resampled, variant, source_hz=TARGET_HZ)
                scaler_cols, scaler_stats = client_scalers[(str(trial["dataset"]), str(trial["client"]))]
                processed = apply_scaling_params(transformed, scaler_cols, scaler_stats)
                feature_cols = scaler_cols
            else:
                processed, feature_cols, _ = preprocess_variant(resampled, variant, source_hz=TARGET_HZ)
            windows = make_windows(
                processed,
                feature_cols=feature_cols,
                dataset=str(trial["dataset"]),
                client=str(trial["client"]),
                trial_id=str(trial["trial_id"]),
                label=int(trial["label"]),
                variant=variant,
            )
            for window in windows:
                window["variant"] = variant.name
                window["feature_set"] = ",".join(feature_cols)
            variant_windows.extend(windows)
        variant_df = pd.DataFrame(variant_windows)
        if variant.balance_per_client:
            variant_df = balance_windows_per_client(variant_df)
        all_windows_by_variant.append(variant_df)

    windows_df = pd.concat(all_windows_by_variant, ignore_index=True)
    trials_df = pd.DataFrame(all_trial_records).drop_duplicates().reset_index(drop=True)

    sisfall_client_coverage = (
        trials_df[trials_df["dataset"] == "SisFall"]
        .groupby("client")
        .agg(
            n_trials=("trial_id", "size"),
            fall_trials=("label", "sum"),
        )
        .reset_index()
    )
    sisfall_client_coverage["has_fall_data"] = sisfall_client_coverage["fall_trials"] > 0
    sisfall_client_coverage["fall_ratio_trials"] = (
        sisfall_client_coverage["fall_trials"] / sisfall_client_coverage["n_trials"]
    )

    return windows_df, trials_df, sisfall_client_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiencias de pre-processamento para deteccao de quedas.")
    parser.add_argument("--output-dir", default="preprocessing_outputs")
    parser.add_argument("--max-trials-per-client", type=int, default=None)
    parser.add_argument("--preview-windows-per-variant", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--variants", nargs="+", choices=[variant.name for variant in VARIANTS])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, _, _ = run_experiments(
        output_dir=args.output_dir,
        max_trials_per_client=args.max_trials_per_client,
        preview_windows_per_variant=args.preview_windows_per_variant,
        selected_datasets=args.datasets,
        selected_variants=args.variants,
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.sort_values(["variant", "dataset"]).to_string(index=False))


if __name__ == "__main__":
    main()
