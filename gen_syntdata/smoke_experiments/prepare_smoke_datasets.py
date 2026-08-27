from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gen_syntdata.build_mixed_datasets import (
    RATIOS,
    feature_columns_from_synthetic,
    normalize_original,
    normalize_synthetic,
    required_synthetic_rows,
    write_dataset,
)
from gen_syntdata.dataset import _resample_sequence, infer_feature_columns
from gen_syntdata.training_split import split_row_counts, training_rows


GENERATOR_DESCRIPTIONS = {
    "timegan": "TimeGAN already trained in gen_syntdata/runs/falls_timegan.",
    "rcgan_lite": "RCGAN-style smoke generator: real fall sequences plus conditional scale/jitter.",
    "crnngan_lite": "C-RNN-GAN-style smoke generator: transition sampling over temporal deltas.",
    "wavegan_lite": "WaveGAN-style smoke generator: frequency-domain phase/noise perturbation.",
    "doppelganger_lite": "DoppelGANger-style smoke generator: sequence attributes plus AR dynamics.",
}


def sample_original_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        sampled = df.copy()
    else:
        rng = np.random.default_rng(seed)
        groups = list(df.groupby(["dataset", "client", "label"], sort=True))
        base_take = max(1, max_rows // max(len(groups), 1))
        pieces: list[pd.DataFrame] = []
        for _, group in groups:
            take = min(len(group), base_take)
            pieces.append(group.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
        sampled = pd.concat(pieces, ignore_index=True)
        if len(sampled) < max_rows:
            remaining = df.drop(sampled.index, errors="ignore")
            if not remaining.empty:
                take = min(max_rows - len(sampled), len(remaining))
                sampled = pd.concat(
                    [sampled, remaining.sample(n=take, random_state=int(rng.integers(0, 1_000_000)))],
                    ignore_index=True,
                )
        if len(sampled) > max_rows:
            sampled = sampled.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return sampled.sort_values(["dataset", "client", "trial_id", "window_start_sec"]).reset_index(drop=True)


def load_fall_sequences(
    original_df: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
    max_sequences: int,
    seed: int,
) -> np.ndarray:
    fall_df = original_df[original_df["label"] == 1].copy()
    sort_columns = [column for column in ["window_start_sec", "window_end_sec"] if column in fall_df.columns]
    sequences: list[np.ndarray] = []
    for _, group in fall_df.groupby(["dataset", "trial_id"], sort=True):
        ordered = group.sort_values(sort_columns) if sort_columns else group
        values = ordered[feature_columns].to_numpy(dtype=np.float32)
        if len(values) > 0:
            sequences.append(_resample_sequence(values, seq_len))
    if not sequences:
        raise ValueError("No fall sequences found in the original data.")
    rng = np.random.default_rng(seed)
    if max_sequences > 0 and len(sequences) > max_sequences:
        indices = rng.choice(len(sequences), size=max_sequences, replace=False)
        sequences = [sequences[int(idx)] for idx in indices]
    return np.stack(sequences, axis=0).astype(np.float32)


def clip_to_reference(sequences: np.ndarray, reference: np.ndarray) -> np.ndarray:
    low = np.nanpercentile(reference, 0.5, axis=(0, 1))
    high = np.nanpercentile(reference, 99.5, axis=(0, 1))
    return np.clip(sequences, low.reshape(1, 1, -1), high.reshape(1, 1, -1)).astype(np.float32)


def generate_rcgan_lite(reference: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, len(reference), size=count)
    base = reference[indices].copy()
    feature_std = np.nanstd(reference, axis=(0, 1)).reshape(1, 1, -1)
    scale = rng.normal(1.0, 0.08, size=(count, 1, 1))
    jitter = rng.normal(0.0, 0.05, size=base.shape) * np.maximum(feature_std, 1e-6)
    return clip_to_reference((base * scale) + jitter, reference)


def generate_crnngan_lite(reference: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    seq_len = reference.shape[1]
    feature_dim = reference.shape[2]
    starts = reference[:, 0, :]
    deltas = np.diff(reference, axis=1).reshape(-1, feature_dim)
    generated = np.empty((count, seq_len, feature_dim), dtype=np.float32)
    for idx in range(count):
        generated[idx, 0] = starts[int(rng.integers(0, len(starts)))]
        delta_indices = rng.integers(0, len(deltas), size=seq_len - 1)
        noise = rng.normal(0.0, 0.03, size=(seq_len - 1, feature_dim)) * np.maximum(np.std(deltas, axis=0), 1e-6)
        generated[idx, 1:] = generated[idx, 0] + np.cumsum(deltas[delta_indices] + noise, axis=0)
    return clip_to_reference(generated, reference)


def generate_wavegan_lite(reference: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    generated = np.empty((count, reference.shape[1], reference.shape[2]), dtype=np.float32)
    for idx in range(count):
        base = reference[int(rng.integers(0, len(reference)))]
        spectrum = np.fft.rfft(base, axis=0)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        phase_noise = rng.normal(0.0, 0.25, size=phase.shape)
        amp_noise = rng.normal(1.0, 0.05, size=magnitude.shape)
        perturbed = magnitude * amp_noise * np.exp(1j * (phase + phase_noise))
        generated[idx] = np.fft.irfft(perturbed, n=reference.shape[1], axis=0).real
    return clip_to_reference(generated, reference)


def generate_doppelganger_lite(reference: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    seq_mean = reference.mean(axis=1)
    seq_std = np.maximum(reference.std(axis=1), 1e-6)
    global_std = np.maximum(reference.std(axis=(0, 1)), 1e-6)
    generated = np.empty((count, reference.shape[1], reference.shape[2]), dtype=np.float32)
    for idx in range(count):
        attr_idx = int(rng.integers(0, len(reference)))
        mean = seq_mean[attr_idx] + rng.normal(0.0, 0.08, size=reference.shape[2]) * global_std
        std = seq_std[attr_idx] * rng.lognormal(mean=0.0, sigma=0.08, size=reference.shape[2])
        phi = rng.uniform(0.70, 0.95, size=reference.shape[2])
        generated[idx, 0] = mean + rng.normal(0.0, 1.0, size=reference.shape[2]) * std
        for timestep in range(1, reference.shape[1]):
            innovation = rng.normal(0.0, 1.0, size=reference.shape[2]) * std * 0.35
            generated[idx, timestep] = mean + phi * (generated[idx, timestep - 1] - mean) + innovation
    return clip_to_reference(generated, reference)


def sequences_to_frame(sequences: np.ndarray, feature_columns: list[str], generator_name: str) -> pd.DataFrame:
    count, seq_len, feature_count = sequences.shape
    frame = pd.DataFrame(sequences.reshape(-1, feature_count), columns=feature_columns)
    frame.insert(0, "label", 1)
    frame.insert(0, "timestep", np.tile(np.arange(seq_len), count))
    frame.insert(
        0,
        "trial_id",
        np.repeat([f"{generator_name}_{index:07d}" for index in range(count)], seq_len),
    )
    frame.insert(0, "client", generator_name)
    frame.insert(0, "dataset", "Synthetic")
    return frame


def build_generator_synthetic(
    generator_name: str,
    reference: np.ndarray,
    count_sequences: int,
    feature_columns: list[str],
    rng: np.random.Generator,
    timegan_path: Path,
) -> pd.DataFrame:
    if generator_name == "timegan":
        synthetic = pd.read_csv(timegan_path)
        required_rows = count_sequences * reference.shape[1]
        if len(synthetic) < required_rows:
            raise ValueError(f"{timegan_path} only has {len(synthetic)} rows; need {required_rows}.")
        return synthetic.head(required_rows).copy()
    if generator_name == "rcgan_lite":
        sequences = generate_rcgan_lite(reference, count_sequences, rng)
    elif generator_name == "crnngan_lite":
        sequences = generate_crnngan_lite(reference, count_sequences, rng)
    elif generator_name == "wavegan_lite":
        sequences = generate_wavegan_lite(reference, count_sequences, rng)
    elif generator_name == "doppelganger_lite":
        sequences = generate_doppelganger_lite(reference, count_sequences, rng)
    else:
        raise KeyError(f"Unknown generator: {generator_name}")
    return sequences_to_frame(sequences, feature_columns, generator_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare small GAN-family smoke datasets.")
    parser.add_argument("--original", type=Path, default=Path("fl/data/preprocessing_results/composed/sequence_ready_nn/windows.csv"))
    parser.add_argument("--timegan-synthetic", type=Path, default=Path("gen_syntdata/runs/falls_timegan/synthetic_falls_for_mixtures.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("gen_syntdata/smoke_experiments/datasets"))
    parser.add_argument("--original-max-rows", type=int, default=20000)
    parser.add_argument("--max-train-sequences", type=int, default=600)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    original_full = pd.read_csv(args.original).dropna(axis=1, how="all")
    feature_columns = infer_feature_columns(original_full)
    original_sample = sample_original_rows(original_full, args.original_max_rows, args.seed)
    generator_training = training_rows(original_sample)
    original = normalize_original(original_sample, feature_columns)
    reference = load_fall_sequences(generator_training, feature_columns, args.seq_len, args.max_train_sequences, args.seed)

    max_required_rows = max(required_synthetic_rows(len(original), ratio) for ratio in RATIOS.values())
    count_sequences = int(math.ceil(max_required_rows / args.seq_len))
    if args.force and args.output_dir.exists():
        import shutil

        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "original": str(args.original),
        "timegan_synthetic": str(args.timegan_synthetic),
        "original_rows_fixed": int(len(original)),
        "generator_training_rows": int(len(generator_training)),
        "real_split_rows": split_row_counts(original_sample),
        "generator_fit_split": "train_only",
        "dev_test_used_for_generation": False,
        "seq_len": int(args.seq_len),
        "synthetic_sequences_per_generator": int(count_sequences),
        "ratios": RATIOS,
        "generators": GENERATOR_DESCRIPTIONS,
    }

    for generator_name in GENERATOR_DESCRIPTIONS:
        generator_dir = args.output_dir / generator_name
        generator_dir.mkdir(parents=True, exist_ok=True)
        synthetic_raw = build_generator_synthetic(
            generator_name=generator_name,
            reference=reference,
            count_sequences=count_sequences,
            feature_columns=feature_columns,
            rng=rng,
            timegan_path=args.timegan_synthetic,
        )
        synthetic_raw.to_csv(generator_dir / "synthetic_raw.csv", index=False)
        synthetic_feature_columns = feature_columns_from_synthetic(synthetic_raw)
        synthetic = normalize_synthetic(
            synthetic_df=synthetic_raw,
            original_columns=original_sample.columns.tolist(),
            feature_columns=synthetic_feature_columns,
            synthetic_dataset="KFall",
            synthetic_client="SA06",
            window_seconds=float(original_sample["window_seconds"].mode().iloc[0]) if "window_seconds" in original_sample.columns else 1.0,
        )
        for ratio_name, synthetic_ratio in RATIOS.items():
            write_dataset(generator_dir / ratio_name, original, synthetic, synthetic_ratio)
        print(f"Prepared {generator_name}: {generator_dir}")

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Smoke datasets ready in {args.output_dir}")


if __name__ == "__main__":
    main()
