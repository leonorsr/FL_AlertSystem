from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gen_syntdata.build_mixed_datasets import RATIOS, required_synthetic_rows
from gen_syntdata.dataset import _resample_sequence, infer_feature_columns
from gen_syntdata.smoke_experiments.prepare_smoke_datasets import (
    GENERATOR_DESCRIPTIONS,
    generate_crnngan_lite,
    generate_doppelganger_lite,
    generate_rcgan_lite,
    generate_wavegan_lite,
    sample_original_rows,
)
from gen_syntdata.training_split import split_row_counts, training_rows


DEFAULT_ROOT = Path("gen_syntdata/experiments_gans")
DEFAULT_OUTPUT = DEFAULT_ROOT / "full_data_datasets"
DEFAULT_TIMEGAN_RUNS = DEFAULT_ROOT / "timegan_runs"
CLASS_LABELS = (0, 1)


def load_class_sequences(
    original: pd.DataFrame,
    feature_columns: list[str],
    label: int,
    seq_len: int,
    max_sequences: int,
    seed: int,
) -> np.ndarray:
    selected = original[original["label"] == label].copy()
    sort_columns = [c for c in ["window_start_sec", "window_end_sec"] if c in selected]
    sequences: list[np.ndarray] = []
    for _, group in selected.groupby(["dataset", "trial_id"], sort=True):
        ordered = group.sort_values(sort_columns) if sort_columns else group
        values = ordered[feature_columns].to_numpy(dtype=np.float32)
        if len(values):
            sequences.append(_resample_sequence(values, seq_len))
    if not sequences:
        raise ValueError(f"No reference sequences found for label={label}")
    if max_sequences > 0 and len(sequences) > max_sequences:
        rng = np.random.default_rng(seed + label)
        indices = rng.choice(len(sequences), max_sequences, replace=False)
        sequences = [sequences[int(i)] for i in indices]
    return np.stack(sequences).astype(np.float32)


def sequences_to_frame(
    sequences: np.ndarray,
    feature_columns: list[str],
    generator: str,
    label: int,
) -> pd.DataFrame:
    count, seq_len, feature_count = sequences.shape
    frame = pd.DataFrame(sequences.reshape(-1, feature_count), columns=feature_columns)
    frame.insert(0, "label", int(label))
    frame.insert(0, "timestep", np.tile(np.arange(seq_len), count))
    frame.insert(
        0,
        "trial_id",
        np.repeat([f"{generator}_class{label}_{index:07d}" for index in range(count)], seq_len),
    )
    frame.insert(0, "client", generator)
    frame.insert(0, "dataset", "Synthetic")
    return frame


def generate_lite(
    generator: str,
    reference: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    functions = {
        "rcgan_lite": generate_rcgan_lite,
        "crnngan_lite": generate_crnngan_lite,
        "wavegan_lite": generate_wavegan_lite,
        "doppelganger_lite": generate_doppelganger_lite,
    }
    return functions[generator](reference, count, rng)


def ensure_timegan_models(
    sample_path: Path,
    runs_dir: Path,
    seq_len: int,
    seed: int,
    embedding_epochs: int,
    supervised_epochs: int,
    joint_epochs: int,
    force: bool,
) -> None:
    for label in CLASS_LABELS:
        run_dir = runs_dir / f"class_{label}"
        if (run_dir / "timegan.pt").exists() and not force:
            continue
        command = [
            sys.executable,
            "-m",
            "gen_syntdata.train_timegan",
            "--input",
            str(sample_path),
            "--output-dir",
            str(run_dir),
            "--label",
            str(label),
            "--seq-len",
            str(seq_len),
            "--embedding-epochs",
            str(embedding_epochs),
            "--supervised-epochs",
            str(supervised_epochs),
            "--joint-epochs",
            str(joint_epochs),
            "--seed",
            str(seed + label),
        ]
        subprocess.run(command, check=True)


def generate_timegan_class(
    runs_dir: Path,
    label: int,
    count: int,
    output_path: Path,
) -> pd.DataFrame:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gen_syntdata.generate_timegan",
            "--run-dir",
            str(runs_dir / f"class_{label}"),
            "--count",
            str(count),
            "--output",
            str(output_path),
            "--label",
            str(label),
            "--sequence-prefix",
            f"timegan_class{label}",
        ],
        check=True,
    )
    frame = pd.read_csv(output_path)
    return frame


def normalize_two_class_synthetic(
    synthetic: pd.DataFrame,
    original: pd.DataFrame,
    feature_columns: list[str],
    generator: str,
) -> pd.DataFrame:
    result = synthetic.copy()
    labels = set(pd.to_numeric(result["label"], errors="raise").astype(int).unique())
    if labels != {0, 1}:
        raise ValueError(f"{generator} must generate labels 0 and 1; found {sorted(labels)}")
    result["label"] = result["label"].astype(int)
    result["dataset"] = "KFall"
    result["client"] = "SA06"
    result["source"] = "synthetic"
    window_seconds = float(original["window_seconds"].mode().iloc[0]) if "window_seconds" in original else 1.0
    if "window_start_sec" in original:
        result["window_start_sec"] = result["timestep"].astype(float) * window_seconds
    if "window_end_sec" in original:
        result["window_end_sec"] = result["window_start_sec"] + window_seconds
    if "window_seconds" in original:
        result["window_seconds"] = window_seconds
    if "variant" in original:
        result["variant"] = generator
    if "feature_set" in original:
        result["feature_set"] = ",".join(feature_columns)
    for column in original.columns:
        if column not in result:
            result[column] = np.nan
    return result[original.columns.tolist() + ["source"]]


def allocate_class_rows(total: int, class_proportions: dict[int, float]) -> dict[int, int]:
    label_zero = int(round(total * class_proportions[0]))
    return {0: label_zero, 1: total - label_zero}


def write_dataset(
    folder: Path,
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    synthetic_ratio: float,
    class_proportions: dict[int, float],
) -> dict[str, object]:
    folder.mkdir(parents=True, exist_ok=True)
    target = required_synthetic_rows(len(original), synthetic_ratio)
    counts = allocate_class_rows(target, class_proportions)
    pieces = []
    for label in CLASS_LABELS:
        available = synthetic[synthetic["label"] == label]
        if len(available) < counts[label]:
            raise ValueError(f"Need {counts[label]} rows for label={label}, found {len(available)}")
        pieces.append(available.head(counts[label]))
    selected = pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    original_out = original.copy()
    original_out["source"] = "original"
    mixed = pd.concat([original_out, selected], ignore_index=True)
    original_out.to_csv(folder / "original_windows.csv", index=False)
    selected.to_csv(folder / "synthetic_windows.csv", index=False)
    mixed.to_csv(folder / "mixed_windows.csv", index=False)
    summary = {
        "original_rows": len(original_out),
        "synthetic_rows": len(selected),
        "total_rows": len(mixed),
        "target_synthetic_ratio": synthetic_ratio,
        "actual_synthetic_ratio_by_rows": len(selected) / len(mixed),
        "original_labels_by_rows": {str(k): int(v) for k, v in original_out["label"].value_counts().sort_index().items()},
        "synthetic_labels_by_rows": {str(k): int(v) for k, v in selected["label"].value_counts().sort_index().items()},
        "synthetic_class_policy": "both_classes",
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 5 x 4 two-class GAN smoke datasets.")
    parser.add_argument("--original", type=Path, default=Path("fl/data/preprocessing_results/composed/sequence_ready_nn/windows.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timegan-runs-dir", type=Path, default=DEFAULT_TIMEGAN_RUNS)
    parser.add_argument("--original-max-rows", type=int, default=0, help="0 uses all real rows.")
    parser.add_argument("--max-train-sequences", type=int, default=0, help="0 uses all training sequences.")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--embedding-epochs", type=int, default=100)
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--joint-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-models", action="store_true")
    args = parser.parse_args()

    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_full = pd.read_csv(args.original).dropna(axis=1, how="all")
    original = sample_original_rows(original_full, args.original_max_rows, args.seed)
    generator_training = training_rows(original)
    sample_path = args.output_dir / "generator_training_rows.csv"
    generator_training.to_csv(sample_path, index=False)
    feature_columns = infer_feature_columns(original)
    proportions = generator_training["label"].value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0.0)
    class_proportions = {label: float(proportions[label]) for label in CLASS_LABELS}
    if not all(value > 0 for value in class_proportions.values()):
        raise ValueError(f"Original sample must contain both classes: {class_proportions}")

    references = {
        label: load_class_sequences(generator_training, feature_columns, label, args.seq_len, args.max_train_sequences, args.seed)
        for label in CLASS_LABELS
    }
    max_rows = max(required_synthetic_rows(len(original), ratio) for ratio in RATIOS.values())
    max_counts = allocate_class_rows(max_rows, class_proportions)
    sequence_counts = {label: math.ceil(max_counts[label] / args.seq_len) for label in CLASS_LABELS}

    ensure_timegan_models(
        sample_path, args.timegan_runs_dir, args.seq_len, args.seed,
        args.embedding_epochs, args.supervised_epochs, args.joint_epochs, args.force_models,
    )
    summaries: list[dict[str, object]] = []
    for generator in GENERATOR_DESCRIPTIONS:
        generator_dir = args.output_dir / generator
        generator_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for label in CLASS_LABELS:
            if generator == "timegan":
                raw_path = generator_dir / f"synthetic_class_{label}.csv"
                frame = generate_timegan_class(args.timegan_runs_dir, label, sequence_counts[label], raw_path)
            else:
                rng = np.random.default_rng(args.seed + label)
                sequences = generate_lite(generator, references[label], sequence_counts[label], rng)
                frame = sequences_to_frame(sequences, feature_columns, generator, label)
            frames.append(frame)
        synthetic_raw = pd.concat(frames, ignore_index=True)
        synthetic_raw.to_csv(generator_dir / "synthetic_raw.csv", index=False)
        synthetic = normalize_two_class_synthetic(synthetic_raw, original, feature_columns, generator)
        for ratio_name, ratio in RATIOS.items():
            summary = write_dataset(generator_dir / ratio_name, original, synthetic, ratio, class_proportions)
            summary.update({"generator": generator, "mixture": ratio_name})
            summaries.append(summary)
        print(f"Prepared both classes for {generator}: {generator_dir}")

    manifest = {
        "experiment": "gans_both_classes_smoke_grid",
        "synthetic_class_policy": "both_classes",
        "required_synthetic_labels": [0, 1],
        "original": str(args.original),
        "original_rows": len(original),
        "generator_training_rows": len(generator_training),
        "real_split_rows": split_row_counts(original),
        "generator_fit_split": "train_only",
        "dev_test_used_for_generation": False,
        "class_proportions": {str(k): v for k, v in class_proportions.items()},
        "ratios": RATIOS,
        "generators": GENERATOR_DESCRIPTIONS,
        "dataset_count": len(GENERATOR_DESCRIPTIONS) * len(RATIOS),
        "timegan_training_epochs": {
            "embedding": args.embedding_epochs,
            "supervised": args.supervised_epochs,
            "joint": args.joint_epochs,
        },
        "datasets": summaries,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared and validated 20 two-class datasets in {args.output_dir}")


if __name__ == "__main__":
    main()
