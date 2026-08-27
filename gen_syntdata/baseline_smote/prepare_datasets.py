from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from gen_syntdata.build_mixed_datasets import RATIOS, required_synthetic_rows
from gen_syntdata.dataset import infer_feature_columns
from gen_syntdata.smoke_experiments.prepare_smoke_datasets import sample_original_rows
from gen_syntdata.training_split import split_row_counts, training_rows


DEFAULT_OUTPUT = Path("gen_syntdata/baseline_smote/full_data_datasets")


def generate_smote(
    training_source: pd.DataFrame,
    feature_columns: list[str],
    count: int,
    k_neighbors: int,
    seed: int,
) -> pd.DataFrame:
    minority = training_source[training_source["label"] == 1].copy().reset_index(drop=True)
    if len(minority) <= k_neighbors:
        raise ValueError(
            f"SMOTE requires more than {k_neighbors} minority samples; found {len(minority)}"
        )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    values = imputer.fit_transform(minority[feature_columns])
    scaled = scaler.fit_transform(values)
    neighbors = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neighbors.fit(scaled)
    neighbor_indices = neighbors.kneighbors(return_distance=False)[:, 1:]

    rng = np.random.default_rng(seed)
    base_indices = rng.integers(0, len(scaled), size=count)
    neighbor_positions = rng.integers(0, k_neighbors, size=count)
    selected_neighbors = neighbor_indices[base_indices, neighbor_positions]
    interpolation = rng.random((count, 1))
    generated_scaled = scaled[base_indices] + interpolation * (
        scaled[selected_neighbors] - scaled[base_indices]
    )
    generated = scaler.inverse_transform(generated_scaled)

    synthetic = pd.DataFrame(generated, columns=feature_columns)
    synthetic["dataset"] = "KFall"
    synthetic["client"] = "SA06"
    synthetic["trial_id"] = [f"smote_{index:07d}" for index in range(count)]
    synthetic["label"] = 1
    synthetic["source"] = "synthetic"
    if "window_start_sec" in training_source:
        synthetic["window_start_sec"] = 0.0
    if "window_end_sec" in training_source:
        synthetic["window_end_sec"] = float(training_source["window_seconds"].mode().iloc[0])
    if "window_seconds" in training_source:
        synthetic["window_seconds"] = float(training_source["window_seconds"].mode().iloc[0])
    if "variant" in training_source:
        synthetic["variant"] = "smote_k5"
    if "feature_set" in training_source:
        synthetic["feature_set"] = ",".join(feature_columns)
    for column in training_source.columns:
        if column not in synthetic:
            synthetic[column] = np.nan
    return synthetic[training_source.columns.tolist() + ["source"]]


def write_dataset(
    folder: Path,
    original: pd.DataFrame,
    synthetic_pool: pd.DataFrame,
    synthetic_ratio: float,
) -> dict[str, object]:
    folder.mkdir(parents=True, exist_ok=True)
    target = required_synthetic_rows(len(original), synthetic_ratio)
    synthetic = synthetic_pool.head(target).copy()
    original_out = original.copy()
    original_out["source"] = "original"
    mixed = pd.concat([original_out, synthetic], ignore_index=True)
    original_out.to_csv(folder / "original_windows.csv", index=False)
    synthetic.to_csv(folder / "synthetic_windows.csv", index=False)
    mixed.to_csv(folder / "mixed_windows.csv", index=False)
    summary = {
        "method": "SMOTE",
        "minority_label": 1,
        "original_rows": len(original_out),
        "synthetic_rows": len(synthetic),
        "total_rows": len(mixed),
        "target_synthetic_ratio": synthetic_ratio,
        "actual_synthetic_ratio_by_rows": len(synthetic) / len(mixed),
        "original_labels_by_rows": {
            str(k): int(v) for k, v in original_out["label"].value_counts().sort_index().items()
        },
        "synthetic_labels_by_rows": {
            str(k): int(v) for k, v in synthetic["label"].value_counts().sort_index().items()
        },
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare four comparable SMOTE datasets.")
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("fl/data/preprocessing_results/composed/sequence_ready_nn/windows.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--original-max-rows", type=int, default=0, help="0 uses all real rows.")
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_full = pd.read_csv(args.original).dropna(axis=1, how="all")
    original = sample_original_rows(original_full, args.original_max_rows, args.seed)
    generator_training = training_rows(original)
    feature_columns = infer_feature_columns(original)
    max_required = max(required_synthetic_rows(len(original), ratio) for ratio in RATIOS.values())
    synthetic_pool = generate_smote(
        generator_training, feature_columns, max_required, args.k_neighbors, args.seed
    )
    if set(synthetic_pool["label"].unique()) != {1}:
        raise AssertionError("SMOTE baseline must generate minority-class falls only")
    synthetic_pool.to_csv(args.output_dir / "synthetic_pool.csv", index=False)

    summaries = []
    for ratio_name, ratio in RATIOS.items():
        summary = write_dataset(args.output_dir / ratio_name, original, synthetic_pool, ratio)
        summary["mixture"] = ratio_name
        summaries.append(summary)
        print(f"Prepared SMOTE {ratio_name}: synthetic={summary['synthetic_rows']}")

    manifest = {
        "method": "SMOTE",
        "implementation": "standardized k-nearest-neighbor interpolation",
        "formula": "x_new = x_i + lambda * (x_neighbor - x_i)",
        "minority_label": 1,
        "k_neighbors": args.k_neighbors,
        "seed": args.seed,
        "source_original": str(args.original),
        "original_rows_fixed": len(original),
        "smote_fit_rows": len(generator_training),
        "real_split_rows": split_row_counts(original),
        "smote_fit_split": "train_only",
        "dev_test_used_for_smote": False,
        "feature_count": len(feature_columns),
        "ratios": RATIOS,
        "datasets": summaries,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"SMOTE datasets exported to {args.output_dir}")


if __name__ == "__main__":
    main()
