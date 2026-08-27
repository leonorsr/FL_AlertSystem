from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from gen_syntdata.build_mixed_datasets import RATIOS
from gen_syntdata.build_mixed_datasets import required_synthetic_rows
from gen_syntdata.experiments_gans_falls.prepare_datasets import validate_falls_only
from gen_syntdata.smoke_experiments.prepare_smoke_datasets import GENERATOR_DESCRIPTIONS
from gen_syntdata.smoke_experiments.prepare_smoke_datasets import sample_original_rows
from gen_syntdata.training_split import split_row_counts, training_rows


DEFAULT_OUTPUT = Path("gen_syntdata/experiments_gans_falls/full_data_datasets")
DEFAULT_TIMEGAN_RUN = Path("gen_syntdata/experiments_gans_falls/timegan_runs/class_1")


def ensure_fall_timegan_source(
    original_path: Path,
    output_dir: Path,
    run_dir: Path,
    seq_len: int,
    seed: int,
    embedding_epochs: int,
    supervised_epochs: int,
    joint_epochs: int,
    force_model: bool,
    original_max_rows: int,
) -> tuple[Path, dict[str, int]]:
    original_full = pd.read_csv(original_path).dropna(axis=1, how="all")
    original = sample_original_rows(original_full, original_max_rows, seed)
    source = training_rows(original)
    run_dir.mkdir(parents=True, exist_ok=True)
    training_path = run_dir / "generator_training_rows.csv"
    source.to_csv(training_path, index=False)
    if force_model or not (run_dir / "timegan.pt").exists():
        subprocess.run(
            [
                sys.executable, "-m", "gen_syntdata.train_timegan",
                "--input", str(training_path), "--output-dir", str(run_dir),
                "--label", "1", "--seq-len", str(seq_len),
                "--embedding-epochs", str(embedding_epochs),
                "--supervised-epochs", str(supervised_epochs),
                "--joint-epochs", str(joint_epochs), "--seed", str(seed),
            ],
            check=True,
        )
    required_rows = max(required_synthetic_rows(len(original), ratio) for ratio in RATIOS.values())
    sequence_count = math.ceil(required_rows / seq_len)
    synthetic_path = run_dir / "synthetic_falls_full.csv"
    subprocess.run(
        [
            sys.executable, "-m", "gen_syntdata.generate_timegan",
            "--run-dir", str(run_dir), "--count", str(sequence_count),
            "--output", str(synthetic_path),
        ],
        check=True,
    )
    return synthetic_path, split_row_counts(original)


def validate_outputs(output_dir: Path) -> None:
    expected = len(GENERATOR_DESCRIPTIONS) * len(RATIOS)
    checked = 0
    for generator in GENERATOR_DESCRIPTIONS:
        raw_path = output_dir / generator / "synthetic_raw.csv"
        for chunk in pd.read_csv(raw_path, usecols=["label"], chunksize=100_000):
            validate_falls_only(chunk, str(raw_path))
        for ratio in RATIOS:
            synthetic_path = output_dir / generator / ratio / "synthetic_windows.csv"
            for chunk in pd.read_csv(synthetic_path, usecols=["label"], chunksize=100_000):
                validate_falls_only(chunk, str(synthetic_path))
            checked += 1
    if checked != expected:
        raise AssertionError(f"Expected {expected} datasets, validated {checked}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 5 x 4 falls-only GAN smoke datasets.")
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("fl/data/preprocessing_results/composed/sequence_ready_nn/windows.csv"),
    )
    parser.add_argument("--timegan-run-dir", type=Path, default=DEFAULT_TIMEGAN_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--original-max-rows", type=int, default=0, help="0 uses all real rows.")
    parser.add_argument("--max-train-sequences", type=int, default=0, help="0 uses all training sequences.")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--embedding-epochs", type=int, default=100)
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--joint-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-model", action="store_true")
    args = parser.parse_args()

    timegan_synthetic, source_split_counts = ensure_fall_timegan_source(
        args.original, args.output_dir, args.timegan_run_dir, args.seq_len, args.seed,
        args.embedding_epochs, args.supervised_epochs, args.joint_epochs, args.force_model,
        args.original_max_rows,
    )

    command = [
        sys.executable,
        "-m",
        "gen_syntdata.smoke_experiments.prepare_smoke_datasets",
        "--original",
        str(args.original),
        "--timegan-synthetic",
        str(timegan_synthetic),
        "--output-dir",
        str(args.output_dir),
        "--original-max-rows",
        str(args.original_max_rows),
        "--max-train-sequences",
        str(args.max_train_sequences),
        "--seq-len",
        str(args.seq_len),
        "--seed",
        str(args.seed),
    ]
    if args.force:
        command.append("--force")
    subprocess.run(command, check=True)
    validate_outputs(args.output_dir)

    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": "gans_falls_only_smoke_grid",
            "synthetic_class": "falls_only",
            "synthetic_label": 1,
            "dataset_count": len(GENERATOR_DESCRIPTIONS) * len(RATIOS),
            "original_rows_fixed": sum(source_split_counts.values()),
            "real_split_rows": source_split_counts,
            "generator_fit_split": "train_only",
            "dev_test_used_for_generation": False,
            "timegan_run_dir": str(args.timegan_run_dir),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Validated 20 falls-only smoke datasets in {args.output_dir}")


if __name__ == "__main__":
    main()
