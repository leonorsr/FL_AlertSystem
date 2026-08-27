from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from gen_syntdata.build_mixed_datasets import normalize_original
from gen_syntdata.dataset import infer_feature_columns
from gen_syntdata.prepare_experiment_gan_datasets import DATASETS, build_dataset


DEFAULT_OUTPUT_DIR = Path("gen_syntdata/experiments_gans_falls")
DEFAULT_ORIGINAL = Path("fl/data/preprocessing_results/composed/sequence_ready_nn/windows.csv")
DEFAULT_TIMEGAN = Path("gen_syntdata/runs/falls_timegan/synthetic_falls_for_mixtures.csv")


def validate_falls_only(frame: pd.DataFrame, source: str) -> None:
    """Reject synthetic input/output containing anything other than fall label 1."""
    if "label" not in frame.columns:
        return
    labels = set(pd.to_numeric(frame["label"], errors="raise").dropna().astype(int).unique())
    if labels != {1}:
        raise ValueError(f"{source} must contain synthetic falls only (label=1); found {sorted(labels)}")


def clear_generated_outputs(output_dir: Path) -> None:
    """Remove generated datasets without deleting this experiment's source files."""
    for dataset_name in DATASETS:
        dataset_dir = output_dir / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated legacy entry point."
    )
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--timegan-synthetic", type=Path, default=DEFAULT_TIMEGAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-reference-sequences", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    parser.error(
        "This command implements the legacy selected-configuration workflow and is disabled "
        "to prevent train/dev/test leakage. Use: venv/bin/python -m "
        "gen_syntdata.experiments_gans_falls.prepare_smoke_datasets --force --force-model"
    )

    if args.force:
        clear_generated_outputs(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_full = pd.read_csv(args.original).dropna(axis=1, how="all")
    timegan_source = pd.read_csv(args.timegan_synthetic)
    validate_falls_only(timegan_source, str(args.timegan_synthetic))

    feature_columns = infer_feature_columns(original_full)
    original = normalize_original(original_full, feature_columns)
    summaries: list[dict[str, object]] = []

    for dataset_name, config in DATASETS.items():
        summary = build_dataset(
            output_dir=args.output_dir,
            dataset_name=dataset_name,
            config=config,
            original_full=original_full,
            original=original,
            feature_columns=feature_columns,
            seq_len=args.seq_len,
            max_reference_sequences=args.max_reference_sequences,
            seed=args.seed,
            original_path=args.original,
            timegan_synthetic_path=args.timegan_synthetic,
        )
        synthetic_path = (
            args.output_dir / dataset_name / str(config["ratio_name"]) / "synthetic_windows.csv"
        )
        validate_falls_only(pd.read_csv(synthetic_path, usecols=["label"]), str(synthetic_path))
        summary["synthetic_class"] = "falls_only"
        summary["synthetic_label"] = 1
        summary_path = synthetic_path.parent / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(
            f"{dataset_name}/{summary['ratio_name']}: original={summary['original_rows']} "
            f"synthetic_falls={summary['synthetic_rows']}"
        )

    manifest = {
        "experiment": "gans_falls_only",
        "output_dir": str(args.output_dir),
        "source_original": str(args.original),
        "source_timegan_synthetic": str(args.timegan_synthetic),
        "synthetic_class": "falls_only",
        "synthetic_label": 1,
        "real_data_classes_preserved": [0, 1],
        "all_original_rows_used": True,
        "seq_len": args.seq_len,
        "max_reference_sequences": args.max_reference_sequences,
        "seed": args.seed,
        "datasets": summaries,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Falls-only GAN datasets exported to: {args.output_dir}")


if __name__ == "__main__":
    main()
