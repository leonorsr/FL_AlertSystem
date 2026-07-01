from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fl_preprocessing_experiments import DATASET_LOADERS, VARIANTS, run_experiments, run_experiments_detailed


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "preprocessing_results"

SIMPLE_VARIANTS = {
    "baseline_zscore",
    "robust_clip",
    "magnitude_features",
    "magnitude_only",
    "client_zscore",
}


def strategy_family(variant_name: str) -> str:
    return "simple" if variant_name in SIMPLE_VARIANTS else "composed"


def build_strategy_catalog() -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        record = asdict(variant)
        record["family"] = strategy_family(variant.name)
        rows.append(record)
    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def export_variant_folder(
    output_root: Path,
    variant_name: str,
    summary_df: pd.DataFrame,
    client_summary_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> None:
    variant = next(v for v in VARIANTS if v.name == variant_name)
    family = strategy_family(variant_name)
    variant_dir = output_root / family / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(variant_dir / "summary.csv", index=False)
    client_summary_df.to_csv(variant_dir / "client_summary.csv", index=False)
    windows_df.to_csv(variant_dir / "windows.csv", index=False)

    config = asdict(variant)
    config["family"] = family
    config["window_seconds"] = variant.window_seconds
    config["target_hz"] = 20
    write_json(variant_dir / "config.json", config)


def create_preprocessing_datasets(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_trials_per_client: int | None = None,
    preview_windows_per_variant: int = 2,
    selected_datasets: list[str] | None = None,
    selected_variants: list[str] | None = None,
) -> None:
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "simple").mkdir(exist_ok=True)
    (output_root / "composed").mkdir(exist_ok=True)
    (output_root / "comparisons").mkdir(exist_ok=True)

    summary, client_summary, _ = run_experiments(
        output_dir=output_root / "comparisons" / "_scratch_exports",
        max_trials_per_client=max_trials_per_client,
        preview_windows_per_variant=preview_windows_per_variant,
        selected_datasets=selected_datasets,
        selected_variants=selected_variants,
    )
    windows_df, trials_df, sisfall_client_coverage = run_experiments_detailed(
        max_trials_per_client=max_trials_per_client,
        selected_datasets=selected_datasets,
        selected_variants=selected_variants,
    )

    strategy_catalog = build_strategy_catalog()
    if selected_variants is not None:
        strategy_catalog = strategy_catalog[strategy_catalog["name"].isin(selected_variants)].copy()
    strategy_catalog.to_csv(output_root / "comparisons" / "strategy_catalog.csv", index=False)
    summary.to_csv(output_root / "comparisons" / "global_summary.csv", index=False)
    client_summary.to_csv(output_root / "comparisons" / "global_client_summary.csv", index=False)
    windows_df.to_csv(output_root / "comparisons" / "all_windows.csv", index=False)
    trials_df.to_csv(output_root / "comparisons" / "trial_catalog.csv", index=False)
    sisfall_client_coverage.to_csv(output_root / "comparisons" / "sisfall_client_coverage.csv", index=False)

    for variant_name in strategy_catalog["name"]:
        variant_summary = summary[summary["variant"] == variant_name].copy()
        variant_client_summary = client_summary[client_summary["variant"] == variant_name].copy()
        variant_windows = windows_df[windows_df["variant"] == variant_name].copy()
        export_variant_folder(
            output_root=output_root,
            variant_name=variant_name,
            summary_df=variant_summary,
            client_summary_df=variant_client_summary,
            windows_df=variant_windows,
        )

    manifest = {
        "output_root": str(output_root),
        "families": ["simple", "composed"],
        "comparison_files": [
            "strategy_catalog.csv",
            "global_summary.csv",
            "global_client_summary.csv",
            "all_windows.csv",
            "trial_catalog.csv",
            "sisfall_client_coverage.csv",
        ],
        "max_trials_per_client": max_trials_per_client,
        "preview_windows_per_variant": preview_windows_per_variant,
        "selected_datasets": selected_datasets,
        "selected_variants": selected_variants,
        "n_variants": int(strategy_catalog.shape[0]),
    }
    write_json(output_root / "comparisons" / "manifest.json", manifest)

    scratch_dir = output_root / "comparisons" / "_scratch_exports"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta datasets pre-processados por estrategia.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-trials-per-client", type=int, default=None)
    parser.add_argument("--preview-windows-per-variant", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--variants", nargs="+", choices=[variant.name for variant in VARIANTS])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_preprocessing_datasets(
        output_dir=args.output_dir,
        max_trials_per_client=args.max_trials_per_client,
        preview_windows_per_variant=args.preview_windows_per_variant,
        selected_datasets=args.datasets,
        selected_variants=args.variants,
    )
    print(f"Preprocessing datasets exported to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
