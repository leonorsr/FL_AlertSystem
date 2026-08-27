from __future__ import annotations

import argparse
from pathlib import Path

import joblib

try:
    from .dataset import save_generated_csv
    from .timegan import TimeGAN, _device
except ImportError:
    from dataset import save_generated_csv
    from timegan import TimeGAN, _device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fall-detection sequences from a trained TimeGAN.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Directory created by train_timegan.py.")
    parser.add_argument("--count", default=100, type=int, help="Number of synthetic sequences to generate.")
    parser.add_argument("--output", default=None, type=Path, help="Output CSV path.")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu, cuda, or mps.")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--sequence-prefix", default="synthetic")
    args = parser.parse_args()

    model, extra = TimeGAN.load(args.run_dir / "timegan.pt", map_location=args.device or "cpu")
    device = _device(args.device)
    model.to(device)
    preprocessor = joblib.load(args.run_dir / "preprocessor.joblib")
    generated_scaled = model.generate(args.count, device=device)
    generated = preprocessor.inverse_transform_array(generated_scaled)

    output = args.output or (args.run_dir / "synthetic_sequences.csv")
    feature_columns = extra.get("feature_columns") or preprocessor.feature_columns
    save_generated_csv(
        generated,
        output,
        feature_columns,
        sequence_prefix=args.sequence_prefix,
        label=args.label,
    )
    print(f"Saved {args.count} synthetic sequences to {output}")


if __name__ == "__main__":
    main()
