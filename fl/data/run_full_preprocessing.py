from __future__ import annotations

from pathlib import Path

from create_preprocessing_datasets import DEFAULT_OUTPUT_DIR, create_preprocessing_datasets


def main() -> None:
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    create_preprocessing_datasets(output_dir=output_dir)
    print(f"Full preprocessing results exported to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
