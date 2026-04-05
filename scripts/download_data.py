#!/usr/bin/env python3
"""Download Rossmann train/store CSVs into data/raw/ (Kaggle Hub, no CLI needed)."""

from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    root = kagglehub.dataset_download("shahpranshu27/rossman-store-sales")
    root = Path(root)
    nested = root / "rossmann-store-sales"
    src = nested if nested.is_dir() else root
    for name in ("train.csv", "store.csv"):
        path = src / name
        if not path.exists():
            raise FileNotFoundError(path)
        dest = DATA_RAW / name
        dest.write_bytes(path.read_bytes())
        print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
