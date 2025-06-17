#!/usr/bin/env python3
"""Prepare timeline dataset and train the sequence autoencoder."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from prepare_autoencoder_dataset import extract_odds_timelines
from sequence_autoencoder import train_sequence_autoencoder


DEFAULT_CACHE_DIR = Path("h2h_data")
DEFAULT_MODEL_OUT = Path("odds_autoencoder.pt")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing cached odds .pkl files",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_OUT,
        help="Where to store the trained autoencoder",
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        help="Optional path to save the aggregated timeline dataset",
    )
    args = parser.parse_args(argv)

    timelines, inspected = extract_odds_timelines(args.cache_dir)
    if not timelines:
        print("No odds timelines found in cache")
        if inspected:
            print("Inspected files: " + ", ".join(sorted(inspected)))
        return

    dataset_out = args.dataset_out or args.cache_dir / "api_cache" / "odds_timelines.pkl"
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_out, "wb") as f:
        pickle.dump(timelines, f)
    print(f"Saved {len(timelines)} timelines to {dataset_out}")

    train_sequence_autoencoder(str(dataset_out), model_out=str(args.model_out))


if __name__ == "__main__":
    main()
