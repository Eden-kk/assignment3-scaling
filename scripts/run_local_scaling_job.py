from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_scaling.local_scaling import ExperimentConfig, run_local_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one local scaling-law experiment on Colab, CUDA, MPS, or CPU."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    metrics = run_local_experiment(config)
    print(metrics)


if __name__ == "__main__":
    main()
