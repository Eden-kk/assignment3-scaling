from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_scaling.local_scaling import ExperimentConfig, run_local_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or attach to a W&B sweep for local scaling-law experiments."
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Path to the base JSON experiment config.",
    )
    parser.add_argument(
        "--sweep-config",
        required=True,
        help="Path to the W&B sweep JSON config.",
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Existing sweep ID. If omitted, a new sweep is created.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Optional max number of agent runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import wandb

    base_config = ExperimentConfig.from_json(args.base_config)

    with open(args.sweep_config) as handle:
        sweep_config = json.load(handle)

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=base_config.wandb.project,
            entity=base_config.wandb.entity,
        )
        print(f"Created sweep: {sweep_id}")

    def train_from_sweep() -> None:
        run_local_experiment(base_config)

    wandb.agent(sweep_id, function=train_from_sweep, count=args.count)


if __name__ == "__main__":
    main()
