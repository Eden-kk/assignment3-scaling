from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a local scaling law from saved run records."
    )
    parser.add_argument(
        "--records",
        required=True,
        help="Path to the JSONL run record file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from cs336_scaling.local_scaling.hooks import fit_scaling_law

    results = fit_scaling_law(args.records)
    print(results)


if __name__ == "__main__":
    main()
