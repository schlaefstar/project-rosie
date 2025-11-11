"""Construct representative samples grouped by device type."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_device_samples(raw_dir: Path, output_dir: Path) -> None:
    """Aggregate sample media for each device type.

    Replace with logic that matches your data layout.
    """
    raise NotImplementedError("Implement device sampling logic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build device sample sets")
    parser.add_argument("raw", type=Path, help="Directory containing raw media grouped by device")
    parser.add_argument("output", type=Path, help="Directory where sample sets should be stored")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_device_samples(args.raw, args.output)


if __name__ == "__main__":
    main()
