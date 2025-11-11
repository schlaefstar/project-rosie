"""High-level orchestration for the video processing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


def process_pipeline(queue_dir: Path, processed_dir: Path) -> None:
    """Run the full pipeline over queued events.

    Replace this stub with your actual pipeline orchestration.
    """
    raise NotImplementedError("Implement pipeline orchestration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process queued video events")
    parser.add_argument("queue", type=Path, help="Directory containing queued events")
    parser.add_argument("processed", type=Path, help="Directory for processed outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_pipeline(args.queue, args.processed)


if __name__ == "__main__":
    main()
