"""Queue random camera events for processing."""

from __future__ import annotations

import argparse
from pathlib import Path
import random


def queue_random_events(source_dir: Path, queue_dir: Path, sample_size: int) -> None:
    """Copy a random sample of event files into the processing queue.

    This is a placeholder implementation. Extend it to match the
    project's data model and storage format.
    """
    raise NotImplementedError("Implement event queueing logic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue random camera events")
    parser.add_argument("source", type=Path, help="Directory containing event files")
    parser.add_argument("queue", type=Path, help="Queue directory where events are staged")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of events to queue")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue_random_events(args.source, args.queue, args.sample_size)


if __name__ == "__main__":
    main()
