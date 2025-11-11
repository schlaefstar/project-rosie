"""Queue random camera events for processing."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.pipeline.logging_utils import configure_logging
from tools.pipeline.queue import QueueManager


def queue_random_events(source_dir: Path, queue_dir: Path, sample_size: int, allow_duplicates: bool) -> int:
    """Copy a random sample of complete events into the processing queue."""
    configure_logging()
    logger = logging.getLogger(__name__)

    manager = QueueManager(source_dir=source_dir, queue_dir=queue_dir)
    staged_paths = manager.queue_random_events(sample_size, allow_duplicates=allow_duplicates)
    logger.info("Queued %d files for processing.", len(staged_paths))
    return len(staged_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue random camera events")
    parser.add_argument("--source", type=Path, default=Path("raw_videos"), help="Directory containing event files")
    parser.add_argument("--queue", type=Path, default=Path("to_process"), help="Queue directory where events are staged")
    parser.add_argument(
        "--count",
        "--sample-size",
        type=int,
        default=10,
        dest="count",
        help="Number of events to queue",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow re-queueing events that are already present in the queue",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue_random_events(args.source, args.queue, args.count, allow_duplicates=args.allow_duplicates)


if __name__ == "__main__":
    main()
