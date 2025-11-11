"""Process a single video event."""

from __future__ import annotations

import argparse
from pathlib import Path


def process_video_event(event_path: Path, output_dir: Path) -> None:
    """Run the per-event processing workflow.

    This is a placeholder for the actual event processing logic.
    """
    raise NotImplementedError("Implement single event processing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a single video event")
    parser.add_argument("event", type=Path, help="Path to the event to process")
    parser.add_argument("output", type=Path, help="Directory where outputs should be saved")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_video_event(args.event, args.output)


if __name__ == "__main__":
    main()
