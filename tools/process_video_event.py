"""Process a single video event."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline.events import EventBundle, discover_events
from pipeline.logging_utils import configure_logging
from pipeline.processor import EventProcessor, ProcessingConfig


def _select_bundle(event_path: Path) -> EventBundle:
    if event_path.is_dir():
        events = discover_events(event_path)
        if not events:
            raise RuntimeError(f"No events discovered in {event_path}")
        if len(events) > 1:
            logging.warning("Multiple events detected in %s; processing the first one found.", event_path)
        return next(iter(events.values()))

    events = discover_events(event_path.parent)
    for bundle in events.values():
        if event_path in bundle.iter_files():
            return bundle

    raise RuntimeError(f"Unable to locate event files for {event_path}")


def process_video_event(event_path: Path, output_dir: Path) -> None:
    """Run the per-event processing workflow."""
    configure_logging()
    bundle = _select_bundle(event_path)

    config = ProcessingConfig(
        raw_dir=Path("raw_videos"),
        queue_dir=event_path.parent,
        processing_root=Path("processing"),
        processed_root=output_dir,
    )
    processor = EventProcessor(config)
    processor.process_bundle(bundle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a single video event")
    parser.add_argument("event", type=Path, help="Path to a file or directory belonging to the event")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed"),
        help="Directory where outputs should be saved (default: processed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_video_event(args.event, args.output)


if __name__ == "__main__":
    main()
