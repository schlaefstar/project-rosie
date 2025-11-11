"""High-level orchestration for the video processing pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.pipeline.events import EventBundle, discover_events, iter_complete_events
from tools.pipeline.logging_utils import configure_logging
from tools.pipeline.processor import EventProcessor, ProcessedEventResult, ProcessingConfig

LOGGER = logging.getLogger(__name__)


def _remove_event_from_queue(bundle: EventBundle, queue_dir: Path) -> None:
    for file_path in bundle.iter_files():
        queue_path = queue_dir / file_path.name
        if queue_path.exists():
            queue_path.unlink()


def process_pipeline(
    queue_dir: Path,
    processed_dir: Path,
    *,
    raw_dir: Path,
    processing_dir: Path,
    run_id: str | None = None,
    run_suffix: str | None = None,
    limit: int | None = None,
    keep_workspace: bool = False,
    dry_run: bool = False,
) -> List[ProcessedEventResult]:
    """Run the full pipeline over queued events."""
    configure_logging(log_files=("pipeline.log", "batch_process.log"))

    events = discover_events(queue_dir)
    bundles = [bundle for bundle in iter_complete_events(events)]
    bundles.sort(key=lambda bundle: bundle.timestamp or bundle.key)

    if not bundles:
        LOGGER.info("No complete events found in %s", queue_dir)
        return []

    if limit is not None:
        bundles = bundles[:limit]

    LOGGER.info("Beginning processing run for %d events (dry-run=%s)", len(bundles), dry_run)

    config = ProcessingConfig(
        raw_dir=raw_dir,
        queue_dir=queue_dir,
        processing_root=processing_dir,
        processed_root=processed_dir,
        run_id=run_id,
        run_suffix=run_suffix,
        keep_intermediate=keep_workspace,
    )

    if dry_run:
        for bundle in bundles:
            LOGGER.info("[DRY RUN] Would process event %s (event_id=%s)", bundle.key, bundle.event_id)
        return []

    processor = EventProcessor(config)
    results: List[ProcessedEventResult] = []
    failures = 0
    for bundle in bundles:
        try:
            result = processor.process_bundle(bundle)
            results.append(result)
            _remove_event_from_queue(bundle, queue_dir)
        except Exception:  # pragma: no cover - logged for inspection
            failures += 1
            LOGGER.exception("Processing failed for event %s", bundle.key)

    LOGGER.info(
        "Processing complete. %d succeeded, %d failed. Outputs stored under %s/%s",
        len(results),
        failures,
        processed_dir,
        config.run_id,
    )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process queued video events")
    parser.add_argument("--queue", type=Path, default=Path("to_process"), help="Directory containing queued events")
    parser.add_argument("--processed", type=Path, default=Path("processed"), help="Directory for processed outputs")
    parser.add_argument("--raw", type=Path, default=Path("raw_videos"), help="Directory with raw source videos")
    parser.add_argument("--processing", type=Path, default=Path("processing"), help="Workspace directory for processing")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run identifier (overrides timestamp)")
    parser.add_argument("--run-suffix", type=str, default=None, help="Suffix appended to the generated run id")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N events in the queue")
    parser.add_argument("--all", action="store_true", help="Process all events in the queue")
    parser.add_argument("--keep-workspace", action="store_true", help="Retain per-event workspaces after completion")
    parser.add_argument("--dry-run", action="store_true", help="Enumerate events without processing media")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_pipeline(
        queue_dir=args.queue,
        processed_dir=args.processed,
        raw_dir=args.raw,
        processing_dir=args.processing,
        run_id=args.run_id,
        run_suffix=args.run_suffix,
        limit=None if args.all else args.limit,
        keep_workspace=args.keep_workspace,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

