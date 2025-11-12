"""Unified pipeline runner providing processing and queue utilities."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List

from tools.pipeline.events import EventBundle, discover_events, iter_complete_events
from tools.pipeline.logging_utils import configure_logging
from tools.pipeline.processor import EventProcessor, ProcessedEventResult, ProcessingConfig
from tools.pipeline.queue import QueueManager

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def run_process_event(event_path: Path, output_dir: Path, run_suffix: str | None = None) -> ProcessedEventResult:
    configure_logging(level=logging.DEBUG, console=False)

    events = discover_events(event_path.parent if event_path.is_file() else event_path)
    bundle: EventBundle | None = None
    if event_path.is_dir():
        bundle = next(iter(iter_complete_events(events)), None)
    else:
        for candidate in events.values():
            if event_path in candidate.iter_files():
                bundle = candidate
                break
    if not bundle:
        raise RuntimeError(f"Unable to resolve event for {event_path}")

    config = ProcessingConfig(
        raw_dir=Path("raw_videos"),
        queue_dir=event_path.parent,
        processing_root=Path("processing"),
        processed_root=output_dir,
        run_suffix=run_suffix,
    )
    processor = EventProcessor(config)
    return processor.process_bundle(bundle)


def run_process_events(event_paths: Iterable[Path], output_dir: Path, run_suffix: str | None = None) -> List[ProcessedEventResult]:
    results: List[ProcessedEventResult] = []
    for path in event_paths:
        LOGGER.info("Processing event %s", path)
        results.append(run_process_event(path, output_dir, run_suffix))
    return results


def _remove_from_queue(bundle: EventBundle, queue_dir: Path) -> None:
    for file_path in bundle.iter_files():
        queue_path = queue_dir / file_path.name
        if queue_path.exists():
            queue_path.unlink()


def run_process_queue(
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
    configure_logging(level=logging.DEBUG, console=False, log_files=("pipeline.log", "batch_process.log"))

    events = discover_events(queue_dir)
    bundles = [bundle for bundle in iter_complete_events(events)]
    bundles.sort(key=lambda b: b.timestamp or b.key)

    if limit is not None and limit > 0:
        bundles = bundles[:limit]

    if not bundles:
        LOGGER.info("No complete events found in %s", queue_dir)
        return []

    if dry_run:
        for bundle in bundles:
            LOGGER.info("[DRY RUN] Would process %s (event_id=%s)", bundle.key, bundle.event_id)
        return []

    config = ProcessingConfig(
        raw_dir=raw_dir,
        queue_dir=queue_dir,
        processing_root=processing_dir,
        processed_root=processed_dir,
        run_id=run_id,
        run_suffix=run_suffix,
        keep_intermediate=keep_workspace,
    )

    processor = EventProcessor(config)
    results: List[ProcessedEventResult] = []
    failures = 0
    for bundle in bundles:
        try:
            result = processor.process_bundle(bundle)
            results.append(result)
            _remove_from_queue(bundle, queue_dir)
        except Exception:  # pragma: no cover - logged for visibility
            failures += 1
            LOGGER.exception("Processing failed for event %s", bundle.key)

    LOGGER.info(
        "Pipeline complete. %d succeeded, %d failed. Outputs stored under %s/%s",
        len(results),
        failures,
        processed_dir,
        config.run_id,
    )
    return results


def run_queue_random(source_dir: Path, queue_dir: Path, count: int, allow_duplicates: bool) -> int:
    configure_logging()
    manager = QueueManager(source_dir=source_dir, queue_dir=queue_dir)
    staged_paths = manager.queue_random_events(count, allow_duplicates=allow_duplicates)
    LOGGER.info("Queued %d files for processing", len(staged_paths))
    return len(staged_paths)


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rosie pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("process-event", help="Process a single event")
    single.add_argument("event", type=Path)
    single.add_argument("--output", type=Path, default=Path("processed"))
    single.add_argument("--run-suffix", type=str, default=None)

    multi = subparsers.add_parser("process-events", help="Process multiple events sequentially")
    multi.add_argument("events", nargs="+", type=Path)
    multi.add_argument("--output", type=Path, default=Path("processed"))
    multi.add_argument("--run-suffix", type=str, default=None)

    queue = subparsers.add_parser("process-queue", help="Process events staged in a queue directory")
    queue.add_argument("--queue", type=Path, default=Path("to_process"))
    queue.add_argument("--processed", type=Path, default=Path("processed"))
    queue.add_argument("--raw", type=Path, default=Path("raw_videos"))
    queue.add_argument("--processing", type=Path, default=Path("processing"))
    queue.add_argument("--run-id", type=str, default=None)
    queue.add_argument("--run-suffix", type=str, default=None)
    queue.add_argument("--limit", type=int, default=None)
    queue.add_argument("--keep-workspace", action="store_true")
    queue.add_argument("--dry-run", action="store_true")

    random_cmd = subparsers.add_parser("queue-random", help="Queue random events from raw storage")
    random_cmd.add_argument("--source", type=Path, default=Path("raw_videos"))
    random_cmd.add_argument("--queue", type=Path, default=Path("to_process"))
    random_cmd.add_argument("--count", type=int, default=10)
    random_cmd.add_argument("--allow-duplicates", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "process-event":
        run_process_event(args.event, args.output, args.run_suffix)
    elif args.command == "process-events":
        run_process_events(args.events, args.output, args.run_suffix)
    elif args.command == "process-queue":
        run_process_queue(
            queue_dir=args.queue,
            processed_dir=args.processed,
            raw_dir=args.raw,
            processing_dir=args.processing,
            run_id=args.run_id,
            run_suffix=args.run_suffix,
            limit=args.limit,
            keep_workspace=args.keep_workspace,
            dry_run=args.dry_run,
        )
    elif args.command == "queue-random":
        run_queue_random(args.source, args.queue, args.count, allow_duplicates=args.allow_duplicates)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
