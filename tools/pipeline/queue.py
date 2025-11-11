"""Queue management helpers for staging events to process."""

from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .events import EventBundle, discover_events, iter_complete_events

LOGGER = logging.getLogger(__name__)


@dataclass
class QueueManager:
    """Coordinates staging events from raw storage into the processing queue."""

    source_dir: Path
    queue_dir: Path

    def __post_init__(self) -> None:
        self.source_dir = self.source_dir.expanduser().resolve()
        self.queue_dir = self.queue_dir.expanduser().resolve()
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def discover_source_events(self) -> List[EventBundle]:
        events = discover_events(self.source_dir)
        return list(iter_complete_events(events))

    def discover_queue_events(self) -> List[EventBundle]:
        events = discover_events(self.queue_dir)
        return list(iter_complete_events(events))

    def pick_random_events(self, count: int, allow_duplicates: bool = False) -> List[EventBundle]:
        source_events = self.discover_source_events()
        if not source_events:
            LOGGER.warning("No complete events discovered in %s", self.source_dir)
            return []

        queue_keys = {bundle.key for bundle in self.discover_queue_events()}
        available = source_events
        if not allow_duplicates:
            available = [bundle for bundle in source_events if bundle.key not in queue_keys]

        if not available:
            LOGGER.warning("No new events available to queue from %s", self.source_dir)
            return []

        if count >= len(available):
            selection = list(available)
        else:
            selection = random.sample(available, count)

        LOGGER.info("Selected %d events for queueing (%d requested)", len(selection), count)
        return selection

    def stage_events(self, events: Sequence[EventBundle]) -> List[Path]:
        staged_paths: List[Path] = []
        for bundle in events:
            for file_path in bundle.iter_files():
                destination = self.queue_dir / file_path.name
                staged_paths.append(destination)
                if destination.exists():
                    LOGGER.debug("Skipping copy; file already exists: %s", destination)
                    continue
                LOGGER.debug("Copying %s -> %s", file_path, destination)
                shutil.copy2(file_path, destination)
        LOGGER.info("Queued %d files for %d events into %s", len(staged_paths), len(events), self.queue_dir)
        return staged_paths

    def queue_random_events(self, count: int, allow_duplicates: bool = False) -> List[Path]:
        selection = self.pick_random_events(count=count, allow_duplicates=allow_duplicates)
        return self.stage_events(selection)


__all__ = ["QueueManager"]

