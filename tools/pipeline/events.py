"""Event discovery and grouping utilities."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping

LOGGER = logging.getLogger(__name__)

# Markers used to strip per-file suffixes when deriving the canonical event key.
EVENT_SUFFIX_MARKERS = (
    "_eventDTO",
    "_va-",
    "_thumbnail",
    "_feedback_",
    "_gdoStateAnnotations",
)

REQUIRED_EVENT_TYPES = {"video", "metadata", "analytics", "thumbnails"}

EVENT_KEY_PATTERN = re.compile(
    r"""
    ^
    (?P<user_id>\d+)_
    (?P<src_id>\d+)_
    (?P<event_camera>\d+[A-Za-z0-9]+)_
    (?P<timestamp>\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})
    (?P<flags>(?:_[^_]+)*)?
    $
    """,
    re.VERBOSE,
)


def _strip_known_suffixes(name_without_ext: str) -> str:
    for marker in EVENT_SUFFIX_MARKERS:
        idx = name_without_ext.find(marker)
        if idx != -1:
            return name_without_ext[:idx]
    return name_without_ext


def _stem(path: Path) -> str:
    """Return the filename without multi-part suffixes."""
    name = path.name
    for ext in (".json.gz", ".json", ".mp4", ".jpg", ".jpeg", ".png"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name


def infer_event_key(path: Path) -> str | None:
    """Infer the canonical event key for a given file path."""
    stem = _stem(path)
    key = _strip_known_suffixes(stem)
    return key.rstrip("_")


def classify_file(path: Path) -> tuple[str, str] | None:
    """Classify a file into its event key and type."""
    if not path.is_file():
        return None

    key = infer_event_key(path)
    if not key:
        return None

    name = path.name.lower()
    if name.endswith(".mp4"):
        return key, "video"
    if name.endswith("_eventdto.json"):
        return key, "metadata"
    if "_va-" in name and (name.endswith(".json.gz") or name.endswith(".json")):
        return key, "analytics"
    if "thumbnail" in name and (name.endswith(".jpg") or name.endswith(".jpeg")):
        return key, "thumbnails"
    if "_feedback_" in name or name.endswith("_gdostateannotations.json"):
        return key, "optional"

    return key, "supplemental"


def parse_event_key(key: str) -> Mapping[str, Any]:
    """Parse an event key into structured components."""
    match = EVENT_KEY_PATTERN.match(key)
    if not match:
        LOGGER.debug("Unable to parse event key: %s", key)
        return {}

    parts = match.groupdict()
    event_camera = parts["event_camera"]
    event_id_match = re.match(r"(?P<event_id>\d+)(?P<camera_type>.*)", event_camera)
    event_id = event_camera
    camera_type = ""
    if event_id_match:
        event_id = event_id_match.group("event_id")
        camera_type = event_id_match.group("camera_type")

    flags = tuple(filter(None, parts.get("flags", "").split("_"))) if parts.get("flags") else ()

    return {
        "user_id": parts["user_id"],
        "src_id": parts["src_id"],
        "event_id": event_id,
        "camera_type": camera_type,
        "timestamp": parts["timestamp"],
        "flags": flags,
    }


@dataclass
class EventBundle:
    """Grouping of files representing a single event."""

    key: str
    files: DefaultDict[str, List[Path]] = field(default_factory=lambda: defaultdict(list))
    metadata: Dict[str, Any] | None = None
    parsed_key: Mapping[str, Any] = field(default_factory=dict)

    def add_file(self, file_type: str, path: Path) -> None:
        self.files[file_type].append(path)

    @property
    def event_id(self) -> str | None:
        return (self.metadata or {}).get("id") or self.parsed_key.get("event_id")

    @property
    def user_id(self) -> str | None:
        return (self.metadata or {}).get("userId") or self.parsed_key.get("user_id")

    @property
    def src_id(self) -> str | None:
        return (self.metadata or {}).get("srcId") or self.parsed_key.get("src_id")

    @property
    def camera_type(self) -> str | None:
        return self.parsed_key.get("camera_type")

    @property
    def timestamp(self) -> str | None:
        dto = self.metadata or {}
        return dto.get("dttm") or self.parsed_key.get("timestamp")

    @property
    def flags(self) -> Iterable[str]:
        dto = self.metadata or {}
        # Harmonize optional metadata-driven flags.
        flag_values: List[str] = []
        if dto.get("status"):
            flag_values.append(f"status={dto['status']}")
        flag_values.extend(self.parsed_key.get("flags", ()))
        return flag_values

    def ensure_metadata(self) -> None:
        if self.metadata is not None:
            return
        metadata_files = self.files.get("metadata", [])
        if not metadata_files:
            return
        # Prefer the first metadata file.
        metadata_path = metadata_files[0]
        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                self.metadata = json.load(fh)
        except Exception as exc:  # pragma: no cover - log and continue
            LOGGER.warning("Failed to load metadata for event %s: %s", self.key, exc)
            self.metadata = {}

    def required_files_missing(self) -> set[str]:
        missing = {category for category in REQUIRED_EVENT_TYPES if not self.files.get(category)}
        return missing

    def is_complete(self) -> bool:
        return not self.required_files_missing()

    def iter_files(self) -> Iterable[Path]:
        for paths in self.files.values():
            yield from paths


def discover_events(directory: Path) -> Dict[str, EventBundle]:
    """Discover events within a directory."""
    events: Dict[str, EventBundle] = {}
    if not directory.exists():
        LOGGER.debug("Directory does not exist: %s", directory)
        return events

    for path in sorted(directory.iterdir()):
        classification = classify_file(path)
        if not classification:
            continue
        key, file_type = classification
        bundle = events.setdefault(key, EventBundle(key=key, parsed_key=parse_event_key(key)))
        bundle.add_file(file_type, path)

    for bundle in events.values():
        bundle.ensure_metadata()

    return events


def is_complete_event(bundle: EventBundle) -> bool:
    """True if the event bundle contains all required file categories."""
    return bundle.is_complete()


def iter_complete_events(events: Mapping[str, EventBundle]) -> Iterable[EventBundle]:
    for bundle in events.values():
        if bundle.is_complete():
            yield bundle


__all__ = [
    "EventBundle",
    "discover_events",
    "is_complete_event",
    "iter_complete_events",
    "parse_event_key",
    "REQUIRED_EVENT_TYPES",
]

