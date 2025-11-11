"""Analytics parsing and interpolation helpers."""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

LOGGER = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a single object detection bounding box."""

    object_id: str
    label: str
    confidence: float
    life: int
    x: float
    y: float
    width: float
    height: float

    def to_serializable(self) -> Mapping[str, object]:
        return {
            "id": self.object_id,
            "label": self.label,
            "confidence": self.confidence,
            "life": self.life,
            "bbox": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
        }


@dataclass
class FrameDetections:
    frame_num: int
    time_sec: float | None
    objects: List[BoundingBox] = field(default_factory=list)

    def to_serializable(self) -> Mapping[str, object]:
        return {
            "frame": self.frame_num,
            "time_sec": self.time_sec,
            "objects": [obj.to_serializable() for obj in self.objects],
        }


@dataclass
class TrackSegment:
    track_id: str
    label: str
    start_sec: float
    end_sec: float
    center_start: Mapping[str, float]
    center_end: Mapping[str, float]
    avg_speed: float | None = None
    max_speed: float | None = None
    total_distance: float | None = None

    def to_serializable(self) -> Mapping[str, object]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "center_start": dict(self.center_start),
            "center_end": dict(self.center_end),
            "avg_speed": self.avg_speed,
            "max_speed": self.max_speed,
            "total_distance": self.total_distance,
        }


@dataclass
class AnalyticsData:
    width: int | None
    height: int | None
    duration_sec: float | None
    normalized: bool
    frames: Dict[int, FrameDetections]
    tracks: Dict[str, TrackSegment]
    raw: Mapping[str, object]
    frames_sorted_by_time: List[FrameDetections] = field(default_factory=list, repr=False)

    def get_frame(self, frame_idx: int) -> FrameDetections | None:
        return self.frames.get(frame_idx)

    def iter_frames(self) -> Iterable[FrameDetections]:
        for frame_idx in sorted(self.frames):
            yield self.frames[frame_idx]

    def iter_tracks(self) -> Iterable[TrackSegment]:
        for track_id in sorted(self.tracks):
            yield self.tracks[track_id]

    def summary(self) -> Mapping[str, object]:
        return {
            "frame_count": len(self.frames),
            "track_count": len(self.tracks),
            "normalized": self.normalized,
            "width": self.width,
            "height": self.height,
            "duration_sec": self.duration_sec,
        }

    def frame_for_time(self, time_sec: float, fps: float | None = None) -> FrameDetections | None:
        if not self.frames_sorted_by_time:
            return None

        def frame_time(frame: FrameDetections) -> float:
            if frame.time_sec is not None:
                return frame.time_sec
            if fps and fps > 0:
                return frame.frame_num / fps
            return float(frame.frame_num)

        return min(self.frames_sorted_by_time, key=lambda frame: abs(frame_time(frame) - time_sec))


def _open_analytics_file(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _parse_bounding_box(obj: Mapping[str, object]) -> BoundingBox:
    return BoundingBox(
        object_id=str(obj.get("id", "")),
        label=str(obj.get("type", "unknown")),
        confidence=float(obj.get("conf", 0.0)),
        life=int(obj.get("life", 0)),
        x=float(obj.get("x", 0.0)),
        y=float(obj.get("y", 0.0)),
        width=float(obj.get("w", obj.get("width", 0.0))),
        height=float(obj.get("h", obj.get("height", 0.0))),
    )


def _parse_track(track_id: str, payload: Mapping[str, object]) -> TrackSegment:
    return TrackSegment(
        track_id=track_id,
        label=str(payload.get("type", "other")),
        start_sec=float(payload.get("startSec", 0.0)),
        end_sec=float(payload.get("endSec", 0.0)),
        center_start=dict(payload.get("centerStart", {"x": 0.0, "y": 0.0})),
        center_end=dict(payload.get("centerEnd", {"x": 0.0, "y": 0.0})),
        avg_speed=(float(payload["avgSpeed"]) if "avgSpeed" in payload else None),
        max_speed=(float(payload["maxSpeed"]) if "maxSpeed" in payload else None),
        total_distance=(float(payload["totalDistance"]) if "totalDistance" in payload else None),
    )


def load_analytics(path: Path) -> AnalyticsData:
    with _open_analytics_file(path) as fh:
        data = json.load(fh)

    frames: Dict[int, FrameDetections] = {}
    for frame_payload in data.get("frames", []):
        frame_idx = int(frame_payload.get("num", 0))
        time_sec = frame_payload.get("timeSec")
        objects_payload = frame_payload.get("objects", [])
        objects = [_parse_bounding_box(obj) for obj in objects_payload]
        frames[frame_idx] = FrameDetections(
            frame_num=frame_idx,
            time_sec=float(time_sec) if time_sec is not None else None,
            objects=objects,
        )

    tracks_payload = data.get("tracks") or {}
    if isinstance(tracks_payload, list):
        tracks_payload = {str(item.get("id", idx)): item for idx, item in enumerate(tracks_payload)}

    tracks: Dict[str, TrackSegment] = {}
    for track_id, track_data in tracks_payload.items():
        try:
            tracks[str(track_id)] = _parse_track(str(track_id), track_data)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to parse track %s: %s", track_id, exc)

    width = data.get("width")
    height = data.get("height")
    duration_sec = data.get("durationSec")

    normalized = bool(data.get("normalized", False))

    frames_sorted_by_time = sorted(
        frames.values(),
        key=lambda frame: frame.time_sec if frame.time_sec is not None else frame.frame_num,
    )

    return AnalyticsData(
        width=int(width) if width is not None else None,
        height=int(height) if height is not None else None,
        duration_sec=float(duration_sec) if duration_sec is not None else None,
        normalized=normalized,
        frames=frames,
        tracks=tracks,
        raw=data,
        frames_sorted_by_time=frames_sorted_by_time,
    )


__all__ = [
    "AnalyticsData",
    "BoundingBox",
    "FrameDetections",
    "TrackSegment",
    "load_analytics",
]

