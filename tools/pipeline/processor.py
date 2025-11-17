"""Core event processing pipeline implementation."""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .analytics import AnalyticsData, BoundingBox, FrameDetections, TrackSegment, load_analytics
from .annotations import AnnotationStyle, draw_bbox_annotation, draw_track_annotations, resolve_track_color
from .events import EventBundle, discover_events
from .llm import ClipTaggerProvider, GeminiProvider, LLMClient, LLMError, MediaEvent, PricingTable
from .logging_utils import log_path

LOGGER = logging.getLogger(__name__)


def _timestamp_run_id(suffix: str | None = None) -> str:
    base = str(int(time.time()))
    if suffix:
        clean = _slugify(suffix.strip())
        if clean:
            return f"{base}_{clean}"
    return base


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


@dataclass
class ProcessingConfig:
    raw_dir: Path
    queue_dir: Path
    processing_root: Path
    processed_root: Path
    run_id: str | None = None
    run_suffix: str | None = None
    keep_intermediate: bool = False

    def __post_init__(self) -> None:
        self.raw_dir = self.raw_dir.expanduser().resolve()
        self.queue_dir = self.queue_dir.expanduser().resolve()
        self.processing_root = self.processing_root.expanduser().resolve()
        self.processing_root.mkdir(parents=True, exist_ok=True)
        self.processed_root = self.processed_root.expanduser().resolve()
        self.processed_root.mkdir(parents=True, exist_ok=True)
        if self.run_suffix:
            cleaned_suffix = _slugify(self.run_suffix.strip())
            self.run_suffix = cleaned_suffix or None
        if not self.run_id:
            self.run_id = _timestamp_run_id(self.run_suffix)


@dataclass
class FrameOutput:
    frame_index: int
    time_sec: float
    bounding_boxes: List[Mapping[str, object]]
    active_tracks: List[str]

    def to_serializable(self) -> Mapping[str, object]:
        return {
            "frame_index": self.frame_index,
            "time_sec": self.time_sec,
            "bounding_boxes": self.bounding_boxes,
            "active_tracks": self.active_tracks,
        }


@dataclass
class ThumbnailOutput:
    filename: str
    source_filename: str
    frame_index: int
    time_sec: float
    bounding_boxes: List[Mapping[str, object]]

    def to_serializable(self) -> Mapping[str, object]:
        return {
            "filename": self.filename,
            "source_thumbnail": self.source_filename,
            "frame_index": self.frame_index,
            "time_sec": self.time_sec,
            "bounding_boxes": self.bounding_boxes,
        }


@dataclass
class ProcessedEventResult:
    bundle: EventBundle
    output_dir: Path
    annotated_video: Path | None
    annotated_thumbnails: List[Path]
    json_path: Path
    report: Mapping[str, object]


class EventProcessor:
    """Executes per-event processing into annotated media and metadata."""

    _EVENT_LOGGERS: Dict[str, logging.Logger] = {}

    def __init__(
        self,
        config: ProcessingConfig,
        *,
        llm_client: LLMClient | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.config = config
        self.style = AnnotationStyle()
        self._llm_client, self._llm_model = self._initialize_llm(llm_client, llm_model)
        self._llm_provider_name: str | None = None
        if self._llm_client and self._llm_model:
            try:
                provider = self._llm_client.get(self._llm_model)
                self._llm_provider_name = provider.provider_name
            except LLMError as exc:
                LOGGER.warning("LLM model '%s' unavailable: %s. Disabling AI outputs.", self._llm_model, exc)
                self._llm_client = None
                self._llm_model = None
                self._llm_provider_name = None

    # Public API ---------------------------------------------------------

    def _initialize_llm(
        self,
        llm_client: LLMClient | None,
        llm_model: str | None,
    ) -> tuple[LLMClient | None, str | None]:
        if llm_client:
            resolved_model = llm_model or os.environ.get("ROSIE_LLM_MODEL")
            if not resolved_model:
                LOGGER.warning("LLM client provided without model; AI outputs disabled.")
                return None, None
            return llm_client, resolved_model

        provider_name = (os.environ.get("ROSIE_LLM_PROVIDER") or "").strip().lower()
        prompt_template_path = os.environ.get("ROSIE_LLM_PROMPT_PATH")
        pricing_path = os.environ.get("ROSIE_LLM_PRICING_PATH")
        pricing = (
            PricingTable.load_from_file(pricing_path)
            if pricing_path
            else PricingTable.load_default()
        )

        if provider_name == "cliptagger" or (
            not provider_name
            and (
                os.environ.get("CLIPTAGGER_API_KEY")
                or os.environ.get("ROSIE_CLIPTAGGER_API_KEY")
                or os.environ.get("INFERENCE_API_KEY")
            )
        ):
            api_key = (
                os.environ.get("CLIPTAGGER_API_KEY")
                or os.environ.get("ROSIE_CLIPTAGGER_API_KEY")
                or os.environ.get("INFERENCE_API_KEY")
            )
            if not api_key:
                LOGGER.warning("ClipTagger selected but no API key configured; AI outputs disabled.")
                return None, None

            resolved_model = llm_model or os.environ.get("ROSIE_LLM_MODEL") or "cliptagger-12b"
            base_url = os.environ.get("CLIPTAGGER_API_BASE_URL", "https://api.inference.net/v1")
            max_frames_env = os.environ.get("CLIPTAGGER_MAX_FRAMES")
            try:
                max_frames = int(max_frames_env) if max_frames_env else None
            except ValueError:
                LOGGER.warning("Invalid CLIPTAGGER_MAX_FRAMES value '%s'; ignoring.", max_frames_env)
                max_frames = None

            try:
                provider = ClipTaggerProvider(
                    api_key=api_key,
                    model_name=resolved_model,
                    base_url=base_url,
                    pricing=pricing,
                    prompt_template_path=Path(prompt_template_path).expanduser()
                    if prompt_template_path
                    else None,
                    max_frames=max_frames,
                )
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Unable to initialize ClipTagger provider: %s. AI outputs disabled.", exc)
                return None, None

            client = LLMClient()
            client.register(provider)
            LOGGER.info(
                "LLM provider '%s' initialized with model '%s'.",
                provider.provider_name,
                provider.model_name,
            )
            return client, provider.model_name

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None, None

        resolved_model = llm_model or os.environ.get("ROSIE_LLM_MODEL") or "models/gemini-2.0-flash"
        try:
            provider = GeminiProvider(
                api_key=api_key,
                model_name=resolved_model,
                pricing=pricing,
                prompt_template_path=Path(prompt_template_path).expanduser()
                if prompt_template_path
                else None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Unable to initialize Gemini provider: %s. AI outputs disabled.", exc)
            return None, None

        client = LLMClient()
        client.register(provider)
        LOGGER.info("LLM provider '%s' initialized with model '%s'.", provider.provider_name, provider.model_name)
        return client, provider.model_name

    def _get_event_logger(self, event_id: str) -> logging.Logger:
        logger_name = f"events.{event_id}"
        logger = self._EVENT_LOGGERS.get(logger_name)
        if logger:
            return logger

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        path = self.config.processed_root / self.config.run_id / _slugify(event_id)
        path.mkdir(parents=True, exist_ok=True)
        log_file = path / f"{event_id}.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        self._EVENT_LOGGERS[logger_name] = logger
        return logger

    def _generate_ai_outputs(
        self,
        bundle: EventBundle,
        video_path: Path,
        analytics: AnalyticsData,
    ) -> Mapping[str, object]:
        event_id = bundle.event_id or bundle.key
        event_logger = self._get_event_logger(event_id)

        if not self._llm_client or not self._llm_model:
            event_logger.info("LLM Input")
            event_logger.info("  Status: disabled (no provider configured). Skipping AI outputs.")
            return {}

        try:
            provider = self._llm_client.get(self._llm_model)
        except LLMError as exc:
            LOGGER.warning("LLM provider lookup failed: %s", exc)
            event_logger.info("LLM Input")
            event_logger.info("  Status: provider lookup failed (%s). Skipping AI outputs.", exc)
            return {}

        frame_samples = self._select_frame_samples(bundle)
        detector_summary = self._compose_detector_summary(analytics)
        event = MediaEvent(
            event_id=bundle.event_id or bundle.key,
            video_path=video_path,
            frame_paths=frame_samples,
            metadata=bundle.metadata or {},
            detector_summary=detector_summary,
        )

        event_logger.info("LLM Input")
        event_logger.info("  Model: %s", self._llm_model)
        event_logger.info("  Video: %s", event.video_path)
        event_logger.info("  Frames: %s", ", ".join(path.name for path in event.frame_paths) or "None")
        event_logger.info("  Detector hints: %s", event.detector_summary or "None")
        event_logger.info(
            "  Metadata snippet: %s",
            json.dumps(event.metadata, default=str, ensure_ascii=False)[:500] if event.metadata else "None",
        )
        prompt_preview = ""
        try:
            prompt_builder = getattr(provider, "prompt_builder", None)
            if prompt_builder:
                prompt_preview = prompt_builder(event)
        except Exception:  # pragma: no cover - defensive
            prompt_preview = "[Unable to render prompt preview]"
        if prompt_preview:
            event_logger.info("  Prompt:\n%s", prompt_preview)

        try:
            start_time = time.time()
            result = self._llm_client.caption_event(self._llm_model, event)
            elapsed = time.time() - start_time
        except LLMError as exc:
            LOGGER.warning("LLM captioning failed for event %s: %s", bundle.key, exc)
            event_logger.error("LLM error: %s", exc)
            return {"error": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected LLM failure for event %s", bundle.key)
            event_logger.exception("Unexpected LLM failure for event %s", bundle.key)
            return {"error": str(exc)}

        output: Dict[str, object] = {
            "provider": provider.provider_name,
            "model": self._llm_model,
            "result": result.as_dict(),
        }
        response_id = getattr(result.raw_response, "response_id", None)
        if response_id:
            output["response_id"] = response_id

        event_logger.info("LLM Processing")
        event_logger.info("  Duration: %.2fs", elapsed)
        event_logger.info(
            "  Tokens - prompt: %s, completion: %s, total: %s",
            result.token_usage.prompt_tokens,
            result.token_usage.completion_tokens,
            result.token_usage.total_tokens,
        )
        event_logger.info("  Token metadata: %s", json.dumps(result.token_usage.metadata, ensure_ascii=False))
        event_logger.info("  Cost (USD): %s", result.cost_usd)

        event_logger.info("LLM Output")
        event_logger.info("  Summary: %s", result.summary)
        event_logger.info("  Steady state: %s", result.steady_state or "None")
        for classification in result.classifications:
            event_logger.info(
                "  Classification: %s (confidence=%.2f) - %s",
                classification.label,
                classification.confidence,
                classification.rationale or "",
            )
        if response_id:
            event_logger.info("  Response ID: %s", response_id)
        return output

    def _select_frame_samples(self, bundle: EventBundle) -> Sequence[Path]:
        frames = bundle.files.get("thumbnails", [])
        samples: List[Path] = []
        for path in frames:
            if path.exists():
                samples.append(path)
            if len(samples) >= 3:
                break
        return tuple(samples)

    def _compose_detector_summary(self, analytics: AnalyticsData) -> str | None:
        label_counts: Counter[str] = Counter()
        label_display: Dict[str, str] = {}

        def _add_label(raw_label: str | None) -> None:
            if not raw_label:
                return
            label = raw_label.strip()
            if not label:
                return
            key = label.lower()
            label_counts[key] += 1
            label_display.setdefault(key, label)

        for track in analytics.iter_tracks():
            _add_label(track.label)

        if not label_counts:
            for frame in analytics.iter_frames():
                if not frame.objects:
                    continue
                for box in frame.objects:
                    _add_label(box.label)

        if not label_counts:
            return None

        items: List[str] = []
        for key in sorted(label_counts.keys()):
            display = label_display.get(key, key)
            count = label_counts[key]
            if count > 1:
                items.append(f"{display} ({count})")
            else:
                items.append(display)

        summary = ", ".join(items)
        return summary[:500] if len(summary) > 500 else summary

    def process_bundle(self, bundle: EventBundle) -> ProcessedEventResult:
        LOGGER.info("Processing event %s", bundle.key)
        workspace = self._prepare_workspace(bundle)
        success = False
        workspace_bundle: EventBundle | None = None
        annotated_video_path: Path | None = None
        annotated_thumbnails: List[Path] = []
        json_output_path: Path | None = None
        report: Mapping[str, object] = {}

        try:
            workspace_bundle = self._populate_workspace(bundle, workspace)
            video_path = self._select_video(workspace_bundle)
            metadata = workspace_bundle.metadata or {}
            analytics_data = self._load_analytics(workspace_bundle)
            fps, frame_size, frame_count = self._probe_video(video_path)

            video_summary, annotated_video_path = self._annotate_video(
                video_path=video_path,
                workspace=workspace,
                bundle=workspace_bundle,
                analytics=analytics_data,
                fps=fps,
                frame_size=frame_size,
                frame_count=frame_count,
            )

            thumbnail_outputs, annotated_thumbnails = self._annotate_thumbnails(
                workspace_bundle=workspace_bundle,
                analytics=analytics_data,
                frame_size=frame_size,
            )

            ai_outputs = self._generate_ai_outputs(
                bundle=workspace_bundle,
                video_path=video_path,
                analytics=analytics_data,
            )

            report = self._build_report(
                bundle=workspace_bundle,
                analytics=analytics_data,
                fps=fps,
                frame_size=frame_size,
                frame_count=frame_count,
                video_summary=video_summary,
                thumbnail_outputs=thumbnail_outputs,
                ai_outputs=ai_outputs,
            )

            output_dir = self._finalize_outputs(
                workspace_bundle,
                workspace,
                annotated_video_path,
                annotated_thumbnails,
                report,
            )

            json_output_path = output_dir / f"{workspace_bundle.event_id}_processed.json"
            success = True

            return ProcessedEventResult(
                bundle=workspace_bundle,
                output_dir=output_dir,
                annotated_video=annotated_video_path,
                annotated_thumbnails=annotated_thumbnails,
                json_path=json_output_path,
                report=report,
            )
        finally:
            if not success:
                LOGGER.error("Processing failed for event %s. Leaving workspace at %s", bundle.key, workspace)
            if success and not self.config.keep_intermediate:
                shutil.rmtree(workspace, ignore_errors=True)

    # Workspace ----------------------------------------------------------

    def _prepare_workspace(self, bundle: EventBundle) -> Path:
        event_slug = _slugify(bundle.event_id or bundle.key)
        workspace = self.config.processing_root / event_slug
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _populate_workspace(self, bundle: EventBundle, workspace: Path) -> EventBundle:
        for file_path in bundle.iter_files():
            destination = workspace / file_path.name
            shutil.copy2(file_path, destination)

        discovered = discover_events(workspace)
        workspace_bundle = discovered.get(bundle.key)
        if not workspace_bundle:
            # Fallback: pick the only bundle
            workspace_bundle = next(iter(discovered.values()))
        return workspace_bundle

    # Source selection ---------------------------------------------------

    def _select_video(self, bundle: EventBundle) -> Path:
        videos = bundle.files.get("video", [])
        if not videos:
            raise RuntimeError(f"No video file found for event {bundle.key}")
        if len(videos) > 1:
            LOGGER.warning("Multiple video files found for event %s. Using first: %s", bundle.key, videos[0])
        return videos[0]

    def _load_analytics(self, bundle: EventBundle) -> AnalyticsData:
        analytics_files = bundle.files.get("analytics", [])
        if not analytics_files:
            raise RuntimeError(f"No analytics files found for event {bundle.key}")
        # Prefer gzipped analytics if available
        analytics_files_sorted = sorted(
            analytics_files,
            key=lambda p: (0 if p.suffix == ".gz" else 1, p.name),
        )
        analytics_data = load_analytics(analytics_files_sorted[0])
        return analytics_data

    def _probe_video(self, video_path: Path) -> Tuple[float, Tuple[int, int], int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        LOGGER.debug("Video probe %s: %sx%s @ %.2f fps (%s frames)", video_path, width, height, fps, frame_count)
        return fps, (width, height), frame_count

    # Video annotation ---------------------------------------------------

    def _annotate_video(
        self,
        video_path: Path,
        workspace: Path,
        bundle: EventBundle,
        analytics: AnalyticsData,
        fps: float,
        frame_size: Tuple[int, int],
        frame_count: int,
    ) -> Tuple[List[FrameOutput], Path]:
        width, height = frame_size
        font_scale = self.style.font_scale(width, height)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video for annotation: {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_filename = f"{bundle.event_id}_annotated.mp4"
        output_path = workspace / output_filename
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open video writer for: {output_path}")

        frame_outputs: List[FrameOutput] = []

        track_points: Dict[str, List[Tuple[int, int]]] = {}
        track_ranges: Dict[str, Tuple[int, int]] = {}
        for track in analytics.iter_tracks():
            start_frame = max(0, int(math.floor(track.start_sec * fps)))
            end_frame = max(start_frame, int(math.ceil(track.end_sec * fps)))
            track_ranges[track.track_id] = (start_frame, end_frame)
            points: List[Tuple[int, int]] = []
            total_frames = max(1, end_frame - start_frame)
            for offset in range(total_frames + 1):
                progress = offset / max(total_frames, 1)
                cx = track.center_start.get("x", 0.0) + (track.center_end.get("x", 0.0) - track.center_start.get("x", 0.0)) * progress
                cy = track.center_start.get("y", 0.0) + (track.center_end.get("y", 0.0) - track.center_start.get("y", 0.0)) * progress
                points.append((int(round(cx)), int(round(cy))))
            if points:
                track_points[track.track_id] = points

        frame_idx = 0
        annotations_generated = 0
        fade_schedule: List[Tuple[float, float]] = [
            (0.75, 0.75),
            (0.75, 0.75),
            (0.50, 0.50),
            (0.50, 0.50),
            (0.50, 0.50),
            (0.25, 0.25),
            (0.25, 0.25),
            (0.25, 0.25),
            (0.10, 0.10),
        ]
        persisted_boxes: Dict[str, Dict[str, Any]] = {}
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                time_sec = frame_idx / fps if fps else 0.0
                frame_data = analytics.frame_for_time(time_sec, fps=fps)
                if frame_data:
                    if frame_data.time_sec is not None:
                        delta = abs(frame_data.time_sec - time_sec)
                        tolerance = max(1.0 / max(fps, 1.0), 0.05)
                    else:
                        delta = abs(frame_data.frame_num - frame_idx) / max(fps, 1.0)
                        tolerance = max(1.0 / max(fps, 1.0), 0.05)
                    if delta > tolerance:
                        frame_data = None

                boxes_output: List[Mapping[str, object]] = []
                active_tracks: List[str] = []
                current_ids: set[str] = set()

                if frame_data and frame_data.objects:
                    annotations_generated += len(frame_data.objects)
                    for box in frame_data.objects:
                        color = resolve_track_color(box.label)
                        bbox = self._clamp_bbox(box, width, height)
                        text_lines = self._format_bbox_annotation(box, bbox)
                        draw_bbox_annotation(frame, bbox, text_lines, self.style, font_scale, color)
                        current_ids.add(str(box.object_id))
                        persisted_boxes[str(box.object_id)] = {
                            "bbox": bbox,
                            "lines": text_lines,
                            "color": color,
                            "fade_index": 0,
                        }
                        boxes_output.append(
                            {
                                "track_id": box.object_id,
                                "label": box.label,
                                "confidence": box.confidence,
                                "life": box.life,
                                "frame": frame_idx,
                                "bbox": {
                                    "x": bbox[0],
                                    "y": bbox[1],
                                    "width": bbox[2],
                                    "height": bbox[3],
                                },
                            }
                        )

                for track in analytics.iter_tracks():
                    start_frame, end_frame = track_ranges.get(track.track_id, (0, 0))
                    if frame_idx < start_frame or frame_idx > end_frame:
                        continue
                    points = track_points.get(track.track_id)
                    if not points:
                        continue
                    offset = frame_idx - start_frame
                    offset = min(max(offset, 0), len(points) - 1)
                    poly_points = points[: offset + 1]
                    start_point = points[0]
                    end_point = points[-1]
                    active_point = points[offset]
                    color = resolve_track_color(track.label)
                    start_label = f"start: {track.start_sec:.2f}s"
                    end_label = f"end: {track.end_sec:.2f}s"
                    draw_track_annotations(
                        frame,
                        track_id=track.track_id,
                        color=color,
                        poly_points=poly_points,
                        start_point=start_point,
                        end_point=end_point,
                        start_label=start_label,
                        end_label=end_label,
                        style=self.style,
                        font_scale=font_scale,
                        active_point=active_point,
                    )
                    active_tracks.append(track.track_id)

                for object_id, entry in list(persisted_boxes.items()):
                    if object_id in current_ids:
                        entry["fade_index"] = 0
                        persisted_boxes[object_id] = entry
                        continue
                    idx = entry.get("fade_index", 0)
                    if idx >= len(fade_schedule):
                        del persisted_boxes[object_id]
                        continue
                    stroke_scale, background_scale = fade_schedule[idx]
                    draw_bbox_annotation(
                        frame,
                        entry["bbox"],
                        entry["lines"],
                        self.style,
                        font_scale,
                        entry["color"],
                        stroke_alpha=stroke_scale,
                        background_alpha_scale=background_scale,
                    )
                    entry["fade_index"] = idx + 1
                    persisted_boxes[object_id] = entry

                if boxes_output or active_tracks:
                    frame_outputs.append(
                        FrameOutput(
                            frame_index=frame_idx,
                            time_sec=time_sec,
                            bounding_boxes=boxes_output,
                            active_tracks=active_tracks,
                        )
                    )

                writer.write(frame)
                frame_idx += 1
        finally:
            writer.release()
            cap.release()

        LOGGER.info(
            "Annotated video %s: %d frames processed with %d annotations",
            bundle.event_id,
            frame_idx,
            annotations_generated,
        )

        return frame_outputs, output_path

    def _clamp_bbox(self, box: BoundingBox, width: int, height: int) -> Tuple[int, int, int, int]:
        x0 = int(round(box.x))
        y0 = int(round(box.y))
        w = max(1, int(round(box.width)))
        h = max(1, int(round(box.height)))

        x1 = x0 + w
        y1 = y0 + h

        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > width:
            x0 = max(0, width - w)
            x1 = min(width, x0 + w)
        if y1 > height:
            y0 = max(0, height - h)
            y1 = min(height, y0 + h)

        x0 = min(max(x0, 0), max(width - 1, 0))
        y0 = min(max(y0, 0), max(height - 1, 0))
        x1 = min(max(x1, x0 + 1), width)
        y1 = min(max(y1, y0 + 1), height)

        return x0, y0, x1 - x0, y1 - y0

    def _format_bbox_annotation(self, box: BoundingBox, clamped_bbox: Tuple[int, int, int, int]) -> List[str]:
        clamp_x, clamp_y, clamp_w, clamp_h = clamped_bbox
        label = box.label.title()
        return [
            f"{label} #{box.object_id}|{box.life} ({box.confidence:.2f})",
            f"({clamp_x},{clamp_y}) {clamp_h}x{clamp_w}",
        ]

    # Thumbnail annotation -----------------------------------------------

    def _annotate_thumbnails(
        self,
        workspace_bundle: EventBundle,
        analytics: AnalyticsData,
        frame_size: Tuple[int, int],
    ) -> Tuple[List[ThumbnailOutput], List[Path]]:
        width, height = frame_size
        font_scale = self.style.font_scale(width, height)

        annotated_paths: List[Path] = []
        thumbnail_outputs: List[ThumbnailOutput] = []
        thumbnails = sorted(workspace_bundle.files.get("thumbnails", []))
        if not thumbnails:
            return thumbnail_outputs, annotated_paths

        frames_with_objects = [frame for frame in analytics.iter_frames() if frame.objects]
        if not frames_with_objects:
            frames_with_objects = list(analytics.iter_frames())
        if not frames_with_objects:
            LOGGER.warning("No analytics frames available for thumbnails: %s", workspace_bundle.key)
            return thumbnail_outputs, annotated_paths

        for idx, thumbnail_path in enumerate(thumbnails):
            frame_data = self._match_thumbnail_to_frame(thumbnail_path.name, frames_with_objects, idx)
            source = cv2.imread(str(thumbnail_path))
            if source is None:
                LOGGER.warning("Failed to load thumbnail %s", thumbnail_path)
                continue

            resized = cv2.resize(source, frame_size, interpolation=cv2.INTER_CUBIC)
            boxes_output: List[Mapping[str, object]] = []
            for box in frame_data.objects:
                color = resolve_track_color(box.label)
                bbox = self._clamp_bbox(box, width, height)
                text_lines = self._format_bbox_annotation(box, bbox)
                draw_bbox_annotation(resized, bbox, text_lines, self.style, font_scale, color)
                boxes_output.append(
                    {
                        "track_id": box.object_id,
                        "label": box.label,
                        "confidence": box.confidence,
                        "life": box.life,
                        "frame": frame_data.frame_num,
                        "bbox": {
                            "x": bbox[0],
                            "y": bbox[1],
                            "width": bbox[2],
                            "height": bbox[3],
                        },
                    }
                )

            annotated_name = thumbnail_path.stem + "_annotated.jpg"
            annotated_path = thumbnail_path.with_name(annotated_name)
            cv2.imwrite(str(annotated_path), resized)

            annotated_paths.append(annotated_path)
            thumbnail_outputs.append(
                ThumbnailOutput(
                    filename=annotated_name,
                    source_filename=thumbnail_path.name,
                    frame_index=frame_data.frame_num,
                    time_sec=frame_data.time_sec or 0.0,
                    bounding_boxes=boxes_output,
                )
            )

        return thumbnail_outputs, annotated_paths

    def _match_thumbnail_to_frame(
        self,
        thumbnail_name: str,
        frames: Sequence[FrameDetections],
        thumbnail_index: int,
    ) -> FrameDetections:
        if not frames:
            raise RuntimeError("No frames available to match thumbnails")

        import re

        match = re.search(r"thumbnail_[a-z]*_?\w*_(\d+)", thumbnail_name.lower())
        if match:
            frame_num_candidate = int(match.group(1))
            for frame in frames:
                if frame.frame_num == frame_num_candidate:
                    return frame

        index = min(thumbnail_index, len(frames) - 1)
        return frames[index]

    # Reporting ----------------------------------------------------------

    def _build_report(
        self,
        bundle: EventBundle,
        analytics: AnalyticsData,
        fps: float,
        frame_size: Tuple[int, int],
        frame_count: int,
        video_summary: List[FrameOutput],
        thumbnail_outputs: List[ThumbnailOutput],
        ai_outputs: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        width, height = frame_size
        metadata = bundle.metadata or {}

        analytics_summary = dict(analytics.summary())
        analytics_summary["bounding_box_frames"] = len([f for f in analytics.iter_frames() if f.objects])

        measured_resolution = {"width": width, "height": height}
        metadata_resolution = self._extract_metadata_resolution(metadata)
        analytics_resolution = {"width": analytics.width, "height": analytics.height}
        matches_metadata = (
            metadata_resolution["width"]
            and metadata_resolution["height"]
            and metadata_resolution["width"] == width
            and metadata_resolution["height"] == height
        )

        run_info = {
            "run_id": self.config.run_id,
            "processed_at": datetime.utcnow().isoformat() + "Z",
        }

        processed_video = {
            "frame_count": frame_count,
            "fps": fps,
            "frames_with_annotations": [frame.to_serializable() for frame in video_summary],
        }

        processed_thumbnails = [thumb.to_serializable() for thumb in thumbnail_outputs]

        identity_provenance = {
            "event_id": bundle.event_id,
            "event_key": bundle.key,
            "run": run_info,
        }

        input_section = {
            "file_info": {
                "source_files": sorted(path.name for path in bundle.iter_files()),
            },
            "event_dto": metadata,
        }

        processed_section = {
            "analytics_summary": analytics_summary,
            "video_resolution": {
                "measured": measured_resolution,
                "metadata": metadata_resolution,
                "analytics": analytics_resolution,
                "matches_metadata": bool(matches_metadata),
            },
            "video_annotations": processed_video,
            "thumbnail_annotations": processed_thumbnails,
            "analytics": {
                "tracks": [track.to_serializable() for track in analytics.iter_tracks()],
            },
        }

        output_files = {
            "annotated_video": f"{bundle.event_id}_annotated.mp4",
            "annotated_thumbnails": [thumb.filename for thumb in thumbnail_outputs],
        }

        json_payload = {
            "identity_provenance": identity_provenance,
            "outputs_ai": ai_outputs or {},
            "input": input_section,
            "processed": processed_section,
            "output_files": output_files,
        }

        return json_payload

    def _extract_metadata_resolution(self, metadata: Mapping[str, object]) -> Mapping[str, int | None]:
        # Attempt to locate resolution information inside event DTO
        media = metadata.get("media") if isinstance(metadata, Mapping) else {}
        resolution_width = None
        resolution_height = None
        if isinstance(media, Mapping):
            video_info = media.get("video") or media.get("videoCdnInfo") or {}
            if isinstance(video_info, Mapping):
                resolution_width = video_info.get("width") or video_info.get("w")
                resolution_height = video_info.get("height") or video_info.get("h")
        return {"width": resolution_width, "height": resolution_height}

    # Finalization -------------------------------------------------------

    def _finalize_outputs(
        self,
        bundle: EventBundle,
        workspace: Path,
        annotated_video_path: Path | None,
        annotated_thumbnails: Sequence[Path],
        report: Mapping[str, object],
    ) -> Path:
        event_slug = _slugify(bundle.event_id or bundle.key)
        output_dir = self.config.processed_root / self.config.run_id / event_slug
        output_dir.mkdir(parents=True, exist_ok=True)

        if annotated_video_path and annotated_video_path.exists():
            shutil.move(str(annotated_video_path), str(output_dir / annotated_video_path.name))

        for thumb_path in annotated_thumbnails:
            destination = output_dir / thumb_path.name
            if destination.exists():
                destination.unlink()
            shutil.move(str(thumb_path), str(destination))

        json_path = output_dir / f"{bundle.event_id}_processed.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        LOGGER.info("Finalized event %s into %s", bundle.key, output_dir)

        return output_dir


__all__ = ["EventProcessor", "ProcessingConfig", "ProcessedEventResult"]

