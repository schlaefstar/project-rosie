"""ClipTagger provider implementation."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

import cv2
import numpy as np
import requests

from .base import Classification, LLMError, LLMProvider, LLMResult, MediaEvent, TokenUsage
from .pricing import PricingTable
from .usage import UsageLedger, UsageRecord

LOGGER = logging.getLogger(__name__)

CLIPTAGGER_SYSTEM_PROMPT = (
    "You are an image annotation API trained to analyze YouTube video keyframes. You will be given instructions "
    "on the output format, what to caption, and how to perform your job. Follow those instructions. For descriptions "
    "and summaries, provide them directly and do not lead them with 'This image shows' or 'This keyframe displays...', "
    "just get right into the details."
)

CLIPTAGGER_USER_PROMPT = """
You are an image annotation API trained to analyze YouTube video keyframes. You must respond with a valid JSON object matching the exact structure below.

Your job is to extract detailed **factual elements directly visible** in the image. Do not speculate or interpret artistic intent, camera focus, or composition. Do not include phrases like "this appears to be", "this looks like", or anything about the image itself. Describe what **is physically present in the frame**, and nothing more.

Return JSON in this structure:

{
    "description": "A detailed, factual account of what is visibly happening (4 sentences max). Only mention concrete elements or actions that are clearly shown. Do not include anything about how the image is styled, shot, or composed. Do not lead the description with something like 'This image shows' or 'this keyframe is...', just get right into the details.",
    "objects": ["object1 with relevant visual details", "object2 with relevant visual details", ...],
    "actions": ["action1 with participants and context", "action2 with participants and context", ...],
    "environment": "Detailed factual description of the setting and atmosphere based on visible cues (e.g., interior of a classroom with fluorescent lighting, or outdoor forest path with snow-covered trees).",
    "content_type": "The type of content it is, e.g. 'real-world footage', 'video game', 'animation', 'cartoon', 'CGI', 'VTuber', etc.",
    "specific_style": "Specific genre, aesthetic, or platform style (e.e., anime, 3D animation, mobile gameplay, vlog, tutorial, news broadcast, etc.)",
    "production_quality": "Visible production level: e.g., 'professional studio', 'amateur handheld', 'webcam recording', 'TV broadcast', etc.",
    "summary": "One clear, comprehensive sentence summarizing the visual content of the frame. Like the description, get right to the point.",
    "logos": ["logo1 with visual description", "logo2 with visual description", ...]
}

Rules:
- Be specific and literal. Focus on what is explicitly visible.
- Do NOT include interpretations of emotion, mood, or narrative unless it's visually explicit.
- No artistic or cinematic analysis.
- Always include the language of any text in the image if present as an object, e.g. "English text", "Japanese text", "Russian text", etc.
- Maximum 10 objects and 5 actions.
- Return an empty array for 'logos' if none are present.
- Always output strictly valid JSON with proper escaping.
- Output **only the JSON**, no extra text or explanation.
"""

DEFAULT_MODEL = "inference-net/cliptagger-12b"
DEFAULT_BASE_URL = "https://api.inference.net/v1"
DEFAULT_TEMPERATURE = 0.1
MAX_IMAGE_BYTES = 1_000_000


@dataclass
class ClipTaggerProvider(LLMProvider):
    """Adapter around the ClipTagger chat-completions API."""

    api_key: str
    model_name: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float | None = 60.0
    pricing: PricingTable | None = None
    ledger: UsageLedger | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "cliptagger"

    @property
    def supports_video(self) -> bool:
        return True

    def caption_event(self, event: MediaEvent) -> LLMResult:
        frame_b64, frame_source = self._resolve_frame(event)
        context_text = self._build_context_text(event, frame_source)

        user_content: List[Mapping[str, Any]] = [
            {"type": "text", "text": CLIPTAGGER_USER_PROMPT.strip()},
        ]
        if context_text:
            user_content.append({"type": "text", "text": context_text})
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "high"},
            }
        )

        messages = [
            {"role": "system", "content": CLIPTAGGER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        payload: MutableMapping[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
        }

        endpoint = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
        except Exception as exc:  # pragma: no cover - network failure
            raise LLMError(f"ClipTagger API call failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(
                f"ClipTagger API returned status {response.status_code}: {response.text.strip()}"
            )

        try:
            result_payload: Mapping[str, Any] = response.json()
        except ValueError as exc:  # pragma: no cover
            raise LLMError(f"ClipTagger response was not valid JSON: {exc}") from exc

        summary, steady_state, classifications = self._extract_json_fields(result_payload)
        token_usage = self._extract_token_usage(result_payload)

        if token_usage.total_tokens is None:
            token_usage.total_tokens = token_usage.prompt_tokens + token_usage.completion_tokens

        cost_usd = self.estimate_cost(token_usage)
        usage_cost = result_payload.get("usage", {}).get("total_cost")
        if usage_cost is not None:
            try:
                cost_usd = float(usage_cost)
            except Exception:  # pragma: no cover - non numeric
                LOGGER.debug("ClipTagger returned non-numeric total_cost: %s", usage_cost)

        if self.ledger and token_usage.total_tokens is not None:
            record = UsageRecord(
                provider=self.provider_name,
                model=self.model_name,
                event_id=event.event_id,
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                total_tokens=token_usage.total_tokens,
                cost_usd=cost_usd or 0.0,
                metadata={"raw_usage": token_usage.metadata},
            )
            self.ledger.record(record)

        return LLMResult(
            summary=summary,
            steady_state=steady_state,
            classifications=classifications,
            token_usage=token_usage,
            raw_response=result_payload,
            cost_usd=cost_usd,
        )

    def estimate_cost(self, usage: TokenUsage) -> float | None:
        if not self.pricing:
            return None
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        return self.pricing.cost_for_tokens(
            self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _resolve_frame(self, event: MediaEvent) -> tuple[str, str]:
        candidates: Sequence[Path] = tuple(path for path in event.frame_paths if path.exists())
        if candidates:
            frame_path = candidates[0]
            return self._encode_image_path(frame_path), frame_path.name

        if event.video_path and event.video_path.exists():
            return self._encode_video_frame(event.video_path), event.video_path.name

        raise LLMError("ClipTaggerProvider requires at least one frame or accessible video file.")

    def _encode_video_frame(self, video_path: Path) -> str:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise LLMError(f"Unable to open video: {video_path}")
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            raise LLMError(f"Unable to read a frame from video: {video_path}")
        return self._encode_image_array(frame)

    def _encode_image_path(self, image_path: Path) -> str:
        try:
            data = image_path.read_bytes()
        except OSError as exc:
            raise LLMError(f"Failed to read image file {image_path}: {exc}") from exc
        return self._encode_image_bytes(data, source=str(image_path))

    def _encode_image_bytes(self, data: bytes, source: str | None = None) -> str:
        if len(data) <= MAX_IMAGE_BYTES:
            return base64.b64encode(data).decode("utf-8")

        img_array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise LLMError(f"Unable to decode image data from {source or 'buffer'}.")
        return self._encode_image_array(image)

    def _encode_image_array(self, image: np.ndarray) -> str:
        quality = 90
        scale = 1.0
        resized = image

        while True:
            success, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not success:
                raise LLMError("Failed to encode frame as JPEG.")
            data = buffer.tobytes()
            if len(data) <= MAX_IMAGE_BYTES or (resized.shape[0] <= 256 and resized.shape[1] <= 256):
                return base64.b64encode(data).decode("utf-8")

            scale *= 0.85
            quality = max(40, int(quality * 0.9))
            new_width = max(1, int(resized.shape[1] * scale))
            new_height = max(1, int(resized.shape[0] * scale))
            resized = cv2.resize(resized, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _build_context_text(self, event: MediaEvent, frame_source: str) -> str:
        parts: List[str] = []
        if event.event_id:
            parts.append(f"Event ID: {event.event_id}")
        if frame_source:
            parts.append(f"Frame source: {frame_source}")
        if event.detector_summary:
            parts.append(f"Detector hints: {event.detector_summary}")
        if event.metadata:
            try:
                metadata_json = json.dumps(event.metadata, default=str)
            except Exception:  # pragma: no cover
                metadata_json = str(event.metadata)
            parts.append(f"Metadata: {metadata_json[:2000]}")
        return "\n".join(parts)

    def _extract_json_fields(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, str | None, List[Classification]]:
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError("ClipTagger response did not include choices.")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise LLMError("ClipTagger response message is empty.")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Failed to decode ClipTagger JSON payload: {exc}") from exc

        summary = str(parsed.get("summary", "")).strip()
        steady_state_candidate = parsed.get("environment") or parsed.get("description")
        steady_state = str(steady_state_candidate).strip() or None

        classifications: List[Classification] = []
        for obj in parsed.get("objects", []) or []:
            text = str(obj).strip()
            if text:
                classifications.append(
                    Classification(label="object", confidence=1.0, rationale=text)
                )
        for action in parsed.get("actions", []) or []:
            text = str(action).strip()
            if text:
                classifications.append(
                    Classification(label="action", confidence=1.0, rationale=text)
                )
        for logo in parsed.get("logos", []) or []:
            text = str(logo).strip()
            if text:
                classifications.append(
                    Classification(label="logo", confidence=1.0, rationale=text)
                )

        return summary, steady_state, classifications

    def _extract_token_usage(self, payload: Mapping[str, Any]) -> TokenUsage:
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens_raw = usage.get("total_tokens")
        total_tokens = int(total_tokens_raw) if total_tokens_raw is not None else None
        metadata = {k: v for k, v in usage.items()}
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata=metadata,
        )


__all__ = ["ClipTaggerProvider"]

