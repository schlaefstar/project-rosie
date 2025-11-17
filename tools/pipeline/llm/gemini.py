"""Gemini provider implementation for captioning events."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Sequence

try:
    import google.generativeai as genai  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - module import guard
    genai = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in runtime environment with dependency installed
    _IMPORT_ERROR = None

from .base import Classification, LLMError, LLMProvider, LLMResult, MediaEvent, TokenUsage
from .pricing import PricingTable
from .usage import UsageLedger, UsageRecord

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROMPT_PATH = _PROJECT_ROOT / "config" / "llm_prompts" / "gemini_caption_prompt.txt"


def _default_schema() -> str:
    return (
        '{"steady_state": str, "summary": str, '
        '"classifications": [{"label": str, "confidence": float, "rationale": str}], '
        '"token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}'
    )


def _read_binary(path: Path) -> bytes:
    return path.read_bytes()


@dataclass
class GeminiProvider(LLMProvider):
    """Adapter around the Gemini API."""

    api_key: str
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.2
    max_output_tokens: int | None = None
    pricing: PricingTable | None = None
    ledger: UsageLedger | None = None
    prompt_template_path: Path | None = None
    prompt_builder: Callable[[MediaEvent], str] | None = None
    read_binary: Callable[[Path], bytes] = _read_binary
    timeout: float | None = None

    def __post_init__(self) -> None:
        if genai is None:  # pragma: no cover - guard for missing dependency
            raise RuntimeError(
                "google-generativeai is not installed. Install the package to use GeminiProvider."
            ) from _IMPORT_ERROR
        self.model_name = self._normalize_model_name(self.model_name)
        genai.configure(api_key=self.api_key)
        generation_config: MutableMapping[str, Any] = {"temperature": self.temperature}
        if self.max_output_tokens:
            generation_config["max_output_tokens"] = self.max_output_tokens
        self._generation_config = dict(generation_config)
        self._model = genai.GenerativeModel(self.model_name, generation_config=generation_config)
        self._prompt_template = self._load_prompt_template()
        if not self._prompt_template:
            LOGGER.error(
                "Prompt template for GeminiProvider is empty. Requests will be sent without system context."
            )
        if self.prompt_builder is None:
            template = self._prompt_template

            def _builder(event: MediaEvent, template: str = template) -> str:
                return _render_prompt_template(template, event)

            self.prompt_builder = _builder

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def supports_video(self) -> bool:
        return True

    def caption_event(self, event: MediaEvent) -> LLMResult:
        if not self.prompt_builder:
            raise LLMError("Prompt builder is not configured for GeminiProvider.")
        prompt = self.prompt_builder(event)
        LOGGER.debug(
            "Gemini prompt for event %s (model=%s): %s",
            event.event_id,
            self.model_name,
            prompt,
        )
        request_parts = [{"text": prompt}]
        media_parts = list(self._media_inputs(event))
        if not media_parts:
            raise LLMError("GeminiProvider requires at least one video or image for captioning.")

        try:
            response = self._model.generate_content(
                request_parts + media_parts,
                **({"request_options": {"timeout": self.timeout}} if self.timeout else {}),
            )
        except Exception as exc:  # pragma: no cover - network errors
            raise LLMError(f"Gemini API call failed: {exc}") from exc

        parsed_json = self._parse_json(response)
        token_usage = self._extract_usage(response)
        cost_usd = self.estimate_cost(token_usage)

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

        classifications = [
            Classification(
                label=item.get("label", ""),
                confidence=float(item.get("confidence", 0.0)),
                rationale=item.get("rationale"),
            )
            for item in parsed_json.get("classifications", [])
        ]
        steady_state = parsed_json.get("steady_state")
        if isinstance(steady_state, str):
            steady_state = steady_state.strip() or None
        else:
            steady_state = None

        result = LLMResult(
            summary=parsed_json.get("summary", "").strip(),
            steady_state=steady_state,
            classifications=classifications,
            token_usage=token_usage,
            raw_response=response,
            cost_usd=cost_usd,
        )
        return result

    def estimate_cost(self, usage: TokenUsage) -> float | None:
        if not self.pricing:
            return None
        total_tokens = usage.total_tokens
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        return self.pricing.cost_for_tokens(
            self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _media_inputs(self, event: MediaEvent) -> Iterable[Mapping[str, Any]]:
        if event.video_path:
            video_bytes = self.read_binary(event.video_path)
            yield {"mime_type": "video/mp4", "data": video_bytes}
        for frame_path in event.frame_paths:
            image_bytes = self.read_binary(frame_path)
            mime_type = _guess_image_mime(frame_path)
            yield {"mime_type": mime_type, "data": image_bytes}

    def _parse_json(self, response: Any) -> MutableMapping[str, Any]:
        candidate_text = getattr(response, "text", "") or ""
        if not candidate_text and getattr(response, "candidates", None):
            first_candidate = response.candidates[0]
            parts = getattr(first_candidate, "content", getattr(first_candidate, "parts", None))
            if parts and hasattr(parts, "__iter__"):
                for part in getattr(parts, "parts", parts):  # type: ignore[attr-defined]
                    text_value = getattr(part, "text", None)
                    if text_value:
                        candidate_text = text_value
                        break
        if not candidate_text:
            raise LLMError("Gemini response did not include textual output.")
        candidate_text = candidate_text.strip()
        candidate_text = _strip_code_fences(candidate_text)
        try:
            return json.loads(candidate_text)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Gemini response was not valid JSON: %s", candidate_text)
            raise LLMError(f"Unable to parse JSON response: {exc}") from exc

    def _extract_usage(self, response: Any) -> TokenUsage:
        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = int(_meta_value(usage_meta, "prompt_token_count") or 0)
        completion_tokens = int(_meta_value(usage_meta, "candidates_token_count") or 0)
        total_value = _meta_value(usage_meta, "total_token_count")
        total_tokens = int(total_value) if total_value is not None else prompt_tokens + completion_tokens
        metadata = {
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": completion_tokens,
            "total_token_count": total_tokens,
        }
        if isinstance(usage_meta, Mapping):
            metadata.update({k: v for k, v in usage_meta.items() if k not in metadata})
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata=metadata,
        )

    def _load_prompt_template(self) -> str:
        candidate_paths: List[Path] = []
        if self.prompt_template_path:
            candidate_paths.append(Path(self.prompt_template_path).expanduser().resolve())
        env_path = os.environ.get("ROSIE_LLM_PROMPT_PATH")
        if env_path:
            candidate_paths.append(Path(env_path).expanduser().resolve())
        candidate_paths.append(_DEFAULT_PROMPT_PATH)

        for path in candidate_paths:
            try:
                if path.exists():
                    return path.read_text(encoding="utf-8")
            except Exception as exc:  # pragma: no cover - configuration issue
                LOGGER.warning("Failed to load prompt template from %s: %s", path, exc)

        LOGGER.error("No prompt template found for GeminiProvider; returning an empty prompt template.")
        return ""

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("model_name must be non-empty.")
        lowered = name.lower()
        if not lowered.startswith("models/"):
            return f"models/{name}"
        return name


def _guess_image_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _meta_value(meta: Any, key: str) -> Any:
    if meta is None:
        return None
    if isinstance(meta, Mapping):
        return meta.get(key)
    return getattr(meta, key, None)


def _strip_code_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _render_prompt_template(template: str, event: MediaEvent) -> str:
    detector_summary = event.detector_summary or "No detections reported."
    frame_list = ", ".join(path.name for path in event.frame_paths) if event.frame_paths else "None provided."
    replacements = {
        "[[EVENT_ID]]": event.event_id or "unknown-event",
        "[[DETECTOR_SUMMARY]]": detector_summary,
        "[[FRAME_LIST]]": frame_list,
        "[[SCHEMA_JSON]]": _default_schema(),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


__all__ = ["GeminiProvider"]


