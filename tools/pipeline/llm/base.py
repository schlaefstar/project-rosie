"""Common interfaces and data structures for LLM-based captioning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol, Sequence, runtime_checkable


@dataclass(slots=True)
class TokenUsage:
    """Token accounting returned by an LLM provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> MutableMapping[str, Any]:
        total = self.total_tokens
        if total is None:
            total = self.prompt_tokens + self.completion_tokens
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": total,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Classification:
    """Structured label predicted by a model."""

    label: str
    confidence: float
    rationale: str | None = None

    def as_dict(self) -> MutableMapping[str, Any]:
        data: MutableMapping[str, Any] = {"label": self.label, "confidence": self.confidence}
        if self.rationale:
            data["rationale"] = self.rationale
        return data


@dataclass(slots=True)
class LLMResult:
    """Standardized return payload from a provider call."""

    summary: str
    classifications: Sequence[Classification]
    token_usage: TokenUsage
    raw_response: Any = None
    cost_usd: float | None = None

    def as_dict(self) -> MutableMapping[str, Any]:
        return {
            "summary": self.summary,
            "classifications": [classification.as_dict() for classification in self.classifications],
            "token_usage": self.token_usage.as_dict(),
            "cost_usd": self.cost_usd,
        }


@dataclass(slots=True)
class MediaEvent:
    """Media payload passed to a provider."""

    event_id: str
    video_path: Path | None = None
    frame_paths: Sequence[Path] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    detector_summary: str | None = None


class LLMError(RuntimeError):
    """Raised when a provider call fails."""


@runtime_checkable
class LLMProvider(Protocol):
    """Contract implemented by each provider adapter."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def supports_video(self) -> bool: ...

    def caption_event(self, event: MediaEvent) -> LLMResult: ...

    def estimate_cost(self, usage: TokenUsage) -> float | None: ...


__all__ = [
    "Classification",
    "LLMError",
    "LLMProvider",
    "LLMResult",
    "MediaEvent",
    "TokenUsage",
]


