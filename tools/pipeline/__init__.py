"""
Pipeline package providing utilities for the video processing workflow.

The modules exposed here encapsulate event discovery, analytics parsing,
media annotation, and orchestration helpers used by the command-line tools
in ``tools/``.
"""

from __future__ import annotations

from .analytics import (  # noqa: F401
    AnalyticsData,
    BoundingBox,
    FrameDetections,
    TrackSegment,
    load_analytics,
)
from .events import (  # noqa: F401
    EventBundle,
    discover_events,
    is_complete_event,
    iter_complete_events,
    parse_event_key,
)
try:  # pragma: no cover - optional dependency guard
    from .llm import (  # noqa: F401
        Classification,
        GeminiProvider,
        LLMClient,
        LLMError,
        LLMProvider,
        LLMResult,
        MediaEvent,
        PricingRule,
        PricingTable,
        TokenUsage,
        UsageLedger,
        UsageRecord,
    )
    __llm_available__ = True
except Exception:  # pragma: no cover - import failure surfaces as missing LLM features
    Classification = None  # type: ignore
    GeminiProvider = None  # type: ignore
    LLMClient = None  # type: ignore
    LLMError = None  # type: ignore
    LLMProvider = None  # type: ignore
    LLMResult = None  # type: ignore
    MediaEvent = None  # type: ignore
    PricingRule = None  # type: ignore
    PricingTable = None  # type: ignore
    TokenUsage = None  # type: ignore
    UsageLedger = None  # type: ignore
    UsageRecord = None  # type: ignore
    __llm_available__ = False
from .logging_utils import configure_logging, log_path  # noqa: F401
from .processor import EventProcessor, ProcessingConfig  # noqa: F401
from .queue import QueueManager  # noqa: F401

__all__ = [
    "Classification",
    "EventBundle",
    "discover_events",
    "is_complete_event",
    "iter_complete_events",
    "parse_event_key",
    "AnalyticsData",
    "BoundingBox",
    "FrameDetections",
    "TrackSegment",
    "load_analytics",
    "GeminiProvider",
    "LLMClient",
    "LLMError",
    "LLMProvider",
    "LLMResult",
    "MediaEvent",
    "configure_logging",
    "log_path",
    "PricingRule",
    "PricingTable",
    "EventProcessor",
    "ProcessingConfig",
    "QueueManager",
    "TokenUsage",
    "UsageLedger",
    "UsageRecord",
]

