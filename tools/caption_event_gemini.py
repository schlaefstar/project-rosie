#!/usr/bin/env python3
"""Caption a single event using Gemini models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.pipeline.llm import (
    GeminiProvider,
    LLMClient,
    LLMError,
    MediaEvent,
    PricingRule,
    PricingTable,
    UsageLedger,
)

DEFAULT_USAGE_DB = Path.home() / ".rosie" / "llm_usage.sqlite"

PRICING_RULES = {
    "gemini-1.5-flash": PricingRule(input_cost_per_million=0.018, output_cost_per_million=0.054),
    "models/gemini-1.5-flash": PricingRule(input_cost_per_million=0.018, output_cost_per_million=0.054),
    "gemini-1.5-pro": PricingRule(input_cost_per_million=3.5, output_cost_per_million=10.5),
    "models/gemini-1.5-pro": PricingRule(input_cost_per_million=3.5, output_cost_per_million=10.5),
    "gemini-2.0-flash": PricingRule(input_cost_per_million=0.018, output_cost_per_million=0.054),
    "models/gemini-2.0-flash": PricingRule(input_cost_per_million=0.018, output_cost_per_million=0.054),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption a video or set of frames with Gemini.")
    parser.add_argument("--event-id", required=True, help="Identifier used for logging and ledger entries.")
    parser.add_argument("--video", type=Path, help="Path to an MP4/WebM clip to send to Gemini.")
    parser.add_argument(
        "--frame",
        dest="frames",
        action="append",
        type=Path,
        default=[],
        help="Optional additional frames or crops to include (can repeat).",
    )
    parser.add_argument(
        "--model",
        default="models/gemini-2.0-flash",
        help="Gemini model name (e.g., models/gemini-2.0-flash, models/gemini-2.0-flash-lite).",
    )
    parser.add_argument(
        "--detector-summary",
        help="Optional textual summary from detectors to include in the prompt.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional JSON file with metadata to include in the prompt.",
    )
    parser.add_argument(
        "--usage-db",
        type=Path,
        default=DEFAULT_USAGE_DB,
        help=f"SQLite ledger path for cost tracking (default: {DEFAULT_USAGE_DB}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Request timeout in seconds.",
    )
    return parser.parse_args()


def load_metadata(path: Path | None) -> dict:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_inputs(video: Path | None, frames: Sequence[Path]) -> None:
    if not video and not frames:
        raise SystemExit("Provide either --video or at least one --frame.")
    if video and not video.exists():
        raise SystemExit(f"Video path not found: {video}")
    for frame in frames:
        if not frame.exists():
            raise SystemExit(f"Frame path not found: {frame}")


def main() -> None:
    args = parse_args()
    validate_inputs(args.video, args.frames)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable before running.")

    pricing = PricingTable(PRICING_RULES)
    ledger = UsageLedger(args.usage_db)
    provider = GeminiProvider(
        api_key=api_key,
        model_name=args.model,
        temperature=args.temperature,
        pricing=pricing,
        ledger=ledger,
        timeout=args.timeout,
    )

    client = LLMClient()
    client.register(provider)

    metadata = load_metadata(args.metadata)
    event = MediaEvent(
        event_id=args.event_id,
        video_path=args.video,
        frame_paths=tuple(args.frames),
        metadata=metadata,
        detector_summary=args.detector_summary,
    )

    try:
        result = client.caption_event(args.model, event)
    except LLMError as exc:
        raise SystemExit(f"LLM call failed: {exc}") from exc

    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()


