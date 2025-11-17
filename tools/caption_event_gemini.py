#!/usr/bin/env python3
"""Caption a single event using configurable LLM providers."""

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

from tools.pipeline.llm import ClipTaggerProvider, GeminiProvider, LLMClient, LLMError, MediaEvent, PricingRule, PricingTable, UsageLedger

DEFAULT_USAGE_DB = Path.home() / ".rosie" / "llm_usage.sqlite"

PRICING_RULES = {
    "gemini-1.5-flash": PricingRule(input_cost_per_million=0.018, output_cost_per_million=0.054),
    "models/gemini-1.5-flash": PricingRule(input_cost_per_million=0.018, output_cost_per_million=0.054),
    "gemini-1.5-pro": PricingRule(input_cost_per_million=3.5, output_cost_per_million=10.5),
    "models/gemini-1.5-pro": PricingRule(input_cost_per_million=3.5, output_cost_per_million=10.5),
    "gemini-2.0-flash": PricingRule(input_cost_per_million=0.018, output_cost_per_million=0.054),
    "models/gemini-2.0-flash": PricingRule(input_cost_per_million=0.018, output_cost_per_million=0.054),
    "cliptagger-12b": PricingRule(input_cost_per_million=0.30, output_cost_per_million=0.50),
    "cliptagger": PricingRule(input_cost_per_million=0.30, output_cost_per_million=0.50),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption a video or set of frames with the configured LLM provider.")
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
        help="Provider model identifier (e.g., models/gemini-2.0-flash, cliptagger-12b).",
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
        help="Generation temperature (if supported by the provider).",
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

    provider_choice = args.model.strip().lower()
    env_provider = (os.environ.get("ROSIE_LLM_PROVIDER") or "").strip().lower()
    pricing = PricingTable(PRICING_RULES)
    ledger = UsageLedger(args.usage_db)

    provider: GeminiProvider | ClipTaggerProvider
    model_key = provider_choice

    if provider_choice.startswith("cliptagger") or env_provider == "cliptagger":
        api_key = (
            os.environ.get("CLIPTAGGER_API_KEY")
            or os.environ.get("ROSIE_CLIPTAGGER_API_KEY")
            or os.environ.get("INFERENCE_API_KEY")
        )
        if not api_key:
            raise SystemExit(
                "ClipTagger selected, but CLIPTAGGER_API_KEY / ROSIE_CLIPTAGGER_API_KEY / INFERENCE_API_KEY is not set."
            )

        requested_model = (args.model or "cliptagger-12b").strip()
        if "/" in requested_model:
            provider_model = requested_model
            model_key = requested_model.split("/", 1)[-1]
        else:
            provider_model = f"inference-net/{requested_model}"
            model_key = requested_model

        base_url = os.environ.get("CLIPTAGGER_API_BASE_URL", "https://api.inference.net/v1")
        clip_temperature = args.temperature if args.temperature is not None else 0.1

        clip_rule = pricing.get("cliptagger-12b")
        if not clip_rule:
            clip_rule = PricingRule(input_cost_per_million=0.30, output_cost_per_million=0.50)
            pricing.register("cliptagger-12b", clip_rule)
        pricing.register(provider_model, clip_rule)

        provider = ClipTaggerProvider(
            api_key=api_key,
            model_name=provider_model,
            base_url=base_url,
            temperature=clip_temperature,
            pricing=pricing,
            ledger=ledger,
            timeout=args.timeout,
        )
    else:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable before running.")

        provider = GeminiProvider(
            api_key=api_key,
            model_name=args.model,
            temperature=args.temperature if args.temperature is not None else 0.2,
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
        result = client.caption_event(model_key, event)
    except LLMError as exc:
        raise SystemExit(f"LLM call failed: {exc}") from exc

    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()


