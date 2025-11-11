"""Construct representative samples grouped by device type."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.pipeline.events import parse_event_key
from tools.pipeline.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def _discover_run(processed_dir: Path, run_id: str | None) -> Path:
    if run_id:
        run_dir = processed_dir / run_id
        if not run_dir.exists():
            raise RuntimeError(f"Run id {run_id} not found in {processed_dir}")
        return run_dir

    run_dirs = [path for path in processed_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No processing runs found in {processed_dir}")
    run_dirs.sort()
    return run_dirs[-1]


def _load_event_payload(event_dir: Path) -> dict:
    json_candidates = list(event_dir.glob("*_processed.json"))
    if not json_candidates:
        raise RuntimeError(f"No processed JSON found in {event_dir}")
    json_path = json_candidates[0]
    with json_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_device_type(payload: dict) -> str:
    event_dto = payload.get("event_dto") or {}
    for key in ("deviceType", "cameraType", "camType", "device_type"):
        value = event_dto.get(key)
        if value:
            return value

    event_key = payload.get("event_key")
    if event_key:
        parsed = parse_event_key(event_key)
        camera_type = parsed.get("camera_type")
        if camera_type:
            return camera_type

    return "Unknown"


def build_device_samples(
    processed_dir: Path,
    output_dir: Path,
    *,
    run_id: str | None = None,
    max_per_device: int = 5,
    overwrite: bool = False,
) -> None:
    """Aggregate annotated media into device-specific sample folders."""

    configure_logging()
    processed_dir = processed_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _discover_run(processed_dir, run_id)
    LOGGER.info("Building device samples from run %s", run_dir.name)

    copied_counts: Dict[str, int] = defaultdict(int)

    for event_dir in sorted(run_dir.iterdir()):
        if not event_dir.is_dir():
            continue
        try:
            payload = _load_event_payload(event_dir)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", event_dir, exc)
            continue

        device_type = _resolve_device_type(payload)
        if copied_counts[device_type] >= max_per_device:
            continue

        destination_dir = output_dir / device_type
        destination_dir.mkdir(parents=True, exist_ok=True)

        for artifact in event_dir.iterdir():
            if not artifact.is_file():
                continue
            if not (
                artifact.name.endswith("_annotated.mp4")
                or artifact.name.endswith("_annotated.jpg")
                or artifact.name.endswith("_processed.json")
            ):
                continue
            destination = destination_dir / artifact.name
            shutil.copy2(artifact, destination)

        copied_counts[device_type] += 1
        LOGGER.info("Copied event %s to device folder %s", event_dir.name, device_type)

    LOGGER.info("Sample library updated: %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build device sample sets")
    parser.add_argument("--processed", type=Path, default=Path("processed"), help="Directory containing processed runs")
    parser.add_argument("--output", type=Path, default=Path("by_device_type"), help="Directory where sample sets should be stored")
    parser.add_argument("--run-id", type=str, default=None, help="Specific run id to source from (default latest)")
    parser.add_argument("--max-per-device", type=int, default=5, help="Maximum samples per device type")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing sample library instead of merging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_device_samples(
        processed_dir=args.processed,
        output_dir=args.output,
        run_id=args.run_id,
        max_per_device=args.max_per_device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

