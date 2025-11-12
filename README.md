# Video Processing Pipeline

This repository hosts the scripts and scaffolding for batch processing and annotating camera events across multiple devices. It intentionally excludes all media and raw event data so the project can be safely shared and versioned.

## Directory Overview

- `raw_videos/` – Local-only storage for original camera captures.
- `to_process/` – Queue of events waiting to be processed.
- `processing/` – Scratch space for intermediate outputs.
- `processed/` – Finalized artifacts ready for review or export.
- `by_device_type/` – Optional organization of data and samples by device.
- `tools/` – Command-line tools and scripts used to drive the pipeline.
- `.github/workflows/` – Continuous integration configuration.

## Tooling

Each script in `tools/` provides a command-line entry point. Extend the placeholders with your project-specific logic.

```bash
# Queue a random sample of events from raw storage into the processing queue
python3 tools/queue_random_events.py --source raw_videos --queue to_process --count 25

# Process all queued events through the pipeline (always include a short run suffix)
python3 tools/process_pipeline.py --queue to_process --processed processed \
  --raw raw_videos --processing processing --run-suffix agent

# Process a single event for debugging
python3 tools/process_video_event.py to_process/event123.json processed --run-suffix agent

# Build representative samples grouped by device type
python3 tools/build_device_samples.py raw_videos by_device_type

# Shell wrapper for running the full pipeline (edit the script to append --run-suffix agent)
./tools/process_all_queued.sh to_process processed
```

## Continuous Integration

The repository includes a placeholder GitHub Actions workflow that runs linting and a basic test suite. Replace the stub commands with your project's actual tooling as you implement the pipeline.

## Data Handling

**Important:** Media assets, raw event files, and generated outputs live outside of Git. The `.gitignore` file protects these directories and file types to keep the repository lightweight and compliant with storage policies.

## Processed JSON Schema (Summary)

- `identity_provenance`: Top-level identifiers for the event plus run metadata (`run_id`, `processed_at`).
- `input`: Source bookkeeping—original filenames referenced during processing and the raw event DTO payload.
- `processed`: Analytics outputs, including aggregate counters, resolution checks, per-frame annotations, thumbnail mappings, and track-level metrics.
- `output_files`: Final artifact names such as the annotated video and generated thumbnails.

Refer to real artifacts under `processed/<run_id>/<event_id>/` for the complete structure and field-level details.

## Agent Notes

- Always include a concise `--run-suffix` such as `agent` on any processing command so generated artifacts are traceable.
- Invoke repository scripts with `python3`; bare `python` may point at an unsupported interpreter.
- LLM helpers (Gemini) are optional. Without the Google client libraries installed, AI captioning features are skipped while core video processing continues to work.
- Install `pytest` locally (`python3 -m pip install pytest`) before running the test suite, as it is not bundled with the system Python.
