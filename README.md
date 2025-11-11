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
python tools/queue_random_events.py raw_videos to_process --sample-size 25

# Process all queued events through the pipeline
python tools/process_pipeline.py to_process processed

# Process a single event for debugging
python tools/process_video_event.py to_process/event123.json processed

# Build representative samples grouped by device type
python tools/build_device_samples.py raw_videos by_device_type

# Shell wrapper for running the full pipeline
./tools/process_all_queued.sh to_process processed
```

## Continuous Integration

The repository includes a placeholder GitHub Actions workflow that runs linting and a basic test suite. Replace the stub commands with your project's actual tooling as you implement the pipeline.

## Data Handling

**Important:** Media assets, raw event files, and generated outputs live outside of Git. The `.gitignore` file protects these directories and file types to keep the repository lightweight and compliant with storage policies.
