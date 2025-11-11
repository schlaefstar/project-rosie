#!/usr/bin/env bash
set -euo pipefail

# Placeholder script to run the full processing pipeline over queued events.
# Extend with project-specific invocation logic.

QUEUE_DIR=${1:-"to_process"}
PROCESSED_DIR=${2:-"processed"}

python tools/process_pipeline.py "$QUEUE_DIR" "$PROCESSED_DIR"
