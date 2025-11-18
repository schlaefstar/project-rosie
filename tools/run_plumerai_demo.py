#!/usr/bin/env python3
"""Run Plumerai video intelligence demo on a video file."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_random_video(raw_videos_dir: Path | None = None) -> Path:
    """Find a random MP4 video file in raw_videos directory."""
    if raw_videos_dir is None:
        raw_videos_dir = PROJECT_ROOT / "raw_videos"
    
    if not raw_videos_dir.exists():
        raise SystemExit(f"raw_videos directory not found: {raw_videos_dir}")
    
    # Find all MP4 files
    video_files = list(raw_videos_dir.glob("*.mp4"))
    
    if not video_files:
        raise SystemExit(f"No MP4 files found in {raw_videos_dir}")
    
    # Select a random video
    selected = random.choice(video_files)
    print(f"Selected random video: {selected.name}", file=sys.stderr)
    return selected


def load_config() -> dict:
    """Load Plumerai configuration from config file or environment variables."""
    config_path = PROJECT_ROOT / "config" / "plumerai.json"
    
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    
    # Allow environment variables to override config file
    config["demo_path"] = os.environ.get("PLUMERAI_DEMO_PATH") or config.get(
        "demo_path", ""
    )
    config["gstreamer_lib_path"] = os.environ.get(
        "PLUMERAI_GSTREAMER_LIB_PATH"
    ) or config.get("gstreamer_lib_path", "/opt/homebrew/lib")
    
    return config


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Get video width and height using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        dims = result.stdout.strip().split("x")
        return int(dims[0]), int(dims[1])
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Failed to get video dimensions: {e.stderr}")
    except (ValueError, IndexError) as e:
        raise SystemExit(f"Failed to parse video dimensions: {e}")


def check_rpath(binary_path: Path, expected_rpath: str) -> bool:
    """Check if binary has the expected rpath."""
    try:
        result = subprocess.run(
            ["otool", "-l", str(binary_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return expected_rpath in result.stdout
    except subprocess.CalledProcessError:
        return False


def fix_rpath(binary_path: Path, rpath: str) -> None:
    """Add rpath to a binary using install_name_tool."""
    try:
        subprocess.run(
            ["install_name_tool", "-add_rpath", rpath, str(binary_path)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Failed to fix rpath for {binary_path}: {e.stderr}")


def ensure_dependencies_fixed(demo_path: Path, gstreamer_lib_path: str) -> None:
    """Ensure all binaries have correct rpaths for GStreamer libraries."""
    binaries = [
        demo_path / "plumerai_demo",
        demo_path / "libosxvideo.dylib",
        demo_path / "libplumerai_gstreamer_plugin.dylib",
    ]
    
    for binary in binaries:
        if not binary.exists():
            continue
        if not check_rpath(binary, gstreamer_lib_path):
            print(f"Fixing rpath for {binary.name}...", file=sys.stderr)
            fix_rpath(binary, gstreamer_lib_path)


def run_demo(
    video_path: Path,
    demo_path: Path,
    width: int | None = None,
    height: int | None = None,
    fix_deps: bool = True,
    gstreamer_lib_path: str = "/opt/homebrew/lib",
) -> None:
    """Run the Plumerai demo on a video file."""
    demo_binary = demo_path / "plumerai_demo"
    if not demo_binary.exists():
        raise SystemExit(f"Demo binary not found at {demo_binary}")
    
    if not video_path.exists():
        raise SystemExit(f"Video file not found: {video_path}")
    
    # Fix dependencies if requested
    if fix_deps:
        ensure_dependencies_fixed(demo_path, gstreamer_lib_path)
    
    # Get video dimensions if not provided
    if width is None or height is None:
        print(f"Getting dimensions for {video_path.name}...", file=sys.stderr)
        width, height = get_video_dimensions(video_path)
        print(f"Video dimensions: {width}x{height}", file=sys.stderr)
    
    # Run the demo from the demo directory to avoid GStreamer scanning .venv
    # Use relative path for video if it's easier, but absolute path works too
    cmd = ["./plumerai_demo", str(video_path.resolve()), str(width), str(height)]
    print(f"Running from {demo_path}: {' '.join(cmd)}", file=sys.stderr)
    
    try:
        subprocess.run(cmd, check=True, cwd=demo_path)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Demo failed with exit code {e.returncode}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Plumerai video intelligence demo on a video file."
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        help="Path to video file (MP4, etc.). If not provided, a random video from raw_videos will be selected.",
    )
    parser.add_argument(
        "--raw-videos-dir",
        type=Path,
        help="Directory containing raw videos (default: PROJECT_ROOT/raw_videos)",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Video width (auto-detected if not provided)",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Video height (auto-detected if not provided)",
    )
    parser.add_argument(
        "--demo-path",
        type=Path,
        help="Path to plumerai demo directory (overrides config)",
    )
    parser.add_argument(
        "--no-fix-deps",
        action="store_true",
        help="Skip automatic dependency fixing",
    )
    parser.add_argument(
        "--gstreamer-lib-path",
        help="Path to GStreamer libraries (overrides config)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = load_config()
    
    # Resolve video path - use random selection if not provided
    if args.video:
        video_path = args.video.expanduser().resolve()
    else:
        raw_videos_dir = (
            args.raw_videos_dir.expanduser().resolve()
            if args.raw_videos_dir
            else None
        )
        video_path = find_random_video(raw_videos_dir)
    demo_path = (
        args.demo_path
        or Path(config["demo_path"]).expanduser().resolve()
        if config.get("demo_path")
        else None
    )
    
    if not demo_path:
        raise SystemExit(
            "Demo path not configured. Set PLUMERAI_DEMO_PATH environment variable "
            "or configure in config/plumerai.json"
        )
    
    gstreamer_lib_path = (
        args.gstreamer_lib_path or config.get("gstreamer_lib_path", "/opt/homebrew/lib")
    )
    
    run_demo(
        video_path=video_path,
        demo_path=demo_path,
        width=args.width,
        height=args.height,
        fix_deps=not args.no_fix_deps,
        gstreamer_lib_path=gstreamer_lib_path,
    )


if __name__ == "__main__":
    main()

