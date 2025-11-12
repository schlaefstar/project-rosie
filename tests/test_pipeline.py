"""Tests for the video processing pipeline utilities."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.pipeline import EventProcessor, ProcessingConfig, discover_events, iter_complete_events, load_analytics
from tools.pipeline.processor import FrameOutput, ThumbnailOutput
from tools.pipeline.queue import QueueManager


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write(path, json.dumps(payload))


class TestEventDiscovery(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pipeline_test_events_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_event(self, stem: str) -> EventBundle:
        metadata_payload = {
            "id": "101010101010",
            "userId": "111",
            "srcId": "222",
            "dttm": "2025-11-01T00:00:00Z",
            "media": {"video": {"width": 1920, "height": 1080}},
        }
        analytics_payload = {
            "width": 1920,
            "height": 1080,
            "durationSec": 4.0,
            "frames": [],
            "tracks": {},
        }

        _write_json(self.temp_dir / f"{stem}_eventDTO.json", metadata_payload)
        _write_json(self.temp_dir / f"{stem}_va-OnDevice.Camera.json.gz".replace(".json.gz", ".json"), analytics_payload)
        (self.temp_dir / f"{stem}.mp4").write_bytes(b"")
        (self.temp_dir / f"{stem}_thumbnail_256_jpeg_1.jpg").write_bytes(b"JPEG")

        events = discover_events(self.temp_dir)
        bundle = events[stem]
        self.assertTrue(bundle.is_complete())
        return bundle

    def test_discover_events_loads_metadata(self) -> None:
        stem = "1234_5678_101010101010DoorbellCam_2025-11-01-00-00-00"
        bundle = self._create_event(stem)
        self.assertEqual(bundle.event_id, "101010101010")
        self.assertEqual(bundle.user_id, "111")
        self.assertEqual(bundle.src_id, "222")
        self.assertIsNotNone(bundle.metadata)

    def test_iter_complete_events_filters_incomplete(self) -> None:
        stem_complete = "123_456_100000000000Cam_2025-11-01-00-00-01"
        bundle = self._create_event(stem_complete)
        incomplete_stem = "123_456_999999999999Cam_2025-11-01-00-00-02"
        (self.temp_dir / f"{incomplete_stem}.mp4").write_bytes(b"")
        events = discover_events(self.temp_dir)
        complete = list(iter_complete_events(events))
        self.assertIn(bundle, complete)
        self.assertEqual(len(complete), 1)


class TestQueueManager(unittest.TestCase):
    def setUp(self) -> None:
        self.source_dir = Path(tempfile.mkdtemp(prefix="pipeline_test_source_"))
        self.queue_dir = Path(tempfile.mkdtemp(prefix="pipeline_test_queue_"))

        self.event_stems = [
            "100_200_300Cam_2025-11-01-00-00-00",
            "101_201_301Cam_2025-11-01-00-00-01",
        ]
        payload = {
            "id": "300",
            "userId": "100",
            "srcId": "200",
            "dttm": "2025-11-01T00:00:00Z",
        }
        analytics_payload = {"frames": [], "tracks": {}, "width": 1920, "height": 1080}
        for stem in self.event_stems:
            _write_json(self.source_dir / f"{stem}_eventDTO.json", payload)
            _write_json(self.source_dir / f"{stem}_va-OnDevice.Camera.json", analytics_payload)
            (self.source_dir / f"{stem}.mp4").write_bytes(b"")
            (self.source_dir / f"{stem}_thumbnail_256_jpeg_1.jpg").write_bytes(b"")

    def tearDown(self) -> None:
        shutil.rmtree(self.source_dir, ignore_errors=True)
        shutil.rmtree(self.queue_dir, ignore_errors=True)

    def test_queue_random_events_copies_files(self) -> None:
        manager = QueueManager(self.source_dir, self.queue_dir)
        with mock.patch("tools.pipeline.queue.random.sample", side_effect=lambda seq, n: list(seq)[:n]):
            staged = manager.queue_random_events(count=1)

        self.assertTrue(staged)
        # Each event has 4 files; ensure they exist in queue
        queued_files = list(self.queue_dir.iterdir())
        self.assertEqual(len(queued_files), 4)
        for path in queued_files:
            self.assertTrue(path.exists())


class TestReportGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path(tempfile.mkdtemp(prefix="pipeline_test_report_"))
        self.raw_dir = self.temp_root / "raw"
        self.queue_dir = self.temp_root / "queue"
        self.processing_dir = self.temp_root / "processing"
        self.processed_dir = self.temp_root / "processed"
        self.raw_dir.mkdir()
        self.queue_dir.mkdir()
        self.processing_dir.mkdir()
        self.processed_dir.mkdir()

        stem = "120_220_320Cam_2025-11-02-01-01-01"
        self.event_dir = self.queue_dir
        metadata_payload = {
            "id": "320",
            "userId": "120",
            "srcId": "220",
            "dttm": "2025-11-02T01:01:01Z",
            "media": {"video": {"width": 1280, "height": 720}},
        }
        analytics_payload = {
            "width": 1280,
            "height": 720,
            "durationSec": 5.0,
            "frames": [
                {
                    "num": 0,
                    "timeSec": 0.0,
                    "objects": [
                        {"id": "1", "type": "person", "conf": 0.95, "life": 3, "x": 10, "y": 20, "w": 30, "h": 40}
                    ],
                }
            ],
            "tracks": {
                "1": {
                    "type": "person",
                    "startSec": 0.0,
                    "endSec": 2.0,
                    "centerStart": {"x": 10, "y": 20},
                    "centerEnd": {"x": 50, "y": 60},
                }
            },
        }

        _write_json(self.event_dir / f"{stem}_eventDTO.json", metadata_payload)
        _write_json(self.event_dir / f"{stem}_va-OnDevice.Camera.json", analytics_payload)
        (self.event_dir / f"{stem}.mp4").write_bytes(b"")
        (self.event_dir / f"{stem}_thumbnail_256_jpeg_1.jpg").write_bytes(b"")

        events = discover_events(self.event_dir)
        self.bundle = next(iter(events.values()))

        self.analytics = load_analytics(self.event_dir / f"{stem}_va-OnDevice.Camera.json")

        self.config = ProcessingConfig(
            raw_dir=self.raw_dir,
            queue_dir=self.queue_dir,
            processing_root=self.processing_dir,
            processed_root=self.processed_dir,
            run_id="test_run",
        )
        self.processor = EventProcessor(self.config)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_build_report_includes_expected_data(self) -> None:
        frame_output = FrameOutput(
            frame_index=0,
            time_sec=0.0,
            bounding_boxes=[
                {
                    "track_id": "1",
                    "label": "person",
                    "confidence": 0.95,
                    "life": 3,
                    "frame": 0,
                    "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
                }
            ],
            active_tracks=["1"],
        )
        thumb_output = ThumbnailOutput(
            filename="thumb_annotated.jpg",
            source_filename="thumb.jpg",
            frame_index=0,
            time_sec=0.0,
            bounding_boxes=frame_output.bounding_boxes,
        )

        report = self.processor._build_report(  # pylint: disable=protected-access
            bundle=self.bundle,
            analytics=self.analytics,
            fps=30.0,
            frame_size=(1280, 720),
            frame_count=120,
            video_summary=[frame_output],
            thumbnail_outputs=[thumb_output],
        )

        identity = report["identity_provenance"]
        processed = report["processed"]

        self.assertEqual(identity["event_id"], "320")
        self.assertEqual(identity["run"]["run_id"], "test_run")
        self.assertEqual(processed["video_resolution"]["measured"], {"width": 1280, "height": 720})
        self.assertEqual(len(processed["analytics"]["tracks"]), 1)
        self.assertEqual(len(processed["video_annotations"]["frames_with_annotations"]), 1)
        self.assertTrue(processed["video_resolution"]["matches_metadata"])


if __name__ == "__main__":
    unittest.main()

