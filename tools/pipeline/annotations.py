"""Drawing utilities for bounding boxes and track overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]


TRACK_COLORS: dict[str, Color] = {
    "person": (0, 0, 255),  # Red (BGR)
    "vehicle": (0, 255, 0),  # Green
    "animal": (255, 0, 0),  # Blue
    "package": (0, 255, 255),  # Yellow
    "other": (0, 165, 255),  # Orange
}


@dataclass
class AnnotationStyle:
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    base_scale: float = 0.6
    thickness: int = 1
    padding: int = 6
    text_color: Color = (0, 0, 0)
    background_color: Color = (255, 255, 255)
    background_alpha: float = 0.5

    def font_scale(self, width: int, height: int) -> float:
        reference = max(width, height)
        scale = self.base_scale * max(reference / 1920.0, 0.5)
        return max(scale, 0.3)


def _ensure_within(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def draw_translucent_rect(
    image,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Color,
    alpha: float,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def draw_multiline_text(
    image,
    origin: Tuple[int, int],
    text_lines: Sequence[str],
    style: AnnotationStyle,
    font_scale: float,
) -> Tuple[int, int]:
    x, y = origin
    max_width = 0
    total_height = 0
    line_sizes: list[Tuple[int, int]] = []
    for line in text_lines:
        size, baseline = cv2.getTextSize(line, style.font, font_scale, style.thickness)
        line_height = size[1] + baseline
        line_sizes.append((size[0], line_height))
        max_width = max(max_width, size[0])
        total_height += line_height

    block_width = max_width + style.padding * 2
    block_height = total_height + style.padding * 2

    frame_height, frame_width = image.shape[:2]
    block_x = _ensure_within(x, 0, frame_width - block_width)
    block_y = _ensure_within(y - block_height, 0, frame_height - block_height)

    draw_translucent_rect(
        image,
        (block_x, block_y),
        (block_x + block_width, block_y + block_height),
        style.background_color,
        style.background_alpha,
    )

    text_y = block_y + style.padding
    for line, (_, line_height) in zip(text_lines, line_sizes):
        text_y += line_height
        cv2.putText(
            image,
            line,
            (block_x + style.padding, text_y),
            style.font,
            font_scale,
            style.text_color,
            style.thickness,
            lineType=cv2.LINE_AA,
        )

    return block_width, block_height


def draw_bbox_annotation(
    image,
    bbox: Tuple[int, int, int, int],
    text_lines: Sequence[str],
    style: AnnotationStyle,
    font_scale: float,
    color: Color,
) -> None:
    x, y, w, h = bbox
    frame_height, frame_width = image.shape[:2]
    top_left = (int(_ensure_within(x, 0, frame_width - 1)), int(_ensure_within(y, 0, frame_height - 1)))
    bottom_right = (
        int(_ensure_within(x + w, 0, frame_width - 1)),
        int(_ensure_within(y + h, 0, frame_height - 1)),
    )

    cv2.rectangle(image, top_left, bottom_right, color, thickness=max(1, int(round(font_scale * 2))))
    label_origin = (top_left[0] + style.padding, bottom_right[1] - style.padding)
    draw_multiline_text(image, label_origin, text_lines, style, font_scale)


def draw_polyline(image, points: Sequence[Tuple[int, int]], color: Color, thickness: int = 2) -> None:
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_marker(
    image,
    point: Tuple[int, int],
    color: Color,
    filled: bool,
    radius: int,
    outline_thickness: int = 2,
) -> None:
    if filled:
        cv2.circle(image, point, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    else:
        cv2.circle(image, point, radius, color, thickness=outline_thickness, lineType=cv2.LINE_AA)
        inner_radius = max(1, radius - outline_thickness)
        cv2.circle(image, point, inner_radius, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)


def draw_track_annotations(
    image,
    track_id: str,
    color: Color,
    poly_points: Sequence[Tuple[int, int]],
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    start_label: str,
    end_label: str,
    style: AnnotationStyle,
    font_scale: float,
    active_point: Tuple[int, int] | None = None,
) -> None:
    draw_polyline(image, poly_points, color=color, thickness=max(2, int(round(font_scale * 2))))

    marker_radius = max(4, int(round(font_scale * 6)))
    draw_marker(image, start_point, color=color, filled=True, radius=marker_radius)
    draw_marker(image, end_point, color=color, filled=False, radius=marker_radius + 2)

    if active_point is not None:
        cv2.circle(image, active_point, max(2, marker_radius // 2), color, thickness=-1, lineType=cv2.LINE_AA)

    marker_font_scale = font_scale * 0.6
    draw_multiline_text(
        image,
        (start_point[0] + style.padding, start_point[1] - style.padding),
        [f"track {track_id}", start_label],
        style,
        marker_font_scale,
    )
    draw_multiline_text(
        image,
        (end_point[0] + style.padding, end_point[1] + style.padding * 3),
        [end_label],
        style,
        marker_font_scale,
    )


def resolve_track_color(label: str) -> Color:
    return TRACK_COLORS.get(label.lower(), TRACK_COLORS["other"])


__all__ = [
    "AnnotationStyle",
    "TRACK_COLORS",
    "draw_bbox_annotation",
    "draw_multiline_text",
    "draw_track_annotations",
    "resolve_track_color",
]

