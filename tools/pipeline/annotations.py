"""Drawing utilities for bounding boxes and track overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

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


def _measure_text_block(
    text_lines: Sequence[str],
    font_scales: Sequence[float],
    style: AnnotationStyle,
) -> Tuple[List[Tuple[str, float, int, int, int]], int, int]:
    metrics: List[Tuple[str, float, int, int, int]] = []
    max_width = 0
    total_height = style.padding * 2
    for line, scale in zip(text_lines, font_scales):
        (width, height), baseline = cv2.getTextSize(line, style.font, scale, style.thickness)
        metrics.append((line, scale, width, height, baseline))
        max_width = max(max_width, width)
        total_height += height + baseline

    block_width = max_width + style.padding * 2
    block_height = total_height
    return metrics, block_width, block_height


def _draw_text_block(
    image,
    anchor: Tuple[int, int],
    text_lines: Sequence[str],
    font_scales: Sequence[float],
    style: AnnotationStyle,
    background_color: Color,
    placement: str,
    background_alpha: float | None = None,
    block_width_override: int | None = None,
) -> None:
    metrics, block_width, block_height = _measure_text_block(text_lines, font_scales, style)
    if block_width_override is not None:
        block_width = max(block_width_override, block_width)
    frame_height, frame_width = image.shape[:2]
    ax, ay = anchor

    if placement == "inside":
        block_x = ax
        if block_x < 0:
            block_x = 0
        if block_x + block_width > frame_width:
            block_x = max(0, frame_width - block_width)
        block_y = ay - block_height
        if block_y < 0:
            block_y = 0
        if block_y + block_height > frame_height:
            block_y = max(0, frame_height - block_height)
    elif placement == "above":
        block_x = _ensure_within(ax - block_width // 2, 0, frame_width - block_width)
        block_y = _ensure_within(ay - block_height - style.padding, 0, frame_height - block_height)
    elif placement == "below":
        block_x = _ensure_within(ax - block_width // 2, 0, frame_width - block_width)
        block_y = _ensure_within(ay + style.padding, 0, frame_height - block_height)
    else:
        block_x = _ensure_within(ax, 0, frame_width - block_width)
        block_y = _ensure_within(ay - block_height, 0, frame_height - block_height)

    draw_translucent_rect(
        image,
        (block_x, block_y),
        (block_x + block_width, block_y + block_height),
        background_color,
        style.background_alpha if background_alpha is None else background_alpha,
    )

    text_y = block_y + style.padding
    for text, scale, _, height, baseline in metrics:
        text_y += height
        cv2.putText(
            image,
            text,
            (block_x + style.padding, text_y),
            style.font,
            scale,
            style.text_color,
            style.thickness,
            lineType=cv2.LINE_AA,
        )
        text_y += baseline


def draw_bbox_annotation(
    image,
    bbox: Tuple[int, int, int, int],
    text_lines: Sequence[str],
    style: AnnotationStyle,
    font_scale: float,
    color: Color,
    stroke_alpha: float = 1.0,
    background_alpha_scale: float = 1.0,
) -> None:
    x, y, w, h = bbox
    frame_height, frame_width = image.shape[:2]
    top_left = (int(_ensure_within(x, 0, frame_width - 1)), int(_ensure_within(y, 0, frame_height - 1)))
    bottom_right = (
        int(_ensure_within(x + w, 0, frame_width - 1)),
        int(_ensure_within(y + h, 0, frame_height - 1)),
    )

    thickness = max(3, int(round(font_scale * 2)) * 3)
    stroke_alpha = max(0.0, min(1.0, stroke_alpha))
    if stroke_alpha >= 1.0:
        cv2.rectangle(image, top_left, bottom_right, color, thickness=thickness)
    else:
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, thickness=thickness)
        cv2.addWeighted(overlay, stroke_alpha, image, 1 - stroke_alpha, 0, dst=image)
    label_anchor = (
        top_left[0],
        bottom_right[1],
    )
    font_scales = [font_scale * 1.15] + [font_scale] * (len(text_lines) - 1)
    _draw_text_block(
        image,
        anchor=label_anchor,
        text_lines=text_lines,
        font_scales=font_scales,
        style=style,
        background_color=color,
        placement="inside",
        background_alpha=style.background_alpha * background_alpha_scale,
        block_width_override=bottom_right[0] - top_left[0],
    )


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
    start_lines = [
        f"Track #{track_id}",
        start_label,
        f"({start_point[0]}, {start_point[1]})",
    ]
    end_lines = [
        f"Track #{track_id}",
        end_label,
        f"({end_point[0]}, {end_point[1]})",
    ]
    font_scales = [marker_font_scale * 1.15, marker_font_scale, marker_font_scale]
    _draw_text_block(
        image,
        anchor=start_point,
        text_lines=start_lines,
        font_scales=font_scales,
        style=style,
        background_color=color,
        placement="above",
        background_alpha=None,
    )
    _draw_text_block(
        image,
        anchor=end_point,
        text_lines=end_lines,
        font_scales=font_scales,
        style=style,
        background_color=color,
        placement="below",
        background_alpha=None,
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

