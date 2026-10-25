"""Top-down virtual pitch, with each detected object projected onto it."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..core.objects.pitch import (
    PITCH_LENGTH_M,
    PITCH_MARGIN_M,
    PITCH_WIDTH_M,
    VirtualPitch,
)

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "referee": (0, 0, 255),
    "player": (0, 200, 0),
    "goalkeeper": (255, 0, 0),
    "ball": (0, 255, 255),
}
LINE_COLOR = (210, 210, 210)
BACKGROUND = 34


def pitch_template(scale: float) -> np.ndarray:
    """Top-down pitch drawing, produced once and copied per frame."""
    margin = int(PITCH_MARGIN_M * scale)
    width = int(PITCH_LENGTH_M * scale) + 2 * margin
    height = int(PITCH_WIDTH_M * scale) + 2 * margin
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)

    def p(x_m: float, y_m: float) -> tuple[int, int]:
        return (int(round(x_m * scale)) + margin,
                int(round(y_m * scale)) + margin)

    def rect(x0, y0, x1, y1, color=LINE_COLOR, thickness=1):
        cv2.rectangle(canvas, p(x0, y0), p(x1, y1), color, thickness)

    cv2.rectangle(canvas, p(0, 0), p(PITCH_LENGTH_M, PITCH_WIDTH_M), LINE_COLOR, 2)
    cv2.line(canvas, p(PITCH_LENGTH_M / 2, 0),
             p(PITCH_LENGTH_M / 2, PITCH_WIDTH_M), LINE_COLOR, 1)

    centre = p(PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2)
    cv2.circle(canvas, centre, int(9.15 * scale), LINE_COLOR, 1)
    cv2.circle(canvas, centre, 2, LINE_COLOR, -1)

    pen_half, goal_half = 40.32 / 2, 18.32 / 2
    cy = PITCH_WIDTH_M / 2
    for left in (True, False):
        x_goal = 0.0 if left else PITCH_LENGTH_M
        x_pen = 16.5 if left else PITCH_LENGTH_M - 16.5
        x_area = 5.5 if left else PITCH_LENGTH_M - 5.5
        rect(min(x_goal, x_pen), cy - pen_half, max(x_goal, x_pen), cy + pen_half)
        rect(min(x_goal, x_area), cy - goal_half, max(x_goal, x_area), cy + goal_half)

        spot = 11.0 if left else PITCH_LENGTH_M - 11.0
        cv2.circle(canvas, p(spot, cy), 2, LINE_COLOR, -1)
        cv2.ellipse(canvas, p(spot, cy),
                    (int(9.15 * scale), int(9.15 * scale)), 0,
                    -60 if left else 120, 60 if left else 240, LINE_COLOR, 1)
        rect(min(x_goal - 2, x_goal), cy - 3.66, max(x_goal + 2, x_goal), cy + 3.66,
             (120, 120, 120), 2)

    return canvas


class PositionVideo:
    """Renders a bird's-eye view of projected object positions."""

    def __init__(self, input_video: str, virtual_pitch: VirtualPitch,
                 fps: float = 30.0, frame_count: int | None = None):
        self.input_video = Path(input_video)
        self.virtual_pitch = virtual_pitch
        self.fps = fps
        self.frame_count = frame_count

    def _frame(self, template: np.ndarray, state) -> np.ndarray:
        scale = self.virtual_pitch.scale
        margin = int(PITCH_MARGIN_M * scale)
        canvas = template.copy()

        if not state.homography_ok:
            cv2.putText(canvas, "no homography this frame", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 220), 2, cv2.LINE_AA)
            self._draw_status(canvas, state)
            return canvas

        for obj in state.all_objects():
            if obj.current_position is None:
                continue
            color = CLASS_COLORS.get(obj.class_name, (200, 200, 200))

            # Trails make direction of travel readable at a glance; the last
            # third of the history is enough without turning the pitch into a
            # tangle of lines.
            trail = getattr(obj, "trail", None) or []
            if len(trail) > 2:
                points = [self._point(x, y, scale, margin) for x, y in trail]
                for index in range(1, len(points)):
                    weight = index / len(points)
                    shade = tuple(int(c * weight) for c in color)
                    cv2.line(canvas, points[index - 1], points[index], shade, 2)

            x, y = obj.current_position
            point = self._point(x, y, scale, margin)
            radius = 4 if obj.class_name == "ball" else 7
            cv2.circle(canvas, point, radius, color, -1)
            cv2.circle(canvas, point, radius, (20, 20, 20), 1)

            track_id = getattr(obj, "track_id", 0)
            if track_id and obj.class_name != "ball":
                cv2.putText(canvas, str(track_id), (point[0] + radius + 2, point[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1,
                            cv2.LINE_AA)

        self._draw_status(canvas, state)
        return canvas

    @staticmethod
    def _point(x_m: float, y_m: float, scale: float, margin: int) -> tuple[int, int]:
        return (int(round(x_m * scale)) + margin, int(round(y_m * scale)) + margin)

    def _draw_status(self, canvas: np.ndarray, state) -> None:
        """Projection count plus the recovered camera pose."""
        cv2.putText(canvas, f"{state.projected} on pitch", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        if state.camera is not None:
            cv2.putText(canvas, state.camera.summary(), (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 200, 170), 1,
                        cv2.LINE_AA)

        if state.homography_held:
            label, color = "homography held over", (90, 170, 220)
        elif state.homography_ok:
            label = f"fit {state.homography_quality:.2f}"
            color = (170, 200, 170)
        else:
            label, color = state.homography_reason or "no fit", (90, 90, 220)
        cv2.putText(canvas, label, (12, canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def render(self, output_video: str) -> Path:
        template = pitch_template(self.virtual_pitch.scale)
        height, width = template.shape[:2]

        out_path = Path(output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {out_path}")

        available = len(self.virtual_pitch.history)
        limit = self.frame_count if self.frame_count is not None else available
        limit = min(limit, available)

        try:
            for index in range(limit):
                writer.write(self._frame(template, self.virtual_pitch.history[index]))
        finally:
            writer.release()

        return out_path
