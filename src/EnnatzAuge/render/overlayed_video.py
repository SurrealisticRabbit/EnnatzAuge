"""Draw detections and trusted pitch keypoints over the source video."""

from __future__ import annotations

from pathlib import Path

import cv2

from ..core.objects.pitch import VirtualPitch

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "referee": (0, 0, 255),
    "player": (0, 200, 0),
    "goalkeeper": (255, 0, 0),
    "ball": (0, 255, 255),
}
DEFAULT_COLOR = (200, 200, 200)
KEYPOINT_COLOR = (0, 255, 255)


class OverlayedVideo:
    """Renders the source frames with boxes and keypoints drawn on top."""

    def __init__(self, input_video: str, virtual_pitch: VirtualPitch,
                 start_frame: int = 0, frame_count: int | None = None):
        self.input_video = Path(input_video)
        self.virtual_pitch = virtual_pitch
        self.start_frame = start_frame
        self.frame_count = frame_count

    def _draw(self, frame, state):
        for obj in state.all_objects():
            if obj.bbox is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in obj.bbox)
            color = CLASS_COLORS.get(obj.class_name, DEFAULT_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            track_id = getattr(obj, "track_id", 0)
            label = f"{obj.class_name} {obj.confidence:.2f}"
            if track_id:
                label = f"#{track_id} {label}"
            speed = getattr(obj, "speed", 0.0)
            if speed > 0.5:
                label = f"{label} {speed:.1f}m/s"
            cv2.putText(frame, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # A dim marker at the projected heel helps see which box produced
            # which dot on the virtual pitch.
            if obj.image_position is not None:
                point = (int(obj.image_position[0]), int(obj.image_position[1]))
                cv2.drawMarker(frame, point, color, cv2.MARKER_CROSS, 10, 1)

        for x, y in state.keypoints:
            cv2.circle(frame, (int(round(x)), int(round(y))), 4,
                       KEYPOINT_COLOR, -1)

        self._draw_status(frame, state)
        return frame

    @staticmethod
    def _draw_status(frame, state) -> None:
        if state.homography_held:
            label, color = "homography HELD OVER", (60, 170, 230)
        elif state.homography_ok:
            label = f"homography {state.homography_quality:.2f}"
            color = (140, 210, 140)
        else:
            label, color = f"no homography: {state.homography_reason}", (90, 90, 220)

        cv2.putText(frame, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
        if state.camera is not None:
            cv2.putText(frame, state.camera.summary(), (14, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                        cv2.LINE_AA)

    def render(self, output_video: str) -> Path:
        cap = cv2.VideoCapture(str(self.input_video))
        if not cap.isOpened():
            raise FileNotFoundError(f"could not open video: {self.input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        out_path = Path(output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"could not open video writer: {out_path}")

        available = len(self.virtual_pitch.history)
        limit = self.frame_count if self.frame_count is not None else available
        limit = min(limit, available)

        try:
            for index in range(limit):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(self._draw(frame, self.virtual_pitch.history[index]))
        finally:
            cap.release()
            writer.release()

        return out_path
