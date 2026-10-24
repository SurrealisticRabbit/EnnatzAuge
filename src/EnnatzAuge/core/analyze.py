"""Pipeline entry: run both models, project onto the virtual pitch, render."""

from __future__ import annotations

from pathlib import Path

import cv2

from .models.pitch import PitchModel
from .models.players import PlayersModel
from .objects.ball import Ball
from .objects.goalkeeper import Goalkeeper
from .objects.pitch import VirtualPitch
from .objects.player import Player
from .objects.referee import Referee
from ..render.overlayed_video import OverlayedVideo
from ..render.position_video import PositionVideo

DEFAULT_IMGSZ = 1280
DEFAULT_CHUNK = 16


def read_frames(video: str, start_frame: int, frame_count: int | None):
    """Read the requested window of frames, plus the source fps.

    Buffers the whole window, which is only safe for short clips: a 1080p BGR
    frame is ~6.2 MB, so one minute is ~9 GB and a 15-minute match is ~127 GB.
    Prefer :func:`iter_frame_chunks` for anything longer than a few seconds.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    try:
        while frame_count is None or len(frames) < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    return frames, fps


def iter_frame_chunks(video: str, start_frame: int, frame_count: int | None,
                      chunk_size: int):
    """Yield ``(chunk, fps)`` without buffering the whole window.

    The renderers re-open the video themselves, so nothing downstream needs the
    frames kept -- only the per-frame state, which is tiny. Streaming keeps peak
    memory at one chunk instead of the entire clip.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    read = 0
    try:
        while frame_count is None or read < frame_count:
            want = chunk_size
            if frame_count is not None:
                want = min(chunk_size, frame_count - read)
            chunk = []
            for _ in range(want):
                ok, frame = cap.read()
                if not ok:
                    break
                chunk.append(frame)
            if not chunk:
                return
            read += len(chunk)
            yield chunk, fps
    finally:
        cap.release()


class Analyzer:
    def __init__(self, pitch_model_path: str, players_model_path: str,
                 input_video: str, start_frame: int = 0,
                 frame_count: int | None = None, imgsz: int = DEFAULT_IMGSZ,
                 chunk_size: int = DEFAULT_CHUNK, scale: float = 10.0,
                 use_openvino: bool = True):
        self.pitch_model = PitchModel(pitch_model_path, imgsz=imgsz,
                                      use_openvino=use_openvino)
        self.players_model = PlayersModel(players_model_path, imgsz=imgsz,
                                          use_openvino=use_openvino)
        self.virtual_pitch = VirtualPitch(scale=scale)
        self.input_video = input_video
        self.start_frame = start_frame
        self.frame_count = frame_count
        self.chunk_size = max(1, chunk_size)
        self.fps = 25.0

    # -- rendering ---------------------------------------------------------

    def _render_overlayed_video(self, output_video: str):
        overlayed_video = OverlayedVideo(self.input_video, self.virtual_pitch,
                                         start_frame=self.start_frame,
                                         frame_count=self.frame_count)
        return overlayed_video.render(output_video)

    def _render_position_video(self, output_video: str):
        position_video = PositionVideo(self.input_video, self.virtual_pitch,
                                       fps=self.fps,
                                       frame_count=self.frame_count)
        return position_video.render(output_video)

    # -- analysis ----------------------------------------------------------

    def _calculate_positions(self):
        """Infer chunk by chunk, streaming so the clip is never fully buffered."""
        processed = 0
        total = self.frame_count

        for chunk, fps in iter_frame_chunks(self.input_video, self.start_frame,
                                            self.frame_count, self.chunk_size):
            self.fps = fps
            pitch_results = self.pitch_model.predict(chunk)
            player_results = self.players_model.predict(chunk)
            for _pitch_frame, _player_frame in zip(pitch_results, player_results):
                self.virtual_pitch.ingest_frame_data(_pitch_frame, _player_frame)
            processed += len(chunk)
            total_text = f"/{total}" if total else ""
            print(f"  processed {processed}{total_text} frames", flush=True)

        if processed == 0:
            raise ValueError("no frames read from the requested window")

    def _calculate_teams(self):
        # Placeholder for later: clustering to assign players to teams.
        pass

    def analyze(self, output_dir: str = "."):
        self._calculate_positions()
        self._calculate_teams()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        overlay = self._render_overlayed_video(str(out / "overlayed_output.mp4"))
        position = self._render_position_video(str(out / "position_output.mp4"))
        return overlay, position
