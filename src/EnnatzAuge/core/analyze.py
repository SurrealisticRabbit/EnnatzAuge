"""Pipeline entry: run both models, project onto the virtual pitch, render."""

from __future__ import annotations

from pathlib import Path

import cv2

from .models import hardware
from .models.pitch import PitchModel
from .models.players import PlayersModel
from .objects.ball import Ball
from .objects.cuts import CutDetector
from .objects.goalkeeper import Goalkeeper
from .objects.pitch import VirtualPitch
from .objects.player import Player
from .objects.referee import Referee
from ..render.overlayed_video import OverlayedVideo
from ..render.position_video import PositionVideo

DEFAULT_IMGSZ = 1280


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
                 chunk_size: int | None = None, scale: float = 10.0,
                 smoothing: float = 0.0, track: bool = True,
                 compensate: bool = False, detect_cuts: bool = True):
        # One backend for both models: they run on the same frames, so choosing
        # per model would only duplicate the export and the memory.
        self.backend = hardware.select()
        self.pitch_model = PitchModel(pitch_model_path, imgsz=imgsz,
                                      backend=self.backend)
        self.players_model = PlayersModel(players_model_path, imgsz=imgsz,
                                          backend=self.backend)
        self.virtual_pitch = VirtualPitch(scale=scale, smoothing=smoothing,
                                          track=track, compensate=compensate)
        self.cut_detector = CutDetector() if detect_cuts else None
        self.input_video = input_video
        self.start_frame = start_frame
        self.frame_count = frame_count
        # Explicit chunk wins; otherwise GPU memory decides for inference and a
        # fixed 16+ frames is decoded between calls.
        self.chunk_size = self.backend.read_chunk(chunk_size)
        self.fps = 25.0

    @property
    def backend_description(self) -> str:
        return self.pitch_model.describe

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
            for frame, _pitch_frame, _player_frame in zip(chunk, pitch_results,
                                                          player_results):
                # The frame itself is needed: a cut is a visual event and the
                # models cannot report it. Cheap compared with inference.
                cut = (self.cut_detector.update(frame)
                       if self.cut_detector is not None else False)
                self.virtual_pitch.ingest_frame_data(_pitch_frame, _player_frame,
                                                     fps=fps, cut=cut)
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
        self._report()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        overlay = self._render_overlayed_video(str(out / "overlayed_output.mp4"))
        position = self._render_position_video(str(out / "position_output.mp4"))
        return overlay, position

    def _report(self):
        """Print what the run achieved, so a bad model is obvious immediately."""
        stats = self.virtual_pitch.stats()
        homography = stats["homography"]
        tracks = stats["tracks"]

        print("\n  analysis summary")
        print(f"    frames              {stats['frames']}")
        print(f"    homography solved   {homography['accepted']}/{homography['frames']}"
              f" ({homography['solve_rate'] * 100:.1f}%),"
              f" {homography['held_over']} held over,"
              f" {homography['jumps']} rejected as jumps")
        print(f"    objects projected   {stats['projected']}"
              f" ({stats['projection_rate'] * 100:.1f}% of tracked boxes)")
        if stats["median_focal_px"]:
            print(f"    median focal length {stats['median_focal_px']:.0f} px")
        fit_quality = stats.get("fit_quality") or {}
        if fit_quality.get("frames"):
            print(f"    keypoint fit        median {fit_quality['median_inliers']} inliers,"
                  f" {fit_quality['median_redundancy']} redundant,"
                  f" spread {fit_quality['median_spread_px']:.0f} px")
            if fit_quality["median_redundancy"] <= 1:
                print(f"    WARNING             the pitch keypoints barely constrain a")
                print(f"                        homography: {fit_quality['median_inliers']}"
                      f" points with only {fit_quality['median_redundancy']} to spare.")
                print(f"                        Positions will drift. Retrain the pitch")
                print(f"                        model so it localises the touchlines and")
                print(f"                        penalty areas, not just the centre circle.")
        print(f"    tracks created      {tracks['created']}"
              f" ({tracks['live_tracks']} still live,"
              f" {tracks['turnover']:.2f} new ids per frame)")
        if stats["resyncs"]:
            print(f"    world frame flips   {stats['resyncs']}"
                  f" (the fit settled on a different solution; identity reset"
                  f" each time)")
        if tracks["mean_lifetime"]:
            print(f"    track lifetime      {tracks['mean_lifetime']:.0f} frames"
                  f" (longest {tracks['longest_lifetime']})")
        if stats["cuts"]:
            print(f"    scene cuts          {stats['cuts']}"
                  f" (temporal state reset at each)")
        if tracks["median_speed_ms"]:
            print(f"    player speed        median {tracks['median_speed_ms']:.2f} m/s,"
                  f" max {tracks['max_speed_ms']:.2f} m/s")
        if tracks["max_ball_speed_ms"]:
            print(f"    fastest ball        {tracks['max_ball_speed_ms']:.2f} m/s")
        gmc = tracks.get("gmc")
        if gmc and gmc["accepted"]:
            print(f"    camera jitter       {gmc['shift_m']:.2f} m corrected"
                  f" on {gmc['accepted']}/{gmc['accepted'] + gmc['rejected']} frames")
        if homography["last_reason"] not in ("ok", ""):
            print(f"    last frame rejected because: {homography['last_reason']}")
