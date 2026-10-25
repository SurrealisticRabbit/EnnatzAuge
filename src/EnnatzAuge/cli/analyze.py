"""Analyse a clip: detections, pitch keypoints, and object positions.

    ennatzauge analyze --video clip.mp4
    ennatzauge analyze --frames 300 --imgsz 1280

Defaults come from configs/analyze.yaml, then from the EnnatzAuge root, so with
datasets and models in place no paths are needed on the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ConfigContext
from ..core.analyze import Analyzer
from ..paths import Paths

DEFAULT_PLAYERS_RUN = "players"
DEFAULT_PITCH_RUN = "pitch"


def _pick_weights(paths: Paths, explicit: str | None, run: str,
                  glob_hint: str) -> str | None:
    """Resolve a checkpoint: explicit path, then the newest matching run."""
    if explicit:
        path = _explicit_path(explicit)
        if not path.exists():
            # Also accept a path relative to the root, e.g. models/runs/x/best.pt
            alt = paths.root / explicit
            if alt.exists():
                return str(alt)
            return None
        return str(path)

    candidate = paths.weights(run, "best")
    if candidate.exists():
        return str(candidate)

    # Fall back to the most recently modified run that has a best.pt, which
    # saves having to know the uuid suffix.
    if paths.runs.exists():
        found = sorted(paths.runs.glob(f"{glob_hint}-*/weights/best.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            return str(found[0])
    return None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    # Defaults are None so configs/analyze.yaml can supply them: argparse
    # cannot tell a user-supplied value from a default one otherwise.
    parser.add_argument("--config", default="analyze",
                        help="config name under configs/ (default: analyze)")
    parser.add_argument("--video", help="source clip (default: <root>/input_video)")
    parser.add_argument("--players", help="players checkpoint (default: newest run)")
    parser.add_argument("--pitch", help="pitch checkpoint (default: newest run)")
    parser.add_argument("--output", help="output directory (default: <root>/models/analysis)")
    parser.add_argument("--start", type=int, default=None, dest="start_frame")
    parser.add_argument("--frames", type=int, default=None, dest="frame_count",
                        help="0 = to the end of the clip")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None, dest="chunk_size",
                        help="frames decoded between inference calls "
                             "(default: auto)")
    parser.add_argument("--scale", type=float, default=None,
                        help="pixels per metre in the virtual pitch")
    parser.add_argument("--no-pitch", action="store_true",
                        help="run detection only")
    parser.add_argument("--smoothing", type=float, default=None,
                        help="temporal prior on the homography, 0 = off "
                             "(default), 0.9 = strongly believe the previous "
                             "frame. Prefer 0; it damps real camera motion.")
    parser.add_argument("--no-track", action="store_true",
                        help="disable identity tracking and path smoothing")
    parser.add_argument("--no-cut-detect", action="store_true",
                        help="treat the clip as one continuous shot")
    parser.add_argument("--compensate", action="store_true",
                        help="subtract whole-frame camera jitter from positions "
                             "(off by default: it also subtracts real motion)")


def _explicit_path(value: str) -> Path:
    """Resolve a user-supplied path against the cwd, like any CLI argument.

    Only *defaults* come from the EnnatzAuge root. Prefixing an explicit path
    with the root would double it when the user already wrote ``.EnnatzAuge/...``.
    """
    return Path(value).expanduser().resolve()


def run(args: argparse.Namespace) -> int:
    paths = Paths()
    try:
        settings = ConfigContext(paths).analyze_config(args.config)
    except (FileNotFoundError, ValueError):
        settings = {}

    def pick(key: str, fallback=None):
        """CLI flag first, then the config file, then the built-in default."""
        value = getattr(args, key, None)
        return settings.get(key, fallback) if value is None else value

    frame_count = pick("frame_count", 0) or 0
    start_frame = int(pick("start_frame", 0) or 0)
    imgsz = int(pick("imgsz", 1280) or 1280)
    chunk_size = pick("chunk_size")
    scale = float(pick("scale", 10.0) or 10.0)
    smoothing = float(pick("smoothing", 0.0))

    video_setting = pick("video")
    if video_setting:
        video = _explicit_path(str(video_setting))
    else:
        candidates = sorted(paths.input_video.glob("*.mp4")) if paths.input_video.exists() else []
        if not candidates:
            print(f"error: no video given and none found in {paths.input_video}")
            return 1
        video = candidates[0]
    if not video.exists():
        print(f"error: video not found: {video}")
        return 1

    players_weights = _pick_weights(paths, pick("players"), DEFAULT_PLAYERS_RUN, "players")
    pitch_weights = None if args.no_pitch else _pick_weights(
        paths, pick("pitch"), DEFAULT_PITCH_RUN, "pitch")

    if players_weights is None and pitch_weights is None:
        print("error: no checkpoints found. Train a model first, or pass "
              "--players / --pitch explicitly.")
        return 1

    if players_weights is None:
        print("warning: no players checkpoint, position output will be empty")

    output_setting = pick("output")
    output = (_explicit_path(str(output_setting)) if output_setting
              else paths.analysis / "latest")

    print(f"video    {video}")
    print(f"players  {players_weights or '-'}")
    print(f"pitch    {pitch_weights or '-'}")
    print(f"output   {output}")

    # Analyzer needs both models; when only one is trained, reuse it so the
    # pipeline still runs and simply produces one kind of overlay.
    analyzer = Analyzer(
        pitch_model_path=pitch_weights or players_weights,
        players_model_path=players_weights or pitch_weights,
        input_video=str(video),
        start_frame=start_frame,
        frame_count=frame_count or None,
        imgsz=imgsz,
        chunk_size=chunk_size,
        scale=scale,
        smoothing=smoothing,
        track=not args.no_track,
        compensate=bool(args.compensate),
        detect_cuts=not args.no_cut_detect,
    )
    print(f"backend  {analyzer.backend_description}")

    overlay, position = analyzer.analyze(str(output))
    print(f"\noverlayed video: {overlay}")
    print(f"position video:  {position}")
    return 0
