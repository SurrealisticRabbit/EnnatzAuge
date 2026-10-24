"""Analyse a clip: detections, pitch keypoints, and object positions.

    ennatzauge analyze --video clip.mp4
    ennatzauge analyze --frames 300 --imgsz 1280

Defaults are resolved from the EnnatzAuge root, so with datasets and models in
place no paths are needed on the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core.analyze import Analyzer
from ..paths import Paths

DEFAULT_PLAYERS_RUN = "players"
DEFAULT_PITCH_RUN = "pitch"


def _pick_weights(paths: Paths, explicit: str | None, run: str,
                  glob_hint: str) -> str | None:
    """Resolve a checkpoint: explicit path, then the newest matching run."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = paths.root / path
        return str(path) if path.exists() else None

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
    parser.add_argument("--video", help="source clip (default: <root>/input_video)")
    parser.add_argument("--players", help="players checkpoint (default: newest run)")
    parser.add_argument("--pitch", help="pitch checkpoint (default: newest run)")
    parser.add_argument("--output", help="output directory (default: <root>/models/analysis)")
    parser.add_argument("--start", type=int, default=0, dest="start_frame")
    parser.add_argument("--frames", type=int, default=0, dest="frame_count",
                        help="0 = to the end of the clip")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--chunk", type=int, default=16, dest="chunk_size")
    parser.add_argument("--scale", type=float, default=10.0,
                        help="pixels per metre in the virtual pitch")
    parser.add_argument("--no-openvino", action="store_true")
    parser.add_argument("--no-pitch", action="store_true",
                        help="run detection only")


def run(args: argparse.Namespace) -> int:
    paths = Paths()

    video = Path(args.video) if args.video else None
    if video is None:
        candidates = sorted(paths.input_video.glob("*.mp4")) if paths.input_video.exists() else []
        if not candidates:
            print(f"error: no video given and none found in {paths.input_video}")
            return 1
        video = candidates[0]
    if not video.is_absolute():
        video = paths.root / video
    if not video.exists():
        print(f"error: video not found: {video}")
        return 1

    players_weights = _pick_weights(paths, args.players, DEFAULT_PLAYERS_RUN, "players")
    pitch_weights = None if args.no_pitch else _pick_weights(
        paths, args.pitch, DEFAULT_PITCH_RUN, "pitch")

    if players_weights is None and pitch_weights is None:
        print("error: no checkpoints found. Train a model first, or pass "
              "--players / --pitch explicitly.")
        return 1

    if players_weights is None:
        print("warning: no players checkpoint, position output will be empty")

    output = Path(args.output) if args.output else paths.analysis / "latest"
    if not output.is_absolute():
        output = paths.root / output

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
        start_frame=args.start_frame,
        frame_count=args.frame_count or None,
        imgsz=args.imgsz,
        chunk_size=args.chunk_size,
        scale=args.scale,
        use_openvino=not args.no_openvino,
    )

    overlay, position = analyzer.analyze(str(output))
    print(f"\noverlayed video: {overlay}")
    print(f"position video:  {position}")
    return 0
