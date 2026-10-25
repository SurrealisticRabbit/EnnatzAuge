"""Command-line entry point.

    ennatzauge paths                 # where everything lives
    ennatzauge doctor                # check this machine can run everything
    ennatzauge build --stride 5      # build the datasets
    ennatzauge train                 # train from configs/train.yaml
    ennatzauge analyze --video x.mp4 # run inference and render
    ennatzauge status                # what runs exist
    ennatzauge stop <run-name>       # stop a run after its current epoch
    ennatzauge exfil                 # package runs and video to download
"""

from __future__ import annotations

import argparse
import sys

# Executed as a plain script, the package root is not on sys.path and the
# relative imports below have no parent. Add the repo root and adopt the
# package identity instead.
if __package__ in (None, ""):
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    __package__ = "src.EnnatzAuge.cli"

from ..paths import Paths
from ..train.manager import clear_stop, list_runs, request_stop
from . import analyze as analyze_cmd
from . import build as build_cmd
from . import doctor as doctor_cmd
from . import exfil as exfil_cmd
from . import train as train_cmd

COMMANDS = {
    "analyze": (analyze_cmd, "run models over a clip and render output"),
    "build": (build_cmd, "build the YOLO datasets from SN-GSR-2025"),
    "doctor": (doctor_cmd, "check the environment and fix what is safe"),
    "exfil": (exfil_cmd, "package runs and video for download"),
    "train": (train_cmd, "train models locally"),
}


def cmd_paths(args: argparse.Namespace) -> int:
    paths = Paths()
    print(paths.describe())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    paths = Paths()
    runs = list_runs(paths.runs)
    if not runs:
        print(f"no runs under {paths.runs}")
        return 0

    print(f"{'run':<22}{'task':<7}{'epochs':>7}{'best ep':>8}"
          f"{'mAP50-95':>10}{'fitness':>9}{'last.pt':>9}  notes")
    for run in runs:
        task = "pose" if run["is_pose"] else "detect"
        notes = [] if run["has_best"] else ["no best.pt"]
        if run["stopped"]:
            notes.append("STOP requested")
        if run["last_pt_mb"] == 0:
            notes.append("no last.pt")
        print(f"{run['run_name']:<22}{task:<7}{run['epochs_done']:>7}"
              f"{run['best_epoch']:>8}{run['best_mAP_50_95']:>10.4f}"
              f"{run['best_fitness']:>9.4f}{run['last_pt_mb']:>9.1f}"
              f"  {', '.join(notes)}")

    if any(run["is_pose"] for run in runs):
        print("\npose runs: mAP50-95 is the keypoint metric. The box metric is "
              "not shown —\nit scores only the whole-frame box and reads ~0.97 "
              "regardless of keypoint quality.")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    paths = Paths()
    if args.clear:
        print(clear_stop(paths.runs, args.run_name))
    else:
        print(request_stop(paths.runs, args.run_name))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ennatzauge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("paths", help="show where data and models live")
    sub.add_parser("status", help="list training runs")

    stop = sub.add_parser("stop", help="stop a run after its current epoch")
    stop.add_argument("run_name")
    stop.add_argument("--clear", action="store_true", help="cancel a stop request")

    for name, (module, help_text) in COMMANDS.items():
        command = sub.add_parser(name, help=help_text)
        module.add_arguments(command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "paths":
        return cmd_paths(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "stop":
        return cmd_stop(args)

    module, _ = COMMANDS[args.command]
    return module.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
