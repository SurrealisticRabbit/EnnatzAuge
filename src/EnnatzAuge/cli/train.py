"""Train models locally, from a config file or explicit arguments.

    ennatzauge train                          # run every job in configs/train.yaml
    ennatzauge train --job players            # one of them
    ennatzauge train --epochs 30 --imgsz 960  # override without editing the file
    ennatzauge train --dry-run                # show what would run

Re-running is resumable: each run directory keeps last.pt, and Ultralytics
continues from it. ``epochs`` is always the TOTAL for the run, never a delta.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from ..config import ConfigContext, filter_fields
from ..paths import Paths
from ..train.jobs import Job
from ..train.manager import Manager, describe_devices, visible_gpu_count

# Keys a caller may override from the command line.
OVERRIDABLE = {
    "epochs": int, "imgsz": int, "batch": float, "gpus": int,
    "patience": int, "workers": int, "save_period": int,
    "dataset_skip_interval": int, "val_stride": int, "val_every": int,
    "mosaic": float, "close_mosaic": int, "fliplr": float, "lr0": float,
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="train",
                        help="config name under configs/ (default: train)")
    parser.add_argument("--job", help="only run the job with this name")
    parser.add_argument("--max-gpus", type=int, default=None,
                        help="GPU slots to schedule against (default: detected)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved jobs and exit")
    parser.add_argument("--model",
                        help="override the starting weights for every job")
    parser.add_argument("--no-chart", action="store_true",
                        help="no live metric chart (useful for logs and CI)")
    for key, kind in OVERRIDABLE.items():
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key, type=kind,
                            help=f"override {key}")


def _jobs_from_args(paths: Paths, args: argparse.Namespace) -> list[Job]:
    context = ConfigContext(paths)
    try:
        raw_jobs = context.train_config(args.config)
    except FileNotFoundError as exc:
        if not args.job:
            print(f"error: {exc}")
            print(f"       create one at {paths.config(args.config)}")
            return []
        raw_jobs = []

    if args.job and raw_jobs:
        raw_jobs = [j for j in raw_jobs if j.get("name") == args.job]
        if not raw_jobs:
            print(f"error: no job named {args.job!r} in the config")
            return []

    if not raw_jobs:
        # Nothing configured: build one from the arguments themselves.
        raw_jobs = [{"name": args.job or "train"}]

    overrides = {k: getattr(args, k) for k in OVERRIDABLE
                 if getattr(args, k, None) is not None}
    if args.model:
        overrides["model"] = args.model
    if args.no_chart:
        overrides["live_chart"] = False

    jobs = []
    for spec in raw_jobs:
        spec = {**spec, **overrides}
        if "data" not in spec:
            # Guess from the job name: anything mentioning pitch uses the pose
            # dataset, everything else the detection one.
            name = str(spec.get("name", "players")).lower()
            dataset = "yolo-pitch" if "pitch" in name else "yolo-players"
            spec["data"] = str(paths.dataset(dataset) / "data.yaml")
        spec.setdefault("project", str(paths.runs))
        jobs.append(Job(**filter_fields(Job, spec)))
    return jobs


def run(args: argparse.Namespace) -> int:
    paths = Paths()
    paths.ensure()

    detected = visible_gpu_count()
    print(f"root     {paths.root}")
    print(f"gpus     {detected if detected else 'none (cpu)'}")
    for line in describe_devices():
        print(f"           {line}")

    jobs = _jobs_from_args(paths, args)
    if not jobs:
        return 1

    missing = [j for j in jobs if not Path(j.data).exists()]
    if missing:
        for job in missing:
            print(f"error: dataset yaml not found: {job.data}")
        print(f"       run 'ennatzauge build' first, or fix the path in "
              f"{paths.config(args.config)}")
        return 1

    print(f"\n{'job':<20}{'epochs':>7}{'imgsz':>7}{'batch':>7}{'gpus':>6}  data")
    for job in jobs:
        print(f"{job.name:<20}{job.epochs:>7}{job.imgsz:>7}"
              f"{str(job.batch):>7}{job.gpus:>6}  {job.data}")

    if args.dry_run:
        print("\ndry run, nothing started")
        return 0

    manager = Manager(max_gpus=args.max_gpus or max(1, detected),
                      project=paths.runs)
    for job in jobs:
        manager.add(job)

    print()
    manager.run()
    print()
    print(manager.summary())
    return 0 if all(j.state.value in ("done", "stopped") for j in manager.jobs) else 2
