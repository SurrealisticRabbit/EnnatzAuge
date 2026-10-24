"""EnnatzAuge: football video analysis and YOLO training.

    import EnnatzAuge

    paths = EnnatzAuge.Paths()
    EnnatzAuge.build_datasets(paths, stride=5)
    EnnatzAuge.train(paths)
    EnnatzAuge.analyze(paths, video="clip.mp4")

The command line is the usual entry point: ``ennatzauge --help``.
"""

from __future__ import annotations

import argparse

__version__ = "0.2.0"

from .config import ConfigContext
from .paths import Paths


def build_datasets(paths: Paths | None = None, datasets=("players", "pitch"),
                   stride: int = 1, splits=("train", "valid", "test"),
                   pitch_images: str = "link", keep_archives: bool = False):
    """Download and convert SN-GSR-2025 into the two YOLO datasets."""
    from .cli.build import run as _run_build

    args = argparse.Namespace(
        datasets=",".join(datasets), splits=",".join(splits), stride=stride,
        pitch_images=pitch_images, keep_archives=keep_archives, raw_dir=None,
    )
    return _run_build(args)


def train(paths: Paths | None = None, config: str = "train",
          max_gpus: int | None = None, job: str | None = None,
          model: str | None = None, dry_run: bool = False, **overrides):
    """Run the jobs in ``configs/<config>.yaml``."""
    from .cli.train import run as _run_train

    values = {key: overrides.get(key) for key in (
        "epochs", "imgsz", "batch", "gpus", "patience", "workers",
        "save_period", "dataset_skip_interval", "val_stride", "val_every",
        "mosaic", "close_mosaic", "fliplr",
    )}
    args = argparse.Namespace(config=config, job=job, max_gpus=max_gpus,
                              dry_run=dry_run, model=model, **values)
    return _run_train(args)


def analyze(paths: Paths | None = None, video: str | None = None,
            output: str | None = None, **options):
    """Render the overlay and virtual-pitch videos for a clip."""
    from .cli.analyze import add_arguments, run as _run_analyze

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    if video is not None:
        args.video = video
    if output is not None:
        args.output = output
    for key, value in options.items():
        setattr(args, key, value)
    return _run_analyze(args)


__all__ = [
    "__version__",
    "ConfigContext",
    "Paths",
    "analyze",
    "build_datasets",
    "train",
]