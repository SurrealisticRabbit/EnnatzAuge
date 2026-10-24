"""Build the YOLO datasets from SoccerNet SN-GSR-2025.

    ennatzauge build --stride 5              # both datasets, all labelled splits
    ennatzauge build --datasets players      # detection only
    ennatzauge build --keep-archives         # leave the zips on disk

One archive is fetched, converted and (by default) deleted before the next, so
peak disk is one ~10 GB zip plus the growing output rather than 70 GB.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from ..data.convert import Converter
from ..data.download import LABELLED_SPLITS, SPLITS, download_split
from ..paths import Paths


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", default="players,pitch",
                        help="comma separated: players, pitch")
    parser.add_argument("--splits", default=",".join(LABELLED_SPLITS),
                        help=f"comma separated (challenge has no labels). "
                             f"Default: {','.join(LABELLED_SPLITS)}")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth frame (5 = 20%%, ~11 GB total)")
    parser.add_argument("--pitch-images", default="link",
                        choices=["link", "copy", "none"],
                        help="how pitch images are provided (default: link)")
    parser.add_argument("--keep-archives", action="store_true",
                        help="do not delete each zip after converting it")
    parser.add_argument("--raw-dir", help="where archives are cached")


def run(args: argparse.Namespace) -> int:
    paths = Paths()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in ("players", "pitch")]
    if unknown:
        print(f"error: unknown dataset(s) {unknown}; choose players, pitch")
        return 1

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    dropped = [s for s in splits if s not in LABELLED_SPLITS]
    if dropped:
        print(f"note: {dropped} has no ground truth, skipping")
        splits = [s for s in splits if s in LABELLED_SPLITS]
    if not splits:
        print("error: no labelled splits selected")
        return 1

    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else paths.datasets / "raw"
    converter = Converter(paths.datasets, datasets=datasets, stride=args.stride,
                          splits=splits, pitch_images=args.pitch_images)

    print(f"root     {paths.root}")
    print(f"output   {paths.datasets}")
    print(f"archives {raw_dir}")
    print(f"stride   {args.stride}   splits {splits}\n")

    started = time.time()
    totals = {"players": {}, "pitch": {}}

    for split in splits:
        print(f"=== {split} ===", flush=True)
        archive = download_split(split, raw_dir)
        counts, stats = converter.convert_split(archive, split)
        for name in totals:
            totals[name].update(counts[name])
        print(f"  done: players={counts['players'][split]} "
              f"pitch={counts['pitch'][split]} "
              f"(skipped {stats['skipped_pitch']} pitch frames, "
              f"{stats['missing_images']} missing images)", flush=True)

        if not args.keep_archives:
            free = shutil.disk_usage(raw_dir).free / 1e9
            archive.unlink()
            print(f"  removed {archive.name} ({free:.1f} GB free)", flush=True)

    converter.finish()

    print(f"\nfinished in {(time.time() - started) / 60:.1f} min")
    for name, values in totals.items():
        if values:
            print(f"  {name}: {values} total={sum(values.values())}")
    for name in datasets:
        print(f"  {paths.datasets / f'yolo-{name}' / 'data.yaml'}")
    return 0
