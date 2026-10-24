"""Package runs and rendered video so they can be downloaded from a remote host.

    ennatzauge exfil                    # bundle everything worth keeping
    ennatzauge exfil --list             # show what would be included
    ennatzauge exfil --include-snapshots  # also epoch*.pt (large)
    ennatzauge exfil --part-size 500     # split into 500 MB parts

A rented GPU box has no persistent storage, and pulling thousands of small files
over SSH is slow and fragile. This writes one archive (optionally split into
parts) with checksums and a manifest, so the transfer is a single resumable
download that can be verified afterwards.

The datasets are excluded by default: they are reproducible with
``ennatzauge build`` and are by far the largest thing on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

from ..paths import Paths

READ_CHUNK = 8 * 1024 * 1024

# Never worth transferring: caches, and Ultralytics' intermediate artefacts.
EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".tmp", ".part", ".lock"}
# Written by Ultralytics during training; regenerable, and occasionally huge.
EXCLUDE_NAMES = {"labels.jpg", "labels_correlogram.jpg", "train_batch0.jpg"}


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(READ_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# collecting
# ---------------------------------------------------------------------------


def _iter_files(root: Path, include_snapshots: bool):
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in item.parts):
            continue
        if item.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        if item.name in EXCLUDE_NAMES:
            continue
        if item.suffix == ".pt" and item.name.startswith("epoch") and not include_snapshots:
            continue
        # Exported OpenVINO graphs are ~2x the checkpoint and regenerate in
        # seconds from it, so they are never worth transferring.
        if any("_openvino_model" in part for part in item.parts):
            continue
        yield item


def _arc(prefix: str, path: Path, base: Path) -> str:
    """Archive member name, always forward-slashed.

    On Windows ``relative_to`` yields backslashes, and tar stores them
    literally -- a Linux extract would produce files named ``weights\\best.pt``
    rather than a ``weights`` directory.
    """
    relative = path.relative_to(base).as_posix()
    return f"{prefix}/{relative}" if prefix else relative


def _run_summary(run_dir: Path) -> dict:
    """Light metadata, so the manifest is readable without the archive."""
    import csv

    summary = {"run_name": run_dir.name}
    results = run_dir / "results.csv"
    if results.exists():
        try:
            with results.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            summary["epochs"] = len(rows)
            values = []
            for row in rows:
                try:
                    values.append(float(row.get("metrics/mAP50-95(B)", 0)))
                except (TypeError, ValueError):
                    pass
            if values:
                summary["best_mAP_50_95"] = round(max(values), 4)
                summary["best_epoch"] = values.index(max(values)) + 1
        except Exception:
            pass

    weights = run_dir / "weights"
    for name in ("best.pt", "last.pt"):
        path = weights / name
        if path.exists():
            summary[f"{name}_mb"] = round(path.stat().st_size / 1e6, 1)
    return summary


def build_plan(paths: Paths, include_runs: bool, include_analysis: bool,
               include_snapshots: bool, include_configs: bool,
               include_datasets: bool) -> list[tuple[Path, str]]:
    """Return ``[(absolute_path, path_inside_archive)]``."""
    plan: list[tuple[Path, str]] = []

    if include_runs and paths.runs.exists():
        for run_dir in sorted(p for p in paths.runs.iterdir() if p.is_dir()):
            for item in _iter_files(run_dir, include_snapshots):
                plan.append((item, _arc(f"runs/{run_dir.name}", item, run_dir)))

    if include_analysis and paths.analysis.exists():
        for item in _iter_files(paths.analysis, include_snapshots):
            plan.append((item, _arc("analysis", item, paths.analysis)))

    if include_configs and paths.configs.exists():
        for item in _iter_files(paths.configs, True):
            plan.append((item, _arc("configs", item, paths.configs)))

    if include_datasets and paths.datasets.exists():
        for name in ("yolo-players", "yolo-pitch"):
            dataset = paths.datasets / name
            if not dataset.exists():
                continue
            # Labels and yaml only: images are reproducible and enormous.
            labels = dataset / "labels"
            if labels.exists():
                for item in _iter_files(labels, True):
                    plan.append((item, _arc(f"datasets/{name}/labels", item, labels)))
            for extra in ("data.yaml", "keypoints.txt"):
                path = dataset / extra
                if path.exists():
                    plan.append((path, f"datasets/{name}/{extra}"))

    return plan


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def write_archive(plan: list[tuple[Path, str]], manifest: dict,
                  out_path: Path) -> int:
    """Write a gzipped tar. Returns the number of files written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    manifest_bytes = json.dumps(manifest, indent=2).encode()

    with tarfile.open(out_path, "w:gz", compresslevel=6) as archive:
        for source, arcname in plan:
            archive.add(str(source), arcname=arcname, recursive=False)
            count += 1

        import io

        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        archive.addfile(info, io.BytesIO(manifest_bytes))
        count += 1

    return count


def split_file(path: Path, part_bytes: int) -> list[Path]:
    """Split a file into parts, so a flaky download only costs one part."""
    parts = []
    index = 0
    with open(path, "rb") as source:
        while True:
            block = source.read(part_bytes)
            if not block:
                break
            part = path.with_suffix(path.suffix + f".{index:03d}")
            part.write_bytes(block)
            parts.append(part)
            index += 1
    if parts:
        path.unlink()  # the parts are the deliverable
    return parts


def write_checksums(files: list[Path], out_path: Path) -> None:
    lines = [f"{sha256_file(p)}  {p.name}" for p in files]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# command
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="destination directory or .tar.gz path "
                                         "(default: <root>/../ennatzauge-exfil)")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="print what would be included and exit")
    parser.add_argument("--include-snapshots", action="store_true",
                        help="include epoch*.pt checkpoints (large)")
    parser.add_argument("--no-runs", action="store_true",
                        help="exclude training runs")
    parser.add_argument("--no-analysis", action="store_true",
                        help="exclude rendered video")
    parser.add_argument("--no-configs", action="store_true",
                        help="exclude configs")
    parser.add_argument("--include-datasets", action="store_true",
                        help="include dataset labels and yaml (not images)")
    parser.add_argument("--part-size", type=int, default=0, metavar="MB",
                        help="split the archive into N MB parts")
    parser.add_argument("--keep-archive", action="store_true",
                        help="keep the unsplit archive alongside the parts")


def run(args: argparse.Namespace) -> int:
    paths = Paths()

    plan = build_plan(
        paths,
        include_runs=not args.no_runs,
        include_analysis=not args.no_analysis,
        include_snapshots=args.include_snapshots,
        include_configs=not args.no_configs,
        include_datasets=args.include_datasets,
    )

    if not plan:
        print("nothing to package.")
        print(f"  runs     {paths.runs}")
        print(f"  analysis {paths.analysis}")
        print("\ntrain or analyse something first.")
        return 1

    total = sum(p.stat().st_size for p, _ in plan)
    runs = sorted({arc.split("/")[1] for _, arc in plan if arc.startswith("runs/")})
    analyses = sorted({arc.split("/")[1] for _, arc in plan
                       if arc.startswith("analysis/")})

    print(f"root     {paths.root}")
    print(f"files    {len(plan)}")
    print(f"size     {_human(total)}")
    if runs:
        print(f"runs     {', '.join(runs)}")
    if analyses:
        print(f"analysis {', '.join(analyses[:5])}"
              + ("..." if len(analyses) > 5 else ""))

    if args.list_only:
        print()
        width = max(len(arc) for _, arc in plan)
        for source, arcname in plan:
            print(f"  {arcname:<{width}}  {_human(source.stat().st_size):>10}")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.output:
        output = Path(args.output).expanduser()
    else:
        output = paths.root.parent / "ennatzauge-exfil"

    if output.suffix in (".gz", ".tgz", ".tar"):
        archive_path = output
        output_dir = output.parent
    else:
        output_dir = output
        archive_path = output_dir / f"ennatzauge-{stamp}.tar.gz"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": str(paths.root),
        "file_count": len(plan),
        "source_bytes": total,
        "includes_snapshots": bool(args.include_snapshots),
        "runs": [_run_summary(paths.run(name)) for name in runs],
        "analysis": analyses,
        "archive_sha256": None,   # filled in after writing
    }

    print(f"\nwriting  {archive_path}")
    count = write_archive(plan, manifest, archive_path)
    size = archive_path.stat().st_size
    print(f"wrote    {count} entries, {_human(size)} raw"
          f" ({_human(total - size)} smaller than the source)")

    deliverables: list[Path] = []
    archive_sha = sha256_file(archive_path)
    manifest["archive_sha256"] = archive_sha

    if args.part_size and args.part_size > 0:
        part_bytes = int(args.part_size) * 1024 * 1024
        deliverables = split_file(archive_path, part_bytes)
        print(f"split    {len(deliverables)} part(s) of {args.part_size} MB")
        if args.keep_archive:
            print("         (archive kept alongside the parts)")
    else:
        deliverables = [archive_path]

    # One checksum line per downloadable file, so a partial or corrupted
    # download is caught before extraction.
    checksum_path = output_dir / f"{archive_path.name}.sha256"
    write_checksums(deliverables, checksum_path)
    print(f"checksum {checksum_path.name} ({len(deliverables)} file(s))")

    manifest_path = output_dir / f"ennatzauge-{stamp}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest {manifest_path.name}")

    print("\nnext:")
    print(f"  download  {output_dir}")
    if len(deliverables) > 1:
        print(f"  verify    sha256sum -c {checksum_path.name}")
        print(f"  join      cat {archive_path.name}.* > {archive_path.name}")
        print(f"  check     echo '{archive_sha}' | sha256sum -c -")
    else:
        print(f"  verify    sha256sum -c {checksum_path.name}")
    print(f"  extract   tar -xzf {archive_path.name}")
    print("\n  runs land under runs/, video under analysis/, configs under configs/.")
    return 0
