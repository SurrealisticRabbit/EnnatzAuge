"""Check that this machine can actually run EnnatzAuge, and fix what is safe to fix.

    ennatzauge doctor                 # check, and create missing dirs/configs
    ennatzauge doctor --install       # also pip-install missing packages
    ennatzauge doctor --json          # machine-readable report

Exit code is 0 when the machine can train and analyse, 1 when something
critical is wrong. Problems that cannot be fixed safely (a CPU-only PyTorch on
a GPU host) are reported with the exact command rather than attempted.

The GPU section suggests settings rather than applying them, because the right
batch size depends on the model and resolution you are about to train -- but it
does set the two process-wide flags that are unambiguous wins (TF32 and cuDNN
autotuning) so a subsequent ``train`` in the same process benefits.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from ..paths import Paths

REQUIRED = {
    "ultralytics": "ultralytics",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "huggingface_hub": "huggingface_hub",
}

MIN_PYTHON = (3, 10)
# Below this there is not enough room for checkpoints plus rendered video.
MIN_FREE_GB = 10.0

NETWORK_FILESYSTEMS = {
    "nfs", "nfs4", "cifs", "smb3", "fuse.sshfs", "fuse.rclone",
    "9p", "vboxsf", "fuse.s3fs", "fuse.gcsfuse", "fuse",
}


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.fixed: list[str] = []

    def add(self, name: str, status: str, detail: str = "", fix: str = "") -> None:
        """status: ok, warn, fail, fixed, skip"""
        self.rows.append({"check": name, "status": status,
                          "detail": detail, "fix": fix})

    @property
    def critical(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "fail"]

    def print(self) -> None:
        icons = {"ok": "ok  ", "warn": "warn", "fail": "FAIL",
                 "fixed": "fix ", "skip": "skip"}
        width = max((len(r["check"]) for r in self.rows), default=10)
        for row in self.rows:
            line = f"  [{icons.get(row['status'], '?')}] {row['check']:<{width}}"
            if row["detail"]:
                line += f"  {row['detail']}"
            print(line)
            if row["fix"]:
                print(f"         -> {row['fix']}")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_platform(report: Report) -> None:
    report.add("platform", "ok", f"{platform.system()} {platform.release()}")
    report.add("cpu", "ok", f"{os.cpu_count()} threads")


def check_python(report: Report) -> None:
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if version[:2] < MIN_PYTHON:
        report.add("python", "fail", text,
                   f"needs >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    else:
        report.add("python", "ok", text)


def check_packages(report: Report, install: bool) -> None:
    missing = []
    for module, package in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if not missing:
        report.add("packages", "ok", f"{len(REQUIRED)} present")
        return

    if not install:
        report.add("packages", "warn", f"missing {', '.join(missing)}",
                   "re-run with --install")
        return

    print(f"  installing {', '.join(missing)} ...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *missing],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        report.add("packages", "fixed", f"installed {', '.join(missing)}")
        report.fixed.append(f"pip install {' '.join(missing)}")
    else:
        tail = (result.stderr or "").strip().splitlines()[-1:] or [""]
        report.add("packages", "fail", f"install failed: {tail[0]}",
                   "check network access, or install manually")


def check_directories(report: Report, paths: Paths) -> None:
    required = {
        "configs": paths.root / "configs",
        "datasets": paths.root / "datasets",
        "input_video": paths.root / "input_video",
        "models/base": paths.root / "models" / "base",
        "models/runs": paths.root / "models" / "runs",
        "models/analysis": paths.root / "models" / "analysis",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if not missing:
        report.add("directories", "ok", str(paths.root))
        return

    paths.ensure()
    report.add("directories", "fixed",
               f"created {', '.join(missing)} under {paths.root}")
    report.fixed.append(f"created {len(missing)} directories")


DEFAULT_TRAIN = """\
# ennatzauge train -- paths are relative to the EnnatzAuge root
jobs:
  - name: players
    run_name: players
    data: datasets/yolo-players/data.yaml
    model: yolo26m.pt
    epochs: 80
    imgsz: 1280
    batch: -1
    dataset_skip_interval: 5
    val_stride: 10
    val_every: 1
"""

DEFAULT_ANALYZE = """\
# ennatzauge analyze
frames: 0
start: 0
imgsz: 1280
chunk: 16
scale: 10
openvino: true
"""


def check_configs(report: Report, paths: Paths) -> None:
    created = []
    for name, template in (("train", DEFAULT_TRAIN), ("analyze", DEFAULT_ANALYZE)):
        target = paths.config(name)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(template, encoding="utf-8")
        created.append(name)

    if created:
        report.add("configs", "fixed", f"wrote {', '.join(created)}")
        report.fixed.append(f"wrote configs: {', '.join(created)}")
    else:
        report.add("configs", "ok", "train.yaml, analyze.yaml")


def check_datasets(report: Report, paths: Paths) -> None:
    found = []
    for name in ("yolo-players", "yolo-pitch"):
        yaml_path = paths.dataset(name) / "data.yaml"
        if not yaml_path.exists():
            continue
        try:
            import yaml

            spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            report.add(f"dataset {name}", "fail", f"unreadable yaml: {exc}")
            continue

        base = yaml_path.parent
        problems = [key for key in ("train", "val")
                    if not (base / str(spec.get(key, ""))).exists()]
        if problems:
            report.add(f"dataset {name}", "warn",
                       f"missing {problems} under {base}",
                       "rebuild with: ennatzauge build")
        else:
            found.append(name)
            report.add(f"dataset {name}", "ok", str(yaml_path))

    if not found:
        report.add("datasets", "warn", "neither dataset is built",
                   "ennatzauge build --stride 5")


def check_models(report: Report, paths: Paths) -> None:
    runs = []
    if paths.runs.exists():
        runs = sorted(p.name for p in paths.runs.iterdir()
                      if (p / "weights" / "best.pt").exists())
    if runs:
        listed = ", ".join(runs[:3]) + ("..." if len(runs) > 3 else "")
        report.add("checkpoints", "ok", f"{len(runs)} run(s): {listed}")
    else:
        report.add("checkpoints", "warn", "no trained weights yet",
                   "train one with: ennatzauge train")

    base = sorted(paths.base_models.glob("*.pt")) if paths.base_models.exists() else []
    if base:
        report.add("base weights", "ok", ", ".join(p.name for p in base[:4]))
    else:
        report.add("base weights", "skip",
                   "none cached; Ultralytics downloads on first use")


def check_disk(report: Report, paths: Paths) -> None:
    try:
        usage = shutil.disk_usage(paths.root)
    except OSError as exc:
        report.add("disk", "fail", str(exc))
        return

    free_gb = usage.free / 1e9
    detail = f"{free_gb:.1f} GB free of {usage.total / 1e9:.0f} GB"
    if free_gb < MIN_FREE_GB:
        report.add("disk", "fail", detail,
                   f"needs {MIN_FREE_GB:.0f} GB for checkpoints and renders")
    elif free_gb < 30:
        report.add("disk", "warn", detail,
                   "tight for two runs plus a dataset build")
    else:
        report.add("disk", "ok", detail)


def check_filesystem(report: Report, paths: Paths) -> None:
    """Warn if the datasets are on a network mount.

    Reading JPEGs over a network filesystem is the biggest cause of GPUs
    idling mid-epoch: the GPU finishes a batch in ~250 ms and then waits.
    """
    if not paths.datasets.exists():
        report.add("dataset storage", "skip", "datasets directory absent")
        return

    mount_type = None
    try:
        if Path("/proc/mounts").exists():
            target = str(paths.datasets.resolve())
            best = ""
            for line in Path("/proc/mounts").read_text().splitlines():
                parts = line.split()
                if len(parts) >= 3 and target.startswith(parts[1]) \
                        and len(parts[1]) > len(best):
                    best = parts[1]
                    mount_type = parts[2]
    except Exception:
        pass

    if mount_type and mount_type.split(".")[0] in NETWORK_FILESYSTEMS:
        report.add("dataset storage", "warn", f"network mount ({mount_type})",
                   "copy the dataset to local disk before training")
    elif mount_type:
        report.add("dataset storage", "ok", f"{mount_type} at {paths.datasets}")
    else:
        report.add("dataset storage", "skip", "mount type not detectable")


def check_openvino(report: Report) -> None:
    try:
        import openvino

        report.add("openvino", "ok", getattr(openvino, "__version__", "present"))
    except ImportError:
        report.add("openvino", "skip", "not installed",
                   'pip install -e ".[cpu]" for much faster CPU inference '
                   '(only used when there is no GPU)')


def check_analysis_backend(report: Report) -> None:
    """What `ennatzauge analyze` will actually run on.

    Worth checking explicitly: the pipeline defaults to whatever is fastest, and
    a missing CUDA build silently means two hours of CPU inference instead of
    two minutes of GPU.
    """
    try:
        from ..core.models import hardware

        backend = hardware.select()
    except Exception as exc:                      # pragma: no cover - defensive
        report.add("analysis", "warn", f"could not select a backend: {exc}")
        return

    detail = (f"{backend.describe()}, batch {backend.max_batch}")
    if backend.format == "torch" and backend.torch_device.startswith("cuda"):
        report.add("analysis", "ok", detail)
    elif backend.format == "openvino":
        report.add("analysis", "ok", detail,
                   "GPU would be much faster here; install a CUDA torch build "
                   "to use one")
    else:
        report.add("analysis", "ok", detail,
                   "slowest option: no CUDA and no OpenVINO available")


def check_gpu(report: Report) -> dict:
    """Inspect CUDA and set the flags that are unambiguous wins."""
    info: dict = {"available": False, "count": 0, "devices": []}

    try:
        import torch
    except ImportError:
        report.add("pytorch", "fail", "torch not importable",
                   "pip install torch")
        return info

    cuda_build = torch.version.cuda
    report.add("pytorch", "ok", f"{torch.__version__} (CUDA {cuda_build or 'none'})")

    info["available"] = bool(torch.cuda.is_available())
    if not info["available"]:
        if cuda_build is None:
            report.add("cuda", "warn", "CPU-only PyTorch build",
                       "on a GPU host: pip install torch --index-url "
                       "https://download.pytorch.org/whl/cu124")
        else:
            report.add("cuda", "warn", "CUDA build present, no device visible",
                       "check the driver and CUDA_VISIBLE_DEVICES")
        report.add("gpu", "skip", "training would run on CPU")
        return info

    count = torch.cuda.device_count()
    info["count"] = count
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        info["devices"].append({
            "index": index,
            "name": props.name,
            "vram_gb": round(props.total_memory / 1e9, 1),
            "capability": f"{props.major}.{props.minor}",
        })

    report.add("gpu", "ok", "; ".join(
        f"{d['index']}: {d['name']} {d['vram_gb']}GB" for d in info["devices"]))

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        info["tf32"] = True
        report.add("tf32", "fixed", "enabled for matmul and cuDNN")
    except Exception as exc:
        report.add("tf32", "warn", str(exc))

    check_topology(report, count)
    check_gpu_headroom(report, torch)
    return info


def check_topology(report: Report, count: int) -> None:
    """For multi-GPU hosts, confirm DDP will not cross the network."""
    if count < 2:
        return
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        report.add("gpu topology", "skip", "nvidia-smi not on PATH")
        return
    try:
        result = subprocess.run([nvidia_smi, "topo", "-m"], capture_output=True,
                                text=True, timeout=20)
        text = result.stdout or ""
    except Exception as exc:
        report.add("gpu topology", "skip", f"could not query: {exc}")
        return

    if "NV" in text:
        report.add("gpu topology", "ok", "NVLink between devices")
    elif "SYS" in text or "NODE" in text:
        report.add("gpu topology", "warn", "devices appear to span hosts",
                   "do not use DDP across hosts; run one job per GPU instead")
    else:
        report.add("gpu topology", "ok", "PCIe-connected, single host")


def check_gpu_headroom(report: Report, torch) -> None:
    try:
        free, total = torch.cuda.mem_get_info(0)
    except Exception:
        return
    used_gb = (total - free) / 1e9
    if used_gb > 1.0:
        report.add("gpu memory", "warn",
                   f"{used_gb:.1f} GB of {total / 1e9:.1f} GB already in use",
                   "another process may be holding the GPU")
    else:
        report.add("gpu memory", "ok", f"{total / 1e9:.1f} GB free")


def recommend(report: Report, gpu: dict) -> None:
    """Turn the detected hardware into concrete training settings."""
    cpu_count = os.cpu_count() or 4

    if gpu["count"]:
        vram = max(d["vram_gb"] for d in gpu["devices"])
        # Measured for yolo26m at imgsz=1280: batch 8 needs ~21 GB, 16 needs
        # ~42 GB. Under 16 GB must drop the batch or the resolution.
        if vram >= 40:
            batch = "8-16 at imgsz=1280"
        elif vram >= 20:
            batch = "8 at imgsz=1280"
        elif vram >= 14:
            batch = "4 at imgsz=1280, or 8 at imgsz=960"
        else:
            batch = "imgsz=960 or lower"
        report.add("recommend batch", "ok", batch)

        # Leave a core for the parent process; Ultralytics opens 4 per worker.
        workers = max(2, min(8, cpu_count // max(1, gpu["count"])))
        report.add("recommend workers", "ok", str(workers))
        report.add("recommend amp", "ok", "leave amp=True (default, ~2x faster)")
        if gpu["count"] > 1:
            report.add("recommend gpus", "ok",
                       f"{gpu['count']} visible; DDP gives ~1.8x for 2x cost, "
                       f"independent jobs usually win")
    else:
        report.add("recommend batch", "ok", "4 on CPU")
        report.add("recommend workers", "ok", "2")


# ---------------------------------------------------------------------------
# command
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--install", action="store_true",
                        help="pip-install missing packages")
    parser.add_argument("--json", action="store_true",
                        help="print the report as JSON")
    parser.add_argument("--no-fix", action="store_true",
                        help="report only; do not create directories or configs")


def run(args: argparse.Namespace) -> int:
    paths = Paths()
    report = Report()

    if not args.json:
        print("EnnatzAuge doctor")
        print(f"root  {paths.root}\n")

    check_platform(report)
    check_python(report)
    check_packages(report, args.install)

    if args.no_fix:
        for name, path in (("configs", paths.root / "configs"),
                           ("datasets", paths.root / "datasets"),
                           ("models/runs", paths.root / "models" / "runs")):
            exists = path.exists()
            report.add(name, "ok" if exists else "warn",
                       str(path) if exists else "missing")
    else:
        check_directories(report, paths)
        check_configs(report, paths)

    check_disk(report, paths)
    check_datasets(report, paths)
    check_models(report, paths)
    check_filesystem(report, paths)
    gpu = check_gpu(report)
    check_openvino(report)
    check_analysis_backend(report)
    recommend(report, gpu)

    if args.json:
        print(json.dumps({
            "root": str(paths.root),
            "ready": not report.critical,
            "checks": report.rows,
            "fixed": report.fixed,
            "gpu": gpu,
        }, indent=2))
    else:
        report.print()
        print()
        if report.fixed:
            print(f"applied {len(report.fixed)} fix(es):")
            for item in report.fixed:
                print(f"  - {item}")
            print()
        if report.critical:
            print(f"{len(report.critical)} blocking problem(s); see the -> hints")
        else:
            print("ready to train and analyse.")

    return 1 if report.critical else 0


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="EnnatzAuge environment check")
    add_arguments(_parser)
    raise SystemExit(run(_parser.parse_args()))
