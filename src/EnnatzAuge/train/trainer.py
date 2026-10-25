"""Runs one training job locally, on whatever GPUs the host has.

This is the portable half of the old Modal manager: the same decisions about
batch sizing, frame striding, validation frequency and checkpointing, without
any dependency on a cloud SDK.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def read_history(results_csv: Path) -> list[dict]:
    """Parse Ultralytics' results.csv into per-epoch rows.

    Handles both detect and pose CSVs. For pose the interesting number is the
    *pose* metric: ``mAP50-95(B)`` only scores the whole-frame bounding box,
    which is trivially easy and reads ~0.97 while the keypoints are still poor.
    ``fitness`` reproduces Ultralytics' own score so "best epoch" matches
    ``best.pt``: 0.1*mAP50 + 0.9*mAP50-95, summed over box and pose for pose
    tasks (see ``PoseMetrics.fitness``).

    Also reports the loss columns each task actually has: detect uses
    box/cls/dfl, pose uses box/cls/dfl plus pose/kobj, and segment uses seg.
    """
    path = Path(results_csv)
    if not path.exists():
        return []

    def num(row: dict, key: str) -> float:
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return 0.0

    def has(row: dict, key: str) -> bool:
        return key in row and row[key] not in ("", None)

    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            try:
                epoch = int(float(row["epoch"]))
            except (KeyError, ValueError):
                continue

            is_pose = has(row, "metrics/mAP50-95(P)")
            box_50 = num(row, "metrics/mAP50(B)")
            box_95 = num(row, "metrics/mAP50-95(B)")

            if is_pose:
                pose_50 = num(row, "metrics/mAP50(P)")
                pose_95 = num(row, "metrics/mAP50-95(P)")
                # Headline numbers follow the task, not the box.
                headline_50, headline_95 = pose_50, pose_95
                fitness = (0.1 * pose_50 + 0.9 * pose_95
                           + 0.1 * box_50 + 0.9 * box_95)
            else:
                pose_50 = pose_95 = None
                headline_50, headline_95 = box_50, box_95
                fitness = 0.1 * box_50 + 0.9 * box_95

            losses = [k for k in ("box", "cls", "dfl", "pose", "kobj", "seg")
                      if has(row, f"train/{k}_loss")]

            rows.append({
                "epoch": epoch,
                "mAP_50": headline_50,
                "mAP_50_95": headline_95,
                "box_mAP_50": box_50,
                "box_mAP_50_95": box_95,
                "pose_mAP_50": pose_50,
                "pose_mAP_50_95": pose_95,
                "fitness": fitness,
                "loss": sum(num(row, f"train/{k}_loss") for k in losses),
                "lr": num(row, "lr/pg0"),
            })
    return rows


def resolve_batch(model, imgsz: int, spec: float, devices) -> tuple[int, int]:
    """Return ``(total_batch, per_gpu_batch)`` for the GPUs in use.

    ``args.batch`` is the TOTAL across all GPUs, so a per-GPU figure has to be
    multiplied back up for Ultralytics and divided back down for reporting.

    ``spec``:
      * ``>= 1``          explicit total, used as given
      * ``0 < spec < 1``  profile using that fraction of VRAM
      * ``<= 0``          profile using 60% of VRAM

    Ultralytics' own ``batch=-1`` is not usable under DDP: it only resolves when
    ``RANK == -1``, so worker ranks keep the literal -1.
    """
    import torch
    from ultralytics.utils.autobatch import check_train_batch_size

    world = len(devices) if isinstance(devices, (list, tuple)) else 1
    if spec and spec >= 1:
        total = int(spec)
        return total, max(1, total // world)

    fraction = spec if isinstance(spec, float) and 0 < spec < 1 else 0.6

    if not torch.cuda.is_available():
        return 4 * world, 4  # nothing to profile; conservative is fine

    primary = devices[0] if isinstance(devices, (list, tuple)) else devices
    try:
        # autobatch reads next(model.parameters()).device, so the module must
        # already be on the GPU or it silently returns the fallback.
        model.model.to(primary)
        per_gpu = check_train_batch_size(
            model=model.model, imgsz=imgsz, amp=True, batch=fraction
        )
    except Exception as exc:  # profiling is best-effort, never fatal
        print(f"  batch auto-size failed ({type(exc).__name__}: {exc}); using 8/GPU",
              flush=True)
        per_gpu = 8
    finally:
        # Profiling caches probe tensors. Leaving them resident shows up as
        # spurious OOM warnings during the first iterations.
        torch.cuda.empty_cache()

    per_gpu = max(1, min(int(per_gpu), 256))
    return per_gpu * world, per_gpu


def write_stride_list(images_dir: Path, stride: int, out_path: Path) -> int:
    """Write a file listing every Nth image of a split.

    Ultralytics accepts a file where a directory is expected, which is the only
    way to sample evenly: ``fraction`` takes a sorted prefix instead, so on this
    dataset it trains on the first N matches rather than a spread. At 25 fps
    neighbouring frames are near duplicates, so striding costs almost nothing.
    """
    files = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() == ".jpg")
    if not files:
        raise FileNotFoundError(f"no .jpg files under {images_dir}")
    keep = files[::max(1, int(stride))]
    Path(out_path).write_text("\n".join(str(p) for p in keep), encoding="utf-8")
    return len(keep)


def resolve_data_yaml(data_path: str, run_name: str,
                      train_stride: int = 1, val_stride: int = 1) -> str:
    """Rewrite a dataset yaml's ``path:`` to the dataset's real location.

    The yamls are generated on Windows and carry paths like
    ``D:/datasets/yolo-players``. On Linux that is not absolute, so Ultralytics
    resolves it against its datasets_dir and fails looking for
    ``/root/datasets/D:/datasets/...``. The yaml's own directory is the
    authority, so ``path`` is overwritten with it.

    With a stride, ``train``/``val`` become generated file lists so the subset
    spreads across all sequences.
    """
    import yaml

    src = Path(data_path)
    if not src.exists():
        raise FileNotFoundError(
            f"dataset yaml not found: {src} - check the datasets directory"
        )

    spec = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    base = src.parent.resolve()
    spec["path"] = str(base)

    problems = []
    for key in ("train", "val", "test"):
        split = spec.get(key)
        if not split:
            if key in ("train", "val"):
                problems.append(f"missing required key {key!r}")
            continue
        if not (base / split).exists():
            problems.append(f"{key} -> {base / split}")
    if problems:
        raise FileNotFoundError(f"{src}: dataset layout incomplete: {problems}")

    tmp = Path(tempfile.gettempdir())
    for key, stride in (("train", train_stride), ("val", val_stride)):
        if stride and int(stride) > 1:
            count = write_stride_list(base / spec[key], int(stride),
                                      tmp / f"{run_name}-{key}.txt")
            spec[key] = str(tmp / f"{run_name}-{key}.txt")
            print(f"  {key}: every {stride}th -> {count} images (all sequences)",
                  flush=True)

    out = tmp / f"{run_name}-data.yaml"
    out.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return str(out)


def epoch_control(trainer, val_every: int, stop_marker: str,
                  chart=None, results_csv: str | None = None) -> None:
    """Per-epoch hook: throttle validation, honour a stop request, redraw chart.

    Runs from ``on_train_epoch_end``, which Ultralytics fires *before* the
    ``if self.args.val`` check, so setting ``trainer.args.val`` here decides
    whether this epoch validates.

    Takes plain values rather than closing over objects: under DDP the trainer
    is pickled into a generated launch script, and a closure holding something
    unpicklable (a socket, an SDK client) breaks the launch. The ``chart``
    argument is a plain object holding only lists, so it pickles fine.
    """
    epoch = trainer.epoch + 1
    every = max(1, int(val_every))
    trainer.args.val = bool(epoch >= trainer.epochs or every == 1
                            or epoch % every == 0)

    if chart is not None and results_csv:
        try:
            rows = read_history(Path(results_csv))
            if rows:
                chart.add(rows[-1])
                chart.draw()
        except Exception:
            pass  # a broken chart must never stop training

    if stop_marker and Path(stop_marker).exists():
        print(f"  [stop] requested at epoch {epoch}; finishing this epoch", flush=True)
        trainer.stop = True


class CheckpointMirror:
    """Copies checkpoints somewhere durable on a timer.

    Rented GPUs have no persistent storage: when the host goes away the run
    directory goes with it. Mirroring every few minutes bounds that loss to the
    interval instead of the whole run.

    Deliberately not an Ultralytics callback -- callbacks are pickled with the
    trainer under DDP.
    """

    def __init__(self, source: Path, target: Path | str | None,
                 stop_marker: Path, interval: float = 300.0) -> None:
        self._source = Path(source)
        self._target = Path(target) if target else None
        self._stop_marker = Path(stop_marker)
        self._interval = max(30.0, float(interval))
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="mirror", daemon=True)

    def _copy(self) -> None:
        if self._target is None:
            return
        self._target.mkdir(parents=True, exist_ok=True)
        for item in self._source.rglob("*"):
            if not item.is_file():
                continue
            dest = self._target / item.relative_to(self._source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

    def _loop(self) -> None:
        while not self._halt.wait(self._interval):
            try:
                self._copy()
                print(f"  [mirror] synced at {time.strftime('%H:%M:%S')}", flush=True)
            except Exception as exc:  # never let housekeeping kill training
                print(f"  [mirror] {type(exc).__name__}: {exc}", flush=True)
            if self._stop_marker.exists():
                return

    def start(self) -> "CheckpointMirror":
        if self._target is not None:
            self._thread.start()
        return self

    def stop(self) -> None:
        self._halt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=30)
        try:
            self._copy()
        except Exception as exc:
            print(f"  [mirror] final sync failed: {exc}", flush=True)


# ---------------------------------------------------------------------------
# the run itself
# ---------------------------------------------------------------------------


def resolve_device(job) -> int | list[int] | str:
    """What to pass as ``device=``, falling back to CPU when there is no GPU.

    A job still asks for GPU indices; on a machine with none, Ultralytics
    raises ``Invalid CUDA 'device=0' requested`` rather than degrading. Falling
    back keeps the package runnable on a laptop or a CPU-only cloud host, which
    matters for smoke tests and for resuming a run somewhere without a GPU.
    """
    try:
        import torch

        available = torch.cuda.device_count()
    except Exception:
        available = 0

    if available == 0:
        return "cpu"

    devices = [d for d in job.resolved_devices() if d < available]
    if not devices:
        devices = list(range(min(job.gpus, available)))
    return devices[0] if len(devices) == 1 else devices


def run_job(job, stop_marker: Path | None = None) -> dict:
    """Train or resume one job for its full epoch budget.

    Returns a dict with the run name, devices, batch, epochs completed and the
    per-epoch history.
    """
    from ultralytics import YOLO

    project = Path(job.project) if job.project else Path.cwd() / "runs"
    run_dir = project / job.run_name
    last_ckpt = run_dir / "weights" / "last.pt"
    resume = bool(job.resume) and last_ckpt.exists()

    device = resolve_device(job)
    devices = device if isinstance(device, list) else [device]

    model = YOLO(str(last_ckpt) if resume else job.model)
    save_period = int(job.save_period or 0)

    skip = max(1, int(job.dataset_skip_interval))
    val_stride = max(1, int(job.val_stride))
    val_every = max(1, int(job.val_every))

    data_yaml = resolve_data_yaml(job.data, job.run_name,
                                  train_stride=skip, val_stride=val_stride)

    batch, per_gpu = resolve_batch(model, job.imgsz, job.batch, device)
    print(f"  batch {batch} total ({per_gpu}/GPU x {len(devices)} device)", flush=True)
    print(f"  data  {data_yaml}", flush=True)
    print(f"  train every {skip}th frame, val every {val_stride}th, "
          f"validate every {val_every} epoch(s)", flush=True)

    marker = Path(stop_marker) if stop_marker else run_dir / "STOP"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if job.clear_stop_on_start and marker.exists():
        marker.unlink()

    results_csv = run_dir / "results.csv"
    chart = None
    if getattr(job, "live_chart", True):
        from .plot import LiveChart

        # Prime the chart with whatever is already on disk, so resuming does
        # not start from an empty plot.
        chart = LiveChart(title=f"  {job.run_name}")
        for row in read_history(results_csv):
            chart.add(row)

    model.add_callback(
        "on_train_epoch_end",
        lambda trainer: epoch_control(trainer, val_every, str(marker),
                                      chart, str(results_csv)),
    )

    mirror = CheckpointMirror(run_dir, job.checkpoint_mirror or None, marker)
    mirror.start()
    print(f"  {'resuming' if resume else 'starting'} {job.epochs} epochs "
          f"on device={device}", flush=True)

    started = time.time()
    tuning = {}
    if job.lr0 and job.lr0 > 0:
        tuning["lr0"] = job.lr0
    if job.optimizer:
        tuning["optimizer"] = job.optimizer
    try:
        model.train(
            data=data_yaml,
            epochs=job.epochs,          # total, not a delta
            imgsz=job.imgsz,
            batch=batch,
            patience=job.patience,
            device=device,
            workers=job.workers,
            cache=job.cache,
            project=str(project),
            name=job.run_name,
            resume=resume,
            exist_ok=True,
            save_period=save_period,
            plots=False,
            mosaic=job.mosaic,
            close_mosaic=job.close_mosaic,
            fliplr=job.fliplr,
            **tuning,
        )
    finally:
        mirror.stop()
        if chart is not None:
            chart.finish()

    history = read_history(run_dir / "results.csv")
    return {
        "run_name": job.run_name,
        "resumed": resume,
        "device": str(device),
        "batch": batch,
        "per_gpu_batch": per_gpu,
        "seconds": round(time.time() - started, 1),
        "epochs_done": history[-1]["epoch"] if history else 0,
        "stopped": marker.exists(),
        "history": history,
    }
