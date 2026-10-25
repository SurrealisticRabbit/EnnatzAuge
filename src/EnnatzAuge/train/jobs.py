"""A single training run's parameters and live state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Job:
    """One training run.

    ``epochs`` is the TOTAL budget for the run, never a delta. Ultralytics
    restores ``epochs`` from the checkpoint on resume (it is absent from
    ``check_resume``'s allowed overrides), so resuming with a smaller number
    than the checkpoint holds finishes immediately having trained nothing.
    """

    name: str
    data: str
    model: str = "yolo26m.pt"
    epochs: int = 100
    imgsz: int = 1280
    # -1 = profile the GPU actually allocated, 0<x<1 = VRAM fraction,
    # >=1 = explicit TOTAL batch (Ultralytics divides it across ranks).
    batch: float = -1
    # GPUs this job claims. 1 = single GPU, >1 = DDP across that many.
    gpus: int = 1
    # Which devices to use. None = the first `gpus` visible GPUs.
    # Accepts "0", "0,1" or a list, matching CUDA_VISIBLE_DEVICES ordering.
    devices: list[int] | str | None = None

    patience: int = 20
    resume: bool = True
    save_period: int = 5
    workers: int = 8
    cache: bool | str = False
    project: str | None = None      # resolved by the config layer

    # --- throughput -------------------------------------------------------
    # Every Nth frame of the training split, applied within each sequence.
    # Ultralytics' own `fraction` truncates a sorted file list, so 0.2 on this
    # dataset trained on the first 12 of 57 matches rather than a sample.
    dataset_skip_interval: int = 1
    val_stride: int = 1
    # Validate every Nth epoch. Keep at 1 if best.pt matters: on a skipped
    # epoch save_model compares a stale fitness and overwrites best.pt.
    val_every: int = 1

    # --- augmentation -----------------------------------------------------
    mosaic: float = 1.0
    close_mosaic: int = 10
    fliplr: float = 0.5

    # --- optimisation -----------------------------------------------------
    # 0 leaves Ultralytics' own default (lr0=0.01 with optimizer=auto). Lowering
    # it is the cheapest thing to try when a small model plateaus early.
    lr0: float = 0.0
    optimizer: str = ""             # auto / SGD / AdamW / ...

    # --- housekeeping -----------------------------------------------------
    # Copy checkpoints somewhere durable every N epochs (a mounted volume, an
    # rclone target, anything the host can lose). Empty = no mirroring.
    checkpoint_mirror: str = ""
    # Redraw a live metric chart each epoch. Only draws on a tty.
    live_chart: bool = True
    clear_stop_on_start: bool = True

    # Pin this in the config to make re-running resume the same run instead of
    # starting a new one. Empty generates a unique name on first use.
    run_name: str = ""

    # --- live state -------------------------------------------------------
    state: JobState = field(default=JobState.PENDING, init=False)
    error: str | None = field(default=None, init=False)
    devices_used: str | None = field(default=None, init=False)
    history: list[dict] = field(default_factory=list, init=False)
    started: float | None = field(default=None, init=False)
    finished: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.run_name:
            self.run_name = f"{self.name}-{uuid.uuid4().hex[:8]}"
        if self.epochs <= 0:
            raise ValueError(f"{self.name}: epochs must be positive")
        if self.gpus < 1:
            raise ValueError(f"{self.name}: gpus must be at least 1")
        if self.gpus > 1 and self.batch >= 1 and self.batch % self.gpus:
            raise ValueError(
                f"{self.name}: batch={self.batch} must be a multiple of "
                f"gpus={self.gpus} (or leave batch=-1 to auto-size)."
            )

    @property
    def duration(self) -> float | None:
        if self.started is None:
            return None
        import time

        return (self.finished or time.time()) - self.started

    @property
    def best(self) -> dict | None:
        """Highest-fitness epoch, matching how Ultralytics picks ``best.pt``."""
        return max(self.history, key=lambda r: r.get("fitness", r["mAP_50_95"]),
                   default=None)

    def resolved_devices(self) -> list[int]:
        """Device indices to train on.

        A list is what makes Ultralytics launch DDP ranks; a bare integer keeps
        it single-process. Passing the list explicitly also means a container
        with more GPUs than the job asked for will not silently use them all.
        """
        if self.devices is None:
            return list(range(self.gpus))
        if isinstance(self.devices, str):
            parts = [p.strip() for p in self.devices.replace(",", " ").split() if p.strip()]
            indices = [int(p) for p in parts]
        else:
            indices = [int(p) for p in self.devices]
        if not indices:
            return [0]
        return indices

    def device_argument(self) -> int | list[int]:
        """What to pass as ``device=`` to Ultralytics."""
        devices = self.resolved_devices()
        return devices[0] if len(devices) == 1 else devices
