"""Schedules training jobs on the local machine.

The Modal version provisioned containers, so it needed a GPU fallback chain and
a remote volume to flush. Locally there is one machine: jobs run as threads and
the GPU budget is simply how many devices this host has.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .jobs import Job, JobState
from .trainer import read_history, run_job


def visible_gpu_count() -> int:
    """How many CUDA devices this host exposes (0 on a CPU-only box)."""
    try:
        import torch

        return max(0, torch.cuda.device_count())
    except Exception:
        return 0


def describe_devices() -> list[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return ["cpu"]
        return [f"{i}: {torch.cuda.get_device_name(i)}"
                for i in range(torch.cuda.device_count())]
    except Exception:
        return ["cpu"]

class HyperParameterTuner:
    """Runs Ultralytics' built-in hyperparameter search for one job.

    ``model.tune()`` is a brute-force search: it samples ``iterations``
    combinations from Ultralytics' search space, trains each for ``epochs``
    epochs, and keeps the one with the best fitness. That is exactly the kind
    of embarrassingly parallel, compute-heavy work a 192 GB MI300X is for, so
    the search is deliberately allowed to run wide.

    The search writes ``tune/`` next to the run directory and leaves the
    winning values in ``tune/best_hyperparameters.yaml``. Those are read back
    and merged into the real training call, so the expensive search is paid
    once and the result is reproducible from the config alone.
    """

    def __init__(self, job: Job, device=None, batch: int | None = None,
                 data_yaml: str | None = None):
        self.job = job
        self.device = device
        self.batch = batch
        self.data_yaml = data_yaml
        self.best: dict = {}

    @property
    def enabled(self) -> bool:
        return int(self.job.tune_iterations or 0) > 0

    def tune(self) -> dict:
        """Run the search and return the winning hyperparameters.

        Returns an empty dict when tuning is disabled or the search fails; a
        failed search must not cost the caller its training run.
        """
        if not self.enabled:
            return {}

        from ultralytics import YOLO

        project = Path(self.job.project) if self.job.project else Path.cwd() / "runs"
        iterations = int(self.job.tune_iterations)
        epochs = max(1, int(self.job.tune_epochs))
        print(f"  tuning: {iterations} iterations x {epochs} epochs "
              f"(~{iterations * epochs} epoch-equivalents)", flush=True)

        model = YOLO(self.job.model)
        kwargs = {
            "data": self.data_yaml,
            "epochs": epochs,
            "iterations": iterations,
            "imgsz": self.job.imgsz,
            "device": self.device,
            "project": str(project),
            "name": f"{self.job.run_name}-tune",
            "exist_ok": True,
            "plots": False,
            "verbose": False,
        }
        if self.batch:
            kwargs["batch"] = self.batch
        if self.job.optimizer:
            kwargs["optimizer"] = self.job.optimizer

        try:
            model.tune(**kwargs)
        except Exception as exc:
            print(f"  tuning failed ({type(exc).__name__}: {exc}); "
                  f"continuing with defaults", flush=True)
            return {}

        self.best = self._best_hyperparameters()
        if self.best:
            shown = ", ".join(f"{k}={v}" for k, v in sorted(self.best.items()))
            print(f"  tuned: {shown}", flush=True)
        return self.best

    def _best_hyperparameters(self) -> dict:
        """Read the winner Ultralytics wrote, if it wrote one."""
        import yaml

        project = Path(self.job.project) if self.job.project else Path.cwd() / "runs"
        candidates = [
            project / f"{self.job.run_name}-tune" / "best_hyperparameters.yaml",
            project / self.job.run_name / "tune" / "best_hyperparameters.yaml",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            # Ultralytics writes {name: {"val": x}}; flatten to {name: x}.
            flat = {}
            for key, value in spec.items():
                if isinstance(value, dict):
                    value = value.get("val", next(iter(value.values()), None))
                if value is not None:
                    flat[key] = value
            if flat:
                return flat
        return {}


class Manager:
    """Runs queued jobs, never using more GPUs at once than ``max_gpus``.

    ``max_gpus`` defaults to the number of devices actually present, so a
    single-GPU box will not try to run four jobs concurrently.

    On a data-centre card the GPU count is the wrong unit of accounting: one
    MI300X is 192 GB, which is eight L40s' worth of VRAM behind a single
    device index. So the manager also tracks a VRAM pool. A job that declares
    ``vram_gb`` (or ``vram_fraction``) is admitted when its budget fits in
    what is left, which lets several experiments share one card without any of
    them profiling a batch for the whole device and OOMing the others.
    """

    def __init__(self, max_gpus: int | None = None,
                 project: Path | str | None = None,
                 vram_pool_gb: float | None = None,
                 max_concurrent: int | None = None):
        self.jobs: list[Job] = []
        self.max_gpus = max_gpus if max_gpus is not None else max(1, visible_gpu_count())
        self.project = Path(project) if project else None
        self._used = 0
        self._cond = threading.Condition()

        from . import accelerator

        self.accelerator = accelerator.detect()
        pool = vram_pool_gb if vram_pool_gb is not None else self.accelerator.pool_vram_gb
        self.vram_pool_gb = max(0.0, float(pool or 0.0))
        self._vram_used = 0.0
        # A cap on simultaneous jobs, independent of the GPU count. Useful when
        # the bottleneck is host RAM or the dataloader rather than VRAM.
        self.max_concurrent = max_concurrent
        self._running = 0

    # -- queue -------------------------------------------------------------

    def add(self, job: Job) -> Job:
        if job.gpus > self.max_gpus:
            raise ValueError(
                f"{job.name}: requests {job.gpus} GPU(s) but max_gpus={self.max_gpus}"
            )
        if self.project is not None and job.project is None:
            job.project = str(self.project)
        self.jobs.append(job)
        return job

    def get(self, identifier: str) -> Job | None:
        for job in self.jobs:
            if job.run_name == identifier or job.name == identifier:
                return job
        return None

    # -- VRAM accounting ---------------------------------------------------

    def vram_budget(self, job: Job) -> float:
        """GB this job may use, resolved from its knobs.

        ``vram_gb`` is an absolute reservation. ``vram_fraction`` is a share of
        the whole pool. Neither set means "one device's worth", which is the
        honest default: a job that has not been told to share should be sized
        as if it owned a card.
        """
        if job.vram_gb and job.vram_gb > 0:
            return float(job.vram_gb)
        if job.vram_fraction and job.vram_fraction > 0:
            return self.vram_pool_gb * float(job.vram_fraction)
        if self.vram_pool_gb <= 0:
            return 0.0
        return self.vram_pool_gb / max(1, self.max_gpus)

    def _shares_device(self, job: Job) -> bool:
        """True when the job declared its own VRAM budget.

        Such a job is partitioning memory rather than claiming a whole device,
        so it must not also consume a GPU slot -- otherwise a single-device
        box serialises every job and the pool never admits a second one.
        """
        return bool((job.vram_gb and job.vram_gb > 0)
                    or (job.vram_fraction and job.vram_fraction > 0))

    def _reserve(self, job: Job) -> float:
        """Block until this job's GPU slots and VRAM budget are both free."""
        budget = self.vram_budget(job)
        slots = 0 if self._shares_device(job) else job.gpus
        with self._cond:
            while True:
                gpu_ok = self._used + slots <= self.max_gpus
                vram_ok = (self.vram_pool_gb <= 0
                           or self._vram_used + budget <= self.vram_pool_gb
                           or self._vram_used == 0)
                slots_ok = (self.max_concurrent is None
                            or self._running < self.max_concurrent)
                if gpu_ok and vram_ok and slots_ok:
                    break
                self._cond.wait()
            self._used += slots
            self._vram_used += budget
            self._running += 1
            job.vram_allocated_gb = round(budget, 2)
            job.sharing = max(1, self._running)
        return budget

    def _release(self, job: Job) -> None:
        with self._cond:
            self._used -= 0 if self._shares_device(job) else job.gpus
            self._vram_used = max(0.0, self._vram_used - job.vram_allocated_gb)
            self._running = max(0, self._running - 1)
            self._cond.notify_all()

    def run(self) -> list[Job]:
        pending = [j for j in self.jobs if j.state is JobState.PENDING]
        if not pending:
            return self.jobs

        threads = [threading.Thread(target=self._run_one, args=(job,),
                                    name=job.run_name)
                   for job in pending]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return self.jobs

    def _run_one(self, job: Job) -> None:
        self._reserve(job)
        try:
            job.state, job.started = JobState.RUNNING, time.time()
            try:
                result = run_job(job)
                job.history = result["history"]
                job.devices_used = result["device"]
                job.resolved_batch = result.get("batch")
                job.state = (JobState.STOPPED if result.get("stopped")
                             else JobState.DONE)
            except Exception as exc:  # one failure must not stop the batch
                job.state = JobState.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()
        finally:
            self._release(job)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        lines = [f"{'run':<28}{'state':<10}{'time':>9}  {'devices':<12}result"]
        for job in self.jobs:
            best = job.best
            secs = f"{job.duration:.0f}s" if job.duration else "-"
            devices = job.devices_used or ",".join(str(d) for d in job.resolved_devices())
            detail = (f"mAP50-95={best['mAP_50_95']:.4f} @ep{best['epoch']}"
                      if best else "no metrics")
            if job.resolved_batch:
                detail += f"  batch={job.resolved_batch}"
            if job.vram_allocated_gb:
                detail += f"  vram={job.vram_allocated_gb:.0f}GB"
            line = (f"{job.run_name:<28}{job.state.value:<10}{secs:>9}  "
                    f"{devices:<12}{detail}")
            if job.error:
                line += f"  ERROR {job.error}"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# run inspection and control (replaces the Modal volume helpers)
# ---------------------------------------------------------------------------


def list_runs(project: Path | str) -> list[dict]:
    """Summarise the runs written under ``project``."""
    root = Path(project)
    if not root.exists():
        return []

    out = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        weights = run_dir / "weights"
        last = weights / "last.pt"
        best = weights / "best.pt"
        rows = read_history(run_dir / "results.csv")
        out.append({
            "run_name": run_dir.name,
            "epochs_done": rows[-1]["epoch"] if rows else 0,
            # fitness, not mAP50-95 alone: it is what picks best.pt.
            "best_fitness": max((r["fitness"] for r in rows), default=0.0),
            "best_mAP_50_95": max((r["mAP_50_95"] for r in rows), default=0.0),
            "best_epoch": (max(rows, key=lambda r: r["fitness"])["epoch"]
                           if rows else 0),
            "is_pose": any(r.get("pose_mAP_50_95") is not None for r in rows),
            "last_pt_mb": round(last.stat().st_size / 1e6, 1) if last.exists() else 0.0,
            "has_best": best.exists(),
            "stopped": (run_dir / "STOP").exists(),
        })
    return out


def request_stop(project: Path | str, run_name: str) -> str:
    """Ask a running job to stop after its current epoch.

    A marker file rather than a signal: the training loop polls for it between
    epochs and shuts down cleanly, saving a final checkpoint, which SIGKILL
    would not.
    """
    run_dir = Path(project) / run_name
    if not run_dir.exists():
        return f"no such run: {run_name}"
    (run_dir / "STOP").write_text("stop", encoding="utf-8")
    return f"stop requested for {run_name}"


def clear_stop(project: Path | str, run_name: str) -> str:
    marker = Path(project) / run_name / "STOP"
    if marker.exists():
        marker.unlink()
    return f"cleared stop for {run_name}"
