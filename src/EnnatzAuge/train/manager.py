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


class Manager:
    """Runs queued jobs, never using more GPUs at once than ``max_gpus``.

    ``max_gpus`` defaults to the number of devices actually present, so a
    single-GPU box will not try to run four jobs concurrently.
    """

    def __init__(self, max_gpus: int | None = None,
                 project: Path | str | None = None):
        self.jobs: list[Job] = []
        self.max_gpus = max_gpus if max_gpus is not None else max(1, visible_gpu_count())
        self.project = Path(project) if project else None
        self._used = 0
        self._cond = threading.Condition()

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

    # -- execution ---------------------------------------------------------

    def _reserve(self, count: int) -> None:
        with self._cond:
            while self._used + count > self.max_gpus:
                self._cond.wait()
            self._used += count

    def _release(self, count: int) -> None:
        with self._cond:
            self._used -= count
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
        self._reserve(job.gpus)
        try:
            job.state, job.started = JobState.RUNNING, time.time()
            try:
                result = run_job(job)
                job.history = result["history"]
                job.devices_used = result["device"]
                job.state = (JobState.STOPPED if result.get("stopped")
                             else JobState.DONE)
            except Exception as exc:  # one failure must not stop the batch
                job.state = JobState.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()
        finally:
            self._release(job.gpus)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        lines = [f"{'run':<28}{'state':<10}{'time':>9}  {'devices':<12}result"]
        for job in self.jobs:
            best = job.best
            secs = f"{job.duration:.0f}s" if job.duration else "-"
            devices = job.devices_used or ",".join(str(d) for d in job.resolved_devices())
            detail = (f"mAP50-95={best['mAP_50_95']:.4f} @ep{best['epoch']}"
                      if best else "no metrics")
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
            "best_mAP_50_95": max((r["mAP_50_95"] for r in rows), default=0.0),
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
