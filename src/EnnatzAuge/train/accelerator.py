"""What accelerates this machine, and what to set before a run starts.

PyTorch exposes ROCm through the ``torch.cuda`` API: ``is_available()``,
``get_device_properties()`` and ``mem_get_info()`` all work on an AMD card, and
``torch.version.hip`` is what tells the two builds apart. Device enumeration is
therefore shared, and only the *tuning* is backend-specific:

* **ROCm** -- MIOpen replaces cuDNN. Its default find mode explores every
  algorithm for every new tensor shape: fast from the second epoch on, but slow
  enough on the first (and on every new batch size) to look like a hang.
  ``MIOPEN_FIND_MODE=FAST`` searches less and still caches the winner. The
  kernel database lives inside the container by default, so a fresh container
  recompiles it from scratch; pointing ``MIOPEN_USER_DB_PATH`` at the
  EnnatzAuge root keeps it across restarts. ``expandable_segments`` matters
  more on a 192 GB card than on a workstation one precisely because that card
  is likely to be running several jobs at once. ``HIPBLASLT_ALLOW_TF32`` lets
  hipBLASLt take the reduced-precision path on gfx942, and
  ``HSA_ENABLE_SDMA=0`` stops the runtime spinning a busy-wait thread per queue
  on a host whose cores the DataLoader wants.
* **CUDA** -- TF32 and cuDNN autotuning. Same allocator setting.
* **MPS / CPU** -- nothing to tune, but still reported so the CLI can say what
  it saw instead of guessing.

Everything here is best-effort. An accelerator this code does not understand
must not stop a run: every probe is wrapped and falls back to the safe answer.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# DataLoader workers are a decode pipeline, not a compute one. Past this many
# per rank the context switching costs more than the JPEG decoding saves, and
# on a 128-core host the extra processes fight the GPU's own threads.
MAX_WORKERS = 16

# Above this fraction of a device's memory a second concurrent job turns from
# "packed tightly" into "the first one that allocates a spike dies".
MAX_VRAM_FRACTION = 0.9

# A card this large is a data-centre part (MI300X is 192 GB, H100 NVL 94 GB),
# not a workstation one, and the rules that keep a 24 GB card safe leave most
# of it idle. Above this threshold the profiler is allowed to keep going and
# the worker pool is allowed to grow, because the bottleneck moves from VRAM to
# the CPU pipeline feeding it.
LARGE_VRAM_GB = 80.0

# Ceiling on the profiled per-GPU batch. Ultralytics' autobatch has no upper
# bound of its own, and on a 192 GB device it will happily return a batch
# larger than the training set: one step per epoch, which destroys the LR
# schedule far more thoroughly than the bigger batch helps. 512 is the point
# where gradient noise has long since stopped being the limiting factor.
MAX_BATCH_LARGE = 512
# Ceiling for everything else. Matches the old implicit behaviour.
MAX_BATCH_DEFAULT = 128

# ROCm environment knobs that are wins on a data-centre part and no-ops
# elsewhere. Kept as data so the doctor report can print exactly what was set.
_ROCM_ENV = {
    # Exhaustive per-shape algorithm search is the single biggest reason an AMD
    # run "hangs" for minutes at the start of a new batch size.
    "MIOPEN_FIND_MODE": "FAST",
    # The kernel database lives inside the container by default, so a fresh
    # container recompiles it from scratch.
    "MIOPEN_USER_DB_PATH": None,          # filled with the cache dir
    "MIOPEN_CUSTOM_CACHE_DIR": None,
    # Matters more on a 192 GB card than a workstation one precisely because
    # that card is likely to be running several jobs at once.
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    # gfx942 (MI300X) has no native fp16 accumulate for every op; letting
    # hipBLASLt pick the reduced-precision path is worth several percent.
    "HIPBLASLT_ALLOW_TF32": "1",
    # Keep the ROCm runtime from spinning a busy-wait thread per queue, which
    # on a 128-core host competes with the DataLoader workers.
    "HSA_ENABLE_SDMA": "0",
}


@dataclass(frozen=True)
class Accelerator:
    """The accelerator (or lack of one) this process will train on."""

    backend: str                 # rocm | cuda | mps | cpu
    devices: tuple[str, ...]     # names as the driver reports them
    total_vram_gb: float         # largest single device
    pool_vram_gb: float          # summed over every visible device
    version: str                 # ROCm or CUDA version, "" for mps/cpu
    arch: str                    # gfx942 / sm_90 / ""

    @property
    def is_gpu(self) -> bool:
        return self.backend in ("rocm", "cuda", "mps")

    @property
    def device_count(self) -> int:
        return len(self.devices)

    @property
    def is_amd(self) -> bool:
        return self.backend == "rocm"

    def describe(self) -> list[str]:
        """Human-readable lines for ``ennatzauge train`` and ``doctor``."""
        if not self.is_gpu:
            return [f"accelerator  {self.backend} "
                    f"({os.cpu_count() or 1} threads, no GPU training)"]
        name = self.devices[0] if len(self.devices) == 1 else (
            f"{len(self.devices)} x {self.devices[0]}")
        free = free_vram_gb()
        head = (f"accelerator  {self.backend} {self.version}".rstrip()
                + (f" ({self.arch})" if self.arch else ""))
        lines = [
            head,
            f"gpu          {name}, {self.total_vram_gb:.1f} GB VRAM"
            + (f", {free:.1f} GB free" if free is not None else ""),
        ]
        if is_large_vram(self.total_vram_gb):
            lines.append(
                f"capacity     data-centre part: batch ceiling "
                f"{batch_ceiling(self.total_vram_gb)}, "
                f"up to {max_workers_for(self.total_vram_gb)} loader workers")
        if self.is_amd:
            lines.append("decode       " + (
                "VCN via FFmpeg vaapi" if ffmpeg_hwaccel() == "vaapi"
                else "software (install ffmpeg for VCN decode)"))
        return lines


_detected: Accelerator | None = None


def _torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def _probe(torch) -> Accelerator:
    if torch is None:
        return Accelerator("cpu", (), 0.0, 0.0, "", "")

    hip = getattr(getattr(torch, "version", None), "hip", None)
    cuda = getattr(getattr(torch, "version", None), "cuda", None)

    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False

    if available:
        try:
            count = int(torch.cuda.device_count())
        except Exception:
            count = 0
        names: list[str] = []
        sizes: list[float] = []
        arch = ""
        for index in range(count):
            try:
                props = torch.cuda.get_device_properties(index)
            except Exception:
                continue
            names.append(str(getattr(props, "name", f"device {index}")))
            sizes.append(float(getattr(props, "total_memory", 0)) / 1e9)
            if not arch:
                # ROCm reports the ISA target (gfx942 for an MI300X), CUDA the
                # compute capability. Both are the useful thing to print.
                if hip:
                    arch = str(getattr(props, "gcnArchName", "") or "").split(":")[0]
                else:
                    arch = f"sm_{getattr(props, 'major', 0)}{getattr(props, 'minor', 0)}"
        if names:
            return Accelerator(
                "rocm" if hip else "cuda",
                tuple(names),
                max(sizes) if sizes else 0.0,
                sum(sizes),
                str(hip or cuda or ""),
                arch,
            )

    try:
        mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    except Exception:
        mps = False
    if mps:
        return Accelerator("mps", ("Apple GPU",), 0.0, 0.0, "", "")

    return Accelerator("cpu", (), 0.0, 0.0, str(cuda or hip or ""), "")


def detect(refresh: bool = False, torch_module=None) -> Accelerator:
    """Cached probe of this machine's accelerator."""
    global _detected

    if torch_module is not None:
        return _probe(torch_module)
    if _detected is None or refresh:
        _detected = _probe(_torch())
    return _detected


def _default_env(key: str, value: str) -> str | None:
    """Set ``key`` unless the caller already chose a value; return what won."""
    if os.environ.get(key):
        return None
    os.environ[key] = value
    return value


def _kernel_cache_dir(kind: str) -> Path:
    root = os.environ.get("ENNATZAUGE_HOME")
    base = Path(root).expanduser() if root else Path.cwd() / ".EnnatzAuge"
    path = base / "cache" / kind
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return path
    return path


# What the last configure() call changed, for the doctor report.
_APPLIED: list[str] = []


def configure(torch_module=None) -> Accelerator:
    """Apply the process-wide settings that are wins on every backend.

    Call this before the first tensor is allocated: the allocator reads its
    configuration once, at initialisation, so a later ``os.environ`` write is
    silently ignored.
    """
    acc = detect(torch_module=torch_module)
    torch = torch_module or _torch()

    applied: list[str] = []
    if acc.backend == "rocm":
        cache = str(_kernel_cache_dir("miopen"))
        for key, value in _ROCM_ENV.items():
            if value is None:
                value = cache
            if _default_env(key, value):
                applied.append(f"{key}={value}" if key != "MIOPEN_USER_DB_PATH"
                               else "MIOPEN_USER_DB_PATH")
    elif acc.backend == "cuda":
        if _default_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"):
            applied.append("expandable_segments")

    if torch is not None and acc.is_gpu:
        try:
            # Kernel autotuning. On ROCm this switches MIOpen to find-mode,
            # which with MIOPEN_FIND_MODE=FAST above is a cheap one-off search
            # instead of the exhaustive default.
            torch.backends.cudnn.benchmark = True
            applied.append("cudnn.benchmark")
        except Exception:
            pass
        if acc.backend in ("rocm", "cuda"):
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                applied.append("tf32")
            except Exception:
                pass

    _APPLIED[:] = applied
    return acc


def applied() -> list[str]:
    return list(_APPLIED)


@contextlib.contextmanager
def cudnn_benchmark(enabled: bool):
    """Temporarily flip kernel autotuning.

    Ultralytics' autobatch refuses to profile while it is on -- the timed search
    and the benchmark want the same workspace -- and silently returns the
    default batch of 16 instead. That is how a 48 GB card ends up training at
    16 images per step, which looks like a slow GPU rather than a skipped probe.
    """
    torch = _torch()
    previous = None
    try:
        if torch is not None:
            previous = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = bool(enabled)
    except Exception:
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            try:
                torch.backends.cudnn.benchmark = previous
            except Exception:
                pass


def workers(requested: int = 0, world: int = 1, sharing: int = 1,
            cpu: int | None = None, loading: bool = True,
            total_vram_gb: float | None = None) -> int:
    """DataLoader workers per rank.

    Striding and mosaic still need JPEGs decoded on the CPU, and that pipeline
    has to stay ahead of a GPU that got several times faster. ``requested=0``
    means auto: enough to keep the accelerator fed, divided between DDP ranks
    and between jobs sharing the machine, capped at ``MAX_WORKERS`` -- or
    higher on a data-centre card, which is fast enough to outrun the default
    pool and then sit idle waiting for it.

    An in-RAM cached dataset already removed the decode, so there a couple of
    workers are enough and the rest are pure scheduling overhead.
    """
    if requested and int(requested) > 0:
        return int(requested)
    total = cpu or os.cpu_count() or 4
    share = max(1, int(world)) * max(1, int(sharing))
    auto = total // share
    if not loading:
        return max(2, min(4, auto))
    return max(2, min(max_workers_for(total_vram_gb, total), auto))


def free_vram_gb(device: int = 0) -> float | None:
    """(free) GB on one device, or None when there is nothing to query."""
    torch = _torch()
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(int(device))
        return float(free) / 1e9
    except Exception:
        return None


def available_ram_gb() -> float:
    """Free system RAM in GB, without adding a dependency."""
    try:  # psutil is often already present via other packages
        import psutil

        return float(psutil.virtual_memory().available) / 1e9
    except Exception:
        pass

    try:
        if os.name == "nt":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return float(status.ullAvailPhys) / 1e9

        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6  # kB -> GB
    except Exception:
        pass
    return 0.0


def free_disk_gb(path: Path | str) -> float:
    try:
        return float(shutil.disk_usage(str(path)).free) / 1e9
    except Exception:
        return 0.0


def vram_fraction_for(budget_gb: float, device_total_gb: float,
                      requested: float = 0.0) -> float:
    """Fraction of one device to profile a batch against.

    ``requested`` wins when set; otherwise the fraction comes from the job's
    VRAM budget so several processes can share one card without each sizing a
    batch against the whole device and then racing for it.
    """
    if requested and 0 < float(requested) < 1:
        return float(requested)
    if budget_gb and budget_gb > 0 and device_total_gb and device_total_gb > 0:
        fraction = float(budget_gb) / float(device_total_gb)
        return max(0.15, min(MAX_VRAM_FRACTION, fraction))
    return 0.6


def is_large_vram(total_gb: float | None) -> bool:
    """True for data-centre parts, where the workstation heuristics are wrong."""
    return bool(total_gb) and float(total_gb) >= LARGE_VRAM_GB


def batch_ceiling(total_gb: float | None) -> int:
    """Largest per-GPU batch worth profiling on a device of this size.

    The profiler answers "what fits", not "what is useful". On a 192 GB card
    the honest answer to the first question is often larger than the training
    set, which turns an epoch into a single optimiser step. This caps the
    answer at something the LR schedule can still work with.
    """
    return MAX_BATCH_LARGE if is_large_vram(total_gb) else MAX_BATCH_DEFAULT


def max_workers_for(total_gb: float | None, cpu: int | None = None) -> int:
    """Worker ceiling, raised on a card big enough to outrun the CPU pipeline.

    A 192 GB part at batch 256 consumes images faster than 16 processes can
    decode them, so the GPU idles between steps. The extra workers are only
    worth it when there is a card that fast to feed.
    """
    if not is_large_vram(total_gb):
        return MAX_WORKERS
    total = cpu or os.cpu_count() or 4
    return max(MAX_WORKERS, min(64, total // 2))


def hardware_decode_available() -> bool:
    """Whether this build can decode video on the accelerator.

    ``cv2.cudacodec`` is the only hardware path OpenCV exposes, and it is
    compiled against the CUDA Video Codec SDK -- so it is absent from every
    ROCm wheel, and on AMD the equivalent (VCN through VA-API) is reached
    through FFmpeg rather than OpenCV. Reporting the truth here is what lets
    the analysis pipeline pick a decoder instead of guessing.
    """
    try:
        import cv2

        return bool(hasattr(cv2, "cudacodec"))
    except Exception:
        return False


def ffmpeg_hwaccel() -> str | None:
    """The FFmpeg hardware-decode backend to ask for, or None for software.

    On ROCm this is ``vaapi`` (VCN through the VA-API driver); on CUDA it is
    ``cuda``. Both are only useful if the ``ffmpeg`` binary is actually on
    PATH, so that is checked rather than assumed.
    """
    acc = detect()
    if not acc.is_gpu:
        return None
    if shutil.which("ffmpeg") is None:
        return None
    return "vaapi" if acc.backend == "rocm" else "cuda"


def rocm_smi() -> str | None:
    """``rocm-smi`` output, or None when the tool is not installed.

    ``torch`` reports what it can see; ``rocm-smi`` reports what the driver
    sees, including the topology (XGMI links between MI300X chiplets) that
    decides whether a multi-GPU run is actually one fast device or several
    slow ones.
    """
    if shutil.which("rocm-smi") is None:
        return None
    import subprocess

    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
