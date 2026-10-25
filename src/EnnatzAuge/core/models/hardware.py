"""Pick the fastest inference path the machine actually offers.

Analysis used to be CPU-only by construction: ``device="cpu"`` was hardcoded and
an OpenVINO IR was always preferred. On a GPU host that was not just slower, it
left the GPU idle while the CPU did all the work.

The choice is a property of the hardware, not something a caller should have to
know, so it is made here once:

* **CUDA** -> run the ``.pt`` directly in PyTorch at fp16. TensorRT would be
  faster still, but the engine build is slow, memory-hungry and has to be redone
  whenever the checkpoint or batch shape changes, which is wrong for a
  short-lived analysis job.
* **Apple MPS** -> PyTorch on the GPU, fp32 (fp16 on MPS is unreliable).
* **CPU** -> OpenVINO if it is installed, which is several times faster than
  PyTorch on an Intel laptop CPU.

``ENNATZAUGE_DEVICE=cuda:1`` (or ``cpu``, ``mps``) overrides the choice, and
``ENNATZAUGE_BACKEND=openvino`` / ``torch`` overrides the backend, which is what
the doctor report uses to prove both paths work.
"""

from __future__ import annotations

import os

import numpy as np

# Conservative allowance for per-frame activations, as a multiple of the raw
# input tensor. Deliberately pessimistic: running out of VRAM mid-clip is much
# worse than using a slightly smaller batch.
_ACTIVATION_FACTOR = 50.0
# Never ask for more than this many frames at once even on a huge GPU: the win
# per extra frame flattens out and the latency of a chunk starts to matter.
_MAX_BATCH = 64
_MIN_BATCH = 4


class Backend:
    """How one process will run inference."""

    def __init__(self, format: str, torch_device: str, half: bool,
                 label: str, max_batch: int):
        self.format = format            # "torch" | "openvino"
        self.torch_device = torch_device
        self.half = half
        self.label = label
        self.max_batch = max_batch

    @property
    def device_argument(self):
        """The value to hand Ultralytics as ``device=``."""
        if self.format == "openvino":
            return "cpu"
        if self.torch_device.startswith("cuda"):
            index = self.torch_device.split(":", 1)[1] if ":" in self.torch_device else "0"
            return int(index)
        return self.torch_device

    def chunk_size(self, imgsz: int, requested: int | None = None) -> int:
        """Frames per inference call.

        This is the model's own limit, not the caller's: a static IR can only
        accept its exported batch. On a GPU it is recomputed for the resolution
        actually being used, since activations scale with it.
        """
        limit = self.max_batch
        if self.torch_device.startswith("cuda"):
            limit = _batch_from_vram(imgsz, self.half)
        if requested and 0 < int(requested) <= limit:
            return int(requested)
        return limit

    def read_chunk(self, requested: int | None = None) -> int:
        """Frames decoded from the video between inference calls.

        Independent of the model's batch: decoding in groups of 16 is cheaper
        than one frame at a time even when the model has to be called 16 times.
        """
        if requested and requested > 0:
            return int(requested)
        return max(16, min(self.max_batch, 64))

    def describe(self) -> str:
        return self.label

    def as_dict(self) -> dict:
        return {
            "format": self.format,
            "device": self.torch_device,
            "half": self.half,
            "batch": self.max_batch,
        }


def _cuda_memory() -> tuple[int, int] | None:
    """(free, total) bytes on the current CUDA device, or None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)
    except Exception:
        return None


def _batch_from_vram(imgsz: int, half: bool) -> int:
    memory = _cuda_memory()
    if memory is None:
        return 8
    free, total = memory
    # Leave headroom: the decoder, the tracker and the renderers all live in the
    # same process as the model.
    usable = max(0.0, min(free * 0.75, total * 0.6))
    per_frame = (imgsz * imgsz * 3 * (2 if half else 4)) * _ACTIVATION_FACTOR
    if per_frame <= 0:
        return _MIN_BATCH
    return int(np.clip(usable // per_frame, _MIN_BATCH, _MAX_BATCH))


def _openvino_available() -> bool:
    try:
        import openvino  # noqa: F401

        return True
    except Exception:
        return False


def select() -> Backend:
    """Choose the fastest available inference path for this machine."""
    import torch

    forced_device = (os.environ.get("ENNATZAUGE_DEVICE") or "").strip()
    forced_backend = (os.environ.get("ENNATZAUGE_BACKEND") or "").strip().lower()

    try:
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    try:
        mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    except Exception:
        mps = False

    if forced_device.startswith("cuda") and cuda:
        device = forced_device
    elif forced_device == "mps" and mps:
        device = "mps"
    elif forced_device == "cpu":
        device = "cpu"
    elif cuda:
        device = "cuda:0"
    elif mps:
        device = "mps"
    else:
        device = "cpu"

    if device.startswith("cuda"):
        name = "cuda"
        try:
            name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            pass
        half = True
        label = f"GPU {name} (PyTorch fp16)"
        if forced_backend == "openvino":
            device = "cpu"      # OpenVINO here is CPU-only in this pipeline
        else:
            return Backend("torch", device, half, label, _batch_from_vram(imgsz_hint(), half))

    if device == "mps":
        label = "Apple GPU (PyTorch MPS)"
        if forced_backend != "openvino":
            return Backend("torch", "mps", False, label, _MIN_BATCH)

    if forced_backend == "torch" or not _openvino_available():
        label = "CPU (PyTorch)" if forced_backend == "torch" else "CPU (PyTorch, OpenVINO unavailable)"
        return Backend("torch", "cpu", False, label, 8)

    return Backend("openvino", "cpu", False, "CPU (OpenVINO)", 1)


def imgsz_hint() -> int:
    """Resolution the batch estimate assumes before a model is loaded."""
    try:
        return int(os.environ.get("ENNATZAUGE_IMGSZ", "1280"))
    except ValueError:
        return 1280


def describe() -> list[str]:
    """One-line-per-fact summary, for the CLI and the doctor report."""
    backend = select()
    lines = [f"analysis backend   {backend.describe()}"]
    memory = _cuda_memory()
    if memory is not None:
        free, total = memory
        lines.append(f"gpu memory         {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")
    try:
        import torch

        lines.append(f"torch              {torch.__version__}"
                     f" (cuda {torch.version.cuda or 'none'})")
    except Exception:
        pass
    return lines
