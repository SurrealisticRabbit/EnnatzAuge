"""Shared YOLO loading and inference for the pitch and players models.

The execution path is chosen by :mod:`.hardware` rather than by the caller: a GPU
host runs the checkpoint directly in PyTorch at fp16, a CPU host uses an OpenVINO
IR when it is available. Three details matter for correctness:

* **The IR must be exported static.** A dynamic-shape graph reshapes to the batch
  it is handed, so a large batch makes OpenVINO allocate activations for the whole
  batch at once -- 240 frames at 1280 needed 14 GB.
* **The IR cache is validated against metadata.yaml**, because Ultralytics names
  the export directory after the model, not the resolution. Without that check a
  640 export is silently reused at 1280.
* **A static IR also fixes the batch size**, so frames have to be fed through it
  in groups that match. Ultralytics does not split for you: handing a 16-frame
  list to a batch-1 graph raises ``The input tensor size is not equal to the
  model input type``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO

from . import hardware


def _static_batch(ov_dir: Path) -> int:
    """Batch size baked into the IR, or 0 when it is dynamic/unknown."""
    try:
        import glob

        import openvino as ov

        xml = next(iter(glob.glob(str(ov_dir / "*.xml"))), None)
        if xml is None:
            return 0
        shape = ov.Core().read_model(xml).input(0).partial_shape
        if shape.is_dynamic:
            return 0
        # Dimension is not an int subclass; get_length() is the accessor.
        return int(shape.get_dimension(0).get_length())
    except Exception:
        return 0


def _ir_is_usable(ov_dir: Path, imgsz: int) -> bool:
    meta_path = ov_dir / "metadata.yaml"
    if not meta_path.exists():
        return False
    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
    except Exception:
        return False
    if (meta.get("args") or {}).get("dynamic"):
        return False
    return [int(v) for v in (meta.get("imgsz") or [])] == [imgsz, imgsz]


class UltralyticsModel:
    """A YOLO checkpoint that runs batched inference over a list of frames."""

    task = "detect"

    def __init__(self, model_path: str, imgsz: int = 1280,
                 backend: hardware.Backend | None = None):
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.backend = backend or hardware.select()
        self.model = self._load(self.model_path, imgsz, self.backend)
        self.sub_batch = self._resolve_sub_batch(imgsz, self.backend)

    @property
    def describe(self) -> str:
        home = self.backend.describe()
        return f"{home}, batch {self.sub_batch}" if self.sub_batch else home

    def _resolve_sub_batch(self, imgsz: int, backend: hardware.Backend) -> int:
        """How many frames may go through the model in one call.

        A static OpenVINO IR accepts exactly its exported batch and nothing else,
        so the caller's chunk size has to be cut down to fit. A GPU benefits from
        a large batch, bounded by free memory.
        """
        if backend.format == "openvino":
            ov_dir = self.model_path.parent / f"{self.model_path.stem}_openvino_model"
            if not ov_dir.exists():
                return 1
            return _static_batch(ov_dir) or 1
        return backend.chunk_size(imgsz)
    def _load(self, weights: Path, imgsz: int, backend: hardware.Backend) -> YOLO:
        if backend.format != "openvino":
            return YOLO(str(weights), task=self.task)

        ov_dir = weights.parent / f"{weights.stem}_openvino_model"
        try:
            if ov_dir.exists() and not _ir_is_usable(ov_dir, imgsz):
                import shutil

                shutil.rmtree(ov_dir, ignore_errors=True)
            if not ov_dir.exists():
                YOLO(str(weights), task=self.task).export(
                    format="openvino", imgsz=imgsz, half=False, dynamic=False)
            # The task must be given explicitly for an exported directory:
            # Ultralytics cannot infer it from an IR and silently falls back to
            # "detect", which drops the keypoints this whole pipeline needs.
            return YOLO(str(ov_dir), task=self.task)
        except Exception:
            # Missing openvino extra or a failed export: plain CPU is slower
            # but always available.
            return YOLO(str(weights), task=self.task)

    @property
    def names(self) -> dict[int, str]:
        return getattr(self.model, "names", {}) or {}

    def predict(self, frames, conf: float = 0.25):
        """Run inference over a list of BGR frames, preserving order."""
        if not frames:
            return []
        frames = list(frames)
        step = self.sub_batch if self.sub_batch > 0 else len(frames)
        options = {
            "imgsz": self.imgsz,
            "conf": conf,
            "device": self.backend.device_argument,
            "verbose": False,
        }
        if self.backend.half and self.backend.format == "torch":
            options["half"] = True

        if step >= len(frames):
            return self.model.predict(frames, **options)
        out = []
        for start in range(0, len(frames), step):
            out.extend(self.model.predict(frames[start:start + step], **options))
        return out
