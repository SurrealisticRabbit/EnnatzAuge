"""Shared YOLO loading and inference for the pitch and players models.

Mirrors how local_runner.py loads models:

* CPU inference prefers a cached OpenVINO IR, which is several times faster
  than PyTorch on an Intel laptop CPU.
* The IR is exported static. A dynamic-shape graph reshapes to the batch it is
  handed, so a large batch makes OpenVINO allocate activations for the whole
  batch at once -- 240 frames at 1280 needed 14 GB.
* The cache is validated against metadata.yaml, because Ultralytics names the
  export directory after the model, not the resolution. Without that check a
  640 export is silently reused at 1280.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO


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
                 use_openvino: bool = True):
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.model = self._load(self.model_path, imgsz, use_openvino)

    @staticmethod
    def _load(weights: Path, imgsz: int, use_openvino: bool) -> YOLO:
        if not use_openvino:
            return YOLO(str(weights))

        ov_dir = weights.parent / f"{weights.stem}_openvino_model"
        try:
            if ov_dir.exists() and not _ir_is_usable(ov_dir, imgsz):
                import shutil

                shutil.rmtree(ov_dir, ignore_errors=True)
            if not ov_dir.exists():
                YOLO(str(weights)).export(format="openvino", imgsz=imgsz,
                                          half=False, dynamic=False)
            return YOLO(str(ov_dir))
        except Exception:
            # Missing openvino extra or a failed export: plain CPU is slower
            # but always available.
            return YOLO(str(weights))

    @property
    def names(self) -> dict[int, str]:
        return getattr(self.model, "names", {}) or {}

    def predict(self, frames, conf: float = 0.25):
        """Run inference over a list of BGR frames, preserving order."""
        if not frames:
            return []
        return self.model.predict(list(frames), imgsz=self.imgsz, conf=conf,
                                  device="cpu", verbose=False)
