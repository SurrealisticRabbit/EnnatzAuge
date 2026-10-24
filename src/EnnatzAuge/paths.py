"""Resolve where EnnatzAuge keeps its data.

Everything the package reads or writes lives under one root, so an installed
copy behaves the same as a development checkout:

    $ENNATZAUGE_HOME        if set, used verbatim
    ./.EnnatzAuge           if it exists (development checkout)
    ~/.EnnatzAuge           otherwise (installed use)

Layout inside the root::

    configs/        analyze.yaml, train.yaml
    datasets/       yolo-players/, yolo-pitch/
    input_video/    source clips
    models/base/    pretrained starting weights
    models/runs/    training output (checkpoints, results.csv)
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "ENNATZAUGE_HOME"
FOLDER_NAME = ".EnnatzAuge"

# Subdirectories created on demand.
CONFIGS = "configs"
DATASETS = "datasets"
INPUT_VIDEO = "input_video"
MODELS = "models"
BASE_MODELS = "models/base"
RUNS = "models/runs"
ANALYSIS = "models/analysis"


def find_home() -> Path:
    """Locate the EnnatzAuge root, without creating anything."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    local = Path.cwd() / FOLDER_NAME
    if local.is_dir():
        return local.resolve()

    return (Path.home() / FOLDER_NAME).resolve()


class Paths:
    """Convenience accessors for the standard locations under one root."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root).expanduser().resolve() if root else find_home()

    def __repr__(self) -> str:
        return f"<Paths {self.root}>"

    def _sub(self, name: str, create: bool = False) -> Path:
        path = self.root / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    # -- directories -------------------------------------------------------

    @property
    def configs(self) -> Path:
        return self._sub(CONFIGS)

    @property
    def datasets(self) -> Path:
        return self._sub(DATASETS)

    @property
    def input_video(self) -> Path:
        return self._sub(INPUT_VIDEO)

    @property
    def models(self) -> Path:
        return self._sub(MODELS)

    @property
    def base_models(self) -> Path:
        return self._sub(BASE_MODELS)

    @property
    def runs(self) -> Path:
        return self._sub(RUNS)

    @property
    def analysis(self) -> Path:
        """Rendered videos, kept out of the training runs directory."""
        return self._sub(ANALYSIS)

    # -- named files -------------------------------------------------------

    def config(self, name: str) -> Path:
        """``paths.config("train")`` -> ``<root>/configs/train.yaml``."""
        return self.configs / (name if name.endswith(".yaml") else f"{name}.yaml")

    def dataset(self, name: str) -> Path:
        return self.datasets / name

    def run(self, run_name: str, create: bool = False) -> Path:
        path = self.runs / run_name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def weights(self, run_name: str, which: str = "best") -> Path:
        """Checkpoint path for a run: ``which`` is best, last or an epoch file."""
        stem = which if which.endswith(".pt") else f"{which}.pt"
        return self.runs / run_name / "weights" / stem

    def ensure(self) -> "Paths":
        """Create every standard directory, for scripts and setup steps."""
        for accessor in (self.configs, self.datasets, self.input_video,
                         self.models, self.base_models, self.runs, self.analysis):
            accessor.mkdir(parents=True, exist_ok=True)
        return self

    def describe(self) -> str:
        lines = [f"root  {self.root}"]
        for name, path in (
            ("configs", self.root / CONFIGS),
            ("datasets", self.root / DATASETS),
            ("input_video", self.root / INPUT_VIDEO),
            ("base models", self.root / BASE_MODELS),
            ("runs", self.root / RUNS),
            ("analysis", self.root / ANALYSIS),
        ):
            state = "ok" if path.exists() else "missing"
            lines.append(f"  {name:<12} {state:<8} {path}")
        return "\n".join(lines)
