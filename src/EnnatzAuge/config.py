"""YAML configuration for the CLI.

Two shapes are supported so a config file stays simple to write by hand:

    # train.yaml -- a single job
    job:
      name: players
      data: datasets/yolo-players/data.yaml
      epochs: 80

    # or several at once
    jobs:
      - name: players
        data: datasets/yolo-players/data.yaml
      - name: pitch
        data: datasets/yolo-pitch/data.yaml

Relative paths inside a config are resolved against the EnnatzAuge root, not
the current directory, so running the CLI from anywhere behaves the same.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .paths import Paths


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def resolve_path(value: str | Path | None, base: Path) -> str | None:
    """Make a config path absolute relative to the EnnatzAuge root."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path)


def filter_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the dataclass does not declare, with a warning.

    Keeps a stale config from failing hard after a rename, while still making
    the typo visible.
    """
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        print(f"  warning: ignoring unknown key(s) for {cls.__name__}: "
              f"{', '.join(unknown)}", flush=True)
    return {k: v for k, v in values.items() if k in known}


def job_dicts(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise ``job:`` or ``jobs:`` into a list."""
    if "jobs" in config:
        jobs = config["jobs"]
        if not isinstance(jobs, list):
            raise ValueError("'jobs' must be a list")
        return [dict(j) for j in jobs]
    if "job" in config:
        job = config["job"]
        if not isinstance(job, dict):
            raise ValueError("'job' must be a mapping")
        return [dict(job)]
    return []


class ConfigContext:
    """Loads configs and resolves their paths against the EnnatzAuge root."""

    def __init__(self, paths: Paths | None = None):
        self.paths = paths or Paths()

    def train_config(self, name: str = "train") -> list[dict[str, Any]]:
        raw = load_yaml(self.paths.config(name))
        jobs = job_dicts(raw)

        if not jobs:
            # Allow a bare mapping at the top level, e.g. just `epochs: 80`.
            if set(raw) - {"jobs", "job"}:
                jobs = [dict(raw)]

        for job in jobs:
            for key in ("data", "model", "project"):
                if key in job:
                    job[key] = resolve_path(job[key], self.paths.root)
            job.setdefault("project", str(self.paths.runs))
        return jobs

    def analyze_config(self, name: str = "analyze") -> dict[str, Any]:
        raw = load_yaml(self.paths.config(name))
        for key in ("video", "output"):
            if key in raw:
                raw[key] = resolve_path(raw[key], self.paths.root)
        for spec in raw.get("models", []) or []:
            if "weights" in spec:
                spec["weights"] = resolve_path(spec["weights"], self.paths.root)
        return raw
