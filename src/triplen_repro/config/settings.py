from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProjectPaths:
    dataset_root: Path
    matlab_source_root: Path
    output_root: Path
    figure_output_dir: Path
    cache_dir: Path
    log_dir: Path


@dataclass(slots=True)
class ProjectConfig:
    paths: ProjectPaths
    analysis: dict[str, Any]
    runtime: dict[str, Any]


def _expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def load_config(config_path: str | Path) -> ProjectConfig:
    config_path = _expand_path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    paths = ProjectPaths(
        dataset_root=_expand_path(raw["paths"]["dataset_root"]),
        matlab_source_root=_expand_path(raw["paths"]["matlab_source_root"]),
        output_root=_expand_path(raw["paths"]["output_root"]),
        figure_output_dir=_expand_path(raw["paths"]["figure_output_dir"]),
        cache_dir=_expand_path(raw["paths"]["cache_dir"]),
        log_dir=_expand_path(raw["paths"]["log_dir"]),
    )
    return ProjectConfig(paths=paths, analysis=raw["analysis"], runtime=raw["runtime"])
