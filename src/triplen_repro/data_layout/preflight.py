from __future__ import annotations

from typing import Iterable

from triplen_repro.config import ProjectConfig
from triplen_repro.data_layout.layout import ResolvedLayout, resolve_layout


def _format_status_line(name: str, status: str, message: str) -> str:
    return f"[{status.upper():11s}] {name}: {message}"


def run_preflight(config: ProjectConfig, require: Iterable[str] | None = None) -> tuple[ResolvedLayout, list[str], list[str]]:
    layout = resolve_layout(config)
    required = set(require or ["processed", "raw_h5", "others"])
    lines: list[str] = []
    failures: list[str] = []
    for name, status in layout.statuses.items():
        lines.append(_format_status_line(name, status.status, status.message))
        if name in required and status.status != "available":
            failures.append(name)
    return layout, lines, failures
