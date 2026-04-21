from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triplen_repro.config import ProjectConfig
from triplen_repro.data_layout import resolve_layout
from triplen_repro.io.dataset import load_h5_session_info


@dataclass(slots=True)
class BasicInfoResult:
    session_labels: list[str]
    trial_counts: np.ndarray
    image_degrees: np.ndarray


def compute_basic_info(config: ProjectConfig) -> BasicInfoResult:
    _ = resolve_layout(config)
    session_labels: list[str] = []
    trial_counts = []
    image_degrees = []
    it_sessions = list(range(1, config.analysis["basic_it_sessions_end"] + 1)) + list(config.analysis["basic_extra_it_sessions"])
    for ss in it_sessions:
        info = load_h5_session_info(config, ss)
        meta = info["meta_data"]
        trial_ml = info["trial_ML"]
        session_labels.append(f"ses{ss:02d}")
        onset_time = np.asarray(meta["onset_time_ms"] if isinstance(meta, dict) else meta.onset_time_ms).reshape(-1)
        trial_counts.append(onset_time.size)
        first_trial = trial_ml[0] if isinstance(trial_ml, np.ndarray) else trial_ml
        if isinstance(first_trial, list):
            first_trial = first_trial[0]
        if hasattr(first_trial, "VariableChanges"):
            variable_changes = first_trial.VariableChanges
        elif isinstance(first_trial, dict):
            variable_changes = first_trial["VariableChanges"]
        else:
            variable_changes = first_trial["VariableChanges"]
        if isinstance(variable_changes, list):
            variable_changes = variable_changes[0]
        if isinstance(variable_changes, dict):
            degree = variable_changes["img_degree_h"]
        else:
            degree = variable_changes.img_degree_h
        image_degrees.append(float(np.asarray(degree).reshape(-1)[0]))
    return BasicInfoResult(session_labels=session_labels, trial_counts=np.asarray(trial_counts), image_degrees=np.asarray(image_degrees))
