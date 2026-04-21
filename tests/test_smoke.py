from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triplen_repro.config import load_config
from triplen_repro.analysis.fig1 import _find_prefix_locations
from triplen_repro.data_layout import run_preflight
from triplen_repro.io.dataset import load_h5_session_info, load_processed_session
from triplen_repro.utils.matlab_compat import matlab_window_indices
from triplen_repro.utils.stats import sample_var, zscore_rows

import numpy as np
import pandas as pd


def test_config_loads() -> None:
    config = load_config(ROOT / "configs" / "default_config.yaml")
    assert config.paths.dataset_root.name == "TripleN"


def test_preflight_core_paths() -> None:
    config = load_config(ROOT / "configs" / "default_config.yaml")
    _, _, failures = run_preflight(config)
    assert "processed" not in failures
    assert "raw_h5" not in failures
    assert "others" not in failures


def test_known_session_loads() -> None:
    config = load_config(ROOT / "configs" / "default_config.yaml")
    processed = load_processed_session(config, "ses01")
    info = load_h5_session_info(config, "ses01")
    assert "response_best" in processed
    assert "img_idx" in info


def test_matlab_window_indices_are_inclusive() -> None:
    idx = matlab_window_indices(4008, 4016)
    assert idx[0] == 4007
    assert idx[-1] == 4015
    assert len(idx) == 9


def test_sample_variance_matches_matlab_ddof1() -> None:
    data = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    out = sample_var(data, axis=1)
    np.testing.assert_allclose(out, np.array([1.0, 4.0]))


def test_zscore_rows_uses_sample_std() -> None:
    data = np.array([[1.0, 2.0, 3.0]])
    out = zscore_rows(data)
    np.testing.assert_allclose(out, np.array([[-1.0, 0.0, 1.0]]) / np.sqrt(1.0), atol=1e-7)


def test_prefix_grouping_handles_unknown_and_prefix_rows() -> None:
    manual = pd.DataFrame({"AREALABEL": ["Unknown", "MB1", "MB2", "AF1"]})
    area_array = np.array([0, 2, 3, 4, 0, 2])
    unknown = _find_prefix_locations(manual, area_array, "Unknown")
    mb = _find_prefix_locations(manual, area_array, "MB")
    np.testing.assert_array_equal(unknown, np.array([0, 4]))
    np.testing.assert_array_equal(mb, np.array([1, 2, 5]))
