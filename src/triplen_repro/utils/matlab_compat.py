from __future__ import annotations

import numpy as np


def matlab_to_python_index(values: np.ndarray) -> np.ndarray:
    return np.asarray(values) - 1


def python_to_matlab_index(values: np.ndarray) -> np.ndarray:
    return np.asarray(values) + 1


def matlab_inclusive_slice(start_1based: int, stop_1based: int) -> slice:
    return slice(int(start_1based) - 1, int(stop_1based))


def matlab_window_indices(start_1based: int, stop_1based: int) -> np.ndarray:
    return np.arange(int(start_1based) - 1, int(stop_1based), dtype=int)
