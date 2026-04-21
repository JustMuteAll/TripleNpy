from __future__ import annotations

import numpy as np


def sample_std(data: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    return np.nanstd(data, axis=axis, ddof=1, keepdims=keepdims)


def sample_var(data: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    return np.nanvar(data, axis=axis, ddof=1, keepdims=keepdims)


def zscore_rows(data: np.ndarray, ddof: int = 1) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    mean = np.nanmean(data, axis=1, keepdims=True)
    std = np.nanstd(data, axis=1, ddof=ddof, keepdims=True)
    std[~np.isfinite(std)] = 1.0
    std[std == 0] = 1.0
    return (data - mean) / std


def nansem(data: np.ndarray, axis: int = 0) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    valid = np.sum(~np.isnan(data), axis=axis)
    valid = np.maximum(valid, 1)
    return np.nanstd(data, axis=axis, ddof=1) / np.sqrt(valid)


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])
