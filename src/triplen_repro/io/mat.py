from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import scipy.io as sio


def is_hdf5_mat(path: str | Path) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False


def load_mat(path: str | Path, simplify_cells: bool = True) -> dict[str, Any]:
    path = Path(path)
    if is_hdf5_mat(path):
        raise ValueError(f"{path} is MATLAB v7.3/HDF5; use h5py-based access")
    return sio.loadmat(path, simplify_cells=simplify_cells)


def whosmat(path: str | Path) -> list[tuple[str, tuple[int, ...], str]]:
    return list(sio.whosmat(str(path)))
