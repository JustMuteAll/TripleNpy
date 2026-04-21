from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def correlation_rdm(data: np.ndarray) -> np.ndarray:
    return squareform(pdist(np.asarray(data), metric="correlation"))
