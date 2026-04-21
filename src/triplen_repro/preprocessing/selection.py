from __future__ import annotations

import numpy as np
import pandas as pd


def valid_trial_mask(info: dict) -> np.ndarray:
    meta = info["meta_data"]
    dataset_valid_idx = np.asarray(meta["dataset_valid_idx"] if isinstance(meta, dict) else meta.dataset_valid_idx).reshape(-1)
    return dataset_valid_idx > 0


def select_units_for_area(processed: dict, area_row: pd.Series, reliability_threshold: float = 0.4) -> np.ndarray:
    pos = np.asarray(processed["pos"]).reshape(-1)
    reliability = np.asarray(processed["reliability_best"]).reshape(-1)
    return np.flatnonzero((pos > float(area_row["y1"])) & (pos < float(area_row["y2"])) & (reliability > reliability_threshold))
