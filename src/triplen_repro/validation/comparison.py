from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


def summarize_array(array: Any) -> dict[str, Any]:
    arr = np.asarray(array)
    summary: dict[str, Any] = {"shape": list(arr.shape), "size": int(arr.size), "dtype": str(arr.dtype)}
    if arr.size == 0:
        summary.update({"finite_count": 0, "nan_count": 0})
        return summary
    if np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        values = arr[finite]
        summary["finite_count"] = int(finite.sum())
        summary["nan_count"] = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
        if values.size:
            summary.update(
                {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return summary


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def load_reference_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if path.suffix.lower() == ".mat":
        return sio.loadmat(path, simplify_cells=True)
    raise ValueError(f"Unsupported reference format: {path}")


def compare_arrays(reference: Any, candidate: Any) -> dict[str, Any]:
    ref = np.asarray(reference)
    cur = np.asarray(candidate)
    report = {
        "reference": summarize_array(ref),
        "candidate": summarize_array(cur),
        "shape_match": list(ref.shape) == list(cur.shape),
    }
    if report["shape_match"] and ref.size and cur.size and np.issubdtype(ref.dtype, np.number) and np.issubdtype(cur.dtype, np.number):
        mask = np.isfinite(ref) & np.isfinite(cur)
        if mask.any():
            diff = np.abs(ref[mask] - cur[mask])
            report["mean_abs_diff"] = float(diff.mean())
            report["max_abs_diff"] = float(diff.max())
    return report


def compare_payloads(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for key, value in candidate.items():
        if key not in reference:
            comparisons[key] = {"status": "missing_in_reference"}
            continue
        ref_value = reference[key]
        if isinstance(value, np.ndarray):
            comparisons[key] = compare_arrays(ref_value, value)
            continue
        if isinstance(value, (int, float, str, bool)):
            comparisons[key] = {"reference": ref_value, "candidate": value, "match": ref_value == value}
            continue
        if isinstance(value, (list, tuple)):
            ref_arr = np.asarray(ref_value)
            cand_arr = np.asarray(value)
            comparisons[key] = compare_arrays(ref_arr, cand_arr) if cand_arr.dtype != object else {"reference": ref_value, "candidate": list(value)}
            continue
        if isinstance(value, dict):
            comparisons[key] = compare_payloads(ref_value, value) if isinstance(ref_value, dict) else {"status": "type_mismatch"}
            continue
        comparisons[key] = {"reference": str(ref_value), "candidate": str(value)}
    return comparisons


def build_stage_report(stage: str, payload: dict[str, Any], reference_path: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"stage": stage, "status": "pass", "payload": {}}
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple, str, int, float, bool)) and not isinstance(value, np.ndarray):
            report["payload"][key] = value
        else:
            report["payload"][key] = summarize_array(value)
    if reference_path is None:
        report["reference_status"] = "missing_reference"
        return report
    if not reference_path.exists():
        report["reference_status"] = "missing_reference"
        return report

    reference = load_reference_payload(reference_path)
    comparisons: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in reference:
            comparisons[key] = {"status": "missing_in_reference"}
            report["status"] = "mismatch"
            continue
        if isinstance(value, np.ndarray):
            comparisons[key] = compare_arrays(reference[key], value)
            if not comparisons[key].get("shape_match", True):
                report["status"] = "mismatch"
    report["reference_status"] = "loaded"
    report["comparisons"] = comparisons
    return report
