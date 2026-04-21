from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import subprocess
from typing import Any
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA

from triplen_repro.config import ProjectConfig
from triplen_repro.data_layout import resolve_layout
from triplen_repro.io.mat import load_mat


@dataclass(slots=True)
class SessionPaths:
    session_label: str
    processed_path: Path
    h5_path: Path
    info_path: Path


def _read_excel_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except ImportError:
        try:
            return _read_excel_via_com(path)
        except Exception:
            return _read_excel_via_powershell(path)


def _read_excel_via_com(path: Path) -> pd.DataFrame:
    import win32com.client  # type: ignore

    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    workbook = None
    try:
        workbook = excel.Workbooks.Open(str(path), ReadOnly=True)
        sheet = workbook.Worksheets(1)
        used = sheet.UsedRange
        values = used.Value2
        if values is None:
            return pd.DataFrame()
        if not isinstance(values, tuple):
            values = ((values,),)
        rows = [list(row) if isinstance(row, tuple) else [row] for row in values]
        if not rows:
            return pd.DataFrame()
        header = [str(x) if x is not None else "" for x in rows[0]]
        body = [list(row) for row in rows[1:]]
        return pd.DataFrame(body, columns=header)
    finally:
        if workbook is not None:
            try:
                workbook.Close(False)
            except Exception:
                pass
        try:
            excel.Quit()
        except Exception:
            pass


def _read_excel_via_powershell(path: Path) -> pd.DataFrame:
    command = rf"""
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$wb = $excel.Workbooks.Open('{path}', $null, $true)
$sheet = $wb.Worksheets.Item(1)
$range = $sheet.UsedRange
$lines = New-Object System.Collections.Generic.List[string]
for ($r = 1; $r -le $range.Rows.Count; $r++) {{
    $vals = New-Object System.Collections.Generic.List[string]
    for ($c = 1; $c -le $range.Columns.Count; $c++) {{
        $text = [string]$range.Cells.Item($r, $c).Text
        $text = $text.Replace("`t", " ").Replace("`r", " ").Replace("`n", " ")
        $vals.Add($text)
    }}
    $lines.Add([string]::Join("`t", $vals))
}}
$wb.Close($false)
$excel.Quit()
$lines -join "`n"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
    )
    text = result.stdout.strip()
    if not text:
        return pd.DataFrame()
    return pd.read_csv(StringIO(text), sep="\t")


def _session_number(session_id: str | int) -> int:
    if isinstance(session_id, int):
        return session_id
    return int(str(session_id).lower().replace("ses", ""))


def find_session_paths(config: ProjectConfig, session_id: str | int) -> SessionPaths:
    layout = resolve_layout(config)
    ss = _session_number(session_id)
    processed = sorted(layout.processed_dir.glob(f"Processed_ses{ss:02d}_*.mat"))
    h5_files = sorted(layout.raw_h5_dir.glob(f"ses{ss:02d}_*.h5"))
    info_files = sorted(layout.raw_h5_dir.glob(f"ses{ss:02d}_*_info.mat"))
    if not processed or not h5_files or not info_files:
        raise FileNotFoundError(f"Missing files for session ses{ss:02d}")
    return SessionPaths(f"ses{ss:02d}", processed[0], h5_files[0], info_files[0])


def load_processed_session(config: ProjectConfig, session_id: str | int) -> dict[str, Any]:
    return load_mat(find_session_paths(config, session_id).processed_path)


def load_h5_session(config: ProjectConfig, session_id: str | int, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    path = find_session_paths(config, session_id).h5_path
    wanted = keys or ["LFP_Data", "raster_matrix_img", "response_matrix_img"]
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in wanted:
            out[key] = np.asarray(f[key])
    return out


def to_matlab_h5_axes(array: np.ndarray, key: str) -> np.ndarray:
    array = np.asarray(array)
    if key in {"raster_matrix_img", "response_matrix_img"} and array.ndim == 3:
        return np.transpose(array, (2, 1, 0))
    if key == "LFP_Data" and array.ndim == 3:
        return np.transpose(array, (2, 1, 0))
    return array


def load_h5_session_info(config: ProjectConfig, session_id: str | int) -> dict[str, Any]:
    return load_mat(find_session_paths(config, session_id).info_path)


def load_area_table(config: ProjectConfig) -> pd.DataFrame:
    return _read_excel_with_fallback(resolve_layout(config).others_dir / "exclude_area.xls")


def load_area_xyz(config: ProjectConfig) -> pd.DataFrame:
    return _read_excel_with_fallback(resolve_layout(config).others_dir / "AreaXYZ.xlsx")


def load_image_pool(config: ProjectConfig) -> list[Any]:
    mat = sio.loadmat(resolve_layout(config).others_dir / "img_pool.mat", simplify_cells=True)
    pool = mat["img_pool"]
    if isinstance(pool, np.ndarray):
        return [pool[i] for i in range(pool.shape[0])]
    return list(pool)


def _zip_file_names(zip_path: Path) -> list[str]:
    with ZipFile(zip_path, "r") as zf:
        return zf.namelist()


def _pca_100(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    n_components = int(min(100, data.shape[0], data.shape[1]))
    if n_components < 1:
        raise ValueError("Cannot run PCA on empty model feature array")
    return PCA(n_components=n_components, svd_solver="full", random_state=0).fit_transform(data)


def _load_llm_embeddings(llm_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    embeddings: list[np.ndarray] = []
    names: list[str] = []
    for path in sorted(llm_dir.glob("LLM_*.mat")):
        mat = sio.loadmat(path, simplify_cells=True)
        raw = np.asarray(mat["embeddings"], dtype=float)
        image_id = np.asarray(mat["image_id"]).reshape(-1).astype(int)
        pooled = np.zeros((1000, raw.shape[1]), dtype=float)
        for img_id in range(1, 1001):
            loc = np.flatnonzero(image_id == img_id)
            if loc.size == 0:
                continue
            pooled[img_id - 1] = raw[loc].mean(axis=0)
        embeddings.append(_pca_100(pooled))
        names.append(str(mat["Model_name"]))
    return embeddings, names


def _append_feat_pca(directory: Path, pattern: str, embeddings: list[np.ndarray], names: list[str]) -> None:
    for path in sorted(directory.glob(pattern)):
        mat = sio.loadmat(path, simplify_cells=True)
        if "feat_pca" not in mat:
            continue
        embeddings.append(np.asarray(mat["feat_pca"], dtype=float))
        names.append(path.stem)


def load_model_embeddings(config: ProjectConfig, load_arrays: bool = False, max_models: int | None = None) -> dict[str, Any]:
    layout = resolve_layout(config)
    status = layout.statuses["model_feature"]
    if layout.model_feature_dir is None:
        if status.path and status.path.suffix.lower() == ".zip":
            return {"available": False, "archive_members": _zip_file_names(status.path), "message": status.message}
        return {"available": False, "message": status.message}
    payload: dict[str, Any] = {"available": True, "root": layout.model_feature_dir}
    if not load_arrays:
        return payload

    model_dir = layout.model_feature_dir
    embeddings, names = _load_llm_embeddings(model_dir / "LLM")

    alexnet = sio.loadmat(model_dir / "alexnet_layer_rsp.mat", simplify_cells=True)
    for idx, score in enumerate(alexnet["score"], start=1):
        embeddings.append(np.asarray(score, dtype=float))
        names.append(f"Alex{idx:02d}")

    _append_feat_pca(model_dir / "R50_DINO", "*layer*.mat", embeddings, names)
    _append_feat_pca(model_dir / "R50_DINO", "*avg*.mat", embeddings, names)
    _append_feat_pca(model_dir / "R50_DINO", "*fc*.mat", embeddings, names)
    _append_feat_pca(model_dir / "R50_IN1k", "*layer*.mat", embeddings, names)
    _append_feat_pca(model_dir / "R50_IN1k", "*avg*.mat", embeddings, names)
    _append_feat_pca(model_dir / "R50_IN1k", "*fc*.mat", embeddings, names)
    _append_feat_pca(model_dir / "ViTB16", "*l*.mat", embeddings, names)
    _append_feat_pca(model_dir / "InceptionV3", "*v3*.mat", embeddings, names)

    if max_models is not None:
        embeddings = embeddings[:max_models]
        names = names[:max_models]
    payload["embeddings"] = embeddings
    payload["model_names"] = names
    return payload


def load_fmri_resources(config: ProjectConfig, load_arrays: bool = False, max_voxels: int | None = None) -> dict[str, Any]:
    layout = resolve_layout(config)
    status = layout.statuses["fmri"]
    if layout.fmri_dir is None:
        if status.path and status.path.suffix.lower() == ".zip":
            return {"available": False, "archive_members": _zip_file_names(status.path), "message": status.message}
        return {"available": False, "message": status.message}
    payload: dict[str, Any] = {"available": True, "root": layout.fmri_dir}
    if not load_arrays:
        return payload

    fmri_dir = layout.fmri_dir
    roi_mat = sio.loadmat(fmri_dir / "ROI_data.mat", simplify_cells=True)
    roi_info = sio.loadmat(fmri_dir / "ROI_info.mat", simplify_cells=True)
    roi_data = np.asarray(roi_mat["ROI_data"], dtype=object)
    roi_names = [str(x) for x in np.asarray(roi_mat["all_interested_roi"]).reshape(-1)]
    subjects = [1, 2, 5, 7]
    hemispheres = ["lh", "rh"]

    rows: list[np.ndarray] = []
    voxel_subject: list[int] = []
    voxel_roi: list[str] = []
    voxel_hemi: list[str] = []
    for roi_idx, roi_name in enumerate(roi_names):
        for hemi_idx, hemi in enumerate(hemispheres):
            for subject in subjects:
                values = np.asarray(roi_data[subject - 1, roi_idx, hemi_idx], dtype=float)
                if values.size == 0:
                    continue
                values = np.atleast_2d(values)
                rows.extend([row for row in values])
                voxel_subject.extend([subject] * values.shape[0])
                voxel_roi.extend([roi_name] * values.shape[0])
                voxel_hemi.extend([hemi] * values.shape[0])

    for hemi in hemispheres:
        for subject in subjects:
            rsp = sio.loadmat(fmri_dir / f"S{subject}_{hemi}_Rsp.mat", simplify_cells=True)
            evc_key = f"S{subject}_{hemi}_EVC"
            mask = np.asarray(roi_info[evc_key]).reshape(-1) > 0
            evc_rows = np.asarray(rsp["mean_brain_data"], dtype=float)[mask]
            if evc_rows.size == 0:
                continue
            rows.extend([row for row in evc_rows])
            voxel_subject.extend([subject] * evc_rows.shape[0])
            voxel_roi.extend(["EVC"] * evc_rows.shape[0])
            voxel_hemi.extend([hemi] * evc_rows.shape[0])

    response = np.asarray(rows, dtype=float)
    if max_voxels is not None:
        response = response[:max_voxels]
        voxel_subject = voxel_subject[:max_voxels]
        voxel_roi = voxel_roi[:max_voxels]
        voxel_hemi = voxel_hemi[:max_voxels]

    response = response / 300.0
    mean = response.mean(axis=1, keepdims=True)
    std = response.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    response = (response - mean) / std

    payload["response"] = response
    payload["voxel_subject"] = np.asarray(voxel_subject)
    payload["voxel_roi"] = np.asarray(voxel_roi, dtype=object)
    payload["voxel_hemi"] = np.asarray(voxel_hemi, dtype=object)
    return payload
