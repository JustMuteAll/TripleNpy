from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from triplen_repro.config import ProjectConfig
from triplen_repro.data_layout import resolve_layout
from triplen_repro.io.dataset import load_area_table, load_fmri_resources, load_h5_session, load_h5_session_info, load_model_embeddings, load_processed_session, to_matlab_h5_axes


@dataclass(slots=True)
class Fig5Result:
    status: str
    resources: dict[str, object]
    model_names: list[str]
    session_scores: dict[int, dict[str, np.ndarray]]
    fmri_scores: dict[str, np.ndarray]
    comparison: dict[str, object]


def _zscore_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    std = values.std()
    if std == 0:
        std = 1.0
    return (values - values.mean()) / std


def _zscore_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = values.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (values - values.mean(axis=0, keepdims=True)) / std


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _fit_pls(features: np.ndarray, response: np.ndarray, n_components: int) -> PLSRegression:
    n_components = int(max(1, min(n_components, features.shape[1], features.shape[0] - 1)))
    model = PLSRegression(n_components=n_components)
    model.fit(features, response)
    return model


def _select_components(feature: np.ndarray, response: np.ndarray, max_components: int = 10) -> int:
    response = _zscore_vector(response)
    max_components = int(max(1, min(max_components, feature.shape[1], feature.shape[0] - 1)))
    n_splits = int(max(2, min(20, feature.shape[0] // 5)))
    cv = KFold(n_splits=n_splits, shuffle=False)
    mse = np.full(max_components, np.inf, dtype=float)
    for comp in range(1, max_components + 1):
        pred = np.full(response.shape[0], np.nan, dtype=float)
        for train_idx, test_idx in cv.split(feature):
            model = _fit_pls(feature[train_idx], response[train_idx], comp)
            pred[test_idx] = model.predict(feature[test_idx]).reshape(-1)
        mse[comp - 1] = np.nanmean((response - pred) ** 2)
    return int(np.argmin(mse) + 1)


def _cross_validated_pls(feature: np.ndarray, response: np.ndarray, cv_num: int) -> tuple[float, float, int]:
    response = _zscore_vector(response)
    best_c = _select_components(feature, response)
    n_splits = int(max(2, min(cv_num, feature.shape[0] // 2)))
    cv = KFold(n_splits=n_splits, shuffle=False)
    pred = np.full(response.shape[0], np.nan, dtype=float)
    for train_idx, test_idx in cv.split(feature):
        model = _fit_pls(feature[train_idx], response[train_idx], best_c)
        pred[test_idx] = model.predict(feature[test_idx]).reshape(-1)
    return _safe_corr(response, pred), _r2_score(response, pred), best_c


def _temporal_pls(feature: np.ndarray, response: np.ndarray, component_number: int) -> np.ndarray:
    response = _zscore_matrix(response)
    cv = KFold(n_splits=2, shuffle=False)
    train_idx, test_idx = next(cv.split(feature))
    pred = np.full((test_idx.size, response.shape[1]), np.nan, dtype=float)
    for t in range(response.shape[1]):
        model = _fit_pls(feature[train_idx], response[train_idx, t], component_number)
        pred[:, t] = model.predict(feature[test_idx]).reshape(-1)
    return np.asarray([_safe_corr(response[test_idx, t], pred[:, t]) for t in range(response.shape[1])], dtype=float)


def _run_session_encoding(
    config: ProjectConfig,
    area_idx: int,
    model_features: list[np.ndarray],
    model_names: list[str],
) -> dict[str, np.ndarray]:
    manual = load_area_table(config)
    row = manual.iloc[area_idx - 1]
    ses_idx = int(row["SesIdx"])
    processed = load_processed_session(config, ses_idx)
    info = load_h5_session_info(config, ses_idx)
    response = to_matlab_h5_axes(load_h5_session(config, ses_idx, keys=["response_matrix_img"])["response_matrix_img"], "response_matrix_img")

    pre_onset = int(np.asarray(info["global_params"]["pre_onset"] if isinstance(info["global_params"], dict) else info["global_params"].pre_onset).reshape(-1)[0])
    time_bins = int(config.analysis.get("encoding_time_bins", 350))
    image_limit = int(config.analysis.get("encoding_image_limit", 1000))
    unit_here = np.flatnonzero(
        (np.asarray(processed["pos"]).reshape(-1) > float(row["y1"]))
        & (np.asarray(processed["pos"]).reshape(-1) < float(row["y2"]))
        & (np.asarray(processed["reliability_best"]).reshape(-1) > float(config.analysis["reliability_threshold"]))
    )
    if unit_here.size == 0:
        return {
            "pred_r_array": np.empty((0, len(model_names))),
            "pred_r2_array": np.empty((0, len(model_names))),
            "pred_r2_array_t": np.empty((0, len(model_names), time_bins)),
            "unit_here": unit_here,
            "r_here": np.empty((0,)),
        }

    all_neuron_rsp = np.asarray(processed["response_best"], dtype=float)[unit_here][:, :image_limit]
    psth = response[unit_here][:, :image_limit, pre_onset + np.arange(1, time_bins + 1)]
    pred_r = np.zeros((unit_here.size, len(model_names)), dtype=float)
    pred_r2 = np.zeros((unit_here.size, len(model_names)), dtype=float)
    pred_r2_t = np.zeros((unit_here.size, len(model_names), time_bins), dtype=float)
    for neuron_idx in range(unit_here.size):
        rsp_now = _zscore_vector(all_neuron_rsp[neuron_idx])
        best_numbers = np.zeros(len(model_names), dtype=int)
        for model_idx, feature in enumerate(model_features):
            pred_r[neuron_idx, model_idx], pred_r2[neuron_idx, model_idx], best_numbers[model_idx] = _cross_validated_pls(
                feature[:image_limit], rsp_now, int(config.analysis["encoding_cv_folds"])
            )
        single_neuron_data = np.asarray(psth[neuron_idx], dtype=float)
        for model_idx, feature in enumerate(model_features):
            pred_r2_t[neuron_idx, model_idx] = _temporal_pls(feature[:image_limit], single_neuron_data, int(best_numbers[model_idx]))
    return {
        "pred_r_array": pred_r,
        "pred_r2_array": pred_r2,
        "pred_r2_array_t": pred_r2_t,
        "unit_here": unit_here,
        "r_here": np.asarray(processed["reliability_best"]).reshape(-1)[unit_here],
    }


def _run_fmri_encoding(config: ProjectConfig, model_features: list[np.ndarray], model_names: list[str]) -> dict[str, np.ndarray]:
    fmri = load_fmri_resources(config, load_arrays=True, max_voxels=config.analysis.get("fig5_max_voxels"))
    roi_data = np.asarray(fmri["response"], dtype=float)
    r = np.zeros((roi_data.shape[0], len(model_names)), dtype=float)
    r2 = np.zeros((roi_data.shape[0], len(model_names)), dtype=float)
    for voxel_idx in range(roi_data.shape[0]):
        for model_idx, feature in enumerate(model_features):
            r[voxel_idx, model_idx], r2[voxel_idx, model_idx], _ = _cross_validated_pls(
                feature[: roi_data.shape[1]], roi_data[voxel_idx], int(config.analysis["encoding_cv_folds"])
            )
    return {
        "r": r,
        "r2": r2,
        "voxel_subject": np.asarray(fmri["voxel_subject"]),
        "voxel_roi": np.asarray(fmri["voxel_roi"], dtype=object),
        "voxel_hemi": np.asarray(fmri["voxel_hemi"], dtype=object),
    }


def run_fig5(config: ProjectConfig) -> Fig5Result:
    layout = resolve_layout(config)
    fmri_status = layout.statuses["fmri"].status
    model_status = layout.statuses["model_feature"].status
    if fmri_status != "available" or model_status != "available":
        return Fig5Result(
            status="blocked",
            resources={"fmri_status": fmri_status, "model_feature_status": model_status},
            model_names=[],
            session_scores={},
            fmri_scores={},
            comparison={"blocked_reason": "missing_resources"},
        )

    model_payload = load_model_embeddings(config, load_arrays=True, max_models=config.analysis.get("fig5_max_models"))
    model_features = [np.asarray(x, dtype=float) for x in model_payload["embeddings"]]
    model_names = list(model_payload["model_names"])
    area_indices = config.analysis.get("fig5_area_indices")
    if area_indices is None:
        area_indices = list(range(1, len(load_area_table(config)) + 1))
    area_limit = config.analysis.get("fig5_max_areas")
    if area_limit is not None:
        area_indices = list(area_indices)[: int(area_limit)]

    session_scores: dict[int, dict[str, np.ndarray]] = {}
    for area_idx in area_indices:
        session_scores[int(area_idx)] = _run_session_encoding(config, int(area_idx), model_features, model_names)

    fmri_scores = _run_fmri_encoding(config, model_features, model_names)
    comparison = {
        "model_count": len(model_names),
        "model_names": model_names,
        "session_area_count": len(session_scores),
        "session_unit_counts": {str(k): int(v["unit_here"].size) for k, v in session_scores.items()},
        "fmri_voxel_count": int(np.asarray(fmri_scores["r"]).shape[0]),
    }
    return Fig5Result(
        status="completed",
        resources={
            "fmri_root": str(layout.fmri_dir),
            "model_feature_root": str(layout.model_feature_dir),
        },
        model_names=model_names,
        session_scores=session_scores,
        fmri_scores=fmri_scores,
        comparison=comparison,
    )


def run_fig5_preflight(config: ProjectConfig) -> dict[str, str]:
    layout = resolve_layout(config)
    fmri = load_fmri_resources(config)
    models = load_model_embeddings(config)
    return {
        "fmri_status": layout.statuses["fmri"].status,
        "model_feature_status": layout.statuses["model_feature"].status,
        "fmri_message": fmri["message"] if not fmri["available"] else str(fmri["root"]),
        "model_message": models["message"] if not models["available"] else str(models["root"]),
    }
