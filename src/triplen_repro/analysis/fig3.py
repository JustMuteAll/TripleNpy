from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

from triplen_repro.config import ProjectConfig
from triplen_repro.io.dataset import (
    load_area_table,
    find_session_paths,
    load_h5_session,
    load_h5_session_info,
    load_image_pool,
    load_processed_session,
    to_matlab_h5_axes,
)
from triplen_repro.utils.stats import nansem, zscore_rows


GROUPED_PREFIX = ["MB", "AB", "MF", "AF", "MO", "AO", "LPP", "PITP", "CLC", "AMC", "Unknown"]
GROUPED_NAME = ["MiddleBody", "AnteriorBody", "MiddleFace", "AnteriorFace", "MiddleObject", "AnteriorObject", "Scene1", "Scene2", "MiddleColor", "AnteriorColor", "Unknown"]
PREF_AREA_PREFIX = ["MO", "AO", "MF", "AF", "MB", "AB", "LP", "PI", "CL", "AM"]
PREF_AREA_NAME = ["M-Object", "A-Object", "M-Face", "A-Face", "M-Body", "A-Body", "Scene1", "Scene2", "M-Color", "A-Color"]
AREA_EXCLUDE = {18, 19, 21, 79}


@dataclass(slots=True)
class AreaRecord:
    area_idx: int
    session_idx: int
    area_label: str
    display_area: str
    y1: float
    y2: float
    reliability: np.ndarray
    response_best: np.ndarray
    unit_type: np.ndarray
    mean_psth: np.ndarray
    psth_pool: np.ndarray
    pos: np.ndarray
    lfp_depth: np.ndarray
    pre_onset: int
    trial_image_count: int
    h5_path: Path


@dataclass(slots=True)
class AreaDiagnostic:
    area_idx: int
    area_label: str
    session_idx: int
    pos_area: np.ndarray
    idx_area: np.ndarray
    depth_edges: np.ndarray
    lfp_time: np.ndarray
    lfp_depth: np.ndarray
    lfp_matrix: np.ndarray
    mi_observed: float
    mi_permutation_median: float


@dataclass(slots=True)
class ClusteringResult:
    all_cluster: np.ndarray
    area_clusters: dict[int, np.ndarray]
    clus_save: dict[int, np.ndarray]
    all_mean_psth: np.ndarray
    selected_units: np.ndarray
    selected_mean_psth: np.ndarray
    sorted_order: np.ndarray
    cluster_boundaries: np.ndarray
    cluster_mean_psth: np.ndarray
    cluster_ci_psth: np.ndarray
    cluster_reliability: list[np.ndarray]
    group_names: list[str]
    group_display_names: list[str]
    group_ratios: np.ndarray
    group_index_all: np.ndarray
    reliability_all: np.ndarray
    brain_area_all: np.ndarray
    pos_all: np.ndarray
    area_records: dict[int, AreaRecord]
    psth_all: np.ndarray
    psth_even: np.ndarray
    psth_odd: np.ndarray
    temporal_similarity: list[np.ndarray]
    mi_observed: np.ndarray
    mi_permutation: np.ndarray
    area_diagnostics: list[AreaDiagnostic]
    example_area: AreaDiagnostic | None
    db_k_list: np.ndarray
    db_scores: np.ndarray
    db_optimal_k: int
    comparison: dict[str, object]


@dataclass(slots=True)
class ImageWiseResult:
    area_label: str
    cluster_ids: tuple[int, int]
    cluster_results: dict[int, dict[str, Any]]
    representative_tiles: np.ndarray
    dnn_layer_names: list[str]
    dnn_correlations: np.ndarray
    comparison: dict[str, object]


@dataclass(slots=True)
class PreferencePanelResult:
    area_names: list[str]
    cluster_ids: list[int]
    panel_data: dict[tuple[int, int], dict[str, Any]]
    comparison: dict[str, object]


def _meta_field(meta: dict | object, key: str) -> np.ndarray:
    if isinstance(meta, dict):
        return np.asarray(meta[key])
    return np.asarray(getattr(meta, key))


def _prefix_match(label: str, prefix: str) -> bool:
    return label == prefix or label.startswith(prefix)


def _remap_clusters(idx: np.ndarray) -> np.ndarray:
    out = idx.copy()
    out[idx == 1] = 4
    out[idx == 2] = 1
    out[out == 4] = 2
    return out


def _matlab_time_index(pre_onset: int, points: np.ndarray) -> np.ndarray:
    return pre_onset + np.asarray(points, dtype=int) - 1


def _safe_mean_rows(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return np.empty((0, data.shape[-1]), dtype=float)
    return np.mean(data, axis=1)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _spearman_corr_matrix(data: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(data, axis=1), axis=1).astype(float)
    ranks = zscore_rows(ranks, ddof=1)
    corr = np.corrcoef(ranks)
    corr[~np.isfinite(corr)] = 0.0
    return corr


def _mutual_info_cluster_position(cluster_idx: np.ndarray, pos: np.ndarray, n_bins: int = 6) -> float:
    cluster_idx = np.asarray(cluster_idx, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float).reshape(-1)
    if cluster_idx.size < 2 or pos.size < 2:
        return float("nan")
    x_edges = np.linspace(0.5, 3.5, 4)
    y_edges = np.linspace(float(np.nanmin(pos)), float(np.nanmax(pos)), n_bins + 1)
    bins, _, _ = np.histogram2d(cluster_idx, pos, bins=[x_edges, y_edges])
    bins = bins + 0.5
    total = np.sum(bins)
    pxy = bins / total
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    independent = px @ py
    return float(np.sum(pxy * np.log2(pxy / independent)))


def _alexnet_prediction_scores(feature_map: dict[str, np.ndarray], analysis_image: np.ndarray, target_latency: np.ndarray) -> tuple[list[str], np.ndarray]:
    layer_names = ["conv5", "fc6", "fc7", "fc8", "output"]
    layer_keys = ["cv5_data", "fc6_4096", "fc7_4096", "fc8_4096", "softmax_1000"]
    scores = []
    for key in layer_keys:
        rsp_here = np.asarray(feature_map[key], dtype=float)[:, analysis_image]
        pc_score = PCA(n_components=min(5, rsp_here.shape[1], rsp_here.shape[0]), svd_solver="full").fit_transform(rsp_here.T)
        predicted = np.zeros(target_latency.size, dtype=float)
        for uu in range(target_latency.size):
            train_set = np.setdiff1d(np.arange(target_latency.size), np.array([uu]), assume_unique=True)
            design = np.column_stack([pc_score[train_set], np.ones(train_set.size, dtype=float)])
            coeff, *_ = np.linalg.lstsq(design, target_latency[train_set], rcond=None)
            predicted[uu] = np.dot(np.append(pc_score[uu], 1.0), coeff)
        order_a = np.argsort(np.argsort(target_latency))
        order_b = np.argsort(np.argsort(predicted))
        scores.append(_safe_corr(order_a, order_b))
    return layer_names, np.asarray(scores, dtype=float)


def _load_alexnet_resp(config: ProjectConfig) -> dict[str, np.ndarray]:
    path = config.paths.dataset_root / "others" / "ModelFeature" / "alexnet_resp.mat"
    mat = sio.loadmat(path, simplify_cells=True)
    return {key: np.asarray(value) for key, value in mat.items() if not key.startswith("__")}


def _build_area_records(config: ProjectConfig) -> list[AreaRecord]:
    manual = load_area_table(config).reset_index(drop=True)
    interested_time_point = np.arange(config.analysis["figure3_time_points_start"], config.analysis["figure3_time_points_stop"] + 1, dtype=int)
    rdm_stop = max(350, int(config.analysis["figure3_time_points_stop"]) + 50)
    interested_time_point_rdm = np.arange(config.analysis["figure3_time_points_start"], rdm_stop + 1, int(config.analysis["figure3_rdm_step"]), dtype=int)

    area_records: list[AreaRecord] = []
    for area_now, row in manual.iterrows():
        if str(row["Area"]) == "EVC":
            continue
        ses_idx = int(row["SesIdx"])
        processed = load_processed_session(config, ses_idx)
        info = load_h5_session_info(config, ses_idx)
        h5 = load_h5_session(config, ses_idx, keys=["response_matrix_img"])
        response = to_matlab_h5_axes(h5["response_matrix_img"], "response_matrix_img")
        session_paths = find_session_paths(config, ses_idx)
        pre_onset = int(_meta_field(info["global_params"], "pre_onset").reshape(-1)[0])
        unit_here = np.flatnonzero(
            (np.asarray(processed["pos"]).reshape(-1) > float(row["y1"]))
            & (np.asarray(processed["pos"]).reshape(-1) < float(row["y2"]))
        )
        if unit_here.size == 0:
            continue
        lfp_meta = info["LFP_META"]
        depth_vals = np.asarray(lfp_meta["depth_vals"] if isinstance(lfp_meta, dict) else getattr(lfp_meta, "depth_vals")).reshape(-1)
        area_records.append(
            AreaRecord(
                area_idx=area_now + 1,
                session_idx=ses_idx,
                area_label=str(row["AREALABEL"]),
                display_area=str(row["Area"]),
                y1=float(row["y1"]),
                y2=float(row["y2"]),
                reliability=np.asarray(processed["reliability_best"]).reshape(-1)[unit_here],
                response_best=np.asarray(processed["response_best"], dtype=float)[unit_here][:, :1000],
                unit_type=np.asarray(processed["UnitType"]).reshape(-1)[unit_here],
                mean_psth=np.asarray(processed["mean_psth"], dtype=float)[unit_here][:, _matlab_time_index(pre_onset, interested_time_point)],
                psth_pool=response[unit_here][:, :1000, _matlab_time_index(pre_onset, interested_time_point_rdm)],
                pos=np.asarray(processed["pos"]).reshape(-1)[unit_here],
                lfp_depth=depth_vals,
                pre_onset=pre_onset,
                trial_image_count=response.shape[1],
                h5_path=session_paths.h5_path,
            )
        )
    return area_records


def compute_psth_clusters(config: ProjectConfig) -> ClusteringResult:
    area_records = _build_area_records(config)
    if not area_records:
        raise RuntimeError("No PSTH data found for Figure 3 clustering")

    reliability_here = []
    all_mean_psth = []
    brain_area = []
    group_index = []
    pos_all = []
    area_by_idx = {record.area_idx: record for record in area_records}

    for aa, prefix in enumerate(GROUPED_PREFIX, start=1):
        for record in area_records:
            if _prefix_match(record.area_label, prefix):
                reliability_here.append(record.reliability)
                all_mean_psth.append(record.mean_psth)
                brain_area.append(np.full(record.reliability.size, record.area_idx, dtype=int))
                group_index.append(np.full(record.reliability.size, aa, dtype=int))
                pos_all.append(record.pos)

    reliability_vec = np.concatenate(reliability_here)
    all_mean_psth_vec = np.concatenate(all_mean_psth, axis=0)
    brain_area_vec = np.concatenate(brain_area)
    group_index_vec = np.concatenate(group_index)
    pos_vec = np.concatenate(pos_all)
    selected = np.flatnonzero(reliability_vec > config.analysis["reliability_threshold"])
    selected_mean_psth = zscore_rows(all_mean_psth_vec[selected], ddof=1)
    model = KMeans(
        n_clusters=config.analysis["clustering_k"],
        random_state=config.analysis["clustering_random_seed"],
        n_init=20,
    )
    idx = _remap_clusters(model.fit_predict(selected_mean_psth) + 1)

    all_cluster = np.zeros(reliability_vec.shape[0], dtype=int)
    all_cluster[selected] = idx
    area_clusters: dict[int, np.ndarray] = {}
    clus_save: dict[int, np.ndarray] = {}
    for area in np.unique(brain_area_vec):
        area_loc = np.flatnonzero(brain_area_vec == area)
        area_clusters[int(area)] = all_cluster[area_loc]
        clus_save[int(area)] = all_cluster[area_loc]

    sorted_cluster = np.argsort(idx, kind="stable")
    cluster_boundaries = np.cumsum([np.sum(idx == cc) for cc in range(1, config.analysis["clustering_k"] + 1)])[:-1]

    cluster_mean_psth = []
    cluster_ci_psth = []
    cluster_reliability = []
    temporal_similarity = []
    for cc in range(1, config.analysis["clustering_k"] + 1):
        cluster_data_chunks = []
        cluster_reliability_chunks = []
        for area_idx, area_cluster in area_clusters.items():
            cluster_mask = area_cluster > 0
            if not np.any(cluster_mask):
                continue
            cluster_area_labels = area_cluster[cluster_mask]
            cluster_unit_mask = cluster_area_labels == cc
            if not np.any(cluster_unit_mask):
                continue
            record = area_by_idx[area_idx]
            reliable_unit_mask = record.reliability > config.analysis["reliability_threshold"]
            cluster_data_chunks.append(_safe_mean_rows(record.psth_pool[reliable_unit_mask][cluster_unit_mask]))
            cluster_reliability_chunks.append(record.reliability[reliable_unit_mask][cluster_unit_mask])
        cluster_data = np.concatenate(cluster_data_chunks, axis=0) if cluster_data_chunks else np.empty((0, 70), dtype=float)
        cluster_reliability.append(np.concatenate(cluster_reliability_chunks) if cluster_reliability_chunks else np.empty(0, dtype=float))
        cluster_data = zscore_rows(cluster_data, ddof=1)
        cluster_mean_psth.append(np.nanmean(cluster_data, axis=0))
        cluster_ci_psth.append(1.96 * nansem(cluster_data, axis=0))

        rdm_vec = []
        time_points = min(70, next(iter(area_by_idx.values())).psth_pool.shape[2])
        for t1 in range(time_points):
            rsp_chunks = []
            for area_idx, area_cluster in area_clusters.items():
                cluster_mask = area_cluster > 0
                if not np.any(cluster_mask):
                    continue
                cluster_area_labels = area_cluster[cluster_mask]
                cluster_unit_mask = cluster_area_labels == cc
                if not np.any(cluster_unit_mask):
                    continue
                record = area_by_idx[area_idx]
                reliable_unit_mask = record.reliability > config.analysis["reliability_threshold"]
                rsp_chunks.append(record.psth_pool[reliable_unit_mask][cluster_unit_mask, :, t1])
            rsp1 = zscore_rows(np.concatenate(rsp_chunks, axis=0), ddof=1)
            rdm1 = squareform(pdist(rsp1.T, metric="correlation"))
            tril = rdm1[np.tril_indices_from(rdm1)]
            rdm_vec.append(tril)
        rdm_vec_arr = np.asarray(rdm_vec, dtype=float)
        temporal_similarity.append(_spearman_corr_matrix(rdm_vec_arr))

    cluster_mean_psth_arr = np.vstack(cluster_mean_psth)
    cluster_ci_psth_arr = np.vstack(cluster_ci_psth)

    group_ratios = np.zeros((len(GROUPED_PREFIX), config.analysis["clustering_k"]), dtype=float)
    for aa in range(1, len(GROUPED_PREFIX) + 1):
        group_loc = np.flatnonzero(group_index_vec[selected] == aa)
        if group_loc.size == 0:
            continue
        for cc in range(1, config.analysis["clustering_k"] + 1):
            group_ratios[aa - 1, cc - 1] = float(np.mean(idx[group_loc] == cc))

    area_diagnostics: list[AreaDiagnostic] = []
    mi_observed = np.full(max(area_by_idx) + 1, np.nan, dtype=float)
    mi_perm = np.full(max(area_by_idx) + 1, np.nan, dtype=float)
    example_area: AreaDiagnostic | None = None
    interested_time_point = np.arange(
        config.analysis["figure3_time_points_start"],
        config.analysis["figure3_time_points_stop"] + 1,
        dtype=int,
    )
    for area_idx in range(1, max(area_by_idx) + 1):
        if area_idx in AREA_EXCLUDE or area_idx not in area_by_idx:
            continue
        record = area_by_idx[area_idx]
        if record.display_area == "EVC":
            continue
        location_here = np.flatnonzero(brain_area_vec[selected] == area_idx)
        if location_here.size == 0:
            continue
        pos_area = pos_vec[selected[location_here]]
        idx_area = idx[location_here]
        depth_edges = np.arange(np.min(pos_area), np.max(pos_area) + 150, 150, dtype=float)
        lfp_points = np.unique(np.floor((_matlab_time_index(record.pre_onset, interested_time_point) + 1) / 2).astype(int) - 1)
        with h5py.File(record.h5_path, "r") as f:
            lfp_ds = f["LFP_Data"]
            lfp_points = lfp_points[(lfp_points >= 0) & (lfp_points < lfp_ds.shape[0])]
            lfp_slice = np.asarray(lfp_ds[lfp_points, :, :1000])
        lfp_matrix = np.squeeze(np.mean(np.transpose(lfp_slice, (2, 1, 0)), axis=0)).T
        mi_here = _mutual_info_cluster_position(idx_area, pos_area)
        perm = []
        rng = np.random.default_rng(1009 + area_idx)
        for _ in range(100):
            perm.append(_mutual_info_cluster_position(idx_area, rng.permutation(pos_area)))
        mi_perm_here = float(np.nanmedian(np.asarray(perm, dtype=float)))
        diagnostic = AreaDiagnostic(
            area_idx=area_idx,
            area_label=record.area_label,
            session_idx=record.session_idx,
            pos_area=pos_area,
            idx_area=idx_area,
            depth_edges=depth_edges,
            lfp_time=np.arange(0, 2 * lfp_points.size, 2, dtype=float)[: lfp_matrix.shape[0]],
            lfp_depth=record.lfp_depth,
            lfp_matrix=lfp_matrix,
            mi_observed=mi_here,
            mi_permutation_median=mi_perm_here,
        )
        area_diagnostics.append(diagnostic)
        mi_observed[area_idx] = mi_here
        mi_perm[area_idx] = mi_perm_here
        example_prefix = str(config.analysis["figure3_example_area_prefix"])
        if example_area is None and diagnostic.idx_area.size > 0 and diagnostic.area_label.startswith(example_prefix):
            example_area = diagnostic
        elif example_area is None and diagnostic.idx_area.size > 0:
            example_area = diagnostic

    db_k_list = np.arange(1, 9, dtype=int)
    db_scores = np.full(db_k_list.shape, np.nan, dtype=float)
    for idx_k, kk in enumerate(db_k_list):
        if kk < 2:
            continue
        kk_model = KMeans(n_clusters=int(kk), random_state=config.analysis["clustering_random_seed"], n_init=20)
        labels = kk_model.fit_predict(selected_mean_psth)
        db_scores[idx_k] = davies_bouldin_score(selected_mean_psth, labels)
    finite_loc = np.flatnonzero(np.isfinite(db_scores))
    db_optimal_k = int(db_k_list[finite_loc[np.argmin(db_scores[finite_loc])]]) if finite_loc.size else int(config.analysis["clustering_k"])

    comparison = {
        "selected_unit_count": int(selected.size),
        "cluster_counts": {f"cluster_{cc}": int(np.sum(idx == cc)) for cc in range(1, config.analysis["clustering_k"] + 1)},
        "area_count": int(len(area_clusters)),
        "group_names": GROUPED_PREFIX,
        "example_area": int(example_area.area_idx) if example_area is not None else None,
        "db_optimal_k": db_optimal_k,
    }
    return ClusteringResult(
        all_cluster=all_cluster,
        area_clusters=area_clusters,
        clus_save=clus_save,
        all_mean_psth=all_mean_psth_vec,
        selected_units=selected,
        selected_mean_psth=selected_mean_psth,
        sorted_order=sorted_cluster,
        cluster_boundaries=cluster_boundaries,
        cluster_mean_psth=cluster_mean_psth_arr,
        cluster_ci_psth=cluster_ci_psth_arr,
        cluster_reliability=cluster_reliability,
        group_names=GROUPED_PREFIX,
        group_display_names=GROUPED_NAME,
        group_ratios=group_ratios,
        group_index_all=group_index_vec,
        reliability_all=reliability_vec,
        brain_area_all=brain_area_vec,
        pos_all=pos_vec,
        area_records=area_by_idx,
        psth_all=np.empty((0, 0, 0), dtype=float),
        psth_even=np.empty((0, 0, 0), dtype=float),
        psth_odd=np.empty((0, 0, 0), dtype=float),
        temporal_similarity=temporal_similarity,
        mi_observed=mi_observed,
        mi_permutation=mi_perm,
        area_diagnostics=area_diagnostics,
        example_area=example_area,
        db_k_list=db_k_list,
        db_scores=db_scores,
        db_optimal_k=db_optimal_k,
        comparison=comparison,
    )


def compute_imagewise_analysis(config: ProjectConfig, clustering: ClusteringResult) -> ImageWiseResult:
    manual = load_area_table(config).reset_index(drop=True)
    area_target = str(config.analysis["figure3_imagewise_area"])
    imagewise_cluster = int(config.analysis["figure3_imagewise_cluster"])
    interested_time_point = np.arange(1, 401, dtype=int)
    img_pool = load_image_pool(config)
    feature_map = _load_alexnet_resp(config)

    cluster_results: dict[int, dict[str, Any]] = {}
    latency_save: dict[tuple[int, int], np.ndarray] = {}

    for interested_unit in (1, imagewise_cluster):
        psth_combined = []
        ses_number = []
        avg_rsp = []
        avg_rsp_by_session: dict[int, np.ndarray] = {}
        for _, row in manual.iterrows():
            if str(row["AREALABEL"]) != area_target:
                continue
            area_idx = int(row.name) + 1
            ses_idx = int(row["SesIdx"])
            processed = load_processed_session(config, ses_idx)
            info = load_h5_session_info(config, ses_idx)
            response = to_matlab_h5_axes(load_h5_session(config, ses_idx, keys=["response_matrix_img"])["response_matrix_img"], "response_matrix_img")
            x1 = float(row["y1"])
            x2 = float(row["y2"])
            good_neuron_idx = np.flatnonzero(
                (np.asarray(processed["pos"]).reshape(-1) > x1)
                & (np.asarray(processed["pos"]).reshape(-1) < x2)
                & (np.asarray(processed["reliability_best"]).reshape(-1) > config.analysis["reliability_threshold"])
            )
            area_cluster = clustering.clus_save.get(area_idx, np.empty(0, dtype=int))
            area_cluster = area_cluster[area_cluster > 0]
            if good_neuron_idx.size == 0 or area_cluster.size == 0:
                continue
            matched = good_neuron_idx[area_cluster == interested_unit]
            if matched.size == 0:
                continue
            pre_onset = int(_meta_field(info["global_params"], "pre_onset").reshape(-1)[0])
            data_here = response[matched][:, :, _matlab_time_index(pre_onset, interested_time_point)]
            psth_combined.append(data_here)
            ses_number.extend([area_idx] * matched.size)
            avg_rsp.append(
                np.asarray(processed["response_best"], dtype=float)[
                    (np.asarray(processed["reliability_best"]).reshape(-1) > config.analysis["reliability_threshold"])
                    & (np.asarray(processed["F_SI"]).reshape(-1) > 0.2),
                    :1000,
                ]
            )
            avg_rsp_by_session[area_idx] = avg_rsp[-1]
        if not psth_combined:
            cluster_results[interested_unit] = {
                "analysis_image": np.empty(0, dtype=int),
                "latency": [],
                "pop_rsp_mean": [],
                "representative_images": np.empty(0, dtype=int),
                "representative_psth_mean": np.empty((0, interested_time_point.size), dtype=float),
                "representative_psth_sem": np.empty((0, interested_time_point.size), dtype=float),
                "session_ids": np.empty(0, dtype=int),
            }
            continue

        psth_combined_arr = np.concatenate(psth_combined, axis=0)
        ses_number_arr = np.asarray(ses_number, dtype=int)
        avg_rsp_arr = zscore_rows(np.concatenate(avg_rsp, axis=0), ddof=1)
        for uu in range(psth_combined_arr.shape[0]):
            maximum = np.nanmax(psth_combined_arr[uu])
            if maximum > 0:
                psth_combined_arr[uu] = psth_combined_arr[uu] / maximum

        analysis_image = np.flatnonzero(np.mean(avg_rsp_arr, axis=0) > 0.7)
        latency = []
        pop_rsp_mean = []
        all_this_ses = np.unique(ses_number_arr)
        for ss, ses_area in enumerate(all_this_ses, start=1):
            pop_rsp = np.squeeze(np.mean(psth_combined_arr[ses_number_arr == ses_area][:, analysis_image, :], axis=0))
            lat = np.argmax(pop_rsp[:, 20:], axis=1) + 21
            latency.append(lat.astype(float))
            latency_save[(ss, interested_unit)] = latency[-1]
            pop_rsp_mean.append(np.mean(np.asarray(avg_rsp_by_session[ses_area], dtype=float)[:, analysis_image], axis=0))

        representative_images = np.empty(0, dtype=int)
        representative_mean = np.empty((0, interested_time_point.size), dtype=float)
        representative_sem = np.empty((0, interested_time_point.size), dtype=float)
        representative_colors = np.empty((0, 3), dtype=float)
        if interested_unit == imagewise_cluster and latency:
            ss = 0
            width = np.flatnonzero(pop_rsp_mean[ss] > 2.3)
            if width.size:
                order = np.argsort(latency[ss][width])[::-1]
                illustration_points = np.r_[np.arange(len(order) - 1, -1, -3), 0]
                illustration_points = np.unique(np.clip(illustration_points, 0, len(order) - 1))
                cm_here = np.flipud(__import__("matplotlib").colormaps["plasma"](np.linspace(40 / 255.0, 200 / 255.0, len(width)))[:, :3])
                representative_images = analysis_image[width[order[illustration_points]]]
                mean_pool = []
                sem_pool = []
                color_pool = []
                for img_idx in representative_images:
                    data = np.squeeze(psth_combined_arr[all_this_ses[ss] == ses_number_arr][:, img_idx, :])
                    mean_pool.append(np.mean(data, axis=0))
                    sem_pool.append(nansem(data, axis=0))
                for point in illustration_points:
                    color_pool.append(cm_here[point])
                representative_mean = np.asarray(mean_pool, dtype=float)
                representative_sem = np.asarray(sem_pool, dtype=float)
                representative_colors = np.asarray(color_pool, dtype=float)

        cluster_results[interested_unit] = {
            "analysis_image": analysis_image,
            "latency": latency,
            "pop_rsp_mean": pop_rsp_mean,
            "representative_images": representative_images,
            "representative_psth_mean": representative_mean,
            "representative_psth_sem": representative_sem,
            "representative_colors": representative_colors,
            "session_ids": all_this_ses,
        }

    target_cluster_result = cluster_results.get(imagewise_cluster, {})
    representative_images = np.asarray(target_cluster_result.get("representative_images", np.empty(0, dtype=int)), dtype=int)
    if representative_images.size:
        representative_tiles = np.stack([np.asarray(img_pool[int(img_idx)], dtype=np.uint8) for img_idx in representative_images], axis=0)
    else:
        representative_tiles = np.empty((0, 0, 0, 3), dtype=np.uint8)
    target_latency = cluster_results.get(imagewise_cluster, {}).get("latency", [])
    if len(target_latency) >= 2:
        target_for_dnn = np.asarray(target_latency[min(1, len(target_latency) - 1)], dtype=float)
    elif len(target_latency) == 1:
        target_for_dnn = np.asarray(target_latency[0], dtype=float)
    else:
        target_for_dnn = np.empty(0, dtype=float)
    analysis_image = np.asarray(cluster_results.get(imagewise_cluster, {}).get("analysis_image", np.empty(0, dtype=int)), dtype=int)
    if target_for_dnn.size and analysis_image.size == target_for_dnn.size:
        dnn_names, dnn_corr = _alexnet_prediction_scores(feature_map, analysis_image, target_for_dnn)
    else:
        dnn_names = ["conv5", "fc6", "fc7", "fc8", "output"]
        dnn_corr = np.full(5, np.nan, dtype=float)

    comparison = {
        "area_label": area_target,
        "cluster_1_session_count": int(len(cluster_results.get(1, {}).get("session_ids", []))),
        "cluster_2_session_count": int(len(cluster_results.get(imagewise_cluster, {}).get("session_ids", []))),
        "analysis_image_count": int(analysis_image.size),
    }
    return ImageWiseResult(
        area_label=area_target,
        cluster_ids=(1, imagewise_cluster),
        cluster_results=cluster_results,
        representative_tiles=representative_tiles,
        dnn_layer_names=dnn_names,
        dnn_correlations=dnn_corr,
        comparison=comparison,
    )


def compute_preference_panel(config: ProjectConfig, clustering: ClusteringResult) -> PreferencePanelResult:
    manual = load_area_table(config).reset_index(drop=True)
    interested_time_point = np.arange(1, 401, dtype=int)
    group_size = int(config.analysis["figure3_preference_group_size"])
    panel_data: dict[tuple[int, int], dict[str, Any]] = {}

    for aa, prefix in enumerate(PREF_AREA_PREFIX):
        for interested_unit in range(1, 4):
            psth_combined = []
            avg_rsp = []
            for _, row in manual.iterrows():
                label = str(row["AREALABEL"])
                if len(label) < 2 or label[:2] != prefix:
                    continue
                area_idx = int(row.name) + 1
                ses_idx = int(row["SesIdx"])
                processed = load_processed_session(config, ses_idx)
                info = load_h5_session_info(config, ses_idx)
                response = to_matlab_h5_axes(load_h5_session(config, ses_idx, keys=["response_matrix_img"])["response_matrix_img"], "response_matrix_img")
                x1 = float(row["y1"])
                x2 = float(row["y2"])
                good_neuron_idx = np.flatnonzero(
                    (np.asarray(processed["pos"]).reshape(-1) > x1)
                    & (np.asarray(processed["pos"]).reshape(-1) < x2)
                    & (np.asarray(processed["reliability_best"]).reshape(-1) > config.analysis["reliability_threshold"])
                )
                area_cluster = clustering.clus_save.get(area_idx, np.empty(0, dtype=int))
                area_cluster = area_cluster[area_cluster > 0]
                if good_neuron_idx.size == 0 or area_cluster.size == 0:
                    continue
                matched = good_neuron_idx[area_cluster == interested_unit]
                if matched.size == 0:
                    continue
                pre_onset = int(_meta_field(info["global_params"], "pre_onset").reshape(-1)[0])
                psth_combined.append(response[matched][:, :, _matlab_time_index(pre_onset, interested_time_point)])
                second_char = prefix[1] if len(prefix) > 1 else ""
                if second_char == "F":
                    si = np.asarray(processed["F_SI"]).reshape(-1)
                elif second_char == "O":
                    si = np.asarray(processed["O_SI"]).reshape(-1)
                elif second_char == "B":
                    si = np.asarray(processed["B_SI"]).reshape(-1)
                else:
                    si = np.asarray(processed["reliability_best"]).reshape(-1)
                avg_rsp.append(
                    np.asarray(processed["response_best"], dtype=float)[
                        (np.asarray(processed["reliability_best"]).reshape(-1) > config.analysis["reliability_threshold"])
                        & (si > 0.2),
                        :1000,
                    ]
                )
            if not psth_combined:
                panel_data[(aa, interested_unit)] = {"count": 0}
                continue

            psth_combined_arr = np.concatenate(psth_combined, axis=0)
            for uu in range(psth_combined_arr.shape[0]):
                data = np.mean(psth_combined_arr[uu, :, :300], axis=0)
                std = np.std(data, ddof=1) if data.size > 1 else 1.0
                if std == 0 or not np.isfinite(std):
                    std = 1.0
                psth_combined_arr[uu, :, :300] = (psth_combined_arr[uu, :, :300] - np.mean(data)) / std

            avg_rsp_arr = zscore_rows(np.concatenate(avg_rsp, axis=0), ddof=1)
            mean_avg = np.mean(avg_rsp_arr, axis=0)
            order = np.argsort(mean_avg)
            low_idx = order[:group_size]
            high_idx = order[-group_size:]
            all_data = np.mean(psth_combined_arr[:, :, :300], axis=0)
            low_data = np.mean(psth_combined_arr[:, low_idx, :300], axis=1)
            high_data = np.mean(psth_combined_arr[:, high_idx, :300], axis=1)
            panel_data[(aa, interested_unit)] = {
                "count": int(psth_combined_arr.shape[0]),
                "overall_mean": np.mean(all_data, axis=0),
                "overall_sem": np.std(all_data, axis=0, ddof=1) / np.sqrt(max(all_data.shape[0], 1)),
                "low_mean": np.mean(low_data, axis=0),
                "low_sem": nansem(low_data, axis=0),
                "high_mean": np.mean(high_data, axis=0),
                "high_sem": nansem(high_data, axis=0),
            }

    comparison = {"panel_count": int(sum(1 for item in panel_data.values() if item.get("count", 0) > 0))}
    return PreferencePanelResult(area_names=PREF_AREA_NAME, cluster_ids=[1, 2, 3], panel_data=panel_data, comparison=comparison)
