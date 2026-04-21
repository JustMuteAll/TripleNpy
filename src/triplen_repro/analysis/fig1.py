from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from textwrap import dedent
from typing import Any

import numpy as np
import scipy.io as sio
from matplotlib import colormaps
from scipy.signal import convolve
from scipy.spatial.distance import pdist, squareform

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - exercised in desktop env without nibabel
    nib = None

from triplen_repro.config import ProjectConfig
from triplen_repro.data_layout import resolve_layout
from triplen_repro.io.dataset import (
    load_area_table,
    load_area_xyz,
    load_h5_session,
    load_h5_session_info,
    load_image_pool,
    load_processed_session,
    to_matlab_h5_axes,
)
from triplen_repro.utils.matlab_compat import matlab_window_indices
from triplen_repro.utils.stats import sample_var, zscore_rows
from triplen_repro.validation.comparison import compare_payloads, summarize_array


LABEL_COLORS = {
    "Scene": np.array([200, 20, 200], dtype=np.uint8),
    "Body": np.array([34, 200, 0], dtype=np.uint8),
    "Face": np.array([0, 155, 248], dtype=np.uint8),
    "Object": np.array([251, 117, 0], dtype=np.uint8),
    "Unknown": np.array([128, 128, 128], dtype=np.uint8),
    "Color": np.array([200, 158, 20], dtype=np.uint8),
}
INTERESTED_AREA = ["Unknown", "AMC", "CLC", "PITP", "LPP", "AO", "MO", "AF", "MF", "AB", "MB"]
INTERESTED_NAME = ["Unknown", "AnteriorColor", "MiddleColor", "Scene2", "Scene1", "AnteriorObject", "MiddleObject", "AnteriorFace", "MiddleFace", "AnteriorBody", "MiddleBody"]
SIMILARITY_AREA = ["MB1", "MB2", "MB3", "AB1", "AB3", "MF1", "MF3", "AF1", "AF3", "MO1s1", "MO1s2", "MO2", "MO5", "AO2", "AO5", "LPP4", "PITP3", "PITP4", "CLC3", "AMC3"]
SIMILARITY_NAME = ["MBody-M1", "MBody-M2", "MBody-M3", "ABody-M1", "ABody-M3", "MFace-M1", "MFace-M3", "AFace-M1", "AFace-M3", "MObjectSite1-M1", "MObjectSite2-M1", "MObject-M2", "MObject-M5", "AObject-M2", "AObject-M5", "Place1-M4", "Place2-M3", "Place2-M4", "CentralColor-M3", "AnteriorColor-M3"]


@dataclass(slots=True)
class AnatomyOverview:
    ap_series: np.ndarray
    slice_indices: np.ndarray
    marker_coords: np.ndarray
    marker_tile_coords: np.ndarray
    marker_subjects: np.ndarray
    marker_labels: np.ndarray
    expected_layout: np.ndarray
    tile_images: list[np.ndarray]
    big_image: np.ndarray
    legend_labels: list[str]
    legend_colors: np.ndarray
    comparison: dict[str, Any]


@dataclass(slots=True)
class SessionIllustration:
    image_strip: np.ndarray
    onset_bar: np.ndarray
    onset_times: np.ndarray
    selected_unit_indices: np.ndarray
    selected_spike_traces: list[np.ndarray]
    recruitment_curve: np.ndarray
    unit_colors: np.ndarray
    image_ids: np.ndarray
    time_window: tuple[float, float]
    trial_window: tuple[int, int]
    comparison: dict[str, Any]


@dataclass(slots=True)
class PopulationSummary:
    reliability: np.ndarray
    fano: np.ndarray
    sparseness: np.ndarray
    snrmax: np.ndarray
    area_labels: np.ndarray
    reliable_area_labels: np.ndarray
    broad_area_names: list[str]
    broad_area_display_names: list[str]
    area_similarity: np.ndarray
    similarity_names: list[str]
    area_wise_rsp: np.ndarray
    within_area: np.ndarray
    comparison: dict[str, Any]


def _meta_field(meta: dict | object, key: str) -> np.ndarray:
    if isinstance(meta, dict):
        return np.asarray(meta[key])
    return np.asarray(getattr(meta, key))


def _matlab_row_numbers(frame) -> np.ndarray:
    return np.arange(1, len(frame) + 1, dtype=int)


def _find_prefix_locations(manual, area_array: np.ndarray, prefix: str) -> np.ndarray:
    area_location: list[np.ndarray] = []
    if prefix == "Unknown":
        area_location.append(np.flatnonzero(area_array == 0))
    for row_number, (_, row) in enumerate(manual.iterrows(), start=1):
        label = str(row["AREALABEL"])
        if len(label) >= len(prefix) and label[: len(prefix)] == prefix:
            area_location.append(np.flatnonzero(area_array == row_number))
    if not area_location:
        return np.empty(0, dtype=int)
    return np.concatenate(area_location)


def _colormap_sample(name: str, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 3), dtype=float)
    cmap = colormaps[name]
    if count == 1:
        return np.asarray([cmap(0.5)[:3]], dtype=float)
    idx = np.floor(200 * (np.arange(1, count + 1) / count)).astype(int)
    return np.asarray([cmap(i / 255.0)[:3] for i in idx], dtype=float)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    return image


def _add_marker(image: np.ndarray, x: int, z: int, label: str) -> np.ndarray:
    output = image.copy()
    color = LABEL_COLORS.get(label, np.array([0, 0, 0], dtype=np.uint8))
    height, width = output.shape[:2]
    for size, fill in ((5, np.array([0, 0, 0], dtype=np.uint8)), (4, color)):
        for xx in range(-size, size + 1):
            for zz in range(-size, size + 1):
                xi = x + xx
                zi = z + zz
                if 0 <= xi < height and 0 <= zi < width:
                    output[xi, zi, :] = fill
    return output


def _rotate_and_crop(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    rotated = np.rot90(image, 1)
    valid = np.any(rotated < 250, axis=2)
    row_idx = np.flatnonzero(valid.any(axis=1))
    col_idx = np.flatnonzero(valid.any(axis=0))
    if row_idx.size == 0 or col_idx.size == 0:
        return rotated, 0, 0
    pad = 12
    row0 = max(int(row_idx[0]) - pad, 0)
    row1 = min(int(row_idx[-1]) + pad + 1, rotated.shape[0])
    col0 = max(int(col_idx[0]) - pad, 0)
    col1 = min(int(col_idx[-1]) + pad + 1, rotated.shape[1])
    return rotated[row0:row1, col0:col1, :], col0, row0


def _build_expected_layout(tile_count: int) -> np.ndarray:
    base = np.arange(1, 25, dtype=int).reshape(6, 4, order="F").T
    base[base > tile_count] = 99
    return base


def _tile_mosaic(tiles: list[np.ndarray], layout: np.ndarray) -> np.ndarray:
    if not tiles:
        return np.empty((0, 0, 3), dtype=np.uint8)
    max_h = max(tile.shape[0] for tile in tiles)
    max_w = max(tile.shape[1] for tile in tiles)

    def _pad(tile: np.ndarray) -> np.ndarray:
        padded = np.full((max_h, max_w, tile.shape[2]), 255, dtype=tile.dtype)
        row0 = (max_h - tile.shape[0]) // 2
        col0 = (max_w - tile.shape[1]) // 2
        padded[row0 : row0 + tile.shape[0], col0 : col0 + tile.shape[1], :] = tile
        return padded

    blank = np.full((max_h, max_w, tiles[0].shape[2]), 255, dtype=tiles[0].dtype)
    tile_lookup = {idx + 1: _pad(tile) for idx, tile in enumerate(tiles)}
    rows = []
    for col in range(layout.shape[1]):
        column_tiles = []
        for row in range(layout.shape[0]):
            key = int(layout[row, col])
            column_tiles.append(tile_lookup.get(key, blank))
        rows.append(np.concatenate(column_tiles, axis=0))
    return np.concatenate(rows, axis=1)


def _load_mri_template_arrays(config: ProjectConfig, brain_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    if nib is not None:
        brain_img = nib.load(str(brain_path))
        mask_img = nib.load(str(mask_path))
        brain_data = np.uint8(np.clip(brain_img.get_fdata() / 4.0, 0, 255))
        brain_mask = mask_img.get_fdata() > 0
        affine = np.asarray(brain_img.affine, dtype=float)
        return brain_data, brain_mask, (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))

    cache_path = config.paths.cache_dir / "fig1_mri_template_cache.mat"
    if not cache_path.exists():
        matlab_cmd = dedent(
            f"""
            brain = niftiread('{_matlab_path(brain_path)}');
            mask = niftiread('{_matlab_path(mask_path)}');
            info = niftiinfo('{_matlab_path(brain_path)}');
            x_offset = info.raw.srow_x(end);
            y_offset = info.raw.srow_y(end);
            z_offset = info.raw.srow_z(end);
            save('{_matlab_path(cache_path)}','brain','mask','x_offset','y_offset','z_offset','-v7');
            exit;
            """
        ).strip()
        subprocess.run(["matlab", "-batch", matlab_cmd], check=True, cwd=str(config.paths.matlab_source_root))
    mat = sio.loadmat(cache_path, simplify_cells=True)
    brain_data = np.uint8(np.clip(np.asarray(mat["brain"], dtype=float) / 4.0, 0, 255))
    brain_mask = np.asarray(mat["mask"]) > 0
    offsets = (float(mat["x_offset"]), float(mat["y_offset"]), float(mat["z_offset"]))
    return brain_data, brain_mask, offsets


def build_area_anatomy(config: ProjectConfig) -> AnatomyOverview:
    area_table = load_area_xyz(config)
    layout = resolve_layout(config)
    if layout.mri_template_dir is None:
        raise FileNotFoundError("Missing MRI template directory for Figure 1 anatomy plot")

    brain_path = layout.mri_template_dir / "NMT_v2.1_sym_SS.nii.gz"
    mask_path = layout.mri_template_dir / "NMT_v2.1_sym_brainmask.nii.gz"
    brain_data, brain_mask, (x_offset, y_offset, z_offset) = _load_mri_template_arrays(config, brain_path, mask_path)

    ap_series = np.sort(np.unique(np.asarray(area_table["A"], dtype=float)))[4:]
    tile_images: list[np.ndarray] = []
    slice_indices = []
    marker_records: list[list[int | float]] = []
    marker_tile_records: list[list[int | float]] = []
    marker_subjects: list[int] = []
    marker_labels: list[str] = []

    for ap_now in ap_series:
        area_here = np.flatnonzero(np.asarray(area_table["A"], dtype=float) == ap_now)
        slice_idx = int(np.floor((ap_now - y_offset) / 0.25))
        slice_indices.append(slice_idx)
        slice_img = np.asarray(brain_data[:, slice_idx, :], dtype=np.uint8)
        slice_mask = np.asarray(brain_mask[:, slice_idx, :], dtype=bool)
        slice_img = slice_img.copy()
        slice_img[~slice_mask] = 255
        rgb = _to_rgb(slice_img)
        tile_id = len(tile_images) + 1
        rotated_h = rgb.shape[1]

        for area_idx in area_here:
            label = str(area_table.iloc[area_idx]["Label"])
            lr_here = float(area_table.iloc[area_idx]["R"])
            si_here = float(area_table.iloc[area_idx]["S"])
            lr_coord = int(np.floor((lr_here - x_offset) / 0.25))
            si_coord = int(np.floor((si_here - z_offset) / 0.25))
            rgb = _add_marker(rgb, lr_coord, si_coord, label)
            marker_records.append([tile_id, lr_coord, si_coord])
            tile_x = lr_coord - 9
            tile_y = (rotated_h - 1 - si_coord) - 149
            marker_tile_records.append([tile_id, tile_x, tile_y])
            marker_subjects.append(int(area_table.iloc[area_idx]["Subject"]))
            marker_labels.append(label)

        yy_max = 40 if ap_now < 2.5 else 22
        rgb[94:165, 18:45, :] = 255
        rgb[:, :yy_max, :] = 255
        tile_img, col0, row0 = _rotate_and_crop(rgb)
        tile_images.append(tile_img)

        tile_marker_idx = np.flatnonzero(np.asarray([record[0] for record in marker_records], dtype=int) == tile_id)
        for marker_idx in tile_marker_idx:
            _, lr_coord, si_coord = marker_records[marker_idx]
            tile_x = int(lr_coord) - col0
            tile_y = (rotated_h - 1 - int(si_coord)) - row0
            marker_tile_records[marker_idx][1] = tile_x
            marker_tile_records[marker_idx][2] = tile_y

    expected_layout = _build_expected_layout(len(tile_images))
    big_image = _tile_mosaic(tile_images, expected_layout)
    legend_labels = ["Scene", "Body", "Face", "Object", "Color", "Unknown"]
    legend_colors = np.asarray([LABEL_COLORS[name] for name in legend_labels], dtype=np.uint8)
    comparison = {
        "ap_series": ap_series.tolist(),
        "slice_indices": slice_indices,
        "marker_count": len(marker_records),
        "layout_shape": list(expected_layout.shape),
        "tile_shape": list(tile_images[0].shape) if tile_images else [0, 0, 0],
    }
    return AnatomyOverview(
        ap_series=np.asarray(ap_series, dtype=float),
        slice_indices=np.asarray(slice_indices, dtype=int),
        marker_coords=np.asarray(marker_records, dtype=int),
        marker_tile_coords=np.asarray(marker_tile_records, dtype=int),
        marker_subjects=np.asarray(marker_subjects, dtype=int),
        marker_labels=np.asarray(marker_labels, dtype=object),
        expected_layout=expected_layout,
        tile_images=tile_images,
        big_image=big_image,
        legend_labels=legend_labels,
        legend_colors=legend_colors,
        comparison=comparison,
    )


def build_session_illustration(config: ProjectConfig, session_id: str | int | None = None) -> SessionIllustration:
    session_id = session_id or config.analysis["figure1_session"]
    processed = load_processed_session(config, session_id)
    info = load_h5_session_info(config, session_id)
    images = load_image_pool(config)

    trial_data = _meta_field(info["meta_data"], "trial_valid_idx").reshape(-1).astype(int)
    onset_time = _meta_field(info["meta_data"], "onset_time_ms").reshape(-1).astype(float)
    good_units = np.flatnonzero(
        (np.asarray(processed["reliability_best"]).reshape(-1) > 0.4)
        & (np.asarray(processed["B_SI"]).reshape(-1) > 0.2)
    )
    good_struct = info["GoodUnitStrc"]
    cm_here = _colormap_sample("plasma", len(good_units))

    best_payload: dict[str, Any] | None = None
    for a_start in range(4008, 5001, 7):
        b_stop = a_start + 8
        window_idx = matlab_window_indices(a_start, b_stop)
        img_idx = trial_data[window_idx]
        if img_idx.size != 9:
            continue
        if img_idx.max() >= 1000 or img_idx.min() <= 0 or np.max(np.diff(onset_time[window_idx])) >= 350:
            continue

        time_start = float(onset_time[window_idx[0]])
        time_end = float(onset_time[window_idx[-1]] + 300)
        data_all = np.zeros(int(time_end - time_start), dtype=float)
        traces: list[np.ndarray] = []
        spike_counts = []

        for unit in good_units:
            spikes = _meta_field(good_struct[unit], "spiketime_ms").reshape(-1).astype(float)
            raster_time = spikes[(spikes > time_start) & (spikes < time_end)]
            traces.append(raster_time)
            spike_counts.append(int(raster_time.size))
            time_here = np.floor(raster_time - time_start).astype(int)
            time_here = time_here[(time_here >= 0) & (time_here < data_all.size)]
            d0 = np.zeros_like(data_all)
            d0[time_here] += 1.0
            data_all += convolve(d0, np.ones(30, dtype=float), mode="same")

        recruitment_curve = data_all / max(len(good_units), 1)
        payload = {
            "window_start": a_start,
            "window_end": b_stop,
            "image_ids": img_idx.copy(),
            "time_start": time_start,
            "time_end": time_end,
            "traces": traces,
            "recruitment_curve": recruitment_curve,
            "spike_counts": np.asarray(spike_counts, dtype=int),
        }
        best_payload = payload
        if recruitment_curve.max(initial=0.0) > 0.3:
            break

    if best_payload is None:
        raise RuntimeError("Failed to find a valid Figure 1 illustration window")

    image_strip_parts = []
    spacer = np.full_like(np.asarray(images[0]), 128)
    onset_bar_parts = []
    for image_id in best_payload["image_ids"]:
        image_strip_parts.append(np.asarray(images[image_id - 1]))
        image_strip_parts.append(spacer)
        onset_bar_parts.append(0.8 * np.ones((50, 227, 3), dtype=float))
        onset_bar_parts.append(0.5 * np.ones((50, 227, 3), dtype=float))
    image_strip = np.concatenate(image_strip_parts, axis=1).astype(np.uint8)
    onset_bar = np.concatenate(onset_bar_parts, axis=1)

    dot_center_row = 113
    dot_centers = np.arange(113, image_strip.shape[1], 227, dtype=int)
    for center_col in dot_centers:
        for xx in range(-6, 7):
            for yy in range(-6, 7):
                rr = dot_center_row + xx
                cc = center_col + yy
                if 0 <= rr < image_strip.shape[0] and 0 <= cc < image_strip.shape[1]:
                    image_strip[rr, cc, :] = np.array([255, 0, 0], dtype=np.uint8)

    comparison = {
        "session_id": str(session_id),
        "window_start_trial": int(best_payload["window_start"]),
        "window_end_trial": int(best_payload["window_end"]),
        "good_unit_count": int(good_units.size),
        "image_ids": best_payload["image_ids"].tolist(),
        "spike_counts": best_payload["spike_counts"].tolist(),
        "recruitment_peak": float(best_payload["recruitment_curve"].max(initial=0.0)),
        "time_start": float(best_payload["time_start"]),
        "time_end": float(best_payload["time_end"]),
    }
    onset_idx = matlab_window_indices(best_payload["window_start"], best_payload["window_end"])
    return SessionIllustration(
        image_strip=image_strip,
        onset_bar=onset_bar,
        onset_times=onset_time[onset_idx],
        selected_unit_indices=good_units,
        selected_spike_traces=best_payload["traces"],
        recruitment_curve=best_payload["recruitment_curve"],
        unit_colors=cm_here,
        image_ids=best_payload["image_ids"],
        time_window=(float(best_payload["time_start"]), float(best_payload["time_end"])),
        trial_window=(int(best_payload["window_start"]), int(best_payload["window_end"])),
        comparison=comparison,
    )


def _compute_fano_from_raster(raster: np.ndarray, img_idx: np.ndarray, response_window: np.ndarray) -> np.ndarray:
    raster_sum = np.sum(raster[:, :, response_window], axis=2)
    fano_across_img = np.full((raster_sum.shape[0], 1000), np.nan, dtype=float)
    for img in range(1, 1001):
        trials = np.flatnonzero(img_idx == img)
        rsp_here = raster_sum[:, trials]
        if rsp_here.shape[1] == 0:
            continue
        var_across_trial = sample_var(rsp_here, axis=1)
        mean_across_trial = np.mean(rsp_here, axis=1)
        fano_across_img[:, img - 1] = var_across_trial / np.where(mean_across_trial == 0, np.nan, mean_across_trial)
    return np.nanmean(fano_across_img, axis=1)


def compute_population_summary(config: ProjectConfig) -> PopulationSummary:
    manual = load_area_table(config).reset_index(drop=True)
    reliability_threshold = float(config.analysis.get("reliability_threshold", 0.4))
    image_limit = int(config.analysis.get("population_summary_image_limit", 1000))

    reliability = []
    fanof = []
    sparseness = []
    snr = []
    area_array = []
    rsp_all = []

    for ses_now in range(1, 91):
        processed = load_processed_session(config, ses_now)
        info = load_h5_session_info(config, ses_now)
        raster = to_matlab_h5_axes(load_h5_session(config, ses_now, keys=["raster_matrix_img"])["raster_matrix_img"], "raster_matrix_img")
        unit_size = len(np.asarray(processed["reliability_basic"]).reshape(-1))
        session_area_array = np.zeros(unit_size, dtype=int)

        reliable_mask = np.asarray(processed["reliability_best"]).reshape(-1) > reliability_threshold
        t1 = int(np.nanmedian(np.asarray(processed["best_r_time1"]).reshape(-1)[reliable_mask]))
        t2 = int(np.nanmedian(np.asarray(processed["best_r_time2"]).reshape(-1)[reliable_mask]))
        pre_onset = int(_meta_field(info["global_params"], "pre_onset").reshape(-1)[0])
        response_window = np.arange(pre_onset + t1 - 1, pre_onset + t2, dtype=int)
        fano_here = _compute_fano_from_raster(raster, np.asarray(info["img_idx"]).reshape(-1), response_window)

        session_rows = manual[manual["SesIdx"] == ses_now]
        for row_number, (_, row) in enumerate(session_rows.iterrows(), start=1):
            absolute_row = int(session_rows.index[row_number - 1]) + 1
            if str(row["AREALABEL"]).lower() == "unknown":
                continue
            x1 = float(row["y1"])
            x2 = float(row["y2"])
            unit_here = (np.asarray(processed["pos"]).reshape(-1) > x1) & (np.asarray(processed["pos"]).reshape(-1) < x2)
            session_area_array[unit_here] = absolute_row

        rsp_here = np.asarray(processed["response_best"], dtype=float)[:, :image_limit]
        reliability.append(np.asarray(processed["reliability_best"]).reshape(-1))
        fanof.append(fano_here)
        mean_rsp = np.mean(rsp_here, axis=1) ** 2
        mean_rsp_sq = np.mean(rsp_here**2, axis=1)
        sparseness.append(1.0 - mean_rsp / np.where(mean_rsp_sq == 0, np.nan, mean_rsp_sq))
        snr.append(np.asarray(processed["snrmax"]).reshape(-1))
        area_array.append(session_area_array)
        rsp_all.append(rsp_here)

    reliability = np.concatenate(reliability)
    fanof = np.concatenate(fanof)
    sparseness = np.concatenate(sparseness)
    snr = np.concatenate(snr)
    area_array = np.concatenate(area_array)
    rsp_all = np.concatenate(rsp_all, axis=0)

    reliability_here = []
    area_label = []
    fanof_here = []
    sparseness_vec = []
    snr_here = []
    group_counts: dict[str, int] = {}
    for aa, prefix in enumerate(INTERESTED_AREA, start=1):
        area_location = _find_prefix_locations(manual, area_array, prefix)
        group_counts[prefix] = int(area_location.size)
        reliability_here.append(reliability[area_location])
        area_label.append(np.full(area_location.size, aa, dtype=int))
        reliable_loc = area_location[np.isin(area_location, np.flatnonzero(reliability > reliability_threshold))]
        fanof_here.append(fanof[reliable_loc])
        sparseness_vec.append(sparseness[reliable_loc])
        snr_here.append(snr[reliable_loc])

    area_wise_rsp = []
    within_area = []
    similarity_names = []
    similarity_counts: dict[str, int] = {}
    for prefix, display_name in zip(SIMILARITY_AREA, SIMILARITY_NAME):
        area_location = _find_prefix_locations(manual, area_array, prefix)
        similarity_counts[prefix] = int(area_location.size)
        if area_location.size == 0:
            continue
        area_rsp = zscore_rows(rsp_all[area_location], ddof=1)
        area_wise_rsp.append(np.mean(area_rsp, axis=0))
        odd = area_rsp[0::2]
        even = area_rsp[1::2]
        if odd.size and even.size:
            within_area.append(float(np.corrcoef(np.mean(odd, axis=0), np.mean(even, axis=0))[0, 1]))
        else:
            within_area.append(float("nan"))
        similarity_names.append(display_name)

    area_wise_rsp_array = np.asarray(area_wise_rsp, dtype=float)
    within_area_array = np.asarray(within_area, dtype=float)
    if len(area_wise_rsp_array) > 1:
        similarity = 1.0 - squareform(pdist(area_wise_rsp_array, metric="correlation"))
        for idx, value in enumerate(within_area_array):
            similarity[idx, idx] = value
    else:
        similarity = np.empty((0, 0), dtype=float)

    reliability_vec = np.concatenate(reliability_here)
    all_area_labels = np.concatenate(area_label)
    reliable_mask = reliability_vec > reliability_threshold
    reliable_area_labels = all_area_labels[reliable_mask]
    comparison = {
        "reliability_count": int(reliability_vec.size),
        "fano_count": int(np.concatenate(fanof_here).size),
        "sparseness_count": int(np.concatenate(sparseness_vec).size),
        "snr_count": int(np.concatenate(snr_here).size),
        "group_counts": group_counts,
        "similarity_counts": similarity_counts,
        "similarity_shape": list(similarity.shape),
    }
    return PopulationSummary(
        reliability=reliability_vec,
        fano=np.concatenate(fanof_here),
        sparseness=np.concatenate(sparseness_vec),
        snrmax=np.concatenate(snr_here),
        area_labels=all_area_labels,
        reliable_area_labels=reliable_area_labels,
        broad_area_names=INTERESTED_AREA,
        broad_area_display_names=INTERESTED_NAME,
        area_similarity=similarity,
        similarity_names=similarity_names,
        area_wise_rsp=area_wise_rsp_array,
        within_area=within_area_array,
        comparison=comparison,
    )


def _json_ready_summary(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            out[key] = summarize_array(value)
        elif isinstance(value, dict):
            out[key] = _json_ready_summary(value)
        else:
            out[key] = value
    return out


def build_fig1_debug_payload(anatomy: AnatomyOverview, illustration: SessionIllustration, population: PopulationSummary) -> dict[str, dict[str, Any]]:
    return {
        "F1_b": {
            "ap_series": anatomy.ap_series,
            "slice_indices": anatomy.slice_indices,
            "marker_coords": anatomy.marker_coords,
            "marker_subjects": anatomy.marker_subjects,
            "marker_labels": anatomy.marker_labels,
            "expected_layout": anatomy.expected_layout,
            "big_image": anatomy.big_image,
        },
        "F1_e": {
            "trial_window": np.asarray(illustration.trial_window, dtype=int),
            "image_ids": illustration.image_ids,
            "good_units": illustration.selected_unit_indices + 1,
            "recruitment_curve": illustration.recruitment_curve,
            "onset_times": illustration.onset_times,
            "time_window": np.asarray(illustration.time_window, dtype=float),
            "spike_counts": np.asarray(illustration.comparison["spike_counts"], dtype=int),
        },
        "F1_fgh_F2_f": {
            "reliability": population.reliability,
            "fano": population.fano,
            "sparseness": population.sparseness,
            "snrmax": population.snrmax,
            "area_labels": population.area_labels,
            "reliable_area_labels": population.reliable_area_labels,
            "area_wise_rsp": population.area_wise_rsp,
            "within_area": population.within_area,
            "area_similarity": population.area_similarity,
        },
    }


def write_fig1_debug_payload(base_dir: Path, payload: dict[str, dict[str, Any]]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for name, values in payload.items():
        npz_ready = {key: value for key, value in values.items() if isinstance(value, np.ndarray)}
        json_ready = {key: value for key, value in values.items() if not isinstance(value, np.ndarray)}
        np.savez(base_dir / f"fig1_{name}_python.npz", **npz_ready)
        with (base_dir / f"fig1_{name}_python.json").open("w", encoding="utf-8") as f:
            import json

            json.dump(_json_ready_summary(json_ready), f, indent=2, default=str)


def _matlab_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def run_matlab_fig1_reference(config: ProjectConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "fig1_export_reference.m"
    matlab_code = dedent(
        f"""
        root_dir = '{_matlab_path(config.paths.dataset_root)}';
        H5_dir = '{_matlab_path(config.paths.dataset_root / "Raw" / "H5FILES")}';
        prep_dir = '{_matlab_path(config.paths.dataset_root / "Processed")}';
        code_dir = '{_matlab_path(config.paths.matlab_source_root)}';
        cd(code_dir); addpath(genpath(pwd));
        save(fullfile(code_dir,'DIRS.mat'),"root_dir","H5_dir","prep_dir","code_dir");
        run(fullfile(code_dir,'code','plot_F1_b.m'));
        ap_series = AP_series;
        slice_indices = zeros(length(AP_series),1);
        marker_coords = [];
        marker_subjects = [];
        marker_labels = {};
        for AP_IDX = 1:length(AP_series)
            AP_NOW = AP_series(AP_IDX);
            slice_indices(AP_IDX) = floor((AP_NOW-NIFTI_DATA.raw.srow_y(end))/0.25);
            area_here = find(Area_DATA.A==AP_NOW);
            for aa = 1:length(area_here)
                LR_HERE = Area_DATA.R(area_here(aa));
                LR_coor = floor((LR_HERE-NIFTI_DATA.raw.srow_x(end))/0.25);
                SI_here = Area_DATA.S(area_here(aa));
                SI_coor = floor((SI_here-NIFTI_DATA.raw.srow_z(end))/0.25);
                marker_coords = [marker_coords; [AP_IDX, LR_coor, SI_coor]];
                marker_subjects = [marker_subjects; Area_DATA.Subject(area_here(aa))];
                marker_labels{end+1,1} = Area_DATA.Label{area_here(aa)};
            end
        end
        big_image = big_img;
        save('{_matlab_path(output_dir / "fig1_F1_b_reference.mat")}', 'ap_series','slice_indices','expected_layout','big_image','marker_coords','marker_subjects','marker_labels');
        run(fullfile(code_dir,'code','plot_F1_e.m'));
        trial_window = [a, b];
        image_ids = trial_data(a:b);
        onset_times = onset_time(a:b);
        time_window = [time_start, time_end];
        spike_counts = zeros(length(good_units),1);
        for uu = 1:length(good_units)
            raster_raw = GoodUnitStrc(good_units(uu)).spiketime_ms;
            raster_time = raster_raw(raster_raw>time_start & raster_raw<time_end);
            spike_counts(uu) = numel(raster_time);
        end
        if exist('data_all','var')
            recruitment_curve = data_all./length(good_units);
        else
            recruitment_curve = [];
        end
        save('{_matlab_path(output_dir / "fig1_F1_e_reference.mat")}', 'trial_window','image_ids','good_units','recruitment_curve','onset_times','time_window','spike_counts');
        run(fullfile(code_dir,'code','plot_F1_fgh_F2_f.m'));
        reliability = reliability_here;
        fano = fanof_here;
        sparseness = Sparseness_vec;
        snrmax = snr_here;
        area_labels = area_label;
        reliable_area_labels = area_label(reliability_here>0.4);
        area_similarity = BIG_RDM;
        save('{_matlab_path(output_dir / "fig1_F1_fgh_F2_f_reference.mat")}', 'reliability','fano','sparseness','snrmax','area_labels','reliable_area_labels','area_wise_rsp','within_area','area_similarity');
        exit;
        """
    ).strip()
    script_path.write_text(matlab_code, encoding="utf-8")
    subprocess.run(["matlab", "-batch", f"run('{_matlab_path(script_path)}')"], check=True, cwd=str(config.paths.matlab_source_root))
    return output_dir


def compare_fig1_payloads(reference_dir: Path, payload: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference_map = {
        "F1_b": reference_dir / "fig1_F1_b_reference.mat",
        "F1_e": reference_dir / "fig1_F1_e_reference.mat",
        "F1_fgh_F2_f": reference_dir / "fig1_F1_fgh_F2_f_reference.mat",
    }
    comparisons: dict[str, Any] = {}
    for key, path in reference_map.items():
        if not path.exists():
            comparisons[key] = {"status": "missing_reference"}
            continue
        reference = sio.loadmat(path, simplify_cells=True)
        reference = {k: v for k, v in reference.items() if not k.startswith("__")}
        comparisons[key] = compare_payloads(reference, payload[key])
    return comparisons
