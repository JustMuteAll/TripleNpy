from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib import transforms
import numpy as np
from scipy import stats

from triplen_repro.analysis.basic import BasicInfoResult
from triplen_repro.analysis.fig1 import AnatomyOverview, PopulationSummary, SessionIllustration
from triplen_repro.analysis.fig3 import ClusteringResult, ImageWiseResult, PreferencePanelResult


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_basic_info(result: BasicInfoResult) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(np.arange(len(result.session_labels)), result.trial_counts, color="0.3")
    axes[0].set_title("Session trial counts")
    axes[0].set_xlabel("IT session index")
    axes[0].set_ylabel("Trials")
    unique_deg, counts = np.unique(result.image_degrees, return_counts=True)
    axes[1].bar(unique_deg.astype(str), counts, color="0.5")
    axes[1].set_title("Image degree")
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_area_anatomy(result: AnatomyOverview) -> plt.Figure:
    rows, cols = result.expected_layout.shape
    fig, axes = plt.subplots(rows, cols, figsize=(18.5, 8.0))
    axes = np.asarray(axes)
    for row_idx in range(rows):
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            tile_id = int(result.expected_layout[row_idx, col_idx])
            ax.axis("off")
            if tile_id == 99:
                ax.imshow(np.full_like(result.tile_images[0], 255))
                continue
            tile = result.tile_images[tile_id - 1]
            ax.imshow(tile.astype(np.uint8))
            ax.text(
                0.5,
                -0.07,
                f"{result.ap_series[tile_id - 1]:.1f} mm",
                fontsize=10,
                fontweight="bold",
                color="black",
                ha="center",
                va="top",
                transform=ax.transAxes,
                clip_on=False,
            )
            tile_marker_idx = np.flatnonzero(result.marker_tile_coords[:, 0] == tile_id)
            for marker_idx in tile_marker_idx:
                _, tile_x, tile_y = result.marker_tile_coords[marker_idx]
                if 0 <= tile_x < tile.shape[1] and 0 <= tile_y < tile.shape[0]:
                    ax.text(
                        float(tile_x),
                        float(tile_y),
                        str(int(result.marker_subjects[marker_idx])),
                        fontsize=7,
                        fontweight="bold",
                        color="white",
                        ha="center",
                        va="center",
                    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.05, wspace=0.005, hspace=0.14)
    return fig


def plot_area_legend(result: AnatomyOverview) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 0.9))
    ax.set_xlim(0.5, 7)
    ax.set_ylim(0.8, 1.2)
    ax.axis("off")
    xs = np.linspace(1, 6, len(result.legend_labels))
    for idx, label in enumerate(result.legend_labels):
        ax.text(xs[idx], 1.0, label, fontsize=18, fontweight="bold", 
                color=result.legend_colors[idx] / 255.0, va="center", ha="center",)
    fig.tight_layout()
    return fig


def plot_session_illustration(result: SessionIllustration) -> plt.Figure:
    fig = plt.figure(figsize=(10, 7.6))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.05, 1.1, 1.1, 1.0, 0.2], hspace=0.18)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_raster = fig.add_subplot(gs[1:3, 0])
    ax_recruit = fig.add_subplot(gs[3, 0], sharex=ax_raster)
    ax_bar = fig.add_subplot(gs[4, 0])

    ax_img.imshow(result.image_strip.astype(np.uint8))
    ax_img.axis("off")

    for unit_idx, spikes in enumerate(result.selected_spike_traces):
        color = result.unit_colors[unit_idx] if unit_idx < len(result.unit_colors) else np.array([0, 0, 0], dtype=float)
        ax_raster.scatter(spikes, np.full(spikes.shape, unit_idx), s=4, marker="s", c=[color], linewidths=0)
    for onset in result.onset_times[1:]:
        ax_raster.axvline(onset, linestyle="--", color="0.5", linewidth=1)
    ax_raster.set_xlim(result.time_window)
    ax_raster.set_ylim(-2, len(result.selected_spike_traces) + 2)
    ax_raster.set_xticks([])
    ax_raster.set_ylabel("# Units")
    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)
    ax_raster.tick_params(direction="out", length=0)

    x = np.arange(result.recruitment_curve.size) + result.time_window[0]
    ax_recruit.plot(x, result.recruitment_curve, linewidth=1, color="0.2")
    for onset in result.onset_times[1:]:
        ax_recruit.axvline(onset, linestyle="--", color="0.5", linewidth=1)
    ax_recruit.set_xlim(result.time_window)
    ticks = np.arange(result.time_window[0], result.time_window[1] + 1, 300)
    ax_recruit.set_xticks(ticks)
    ax_recruit.set_xticklabels([str(int(t - result.time_window[0])) for t in ticks])
    ax_recruit.set_ylim(0, 0.8)
    ax_recruit.set_yticks([0, 0.8])
    ax_recruit.set_ylabel("% Unit recruited")
    ax_recruit.set_xlabel("Time (ms)")
    ax_recruit.spines["top"].set_visible(False)
    ax_recruit.spines["right"].set_visible(False)
    ax_recruit.tick_params(direction="out", length=0)

    ax_bar.imshow(result.onset_bar, aspect="auto")
    ax_bar.axis("off")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.07)
    return fig


def _set3_colors(count: int) -> list[tuple[float, float, float, float]]:
    cmap = colormaps["Set3"]
    return [cmap(i / max(count - 1, 1)) for i in range(count)]


def _colored_boxplot(ax: plt.Axes, values: np.ndarray, labels: np.ndarray, colors: list[tuple[float, float, float, float]]) -> None:
    grouped = [values[(labels == idx) & np.isfinite(values)] for idx in range(1, len(colors) + 1)]
    bp = ax.boxplot(grouped, vert=False, patch_artist=True, showfliers=False, whis=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1)
    for median in bp["medians"]:
        x = np.mean(median.get_xdata())
        y = np.mean(median.get_ydata())
        median.set_color("white")
        median.set_linewidth(0)
        ax.scatter(x, y, s=15, c="white", edgecolors="black", linewidths=0.5, zorder=3)
    for whisker in bp["whiskers"]:
        whisker.set_color("0.5")
        whisker.set_linestyle("-")
        whisker.set_linewidth(1)
    for cap in bp["caps"]:
        cap.set_visible(False)


def plot_population_summary(result: PopulationSummary) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.6))
    colors = _set3_colors(len(result.broad_area_names))

    _colored_boxplot(axes[0], result.reliability, result.area_labels, colors)
    axes[0].set_yticks(np.arange(1, len(result.broad_area_names) + 1))
    axes[0].set_yticklabels(result.broad_area_display_names)
    axes[0].set_xlim(-0.1, 1.0)
    axes[0].set_xticks(np.arange(0, 1.01, 0.2))
    axes[0].set_title("Reliability (r)")

    _colored_boxplot(axes[1], result.fano, result.reliable_area_labels, colors)
    axes[1].set_xlim(0.8, 1.6)
    axes[1].set_title("Fano factor")

    _colored_boxplot(axes[2], result.sparseness, result.reliable_area_labels, colors)
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_title("Sparseness")

    _colored_boxplot(axes[3], result.snrmax, result.reliable_area_labels, colors)
    axes[3].set_xlim(3, 50)
    axes[3].set_title("SNRmax")

    for idx, ax in enumerate(axes):
        ax.tick_params(direction="out", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx > 0:
            ax.set_yticks([])
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.25)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.86, bottom=0.16, wspace=0.18)
    return fig


def _orange_bao_cmap() -> ListedColormap:
    blues = colormaps["Blues"](np.linspace(1, 0, 128))
    oranges = colormaps["Oranges"](np.linspace(0, 1, 128))
    return ListedColormap(np.vstack([blues, oranges]))


def plot_area_similarity(result: PopulationSummary) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(result.area_similarity, cmap=_orange_bao_cmap(), vmin=-0.6, vmax=0.6, aspect="equal")
    ax.set_xticks(np.arange(len(result.similarity_names)))
    ax.set_yticks(np.arange(len(result.similarity_names)))
    ax.set_xticklabels(result.similarity_names, rotation=90, fontsize=8)
    ax.set_yticklabels(result.similarity_names, fontsize=8)
    ax.tick_params(direction="out", length=0)
    ax.set_xlim(-0.5, len(result.similarity_names) - 0.5)
    ax.set_ylim(len(result.similarity_names) - 0.5, -0.5)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([-0.6, 0, 0.6])
    cbar.set_label("Correlation")
    fig.tight_layout()
    return fig


def _fig3_cluster_colors() -> np.ndarray:
    return np.asarray(
        [
            (0.89, 0.10, 0.11),
            (0.25, 0.50, 0.75),
            (0.35, 0.72, 0.35),
        ],
        dtype=float,
    )


def _spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = min(a.size, b.size)
    if n < 2:
        return float("nan")
    return float(stats.spearmanr(a[:n], b[:n], nan_policy="omit").statistic)


def _plot_fig3_violin(ax: plt.Axes, grouped: list[np.ndarray], colors: np.ndarray) -> None:
    vp = ax.violinplot(grouped, positions=[1, 2, 3], widths=0.8, showmeans=False, showextrema=False, showmedians=False)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.95)
    medians = [np.nanmedian(values) if values.size else np.nan for values in grouped]
    ax.scatter([1, 2, 3], medians, s=30, c="white", edgecolors="0.5", linewidths=0.5, zorder=3)
    ax.set_xlim(0.5, 3.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["C1", "C2", "C3"])
    ax.set_ylabel("Reliability")
    ax.set_ylim(0.4, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_fig3_mi_panel(ax: plt.Axes, result: ClusteringResult) -> None:
    observed = result.mi_observed[np.isfinite(result.mi_observed)]
    permuted = result.mi_permutation[np.isfinite(result.mi_permutation)]
    n = min(observed.size, permuted.size)
    observed = observed[:n]
    permuted = permuted[:n]
    bp = ax.boxplot(
        [observed, permuted],
        positions=[1, 2],
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        whis=0,
        medianprops={"color": "#ff7f0e", "linewidth": 1.0},
    )
    box_color = colormaps["Oranges"](0.45)
    for patch in bp["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("0.25")
    for whisker in bp["whiskers"]:
        whisker.set_color("0.2")
    for cap in bp["caps"]:
        cap.set_color("0.2")
    for idx in range(n):
        ax.plot([1, 2], [observed[idx], permuted[idx]], color=(0.6, 0.6, 0.6, 0.28), linewidth=0.6)
    ax.scatter(np.ones(n), observed, s=10, color=(0.6, 0.6, 0.6, 0.4))
    ax.scatter(np.full(n, 2), permuted, s=10, color=(0.6, 0.6, 0.6, 0.4))
    pvalue = float(stats.ranksums(observed, permuted).pvalue) if n else float("nan")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Observed", "Permutation"])
    ax.set_ylabel("MI(cluster, position)")
    ax.set_ylim(0, 0.5)
    ax.set_title(f"p-value: {pvalue:.6e}", loc="left", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _build_fig3_main(result: ClusteringResult, include_mi: bool) -> plt.Figure:
    fig = plt.figure(figsize=(14.2, 5.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.55, 1.12, 1.2, 1.12, 1.3], hspace=0.48, wspace=0.48)
    colors = _fig3_cluster_colors()
    time_axis = np.arange(5, 351, 5, dtype=float)
    counts = [int(np.isfinite(values).sum()) for values in result.cluster_reliability]

    ax_heat = fig.add_subplot(gs[:, 0])
    c1 = result.sorted_order[: counts[0]]
    c2 = result.sorted_order[counts[0] : counts[0] + counts[1]]
    c3 = result.sorted_order[counts[0] + counts[1] :]
    heat_order = np.concatenate([c3, c2, c1])
    sorted_data = result.selected_mean_psth[heat_order]
    ax_heat.imshow(sorted_data, aspect="auto", cmap=_orange_bao_cmap(), vmin=-3, vmax=3, extent=[0, 300, sorted_data.shape[0], 0])
    for boundary in np.cumsum([counts[2], counts[1]]):
        ax_heat.axhline(boundary, color="0.45", linestyle="--", linewidth=1)
    centers = np.asarray(
        [
            counts[2] / 2.0,
            counts[2] + counts[1] / 2.0,
            counts[2] + counts[1] + counts[0] / 2.0,
        ],
        dtype=float,
    )
    trans = transforms.blended_transform_factory(ax_heat.transAxes, ax_heat.transData)
    for label, center in zip(["Cluster3", "Cluster2", "Cluster1"], centers):
        ax_heat.text(-0.07, center, label, rotation=90, va="center", ha="center", transform=trans, fontsize=11)
    ax_heat.set_xlabel("Time (ms)")
    ax_heat.set_yticks([])
    ax_heat.set_xlim(0, 300)

    ax_curve = fig.add_subplot(gs[0, 1])
    for cc in range(3):
        mean = result.cluster_mean_psth[cc]
        ci = result.cluster_ci_psth[cc]
        x = time_axis[: mean.size]
        ax_curve.plot(x, mean, color=colors[cc], linewidth=1.3, label=f"Cluster{cc + 1}")
        ax_curve.fill_between(x, mean - ci, mean + ci, color=colors[cc], alpha=0.18)
    ax_curve.set_xlim(0, 300)
    ax_curve.set_ylim(-2, 2)
    ax_curve.set_xlabel("Time (ms)")
    ax_curve.set_ylabel("Norm. firing rate (a.u.)")
    ax_curve.legend(frameon=False, loc="lower left", fontsize=8, handlelength=3, handletextpad=0.4)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)

    ax_ratio = fig.add_subplot(gs[0, 2])
    display = np.flipud(result.group_ratios)
    left = np.zeros(display.shape[0], dtype=float)
    for cc in range(3):
        ax_ratio.barh(np.arange(display.shape[0]), display[:, cc], left=left, color=colors[cc], edgecolor="none")
        left += display[:, cc]
    ax_ratio.set_yticks(np.arange(display.shape[0]))
    ax_ratio.set_yticklabels(list(reversed(result.group_display_names)))
    ax_ratio.tick_params(axis="y", labelsize=8)
    ax_ratio.set_xlim(0, 1)
    ax_ratio.set_xlabel("Ratio")
    ax_ratio.spines["top"].set_visible(False)
    ax_ratio.spines["right"].set_visible(False)

    ax_rel = fig.add_subplot(gs[0, 3])
    grouped = [values[np.isfinite(values)] for values in result.cluster_reliability]
    _plot_fig3_violin(ax_rel, grouped, colors)

    im = None
    sim_axes = []
    for cc in range(3):
        ax = fig.add_subplot(gs[1, 1 + cc])
        plot_matrix = 1.0 - result.temporal_similarity[cc]
        x = np.linspace(1, 350, plot_matrix.shape[0])
        y = np.linspace(1, 350, plot_matrix.shape[1])
        im = ax.imshow(plot_matrix, cmap=colormaps["Spectral_r"], vmin=0, vmax=1, extent=[1, 350, 350, 1], aspect="equal")
        ax.contour(x, y, plot_matrix, levels=[0.3, 0.5], colors=["0.35", "0.35"], linewidths=0.45)
        ax.set_title(f"Cluster{cc + 1}, n = {counts[cc]}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Time (ms)")
        ax.set_xlim(1, 350)
        ax.set_ylim(350, 1)
        ax.set_xticks([50, 150, 250, 350])
        ax.set_yticks([50, 150, 250, 350])
        sim_axes.append(ax)
    if im is not None:
        cbar = fig.colorbar(im, ax=sim_axes, fraction=0.025, pad=0.015)
        cbar.set_ticks([0, 0.4, 0.6, 1.0])

    if include_mi:
        ax_mi = fig.add_subplot(gs[:, 4])
        _plot_fig3_mi_panel(ax_mi, result)
    else:
        ax_blank = fig.add_subplot(gs[:, 4])
        ax_blank.axis("off")

    return fig


def plot_fig3_summary(result: ClusteringResult) -> plt.Figure:
    return _build_fig3_main(result, include_mi=False)


def plot_fig3_cluster_size(result: ClusteringResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    valid = np.isfinite(result.db_scores)
    x = result.db_k_list[valid]
    y = result.db_scores[valid]
    ax.plot(x, y, color="k", linewidth=1)
    ax.scatter(x, y, facecolors="none", edgecolors="k", s=34, linewidths=0.8)
    if result.db_optimal_k in x:
        opt_idx = np.where(result.db_k_list == result.db_optimal_k)[0][0]
        ax.scatter(result.db_optimal_k, result.db_scores[opt_idx], color="red", s=38, zorder=3)
    ax.set_title("DaviesBouldin", pad=4)
    ax.set_xlabel("Cluster number")
    ax.set_xticks(np.arange(2, result.db_k_list[-1]))
    ax.set_xlim(1.5, 7.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_fig3_mi_summary(result: ClusteringResult) -> plt.Figure:
    return _build_fig3_main(result, include_mi=True)


def plot_fig3_area_example(result: ClusteringResult) -> plt.Figure:
    if result.example_area is None:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")
        return fig

    example = result.example_area
    colors = _fig3_cluster_colors()
    fig = plt.figure(figsize=(9.0, 2.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.62], wspace=0.35)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_lfp = fig.add_subplot(gs[0, 1])
    ax_blank = fig.add_subplot(gs[0, 2])

    for cc in range(1, 4):
        loc = np.flatnonzero(example.idx_area == cc)
        if loc.size:
            ax_hist.hist(
                example.pos_area[loc],
                bins=example.depth_edges,
                orientation="horizontal",
                color=colors[cc - 1],
                alpha=0.55,
                label=f"Cluster{cc}",
            )
    yl = [np.min(example.pos_area), np.max(example.pos_area)]
    ax_hist.set_ylim(yl)
    ax_hist.set_xlabel("# Units")
    ax_hist.set_ylabel("Depth (um)")
    ax_hist.legend(frameon=False, loc="lower right", fontsize=8, handlelength=1.8, handletextpad=0.4)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    time_extent = min(300, int(2 * max(example.lfp_matrix.shape[0] - 1, 1)))
    im = ax_lfp.imshow(
        example.lfp_matrix.T,
        aspect="auto",
        origin="lower",
        extent=[0, time_extent, example.lfp_depth[0], example.lfp_depth[-1]],
        cmap=colormaps["plasma"],
    )
    ax_lfp.set_title("Local Field Potential (uv)", pad=4)
    ax_lfp.set_xlabel("Time (ms)")
    ax_lfp.set_ylim(yl)
    cbar = fig.colorbar(im, ax=ax_lfp, fraction=0.08, pad=0.10)
    cbar.ax.tick_params(length=0)
    ax_blank.axis("off")
    fig.suptitle(f"session {example.session_idx}, area = {example.area_label}, MI={example.mi_observed:.02f}", y=0.96, fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.82, wspace=0.42)
    return fig


def _bordered_tile_strip(tiles: np.ndarray, colors: np.ndarray, border: int = 7, gap: int = 10) -> np.ndarray:
    if tiles.size == 0:
        return np.full((120, 120, 3), 255, dtype=np.uint8)
    parts = []
    for tile, color in zip(tiles, colors):
        framed = np.pad(tile.astype(np.uint8), ((border, border), (border, border), (0, 0)), constant_values=255)
        frame_color = np.round(np.asarray(color) * 255).astype(np.uint8)
        framed[:border, :, :] = frame_color
        framed[-border:, :, :] = frame_color
        framed[:, :border, :] = frame_color
        framed[:, -border:, :] = frame_color
        parts.append(framed)
        parts.append(np.full((framed.shape[0], gap, 3), 255, dtype=np.uint8))
    return np.concatenate(parts[:-1], axis=1)


def _style_empty_panel(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_fig3_imagewise(result: ImageWiseResult) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.8))
    target_cluster = result.cluster_ids[1]
    target = result.cluster_results.get(target_cluster, {})
    clus1 = result.cluster_results.get(1, {})
    rep_colors = np.asarray(target.get("representative_colors", np.empty((0, 3))), dtype=float)
    tile_count = min(result.representative_tiles.shape[0], rep_colors.shape[0])
    colors = rep_colors[:tile_count] if tile_count else np.flipud(colormaps["plasma"](np.linspace(40 / 255.0, 200 / 255.0, 1))[:, :3])

    analysis_image = np.asarray(target.get("analysis_image", np.empty(0, dtype=int)), dtype=int)
    latencies = target.get("latency", [])
    pop_rsp = target.get("pop_rsp_mean", [])
    rep_images = np.asarray(target.get("representative_images", np.empty(0, dtype=int)), dtype=int)

    axes[0, 0].set_xlabel("Peak Latency (ms)")
    axes[0, 0].set_ylabel("Firing Rate (a.u.)")
    if latencies and pop_rsp:
        lat = np.asarray(latencies[0], dtype=float)
        rsp = np.asarray(pop_rsp[0], dtype=float)
        axes[0, 0].scatter(lat, rsp, s=8, facecolors="none", edgecolors="k", linewidths=0.5)
        for idx, img_idx in enumerate(rep_images[:tile_count]):
            loc = np.flatnonzero(analysis_image == img_idx)
            if loc.size:
                axes[0, 0].scatter(lat[loc[0]], rsp[loc[0]], s=28, color=colors[idx], edgecolors="none", zorder=3)
    axes[0, 0].set_xlim(100, 255)
    axes[0, 0].set_title("")
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)

    rep_mean = np.asarray(target.get("representative_psth_mean", np.empty((0, 0))), dtype=float)
    rep_sem = np.asarray(target.get("representative_psth_sem", np.empty((0, 0))), dtype=float)
    axes[0, 1].set_title(f"Cluster {target_cluster}", pad=4)
    if rep_mean.size:
        draw_count = min(rep_mean.shape[0], tile_count)
        for ii in range(draw_count):
            x = np.arange(1, rep_mean.shape[1] + 1)
            axes[0, 1].plot(x, rep_mean[ii], color=colors[ii], linewidth=1.25)
            axes[0, 1].fill_between(x, rep_mean[ii] - rep_sem[ii], rep_mean[ii] + rep_sem[ii], color=colors[ii], alpha=0.22)
    axes[0, 1].set_xlim(50, 250)
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Norm. firing rate")
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)

    display_count = tile_count
    if tile_count > 5:
        display_idx = np.linspace(0, tile_count - 1, 5, dtype=int)
        display_count = 5
        display_tiles = result.representative_tiles[display_idx]
        display_colors = colors[display_idx]
    else:
        display_tiles = result.representative_tiles[:tile_count]
        display_colors = colors[:tile_count]
    strip = _bordered_tile_strip(display_tiles, display_colors)
    axes[0, 2].imshow(strip.astype(np.uint8))
    axes[0, 2].axis("off")

    axes[1, 0].set_xlabel(f"Latency Clus{target_cluster} Ses1 (ms)")
    axes[1, 0].set_ylabel(f"Latency Clus{target_cluster} Ses2 (ms)")
    if len(target.get("latency", [])) >= 2:
        a = np.asarray(target["latency"][0], dtype=float)
        b = np.asarray(target["latency"][1], dtype=float)
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
        lo = min(np.min(a), np.min(b)) - 5
        hi = max(np.max(a), np.max(b)) + 5
        axes[1, 0].scatter(a, b, s=10, facecolors="none", edgecolors="k", linewidths=0.5)
        axes[1, 0].plot([lo, hi], [lo, hi], linestyle="--", color="0.55", linewidth=1)
        axes[1, 0].set_xlim(lo, hi)
        axes[1, 0].set_ylim(lo, hi)
        corr = _spearman_r(a, b)
        axes[1, 0].text(0.52, 0.04, f"corr {corr:.2f}", transform=axes[1, 0].transAxes, fontsize=11)
    else:
        _style_empty_panel(axes[1, 0], "")

    axes[1, 1].set_xlabel("Latency Clus1 Ses1 (ms)")
    axes[1, 1].set_ylabel(f"Latency Clus{target_cluster} Ses1 (ms)")
    if clus1.get("latency") and target.get("latency"):
        a = np.asarray(clus1["latency"][0], dtype=float)
        b = np.asarray(target["latency"][0], dtype=float)
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
        lo = min(np.min(a), np.min(b)) - 5
        hi = max(np.max(a), np.max(b)) + 5
        axes[1, 1].scatter(a, b, s=10, facecolors="none", edgecolors="k", linewidths=0.5)
        axes[1, 1].plot([lo, hi], [lo, hi], linestyle="--", color="0.55", linewidth=1)
        axes[1, 1].set_xlim(lo, hi)
        axes[1, 1].set_ylim(lo, hi)
        corr = _spearman_r(a, b)
        axes[1, 1].text(0.52, 0.04, f"corr {corr:.2f}", transform=axes[1, 1].transAxes, fontsize=11)
    else:
        _style_empty_panel(axes[1, 1], "")

    axes[1, 2].bar(np.arange(1, len(result.dnn_layer_names) + 1), result.dnn_correlations, edgecolor="none", color="#1f77b4")
    axes[1, 2].set_xticks(np.arange(1, len(result.dnn_layer_names) + 1))
    axes[1, 2].set_xticklabels(result.dnn_layer_names, rotation=45, ha="right")
    axes[1, 2].set_xlabel("layer")
    axes[1, 2].set_ylabel("Predict Accuracy (r)")
    axes[1, 2].set_ylim(0, max(0.3, float(np.nanmax(result.dnn_correlations)) + 0.03))
    axes[1, 2].spines["top"].set_visible(False)
    axes[1, 2].spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.12, hspace=0.32, wspace=0.28)
    return fig


def plot_fig3_preference_panel(result: PreferencePanelResult) -> plt.Figure:
    fig = plt.figure(figsize=(12.6, 6.6))
    gs = fig.add_gridspec(5, 7, width_ratios=[1, 1, 1, 0.32, 1, 1, 1], hspace=0.62, wspace=0.55)
    colors = _fig3_cluster_colors()

    for row in range(5):
        for side in range(2):
            area_idx = row * 2 + side
            area_name = result.area_names[area_idx]
            base_col = 0 if side == 0 else 4
            for cluster_id in result.cluster_ids:
                ax = fig.add_subplot(gs[row, base_col + cluster_id - 1])
                entry = result.panel_data.get((area_idx, cluster_id), {"count": 0})
                if entry.get("count", 0) == 0:
                    ax.axis("off")
                    continue
                x = np.arange(1, entry["overall_mean"].size + 1)
                ax.plot(x, entry["low_mean"], color=colors[cluster_id - 1], linewidth=1.0, alpha=0.45)
                ax.fill_between(x, entry["low_mean"] - entry["low_sem"], entry["low_mean"] + entry["low_sem"], color=colors[cluster_id - 1], alpha=0.10)
                ax.plot(x, entry["overall_mean"], color="0.65", linewidth=1.0)
                ax.fill_between(x, entry["overall_mean"] - entry["overall_sem"], entry["overall_mean"] + entry["overall_sem"], color="0.75", alpha=0.12)
                ax.plot(x, entry["high_mean"], color=colors[cluster_id - 1], linewidth=1.35)
                ax.fill_between(x, entry["high_mean"] - entry["high_sem"], entry["high_mean"] + entry["high_sem"], color=colors[cluster_id - 1], alpha=0.18)
                ax.set_xlim(0, 300)
                if row == 0:
                    ax.set_title(f"n={entry['count']}", pad=3)
                else:
                    ax.set_title(f"n={entry['count']}", pad=2)
                if cluster_id == 1:
                    ax.set_ylabel(f"{area_name}\nNorm. firing rate")
                else:
                    ax.set_yticklabels([])
                if row == 4:
                    ax.set_xlabel("Time (ms)")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if row == 0:
                    pass

    fig.tight_layout()
    return fig
