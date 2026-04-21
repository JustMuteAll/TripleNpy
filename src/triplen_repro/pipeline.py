from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from triplen_repro.analysis import (
    build_area_anatomy,
    build_fig1_debug_payload,
    build_session_illustration,
    compare_fig1_payloads,
    compute_imagewise_analysis,
    compute_basic_info,
    compute_preference_panel,
    compute_population_summary,
    compute_psth_clusters,
    run_fig5,
    run_fig5_preflight,
    run_matlab_fig1_reference,
    write_fig1_debug_payload,
)
from triplen_repro.config import ProjectConfig
from triplen_repro.data_layout import run_preflight
from triplen_repro.plotting import (
    plot_area_anatomy,
    plot_area_legend,
    plot_area_similarity,
    plot_basic_info,
    plot_fig3_area_example,
    plot_fig3_cluster_size,
    plot_fig3_imagewise,
    plot_fig3_mi_summary,
    plot_fig3_preference_panel,
    plot_fig3_summary,
    plot_population_summary,
    plot_session_illustration,
    save_figure,
)
from triplen_repro.validation import build_stage_report, write_report


def _ensure_output_dirs(config: ProjectConfig) -> None:
    for path in [config.paths.output_root, config.paths.figure_output_dir, config.paths.cache_dir, config.paths.log_dir]:
        path.mkdir(parents=True, exist_ok=True)
    (config.paths.output_root / "validation").mkdir(parents=True, exist_ok=True)
    (config.paths.output_root / "matlab_reference").mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def run_stage(config: ProjectConfig, stage: str, fig1_debug_compare: bool = False, matlab_compare: bool = False) -> dict[str, Any]:
    _ensure_output_dirs(config)
    stage = stage.lower()
    summary: dict[str, Any] = {"stage": stage, "completed": [], "blocked": [], "notes": []}
    if stage == "preflight":
        _, lines, failures = run_preflight(config)
        summary["notes"] = lines
        summary["blocked"] = failures
        return summary

    _, lines, failures = run_preflight(config)
    summary["notes"].extend(lines)
    if failures:
        summary["blocked"].extend(failures)
        return summary

    if stage in {"basic", "all"}:
        basic = compute_basic_info(config)
        save_figure(plot_basic_info(basic), config.paths.figure_output_dir / "basic_info.png")
        np.savez(config.paths.cache_dir / "basic_info.npz", session_labels=np.array(basic.session_labels, dtype=object), trial_counts=basic.trial_counts, image_degrees=basic.image_degrees)
        write_report(config.paths.output_root / "validation" / "basic_report.json", build_stage_report("basic", {"trial_counts": basic.trial_counts, "image_degrees": basic.image_degrees}))
        summary["completed"].append("basic")

    if stage in {"fig1", "all"}:
        for obsolete in ["F1.png", "fig1_session_illustration.png", "fig1_population_summary.png"]:
            _remove_if_exists(config.paths.figure_output_dir / obsolete)
        anatomy = build_area_anatomy(config)
        save_figure(plot_area_anatomy(anatomy), config.paths.figure_output_dir / "F1B_MRI.png")
        save_figure(plot_area_legend(anatomy), config.paths.figure_output_dir / "F1B_LB.png")
        illustration = build_session_illustration(config)
        save_figure(plot_session_illustration(illustration), config.paths.figure_output_dir / "F1e.png")
        population = compute_population_summary(config)
        save_figure(plot_population_summary(population), config.paths.figure_output_dir / "F1low.png")
        save_figure(plot_area_similarity(population), config.paths.figure_output_dir / "F2F.png")
        np.savez(
            config.paths.cache_dir / "population_summary.npz",
            reliability=population.reliability,
            fano=population.fano,
            sparseness=population.sparseness,
            snrmax=population.snrmax,
            area_similarity=population.area_similarity,
            area_labels=population.area_labels,
        )
        debug_payload = build_fig1_debug_payload(anatomy, illustration, population)
        if fig1_debug_compare:
            write_fig1_debug_payload(config.paths.output_root / "matlab_reference", debug_payload)
        compare_report: dict[str, Any] = {"status": "not_requested"}
        if matlab_compare:
            reference_dir = run_matlab_fig1_reference(config, config.paths.output_root / "matlab_reference")
            compare_report = compare_fig1_payloads(reference_dir, debug_payload)
            write_report(config.paths.output_root / "validation" / "fig1_compare.json", {"stage": "fig1", "comparisons": compare_report})
        write_report(
            config.paths.output_root / "validation" / "fig1_report.json",
            build_stage_report(
                "fig1",
                {
                    "anatomy": anatomy.comparison,
                    "reliability": population.reliability,
                    "fano": population.fano,
                    "sparseness": population.sparseness,
                    "snrmax": population.snrmax,
                    "area_similarity": population.area_similarity,
                    "illustration_window": illustration.comparison,
                    "population_summary": population.comparison,
                    "compare_status": compare_report if matlab_compare else "not_requested",
                },
            ),
        )
        summary["completed"].append("fig1")

    if stage in {"fig3", "all"}:
        clustering_cache_path = config.paths.cache_dir / "fig3_clustering.pkl"
        if clustering_cache_path.exists():
            clustering = _load_pickle(clustering_cache_path)
            summary["notes"].append(f"[CACHE HIT  ] fig3_clustering: {clustering_cache_path}")
        else:
            clustering = compute_psth_clusters(config)
            _save_pickle(clustering_cache_path, clustering)
            summary["notes"].append(f"[CACHE SAVE ] fig3_clustering: {clustering_cache_path}")
        imagewise = compute_imagewise_analysis(config, clustering)
        preference = compute_preference_panel(config, clustering)
        save_figure(plot_fig3_summary(clustering), config.paths.figure_output_dir / "F3A.png")
        save_figure(plot_fig3_cluster_size(clustering), config.paths.figure_output_dir / "F3AS1.png")
        save_figure(plot_fig3_mi_summary(clustering), config.paths.figure_output_dir / "F3AS3.png")
        save_figure(plot_fig3_imagewise(imagewise), config.paths.figure_output_dir / "F3E.png")
        save_figure(plot_fig3_preference_panel(preference), config.paths.figure_output_dir / "F3SS.png")
        if clustering.example_area is not None:
            save_figure(plot_fig3_area_example(clustering), config.paths.figure_output_dir / f"F3S_{clustering.example_area.area_idx}.png")
        np.savez(
            config.paths.cache_dir / "fig3_clusters.npz",
            all_cluster=clustering.all_cluster,
            selected_units=clustering.selected_units,
            all_mean_psth=clustering.all_mean_psth,
            cluster_mean_psth=clustering.cluster_mean_psth,
            cluster_ci_psth=clustering.cluster_ci_psth,
            group_ratios=clustering.group_ratios,
            sorted_order=clustering.sorted_order,
            cluster_boundaries=clustering.cluster_boundaries,
            db_k_list=clustering.db_k_list,
            db_scores=clustering.db_scores,
            mi_observed=clustering.mi_observed,
            mi_permutation=clustering.mi_permutation,
            dnn_correlations=imagewise.dnn_correlations,
            area_keys=np.array(list(clustering.area_clusters.keys())),
            area_values=np.array([clustering.area_clusters[k] for k in clustering.area_clusters], dtype=object),
        )
        write_report(
            config.paths.output_root / "validation" / "fig3_report.json",
            build_stage_report(
                "fig3",
                {
                    "all_cluster": clustering.all_cluster,
                    "selected_units": clustering.selected_units,
                    "all_mean_psth": clustering.all_mean_psth,
                    "cluster_mean_psth": clustering.cluster_mean_psth,
                    "group_ratios": clustering.group_ratios,
                    "db_scores": clustering.db_scores,
                    "mi_observed": clustering.mi_observed[np.isfinite(clustering.mi_observed)],
                    "dnn_correlations": imagewise.dnn_correlations,
                    "comparison": clustering.comparison,
                    "imagewise": imagewise.comparison,
                    "preference": preference.comparison,
                },
            ),
        )
        summary["completed"].append("fig3")

    if stage in {"fig5", "all"}:
        fig5 = run_fig5_preflight(config)
        _save_json(config.paths.cache_dir / "fig5_preflight.json", fig5)
        if fig5["fmri_status"] != "available" or fig5["model_feature_status"] != "available":
            summary["blocked"].append("fig5_resources")
            summary["notes"].append(fig5)
        else:
            result = run_fig5(config)
            for area_idx, payload in result.session_scores.items():
                np.savez(config.paths.cache_dir / f"fig5_session_area_{area_idx:03d}.npz", **payload)
            if result.fmri_scores:
                np.savez(config.paths.cache_dir / "fig5_fmri_summary.npz", **result.fmri_scores)
            write_report(
                config.paths.output_root / "validation" / "fig5_report.json",
                build_stage_report(
                    "fig5",
                    {
                        "fmri_r": result.fmri_scores.get("r", np.empty((0, 0))),
                        "fmri_r2": result.fmri_scores.get("r2", np.empty((0, 0))),
                        "model_names": result.model_names,
                        "comparison": result.comparison,
                    },
                ),
            )
            summary["completed"].append("fig5")

    _save_json(config.paths.log_dir / f"{stage}_summary.json", summary)
    return summary
