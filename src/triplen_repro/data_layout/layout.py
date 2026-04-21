from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from triplen_repro.config import ProjectConfig


@dataclass(slots=True)
class LayoutStatus:
    name: str
    status: str
    path: Path | None
    message: str


@dataclass(slots=True)
class ResolvedLayout:
    processed_dir: Path
    raw_h5_dir: Path
    others_dir: Path
    fmri_dir: Path | None
    model_feature_dir: Path | None
    stimuli_dir: Path | None
    mri_template_dir: Path | None
    statuses: dict[str, LayoutStatus]


def _resolve_optional_dir(dataset_root: Path, others_dir: Path, direct_rel: str, zip_name: str) -> LayoutStatus:
    direct = dataset_root / direct_rel
    if direct.exists():
        return LayoutStatus(zip_name, "available", direct, f"Using extracted directory: {direct}")
    alt = others_dir / zip_name.replace(".zip", "")
    if alt.exists():
        return LayoutStatus(zip_name, "available", alt, f"Using extracted directory: {alt}")
    archive = others_dir / zip_name
    if archive.exists():
        return LayoutStatus(zip_name, "extractable", archive, f"Archive found but not extracted: {archive}")
    return LayoutStatus(zip_name, "missing", None, f"Missing required resource or archive: {zip_name}")


def resolve_layout(config: ProjectConfig) -> ResolvedLayout:
    dataset_root = config.paths.dataset_root
    processed_dir = dataset_root / "Processed"
    raw_h5_dir = dataset_root / "Raw" / "H5FILES"
    others_dir = dataset_root / "others"

    statuses = {
        "processed": LayoutStatus("processed", "available" if processed_dir.exists() else "missing", processed_dir if processed_dir.exists() else None, str(processed_dir)),
        "raw_h5": LayoutStatus("raw_h5", "available" if raw_h5_dir.exists() else "missing", raw_h5_dir if raw_h5_dir.exists() else None, str(raw_h5_dir)),
        "others": LayoutStatus("others", "available" if others_dir.exists() else "missing", others_dir if others_dir.exists() else None, str(others_dir)),
    }
    fmri = _resolve_optional_dir(dataset_root, others_dir, "Data/FMRI", "FMRI.zip")
    model_feature = _resolve_optional_dir(dataset_root, others_dir, "Data/others/ModelFeature", "ModelFeature.zip")
    stimuli = _resolve_optional_dir(dataset_root, others_dir, "Data/others/StimuliNNN", "StimuliNNN.zip")
    statuses["fmri"] = fmri
    statuses["model_feature"] = model_feature
    statuses["stimuli"] = stimuli

    mri_template_dir = config.paths.matlab_source_root / "utils" / "downloaded" / "MRI"
    statuses["mri_templates"] = LayoutStatus("mri_templates", "available" if mri_template_dir.exists() else "missing", mri_template_dir if mri_template_dir.exists() else None, str(mri_template_dir))

    return ResolvedLayout(
        processed_dir=processed_dir,
        raw_h5_dir=raw_h5_dir,
        others_dir=others_dir,
        fmri_dir=fmri.path if fmri.status == "available" else None,
        model_feature_dir=model_feature.path if model_feature.status == "available" else None,
        stimuli_dir=stimuli.path if stimuli.status == "available" else None,
        mri_template_dir=mri_template_dir if mri_template_dir.exists() else None,
        statuses=statuses,
    )
