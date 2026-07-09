"""fsaverage6 HCP-MMP1 ROI vertex extraction.

The narratives held-out fMRI lives at the full 81,924-vertex fs6 resolution
(40,962 per hemisphere). To compare against RABBiT's predictions we have to
slice the same 30 ROI blocks (15 bilateral ROI bases × LH/RH) out of the
held-out fMRI in the same order the model emits them (interleaved LH/RH per
ROI base — see
``rabbit.model.roi_layout.ROI_PARCELS``).

The extraction here is paper-canonical: ``mne`` reads the HCP-MMP1
annotation on the fsaverage6 surface, parcels are concatenated per ROI base,
and each ROI block is z-scored along time. Matches the recipe used to
produce the training-time per-clip fMRI .npz files.

Requires the ``fs6`` optional install: ``pip install rabbit[fs6]``.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np

from ..model.roi_layout import ROI_PARCELS

# 40,962 vertices per hemisphere on fsaverage6.
NV_FS6 = 40962


__all__ = [
    "NV_FS6",
    "build_fs6_roi_vertex_indices",
    "extract_roi_from_fs6",
]


def build_fs6_roi_vertex_indices(
    subjects_dir: Optional[str] = None,
    roi_parcels: "OrderedDict[str, tuple[str, ...]]" = ROI_PARCELS,
) -> "OrderedDict[str, np.ndarray]":
    """Return the fs6 vertex indices for each of the 30 bilateral ROIs.

    Keys are ordered LH-then-RH **per ROI base** to match the flat output
    layout of ``rabbit.model.RABBiT`` (i.e. ``aud_primary_lh, aud_primary_rh,
    aud_belt_lh, aud_belt_rh, …``).

    Args:
        subjects_dir: optional FreeSurfer subjects directory containing
            ``fsaverage6/label/[lr]h.HCPMMP1.annot``. If None, falls back to
            ``mne.datasets.sample.data_path()/'subjects'``.
        roi_parcels: mapping ``roi_base -> tuple of HCP-MMP1 parcel stems``
            (no ``L_/R_`` prefix). Defaults to the paper's 15-ROI definition.

    Returns:
        ``OrderedDict[roi_name_with_hemi -> np.ndarray of vertex indices]`` of
        length 30. Per-hemi indices index into the corresponding fs6
        hemisphere (0..NV_FS6).
    """
    try:
        import mne
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fs6 ROI extraction requires `mne`. Install with "
            "`pip install rabbit[fs6]`."
        ) from e

    if subjects_dir is None:
        subjects_dir = str(mne.datasets.sample.data_path() / "subjects")

    labels_by_hemi: dict[str, dict[str, "mne.Label"]] = {}
    for hemi in ("lh", "rh"):
        labels = mne.read_labels_from_annot(
            "fsaverage6", "HCPMMP1", hemi, subjects_dir=subjects_dir,
        )
        labels_by_hemi[hemi] = {lab.name.split("-")[0]: lab for lab in labels}

    roi_verts: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for roi_base, parcels in roi_parcels.items():
        for hemi, prefix in (("lh", "L_"), ("rh", "R_")):
            verts: list[int] = []
            for parcel_stem in parcels:
                full = f"{prefix}{parcel_stem}_ROI"
                if full in labels_by_hemi[hemi]:
                    verts.extend(labels_by_hemi[hemi][full].vertices.tolist())
            roi_verts[f"{roi_base}_{hemi}"] = np.array(
                sorted(set(verts)), dtype=np.int64,
            )
    return roi_verts


def extract_roi_from_fs6(
    full_cortex_fmri: np.ndarray,
    roi_verts: "OrderedDict[str, np.ndarray]",
    z_score: bool = True,
) -> np.ndarray:
    """Slice a full-cortex fs6 fMRI tensor into the 30-ROI flat layout.

    Args:
        full_cortex_fmri: ``(n_TRs, 81924)`` fs6 fMRI (LH then RH along the
            vertex axis; 40,962 per hemisphere).
        roi_verts: from ``build_fs6_roi_vertex_indices``. Determines the ROI
            slicing order in the output.
        z_score: when True, z-score each ROI's voxel block along time.

    Returns:
        ``(n_TRs, sum(len(v) for v in roi_verts.values()))`` float32 tensor.
        Concatenated in the order of ``roi_verts.keys()``.
    """
    if full_cortex_fmri.shape[1] != 2 * NV_FS6:
        raise ValueError(
            f"Expected fs6 fMRI with {2 * NV_FS6} vertices, got "
            f"{full_cortex_fmri.shape[1]}."
        )
    lh = full_cortex_fmri[:, :NV_FS6]
    rh = full_cortex_fmri[:, NV_FS6:]

    blocks: list[np.ndarray] = []
    for roi_key, verts in roi_verts.items():
        block = (lh if roi_key.endswith("_lh") else rh)[:, verts]
        if z_score:
            m = block.mean(axis=0, keepdims=True)
            s = block.std(axis=0, keepdims=True)
            s = np.where(s < 1e-12, 1.0, s)
            block = (block - m) / s
        blocks.append(block.astype(np.float32))
    return np.concatenate(blocks, axis=1)
