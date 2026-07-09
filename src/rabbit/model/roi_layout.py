"""ROI layout: the canonical 30-ROI specification used by RABBiT.

Each ROI is a bilateral HCP-MMP1 grouping on ``fsaverage6``. There are 15
bilateral ROIs and therefore 30 layout entries (``<base>_lh`` and
``<base>_rh``). The ``ROILayout`` dataclass records the per-ROI vertex
counts, slices into the flat prediction vector, and book-keeping for the
parcel groupings.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Canonical fs6 30-ROI definition (HCP-MMP1, Glasser et al. 2016).
#
# Mapping: base ROI name -> ordered list of HCP-MMP1 parcels (without L_/R_
# hemisphere prefix). The vertex counts below come from ``mne`` annotation
# extraction on ``fsaverage6`` and are what RABBiT trains on.
# ─────────────────────────────────────────────────────────────────────────────

ROI_PARCELS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    {
        "aud_primary":    ("A1",),
        "aud_belt":       ("A4", "A5", "LBelt", "MBelt", "PBelt"),
        "stg_sts":        ("STGa", "STSda", "STSdp", "STSva", "STSvp", "STV"),
        "temporal_pole":  ("TE1a", "TE1m", "TE1p", "TE2a", "TE2p", "TGd", "TGv"),
        "posterior_temp": ("PHT", "TPOJ1", "TPOJ2", "TPOJ3"),
        "angular_gyrus":  ("PGi", "PGs", "PGp", "PFm"),
        "supramarginal":  ("PFop", "PFt", "PFcm", "PSL"),
        "ifg":            ("44", "45", "47l"),
        "mfg_dlpfc":      ("55b", "IFJa", "IFSp", "SFL", "8Av", "8BL"),
        "mpfc_tom":       ("9m", "10r", "p32pr", "d32", "a24", "p24"),
        "precuneus_pcc":  ("7m", "7Am", "7Pm", "31pd", "31pv", "31a", "PCV"),
        "motor":          ("1", "2", "3a", "3b", "4", "6r", "6v", "6d", "6ma", "6mp", "5m", "5mv"),
        "insula_fop":     ("FOP1", "FOP2", "FOP3", "FOP4", "FOP5", "RI", "PI", "Ig"),
        "visual_early":   ("V1", "V2", "V3", "V3A", "V3B", "V4"),
        "visual_higher":  ("V6", "V6A", "V7", "V8", "LO1", "LO2", "MT", "MST", "FST"),
    }
)


# Vertex counts after fs6 HCP-MMP1 extraction (fixed; produced by the offline
# vertex-set builder, reproduced under the same atlas + fsaverage6 surface).
FS6_ROI_DIMS: "OrderedDict[str, int]" = OrderedDict(
    {
        "aud_primary_lh": 101,  "aud_primary_rh": 77,
        "aud_belt_lh": 902,     "aud_belt_rh": 812,
        "stg_sts_lh": 1074,     "stg_sts_rh": 1128,
        "temporal_pole_lh": 1770, "temporal_pole_rh": 1793,
        "posterior_temp_lh": 678, "posterior_temp_rh": 942,
        "angular_gyrus_lh": 1554, "angular_gyrus_rh": 1615,
        "supramarginal_lh": 912,  "supramarginal_rh": 847,
        "ifg_lh": 571,            "ifg_rh": 457,
        "mfg_dlpfc_lh": 1053,     "mfg_dlpfc_rh": 1035,
        "mpfc_tom_lh": 905,       "mpfc_tom_rh": 1024,
        "precuneus_pcc_lh": 1074, "precuneus_pcc_rh": 1159,
        "motor_lh": 4943,         "motor_rh": 5188,
        "insula_fop_lh": 1242,    "insula_fop_rh": 1158,
        "visual_early_lh": 2931,  "visual_early_rh": 2908,
        "visual_higher_lh": 789,  "visual_higher_rh": 752,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Layout dataclass + builder
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ROILayout:
    """Canonical ROI layout shared by the model, dataset, and inference code.

    One learned query token per ROI. The flat prediction vector concatenates
    each ROI's voxel block in ``roi_names`` order, sliced by ``roi_slices``.
    """

    roi_names:        tuple[str, ...]
    roi_dims:         "OrderedDict[str, int]"
    roi_dims_dict:    "OrderedDict[str, int]"
    roi_slices:       "OrderedDict[str, slice]"
    flat_dim:         int
    num_queries:      int

    def split(
        self,
        flat_tensor: "torch.Tensor | np.ndarray",
    ) -> "OrderedDict[str, torch.Tensor | np.ndarray]":
        """Slice a flat ROI tensor back into per-ROI blocks."""
        out: "OrderedDict[str, torch.Tensor | np.ndarray]" = OrderedDict()
        for name, slc in self.roi_slices.items():
            out[name] = flat_tensor[..., slc]
        return out


def build_flat_roi_layout(roi_dims: Mapping[str, int]) -> ROILayout:
    """Build an ROI layout from a (roi_name -> n_voxels) mapping.

    The order of ``roi_dims`` keys defines the order of slices in the flat
    prediction tensor. The most common usage is ``build_flat_roi_layout(FS6_ROI_DIMS)``.
    """
    roi_dims = OrderedDict((str(name), int(dim)) for name, dim in roi_dims.items())
    if not roi_dims:
        raise ValueError("roi_dims must contain at least one ROI.")

    roi_names = tuple(roi_dims.keys())
    roi_slices: "OrderedDict[str, slice]" = OrderedDict()
    start = 0
    for name, dim in roi_dims.items():
        if dim <= 0:
            raise ValueError(f"ROI '{name}' must have positive size, got {dim}.")
        roi_slices[name] = slice(start, start + dim)
        start += dim

    return ROILayout(
        roi_names=roi_names,
        roi_dims=roi_dims,
        roi_dims_dict=roi_dims,
        roi_slices=roi_slices,
        flat_dim=start,
        num_queries=len(roi_names),
    )


def build_fs6_layout() -> ROILayout:
    """The canonical fs6 30-ROI layout used in the paper."""
    return build_flat_roi_layout(FS6_ROI_DIMS)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def initialise_roi_queries(
    hidden_dim: int,
    roi_layout: ROILayout,
    init_std: float = 0.02,
) -> nn.Embedding:
    """Learned position embedding for each ROI query (one row per ROI)."""
    query_embed = nn.Embedding(roi_layout.num_queries, hidden_dim)
    nn.init.normal_(query_embed.weight, mean=0.0, std=init_std)
    return query_embed


def flatten_roi_blocks(
    roi_blocks: Mapping[str, "torch.Tensor | np.ndarray"],
    roi_layout: ROILayout,
) -> "torch.Tensor | np.ndarray":
    """Concatenate per-ROI blocks back into the flat prediction tensor."""
    ordered = [roi_blocks[name] for name in roi_layout.roi_names]
    first = ordered[0]
    if torch.is_tensor(first):
        return torch.cat([torch.as_tensor(b) for b in ordered], dim=-1)
    return np.concatenate([np.asarray(b) for b in ordered], axis=-1)


def base_of(roi_name: str) -> str:
    """`aud_primary_lh` -> `aud_primary`."""
    if roi_name.endswith("_lh") or roi_name.endswith("_rh"):
        return roi_name[:-3]
    return roi_name


def hemisphere_of(roi_name: str) -> Optional[str]:
    """`aud_primary_lh` -> 'lh'. Returns None for un-suffixed names."""
    if roi_name.endswith("_lh"):
        return "lh"
    if roi_name.endswith("_rh"):
        return "rh"
    return None
