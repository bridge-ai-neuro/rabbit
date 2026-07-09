"""Evaluation pipeline for RABBiT.

Public API::

    from rabbit.eval import evaluate_on_narratives, NarrativesStory

    story = NarrativesStory(
        name="21styear",
        audio_path="path/to/stim_audio_16k.npy",
        report_path="path/to/21styear.report",
        fmri_paths=[Path(f"path/to/sub_{s}.npy") for s in SUBJECT_IDS],
    )
    result = evaluate_on_narratives(model, story, tr_length=1.49, hrf_delay=6)
    # result.vertex_corr   -> (41394,) per-vertex Pearson r
    # result.roi_corrs     -> per-ROI mean r
"""

from .audio_align import (
    align_wav_to_trs_eval,
    parse_report,
    resample_fmri_to_target_tr,
)
from .fs6 import (
    NV_FS6,
    build_fs6_roi_vertex_indices,
    extract_roi_from_fs6,
)
from .metrics import correlation_4fold, pearson_r_per_voxel, zscore
from .narratives import (
    NarrativesResult,
    NarrativesStory,
    evaluate_on_narratives,
    load_narratives_audio,
    load_narratives_mean_fmri,
)

__all__ = [
    # high-level
    "NarrativesStory",
    "NarrativesResult",
    "evaluate_on_narratives",
    "load_narratives_audio",
    "load_narratives_mean_fmri",
    # alignment helpers
    "parse_report",
    "resample_fmri_to_target_tr",
    "align_wav_to_trs_eval",
    # fs6
    "NV_FS6",
    "build_fs6_roi_vertex_indices",
    "extract_roi_from_fs6",
    # metrics
    "pearson_r_per_voxel",
    "correlation_4fold",
    "zscore",
]
