"""RABBiT — speech → fMRI ROI brain encoder.

Quick start::

    from rabbit.inference import ROIPredictor

    predictor = ROIPredictor.from_checkpoint(
        checkpoint_path="path/to/best.pt",
        config_path="configs/friends_shared_dev.yaml",
        use_avg_dev=True,
    )
    result = predictor.predict("clip.wav", tr_length=1.49, hrf_delay=6)
    # result.fmri    -> (n_TRs, 41394)
    # result.by_roi  -> OrderedDict[roi_name -> (n_TRs, V_roi)]
"""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .inference import RABBiTPrediction, ROIPredictor, load_from_checkpoint
from .model import RABBiT, ROILayout, build_fs6_layout

try:
    __version__ = _pkg_version("rabbit")
except PackageNotFoundError:  # editable install pre-build
    __version__ = "0.1.0"


def predict(
    audio,
    checkpoint_path,
    config_path,
    *,
    tr_length: float = 1.49,
    hrf_delay: int = 6,
    trim_start: int = 10,
    trim_end: int = 9,
    audio_onset: float = 0.0,
    shift: int = 0,
    device: str = None,
    use_avg_dev: bool = True,
    batch_size: int = 32,
    return_attention: bool = False,
) -> "RABBiTPrediction":
    """One-shot inference convenience: load a checkpoint, predict on audio,
    return a :class:`RABBiTPrediction`. No state retained between calls.

    For multiple predictions on the same checkpoint, instantiate a
    :class:`ROIPredictor` once and call ``predictor.predict(...)`` repeatedly.

    Args:
        audio: a filesystem path, a ``Path``, a 1-D ``np.ndarray`` waveform at
            16 kHz, or a 1-D ``torch.Tensor`` at 16 kHz.
        checkpoint_path: ``.pt`` file (legacy-style or RABBiT-native).
        config_path: YAML config matching the checkpoint.
        tr_length: TR period in seconds. Default 1.49 (Friends).
        hrf_delay: HRF lag in TRs. Default 6.
        trim_start, trim_end: edge-TR trimming.
        audio_onset: leading silence in seconds before the stimulus starts.
        shift: integer TR offset applied to the returned ``tr_times`` only.
        device: ``'cuda'`` or ``'cpu'``. Auto when None.
        use_avg_dev: apply the avg-dev trick for held-out subjects (default).
        batch_size: TR batches sent through the GPU at once.
        return_attention: include cross-attention weights in the result.
    """
    import torch as _torch  # local: keep top-level import light
    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    predictor = ROIPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path, config_path=config_path,
        device=device, use_avg_dev=use_avg_dev,
    )
    return predictor.predict(
        audio,
        tr_length=tr_length, hrf_delay=hrf_delay,
        trim_start=trim_start, trim_end=trim_end,
        audio_onset=audio_onset, shift=shift,
        batch_size=batch_size, return_attention=return_attention,
    )


__all__ = [
    "RABBiT",
    "ROILayout",
    "ROIPredictor",
    "RABBiTPrediction",
    "build_fs6_layout",
    "load_from_checkpoint",
    "predict",
    "__version__",
]


def __getattr__(name):
    # Lazy import of rabbit.eval to keep top-level import light (mne is optional).
    if name in {"evaluate_on_narratives", "NarrativesStory", "NarrativesResult"}:
        from . import eval as _eval
        return getattr(_eval, name)
    raise AttributeError(name)
