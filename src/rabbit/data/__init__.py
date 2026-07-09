"""Dataset adapters for RABBiT training and evaluation."""

from .friends import (
    FriendsClipDataset,
    FriendsClipManifest,
    build_clip_audio_path,
    build_clip_fmri_path,
    parse_clip_name,
)

__all__ = [
    "FriendsClipManifest",
    "FriendsClipDataset",
    "parse_clip_name",
    "build_clip_audio_path",
    "build_clip_fmri_path",
]
