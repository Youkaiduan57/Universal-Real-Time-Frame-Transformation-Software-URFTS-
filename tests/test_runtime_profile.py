"""Tests for persisted runtime profiles."""

from __future__ import annotations

from pathlib import Path

import main

from config import (
    FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
    FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
)
from runtime_profile import RuntimeProfile


def test_runtime_profile_save_and_load_round_trip(tmp_path: Path) -> None:
    profile = RuntimeProfile(
        capture_backend="auto",
        opencv_threads=4,
        upscaling_method="bicubic",
        processing_backend="opencv_cpu",
        fsr1_like_sharpening_enabled=False,
        fsr1_like_sharpening_strength=0.4,
        fsr1_like_edge_strength=0.6,
    )

    profile_path = tmp_path / "configs" / "runtime_profile.json"
    profile.save(profile_path)

    loaded_profile = RuntimeProfile.load(profile_path)

    assert loaded_profile == profile


def test_runtime_profile_defaults_missing_processing_backend() -> None:
    profile = RuntimeProfile.from_dict(
        {
            "capture_backend": "auto",
            "opencv_threads": 8,
            "upscaling_method": "bicubic",
        }
    )

    assert profile.processing_backend == "auto"


def test_old_runtime_profile_defaults_missing_fsr1_like_settings() -> None:
    profile = RuntimeProfile.from_dict(
        {
            "capture_backend": "auto",
            "opencv_threads": 8,
            "upscaling_method": "bicubic",
            "processing_backend": "opencv_cpu",
        }
    )

    assert profile.fsr1_like_sharpening_enabled is FSR1_LIKE_DEFAULT_SHARPENING_ENABLED
    assert profile.fsr1_like_sharpening_strength == FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH
    assert profile.fsr1_like_edge_strength == FSR1_LIKE_DEFAULT_EDGE_STRENGTH


def test_retune_preserves_saved_upscaler_settings(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime_profile.json"
    profile = RuntimeProfile(
        capture_backend="auto",
        opencv_threads=4,
        upscaling_method="fsr1_like",
        processing_backend="opencv_cpu",
        fsr1_like_sharpening_enabled=False,
        fsr1_like_sharpening_strength=0.35,
        fsr1_like_edge_strength=0.55,
    )
    profile.save(profile_path)

    loaded_profile = main._load_runtime_profile(profile_path, force_retune=True)

    assert loaded_profile == profile
