"""Tests for configuration model validation."""

from __future__ import annotations

import pytest

from config import (
    ApplicationConfig,
    CaptureRegion,
    normalize_upscaling_method,
    validate_fsr1_like_strength,
)


def test_capture_region_requires_positive_dimensions() -> None:
    with pytest.raises(ValueError):
        CaptureRegion(width=0)


def test_application_config_requires_positive_output_dimensions() -> None:
    with pytest.raises(ValueError):
        ApplicationConfig(output_width=0)


def test_application_config_requires_thread_candidates() -> None:
    with pytest.raises(ValueError):
        ApplicationConfig(opencv_thread_candidates=())


def test_fsr1_like_is_a_supported_upscaling_method() -> None:
    assert normalize_upscaling_method("FSR1_LIKE") == "fsr1_like"


@pytest.mark.parametrize("strength", (-0.01, 1.01))
def test_fsr1_like_strength_rejects_values_outside_normalized_range(
    strength: float,
) -> None:
    with pytest.raises(ValueError):
        validate_fsr1_like_strength(strength, "test_strength")
