"""Tests for the lightweight edge-adaptive and sharpening stages."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from fsr1_like import (
    apply_fsr1_like_sharpening,
    edge_adaptive_upscale,
    fsr1_like_upscale,
)
from processing_backend import OpenCVProcessingBackend


def sample_frame() -> np.ndarray:
    y_coordinates, x_coordinates = np.indices((17, 23))
    return np.stack(
        (
            (x_coordinates * 11) % 256,
            (y_coordinates * 17) % 256,
            ((x_coordinates + y_coordinates) * 7) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def test_fsr1_like_output_dimensions() -> None:
    backend = OpenCVProcessingBackend(
        output_width=71,
        output_height=49,
        upscaling_method="fsr1_like",
    )

    output = backend.process(sample_frame())

    assert output.shape == (49, 71, 3)


def test_fsr1_like_preserves_uint8_dtype() -> None:
    output = fsr1_like_upscale(sample_frame(), output_width=46, output_height=34)

    assert output.dtype == np.uint8


@pytest.mark.parametrize(
    ("output_width", "output_height"),
    ((31, 29), (53, 41), (92, 68)),
)
def test_fsr1_like_supports_arbitrary_scale_factors(
    output_width: int,
    output_height: int,
) -> None:
    output = fsr1_like_upscale(
        sample_frame(),
        output_width=output_width,
        output_height=output_height,
    )

    assert output.shape == (output_height, output_width, 3)


def test_fsr1_like_sharpening_can_be_disabled() -> None:
    frame = sample_frame()
    edge_only = edge_adaptive_upscale(frame, 51, 37, strength=0.4)
    full_path_with_sharpening_disabled = fsr1_like_upscale(
        frame,
        output_width=51,
        output_height=37,
        edge_strength=0.4,
        sharpening_strength=0.9,
        sharpening_enabled=False,
    )

    assert np.array_equal(full_path_with_sharpening_disabled, edge_only)


@pytest.mark.parametrize("strength", (-0.1, 1.1))
def test_sharpening_strength_validation(strength: float) -> None:
    with pytest.raises(ValueError):
        apply_fsr1_like_sharpening(sample_frame(), strength=strength)


@pytest.mark.parametrize("strength", (-0.1, 1.1))
def test_edge_adaptive_strength_validation(strength: float) -> None:
    with pytest.raises(ValueError):
        edge_adaptive_upscale(sample_frame(), 46, 34, strength=strength)


def test_fsr1_like_is_deterministic_for_identical_input() -> None:
    frame = sample_frame()

    first_output = fsr1_like_upscale(frame, output_width=57, output_height=43)
    second_output = fsr1_like_upscale(frame, output_width=57, output_height=43)

    assert np.array_equal(first_output, second_output)


def test_edge_adaptive_stage_is_not_plain_bilinear() -> None:
    frame = sample_frame()
    adaptive_output = edge_adaptive_upscale(frame, 57, 43, strength=0.5)
    bilinear_output = cv2.resize(frame, (57, 43), interpolation=cv2.INTER_LINEAR)

    assert not np.array_equal(adaptive_output, bilinear_output)


def test_sharpening_does_not_create_values_outside_local_extrema() -> None:
    frame = sample_frame()
    sharpened = apply_fsr1_like_sharpening(frame, strength=1.0)

    assert sharpened.dtype == np.uint8
    assert sharpened.min() >= frame.min()
    assert sharpened.max() <= frame.max()
