"""Lightweight, deterministic spatial upscaling inspired by FSR 1 stages.

This is an original EASU/RCAS-like implementation. It is not AMD FidelityFX
FSR 1 and does not claim compatibility with AMD's published implementation.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
    FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
    validate_fsr1_like_sharpening_enabled,
    validate_fsr1_like_strength,
)

_EDGE_THRESHOLD = 12
_CROSS_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array.")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR frame with shape HxWx3.")

    if frame.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR frame.")

    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("Frame dimensions must be greater than zero.")


def _validate_dimension(value: int, field_name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")

    return value


def edge_adaptive_upscale(
    frame: np.ndarray,
    output_width: int,
    output_height: int,
    strength: float = FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
) -> np.ndarray:
    """Upscale a BGR frame using a local gradient-adaptive interpolation blend.

    Bilinear interpolation provides the stable base and nearest-neighbor
    interpolation provides a non-ringing edge-preserving candidate. A Sobel
    luma mask blends toward that candidate only around detected source edges;
    flat regions remain bilinear. The bounded blend cannot overshoot either
    interpolation candidate, avoiding the halos of an unconstrained edge kernel.
    """

    _validate_frame(frame)
    output_width = _validate_dimension(output_width, "output_width")
    output_height = _validate_dimension(output_height, "output_height")
    strength = validate_fsr1_like_strength(strength, "fsr1_like_edge_strength")

    output_size = (output_width, output_height)
    if strength == 0.0 or frame.shape[:2] == (output_height, output_width):
        return cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)

    luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.convertScaleAbs(
        cv2.Sobel(luminance, cv2.CV_16S, 1, 0, ksize=3)
    )
    gradient_y = cv2.convertScaleAbs(
        cv2.Sobel(luminance, cv2.CV_16S, 0, 1, ksize=3)
    )
    gradient_magnitude = cv2.add(gradient_x, gradient_y)
    edge_mask = gradient_magnitude > _EDGE_THRESHOLD

    if not bool(np.any(edge_mask)):
        return cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)

    source_edge_mask = np.where(edge_mask, 255, 0).astype(np.uint8)
    resized_edge_mask = cv2.resize(
        source_edge_mask,
        output_size,
        interpolation=cv2.INTER_NEAREST,
    )
    bilinear = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
    nearest = cv2.resize(frame, output_size, interpolation=cv2.INTER_NEAREST)
    edge_candidate = cv2.addWeighted(
        bilinear,
        1.0 - strength,
        nearest,
        strength,
        0.0,
    )
    cv2.copyTo(edge_candidate, resized_edge_mask, bilinear)

    return bilinear


def apply_fsr1_like_sharpening(
    frame: np.ndarray,
    strength: float = FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
) -> np.ndarray:
    """Apply a conservative, locally clamped unsharp pass to a BGR frame."""

    _validate_frame(frame)
    strength = validate_fsr1_like_strength(
        strength,
        "fsr1_like_sharpening_strength",
    )

    if strength == 0.0:
        return frame.copy()

    blurred = cv2.GaussianBlur(
        frame,
        (3, 3),
        sigmaX=0.0,
        borderType=cv2.BORDER_REFLECT101,
    )
    sharpened = cv2.addWeighted(
        frame,
        1.0 + strength,
        blurred,
        -strength,
        0.0,
    )

    local_minimum = cv2.erode(frame, _CROSS_KERNEL)
    local_maximum = cv2.dilate(frame, _CROSS_KERNEL)
    np.maximum(sharpened, local_minimum, out=sharpened)
    np.minimum(sharpened, local_maximum, out=sharpened)

    return sharpened


def fsr1_like_upscale(
    frame: np.ndarray,
    output_width: int,
    output_height: int,
    edge_strength: float = FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    sharpening_strength: float = FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
    sharpening_enabled: bool = FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
) -> np.ndarray:
    """Run the explicit edge-adaptive and optional sharpening stages."""

    sharpening_enabled = validate_fsr1_like_sharpening_enabled(sharpening_enabled)
    upscaled = edge_adaptive_upscale(
        frame=frame,
        output_width=output_width,
        output_height=output_height,
        strength=edge_strength,
    )

    if not sharpening_enabled:
        return upscaled

    return apply_fsr1_like_sharpening(
        upscaled,
        strength=sharpening_strength,
    )
