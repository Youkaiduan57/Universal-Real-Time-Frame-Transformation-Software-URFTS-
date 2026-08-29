"""Runtime and numerical validation for the converted official RIFE v3.6 model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from frame_interpolator import RIFEInterpolator
from rife_v3_6_onnx import validate_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "models" / "RIFE_v3.6_flownet.pkl"
ONNX_PATH = PROJECT_ROOT / "models" / "RIFE_v3.6.onnx"


def test_converted_rife_v3_6_runs_real_cpu_interpolation() -> None:
    interpolator = RIFEInterpolator(ONNX_PATH, provider="cpu")
    interpolator.initialize()
    try:
        for height, width in ((19, 27), (35, 51)):
            frame_a = np.random.default_rng(height).integers(
                0, 256, size=(height, width, 3), dtype=np.uint8
            )
            frame_b = np.random.default_rng(width).integers(
                0, 256, size=(height, width, 3), dtype=np.uint8
            )
            output = interpolator.interpolate(frame_a, frame_b)

            assert output.shape == frame_a.shape
            assert output.dtype == np.uint8
            assert np.isfinite(output).all()
    finally:
        interpolator.shutdown()


def test_converted_rife_v3_6_matches_official_pytorch_weights() -> None:
    maximum_error, mean_error = validate_export(
        WEIGHTS_PATH,
        ONNX_PATH,
        height=32,
        width=32,
        tolerance=1e-4,
    )

    assert maximum_error <= 1e-4
    assert mean_error <= 1e-5
