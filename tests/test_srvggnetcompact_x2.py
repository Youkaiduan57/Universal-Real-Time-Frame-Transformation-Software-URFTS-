"""Validation for the locally converted official SRVGGNetCompact x2 model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

pytest.importorskip("torch")

from ai_processor import AIProcessor
from srvggnetcompact_x2_onnx import (
    OFFICIAL_WEIGHTS_SHA256,
    OFFICIAL_WEIGHTS_URL,
    file_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "models" / "RealESRGANv2-animevideo-xsx2.pth"
ONNX_PATH = PROJECT_ROOT / "models" / "SRVGGNetCompact_x2.onnx"


def test_official_weights_and_converted_model_provenance() -> None:
    assert WEIGHTS_PATH.is_file()
    assert file_sha256(WEIGHTS_PATH) == OFFICIAL_WEIGHTS_SHA256
    assert ONNX_PATH.is_file()

    model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model)
    metadata = {entry.key: entry.value for entry in model.metadata_props}

    assert metadata["architecture"] == "SRVGGNetCompact"
    assert metadata["scale"] == "2"
    assert metadata["source_weights"] == OFFICIAL_WEIGHTS_URL
    assert metadata["source_weights_sha256"] == OFFICIAL_WEIGHTS_SHA256


def test_converted_model_loads_and_exposes_expected_metadata() -> None:
    processor = AIProcessor(
        ONNX_PATH,
        input_layout="nchw",
        output_layout="nchw",
        color_order="rgb",
        provider="cpu",
        scale=2,
    )
    processor.initialize()

    assert processor.active_providers == ("CPUExecutionProvider",)
    assert processor.input_metadata is not None
    assert processor.input_metadata.name == "input"
    assert processor.input_metadata.shape == (1, 3, "height", "width")
    assert processor.output_metadata is not None
    assert processor.output_metadata.name == "output"
    assert processor.output_metadata.shape == (
        1,
        3,
        "output_height",
        "output_width",
    )

    processor.shutdown()
    processor.shutdown()
    assert processor.initialized is False


def test_converted_model_cpu_inference_produces_true_x2_output() -> None:
    processor = AIProcessor(ONNX_PATH, provider="cpu", scale="auto")
    processor.initialize()
    frame = np.random.default_rng(2026).integers(
        0,
        256,
        size=(8, 12, 3),
        dtype=np.uint8,
    )

    output = processor.process(frame)

    assert output.shape == (16, 24, 3)
    assert output.dtype == np.uint8
    assert processor.detected_scale == 2
    assert processor.output_dimensions == (24, 16)
    processor.shutdown()


@pytest.mark.skipif(
    "DmlExecutionProvider" not in ort.get_available_providers(),
    reason="DirectML is unavailable",
)
def test_converted_model_directml_inference_produces_true_x2_output() -> None:
    processor = AIProcessor(
        ONNX_PATH,
        provider="directml",
        device_id=0,
        scale=2,
    )
    processor.initialize()
    frame = np.zeros((8, 12, 3), dtype=np.uint8)

    output = processor.process(frame)

    assert "DmlExecutionProvider" in processor.active_providers
    assert output.shape == (16, 24, 3)
    assert processor.detected_scale == 2
    processor.shutdown()
