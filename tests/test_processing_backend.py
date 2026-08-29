"""Tests for processing backends and backend selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import main
from config import normalize_processing_backend
from frame_processor import FrameProcessor
from processing_backend import (
    OpenCVProcessingBackend,
    TorchCudaProcessingBackend,
)


def test_processing_backend_rejects_unsupported_name() -> None:
    with pytest.raises(ValueError):
        normalize_processing_backend("unsupported-backend")


def test_opencv_processing_backend_preserves_dimensions_and_dtype() -> None:
    backend = OpenCVProcessingBackend(
        output_width=64,
        output_height=36,
        upscaling_method="bicubic",
    )
    frame = np.zeros((18, 32, 3), dtype=np.uint8)

    output_frame = backend.process(frame)

    assert output_frame.shape == (36, 64, 3)
    assert output_frame.dtype == np.uint8


def test_frame_processor_uses_selected_backend() -> None:
    class DummyBackend:
        backend_name = "dummy"
        display_name = "Dummy Backend"

        def __init__(self) -> None:
            self.process_calls = 0

        def process(self, frame):
            self.process_calls += 1
            return frame + 1

    backend = DummyBackend()
    processor = FrameProcessor(processing_backend=backend)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    output_frame = processor.process(frame)

    assert backend.process_calls == 1
    assert np.array_equal(output_frame, frame + 1)


def test_cuda_backend_unavailable_behavior(monkeypatch) -> None:
    dummy_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            synchronize=lambda device=None: None,
            empty_cache=lambda: None,
        )
    )
    monkeypatch.setattr("processing_backend.torch", dummy_torch)
    monkeypatch.setattr("processing_backend.torch_functional", None)

    assert TorchCudaProcessingBackend.is_available() is False

    with pytest.raises(RuntimeError):
        TorchCudaProcessingBackend()


def test_cuda_backend_explicitly_rejects_fsr1_like_without_substitution() -> None:
    with pytest.raises(ValueError, match="does not support fsr1_like"):
        TorchCudaProcessingBackend(upscaling_method="fsr1_like")


@pytest.mark.skipif(not TorchCudaProcessingBackend.is_available(), reason="CUDA is not available")
def test_cuda_processing_backend_preserves_dimensions_and_dtype() -> None:
    backend = TorchCudaProcessingBackend(
        output_width=64,
        output_height=36,
        upscaling_method="bilinear",
    )
    frame = np.zeros((18, 32, 3), dtype=np.uint8)

    output_frame = backend.process(frame)

    assert output_frame.shape == (36, 64, 3)
    assert output_frame.dtype == np.uint8


def test_saved_unavailable_processing_backend_falls_back(monkeypatch) -> None:
    class DummyBackend:
        backend_name = "opencv_cpu"
        display_name = "OpenCV CPU"

        def is_available(self) -> bool:
            return True

    class DummyTuner:
        def __init__(self, *args, **kwargs) -> None:
            self.called = True

        def tune(self, *args, **kwargs):
            return DummyBackend()

    monkeypatch.setattr(main, "ProcessingBackendTuner", DummyTuner)
    monkeypatch.setattr(
        main,
        "create_processing_backend",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("backend missing")),
    )

    runtime_profile = SimpleNamespace(
        processing_backend="torch_cuda",
        upscaling_method="bicubic",
    )
    app_config = SimpleNamespace(output_width=64, output_height=36)
    test_frame = np.zeros((18, 32, 3), dtype=np.uint8)

    selected_backend, profile_changed = main._select_processing_backend(
        runtime_profile=runtime_profile,
        test_frame=test_frame,
        app_config=app_config,
        force_retune=False,
    )

    assert selected_backend.backend_name == "opencv_cpu"
    assert profile_changed is True


def test_saved_cuda_backend_warns_and_falls_back_for_fsr1_like(
    monkeypatch,
    caplog,
) -> None:
    observed_methods = []

    class DummyBackend:
        backend_name = "opencv_cpu"
        display_name = "OpenCV CPU"

    class DummyTuner:
        def tune(self, *args, **kwargs):
            observed_methods.append(kwargs["upscaling_method"])
            return DummyBackend()

    def reject_cuda_backend(**kwargs):
        observed_methods.append(kwargs["upscaling_method"])
        raise ValueError("PyTorch CUDA processing does not support fsr1_like")

    monkeypatch.setattr(main, "ProcessingBackendTuner", DummyTuner)
    monkeypatch.setattr(main, "create_processing_backend", reject_cuda_backend)
    runtime_profile = SimpleNamespace(
        processing_backend="torch_cuda",
        upscaling_method="fsr1_like",
        fsr1_like_sharpening_enabled=True,
        fsr1_like_sharpening_strength=0.2,
        fsr1_like_edge_strength=0.35,
    )
    app_config = SimpleNamespace(output_width=64, output_height=36)
    test_frame = np.zeros((18, 32, 3), dtype=np.uint8)

    with caplog.at_level("WARNING"):
        selected_backend, profile_changed = main._select_processing_backend(
            runtime_profile=runtime_profile,
            test_frame=test_frame,
            app_config=app_config,
            force_retune=False,
        )

    assert selected_backend.backend_name == "opencv_cpu"
    assert profile_changed is True
    assert observed_methods == ["fsr1_like", "fsr1_like"]
    assert "does not support fsr1_like" in caplog.text
