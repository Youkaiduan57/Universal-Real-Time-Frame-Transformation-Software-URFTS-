"""Focused tests for the ONNX image-to-image processing infrastructure."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ai_processor
import main
from ai_processor import AIProcessor, AIProcessorError
from identity_onnx_model import create_identity_image_model
from resize_onnx_model import create_resize_image_model


class _FakeSession:
    def __init__(
        self,
        model_path: str,
        sess_options=None,
        providers=None,
        input_shape=None,
        output_shape=None,
        tensor_type: str = "tensor(float)",
        output_value=None,
        active_providers=None,
    ) -> None:
        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers or ["CPUExecutionProvider"]
        self.active_providers = active_providers
        if self.active_providers is None:
            self.active_providers = [
                provider[0] if isinstance(provider, tuple) else provider
                for provider in self.providers
            ]
        self.input_shape = input_shape or [1, 3, "height", "width"]
        self.output_shape = output_shape or [1, 3, "height", "width"]
        self.tensor_type = tensor_type
        self.output_value = output_value
        self.feeds = []

    def get_inputs(self):
        return [
            SimpleNamespace(
                name="input",
                shape=self.input_shape,
                type=self.tensor_type,
            )
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(
                name="output",
                shape=self.output_shape,
                type=self.tensor_type,
            )
        ]

    def get_providers(self):
        return self.active_providers

    def run(self, output_names, input_feed):
        self.feeds.append((output_names, input_feed))
        output = self.output_value
        if output is None:
            output = input_feed["input"]
        return [output]


def test_model_loading_prepares_rgb_nchw_and_uses_one_cpu_session(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    sessions = []

    def create_session(path, sess_options, providers):
        session = _FakeSession(path, sess_options, providers)
        sessions.append(session)
        return session

    processor = AIProcessor(model_path, session_factory=create_session)
    processor.initialize()
    processor.initialize()
    frame = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    result = processor.process(frame)

    assert len(sessions) == 1
    assert sessions[0].model_path == str(model_path)
    assert sessions[0].providers == ["CPUExecutionProvider"]
    assert processor.active_providers == ("CPUExecutionProvider",)
    output_names, input_feed = sessions[0].feeds[0]
    input_tensor = input_feed["input"]
    assert output_names == ["output"]
    assert input_tensor.shape == (1, 3, 1, 2)
    assert input_tensor.dtype == np.float32
    assert input_tensor[0, 0, 0, 0] == pytest.approx(30 / 255)
    assert np.array_equal(result, frame)


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_real_identity_model_round_trip_preserves_bgr_frame(
    tmp_path: Path,
    layout: str,
) -> None:
    model_path = create_identity_image_model(
        tmp_path / f"identity_{layout}.onnx",
        layout=layout,
    )
    processor = AIProcessor(
        model_path,
        input_layout=layout,
        output_layout=layout,
        color_order="rgb",
    )
    processor.initialize()
    frame = np.random.default_rng(7).integers(
        0,
        256,
        size=(13, 17, 3),
        dtype=np.uint8,
    )

    output = processor.process(frame)

    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert processor.detected_scale == 1
    assert processor.output_dimensions == (frame.shape[1], frame.shape[0])
    assert processor.input_metadata is not None
    assert processor.output_metadata is not None
    assert processor.input_metadata.layout == layout
    assert processor.output_metadata.layout == layout
    maximum_error = np.abs(output.astype(np.int16) - frame.astype(np.int16)).max()
    assert maximum_error <= 1


def test_bgr_model_interpretation_does_not_swap_channels(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    session = _FakeSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    processor = AIProcessor(
        model_path,
        color_order="bgr",
        session_factory=lambda *args, **kwargs: session,
    )
    processor.initialize()
    frame = np.array([[[11, 22, 33]]], dtype=np.uint8)

    output = processor.process(frame)

    input_tensor = session.feeds[0][1]["input"]
    assert input_tensor[0, :, 0, 0].tolist() == pytest.approx(
        [11 / 255, 22 / 255, 33 / 255]
    )
    assert np.array_equal(output, frame)


def test_missing_model_is_clear_error() -> None:
    with pytest.raises(AIProcessorError, match="requires an ONNX model"):
        AIProcessor().initialize()


def test_invalid_model_path_is_clear_error(tmp_path: Path) -> None:
    invalid_path = tmp_path / "missing.onnx"

    with pytest.raises(AIProcessorError, match="does not exist"):
        AIProcessor(invalid_path).initialize()


def test_backend_selection_initializes_configured_ai_processor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    initialized = []

    class DummyAIProcessor:
        def __init__(
            self,
            model_path,
            input_layout,
            output_layout,
            color_order,
            provider,
            device_id,
            scale,
            input_width,
            input_height,
            tile,
            tile_size,
            tile_overlap,
        ):
            self.model_path = model_path
            self.input_layout = input_layout
            self.output_layout = output_layout
            self.color_order = color_order
            self.provider = provider
            self.device_id = device_id
            self.scale = scale
            self.input_width = input_width
            self.input_height = input_height
            self.tile = tile
            self.tile_size = tile_size
            self.tile_overlap = tile_overlap

        def initialize(self):
            initialized.append(self.model_path)

    monkeypatch.setattr(main, "AIProcessor", DummyAIProcessor)

    processor = main._create_runtime_processor(
        processor_name="ai",
        model_path=model_path,
        processing_backend=None,
        app_config=SimpleNamespace(output_width=64, output_height=36),
        upscaling_method="bilinear",
        ai_input_layout="nhwc",
        ai_output_layout="nhwc",
        ai_color_order="bgr",
        ai_provider="directml",
        ai_device_id=2,
        ai_scale="4",
        ai_input_width=320,
        ai_input_height=180,
        ai_tile="192",
        ai_tile_overlap=12,
    )

    assert isinstance(processor, DummyAIProcessor)
    assert processor.input_layout == "nhwc"
    assert processor.output_layout == "nhwc"
    assert processor.color_order == "bgr"
    assert processor.provider == "directml"
    assert processor.device_id == 2
    assert processor.scale == "4"
    assert processor.input_width == 320
    assert processor.input_height == 180
    assert processor.tile == "192"
    assert processor.tile_size is None
    assert processor.tile_overlap == 12
    assert initialized == [model_path]


@pytest.mark.parametrize(
    ("input_shape", "error_pattern"),
    [
        ([1, "height", "width"], "rank-4"),
        ([1, 4, "height", "width"], "exactly 3 image channels"),
    ],
)
def test_unsupported_model_rank_and_channels_are_rejected(
    tmp_path: Path,
    input_shape,
    error_pattern: str,
) -> None:
    model_path = tmp_path / "unsupported.onnx"
    model_path.touch()

    def create_session(path, sess_options, providers):
        return _FakeSession(
            path,
            sess_options,
            providers,
            input_shape=input_shape,
        )

    with pytest.raises(AIProcessorError, match=error_pattern):
        AIProcessor(model_path, session_factory=create_session).initialize()


def test_unsupported_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported input layout"):
        AIProcessor("model.onnx", input_layout="chw")


def test_unsupported_model_dtype_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "uint8.onnx"
    model_path.touch()

    def create_session(path, sess_options, providers):
        return _FakeSession(
            path,
            sess_options,
            providers,
            tensor_type="tensor(uint8)",
        )

    with pytest.raises(AIProcessorError, match="float32"):
        AIProcessor(model_path, session_factory=create_session).initialize()


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((3, 4), dtype=np.uint8),
        np.zeros((3, 4, 4), dtype=np.uint8),
        np.zeros((3, 4, 3), dtype=np.float32),
    ],
)
def test_invalid_input_frame_rank_channel_and_dtype_are_rejected(
    tmp_path: Path,
    frame: np.ndarray,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    processor = AIProcessor(model_path, session_factory=_FakeSession)
    processor.initialize()

    with pytest.raises(AIProcessorError, match="Input frame"):
        processor.process(frame)


def test_invalid_runtime_output_shape_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    invalid_output = np.zeros((1, 4, 2, 2), dtype=np.float32)

    def create_session(path, sess_options, providers):
        return _FakeSession(
            path,
            sess_options,
            providers,
            output_value=invalid_output,
        )

    processor = AIProcessor(model_path, session_factory=create_session)
    processor.initialize()

    with pytest.raises(AIProcessorError, match="exactly 3 channels"):
        processor.process(np.zeros((2, 2, 3), dtype=np.uint8))


def test_shutdown_releases_session_and_is_idempotent(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    processor = AIProcessor(model_path, session_factory=_FakeSession)
    processor.initialize()

    processor.shutdown()
    processor.shutdown()

    assert processor.initialized is False
    assert processor.active_providers == ()
    assert processor.input_metadata is None
    assert processor.output_metadata is None
    assert processor.detected_scale is None
    assert processor.output_dimensions is None
    assert processor.last_inference_ms is None
    with pytest.raises(AIProcessorError, match="not initialized"):
        processor.process(np.zeros((1, 1, 3), dtype=np.uint8))


def test_directml_provider_selection_and_required_session_options(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    sessions = []

    monkeypatch.setattr(
        ai_processor.ort,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )

    def create_session(path, sess_options, providers):
        session = _FakeSession(
            path,
            sess_options,
            providers,
            active_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        sessions.append(session)
        return session

    processor = AIProcessor(
        model_path,
        provider="directml",
        device_id=2,
        session_factory=create_session,
    )
    processor.initialize()

    assert len(sessions) == 1
    assert sessions[0].providers == [
        ("DmlExecutionProvider", {"device_id": 2})
    ]
    assert sessions[0].sess_options.enable_mem_pattern is False
    assert (
        sessions[0].sess_options.execution_mode
        == ai_processor.ort.ExecutionMode.ORT_SEQUENTIAL
    )
    assert (
        sessions[0].sess_options.graph_optimization_level
        == ai_processor.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    assert processor.active_providers == (
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    )


def test_directml_unavailable_raises_without_creating_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    session_created = False

    monkeypatch.setattr(
        ai_processor.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    def create_session(*args, **kwargs):
        nonlocal session_created
        session_created = True
        raise AssertionError("Session creation must not be attempted.")

    processor = AIProcessor(
        model_path,
        provider="directml",
        session_factory=create_session,
    )

    with pytest.raises(AIProcessorError, match="DmlExecutionProvider is unavailable"):
        processor.initialize()

    assert session_created is False
    assert processor.initialized is False


def test_directml_active_provider_verification_rejects_cpu_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    monkeypatch.setattr(
        ai_processor.ort,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )

    def create_session(path, sess_options, providers):
        return _FakeSession(
            path,
            sess_options,
            providers,
            active_providers=["CPUExecutionProvider"],
        )

    processor = AIProcessor(
        model_path,
        provider="directml",
        session_factory=create_session,
    )

    with pytest.raises(AIProcessorError, match="DmlExecutionProvider is not active"):
        processor.initialize()

    assert processor.initialized is False
    assert processor.active_providers == ()


def test_negative_directml_device_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero or greater"):
        AIProcessor("model.onnx", provider="directml", device_id=-1)


@pytest.mark.parametrize(
    ("scale", "layout"),
    [
        (2, "nchw"),
        (2, "nhwc"),
        (3, "nchw"),
        (4, "nchw"),
        (4, "nhwc"),
    ],
)
def test_real_resize_model_auto_detects_supported_scale(
    tmp_path: Path,
    scale: int,
    layout: str,
) -> None:
    model_path = create_resize_image_model(
        tmp_path / f"resize_{layout}_{scale}x.onnx",
        scale=scale,
        layout=layout,
    )
    processor = AIProcessor(
        model_path,
        input_layout=layout,
        output_layout=layout,
        scale="auto",
    )
    processor.initialize()
    frame = np.random.default_rng(19).integers(
        0,
        256,
        size=(3, 5, 3),
        dtype=np.uint8,
    )

    output = processor.process(frame)

    assert output.shape == (3 * scale, 5 * scale, 3)
    assert processor.detected_scale == scale
    assert processor.output_width == 5 * scale
    assert processor.output_height == 3 * scale
    assert processor.output_dimensions == (5 * scale, 3 * scale)


def test_explicit_scale_accepts_match_and_rejects_mismatch(tmp_path: Path) -> None:
    model_path = create_resize_image_model(
        tmp_path / "resize_2x.onnx",
        scale=2,
    )
    frame = np.zeros((3, 5, 3), dtype=np.uint8)
    matching_processor = AIProcessor(model_path, scale=2)
    matching_processor.initialize()

    output = matching_processor.process(frame)

    assert output.shape == (6, 10, 3)
    assert matching_processor.detected_scale == 2

    mismatched_processor = AIProcessor(model_path, scale="4")
    mismatched_processor.initialize()
    with pytest.raises(AIProcessorError, match="2x, but 4x was requested"):
        mismatched_processor.process(frame)
    assert mismatched_processor.detected_scale is None
    assert mismatched_processor.output_dimensions is None


def test_mismatched_horizontal_and_vertical_scales_are_rejected(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "mismatched.onnx"
    model_path.touch()
    output_value = np.zeros((1, 3, 4, 6), dtype=np.float32)

    def create_session(path, sess_options, providers):
        return _FakeSession(
            path,
            sess_options,
            providers,
            output_value=output_value,
        )

    processor = AIProcessor(model_path, session_factory=create_session)
    processor.initialize()

    with pytest.raises(AIProcessorError, match="mismatched horizontal and vertical"):
        processor.process(np.zeros((2, 2, 3), dtype=np.uint8))


def test_non_integer_output_scale_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "non_integer.onnx"
    model_path.touch()
    output_value = np.zeros((1, 3, 5, 4), dtype=np.float32)

    def create_session(path, sess_options, providers):
        return _FakeSession(
            path,
            sess_options,
            providers,
            output_value=output_value,
        )

    processor = AIProcessor(model_path, session_factory=create_session)
    processor.initialize()

    with pytest.raises(AIProcessorError, match="must be an integer"):
        processor.process(np.zeros((2, 2, 3), dtype=np.uint8))


def test_output_metadata_and_dimensions_are_exposed(tmp_path: Path) -> None:
    model_path = create_resize_image_model(
        tmp_path / "resize_2x.onnx",
        scale=2,
    )
    processor = AIProcessor(model_path)
    processor.initialize()

    assert processor.input_metadata is not None
    assert processor.input_metadata.name == "input"
    assert processor.input_metadata.dtype == "tensor(float)"
    assert processor.input_metadata.layout == "nchw"
    assert processor.output_metadata is not None
    assert processor.output_metadata.name == "output"
    assert processor.output_metadata.dtype == "tensor(float)"
    assert processor.output_metadata.layout == "nchw"

    processor.process(np.zeros((4, 6, 3), dtype=np.uint8))

    assert processor.detected_scale == 2
    assert processor.output_dimensions == (12, 8)


@pytest.mark.parametrize(
    ("input_width", "input_height"),
    [(320, None), (None, 180)],
)
def test_internal_resolution_requires_paired_dimensions(
    input_width: int | None,
    input_height: int | None,
) -> None:
    with pytest.raises(ValueError, match="supplied together"):
        AIProcessor(
            "model.onnx",
            input_width=input_width,
            input_height=input_height,
        )


@pytest.mark.parametrize(
    ("input_width", "input_height"),
    [(0, 180), (-1, 180), (320, 0), (320, -1)],
)
def test_internal_resolution_requires_positive_dimensions(
    input_width: int,
    input_height: int,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        AIProcessor(
            "model.onnx",
            input_width=input_width,
            input_height=input_height,
        )


def test_internal_resolution_resizes_before_inference_and_records_dimensions(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "resize.onnx"
    model_path.touch()
    sessions = []

    class TwoXSession(_FakeSession):
        def run(self, output_names, input_feed):
            self.feeds.append((output_names, input_feed))
            tensor = input_feed["input"]
            return [np.repeat(np.repeat(tensor, 2, axis=2), 2, axis=3)]

    def create_session(path, sess_options, providers):
        session = TwoXSession(path, sess_options, providers)
        sessions.append(session)
        return session

    processor = AIProcessor(
        model_path,
        scale=2,
        input_width=5,
        input_height=3,
        session_factory=create_session,
    )
    processor.initialize()
    frame = np.random.default_rng(31).integers(
        0,
        256,
        size=(8, 12, 3),
        dtype=np.uint8,
    )

    output = processor.process(frame)

    assert sessions[0].feeds[0][1]["input"].shape == (1, 3, 3, 5)
    assert output.shape == (6, 10, 3)
    assert processor.original_capture_dimensions == (12, 8)
    assert processor.ai_input_dimensions == (5, 3)
    assert processor.output_dimensions == (10, 6)
    assert processor.detected_scale == 2


def test_no_internal_resolution_preserves_captured_dimensions(tmp_path: Path) -> None:
    model_path = tmp_path / "identity.onnx"
    model_path.touch()
    processor = AIProcessor(model_path, session_factory=_FakeSession)
    processor.initialize()
    frame = np.random.default_rng(37).integers(
        0,
        256,
        size=(4, 7, 3),
        dtype=np.uint8,
    )

    output = processor.process(frame)

    assert np.array_equal(output, frame)
    assert processor.original_capture_dimensions == (7, 4)
    assert processor.ai_input_dimensions == (7, 4)
    assert processor.output_dimensions == (7, 4)
    assert processor.detected_scale == 1


def test_large_unbounded_ai_input_logs_one_clear_warning(
    monkeypatch,
    caplog,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "identity.onnx"
    model_path.touch()
    monkeypatch.setattr(ai_processor, "AI_LARGE_INPUT_PIXEL_THRESHOLD", 5)
    processor = AIProcessor(model_path, session_factory=_FakeSession)
    processor.initialize()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    with caplog.at_level("WARNING"):
        processor.process(frame)
        processor.process(frame)

    warnings = [
        record
        for record in caplog.records
        if "without an internal resolution limit" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "3x2 (6 pixels)" in warnings[0].getMessage()
    assert "--ai-input-width" in warnings[0].getMessage()


def test_internal_resolution_metadata_is_cleared_by_idempotent_shutdown(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "identity.onnx"
    model_path.touch()
    processor = AIProcessor(
        model_path,
        input_width=4,
        input_height=3,
        session_factory=_FakeSession,
    )
    processor.initialize()
    processor.process(np.zeros((6, 8, 3), dtype=np.uint8))

    processor.shutdown()
    processor.shutdown()

    assert processor.original_capture_dimensions is None
    assert processor.ai_input_dimensions is None


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_tiled_stitching_preserves_two_x_output_and_image_content(
    layout: str,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / f"two_x_{layout}.onnx"
    model_path.touch()

    class TwoXSession(_FakeSession):
        def __init__(self, *args, **kwargs):
            shape = [1, 3, "height", "width"]
            if layout == "nhwc":
                shape = [1, "height", "width", 3]
            super().__init__(*args, input_shape=shape, output_shape=shape, **kwargs)

        def run(self, output_names, input_feed):
            self.feeds.append((output_names, input_feed))
            tensor = input_feed["input"]
            spatial_axes = (2, 3) if layout == "nchw" else (1, 2)
            output = np.repeat(tensor, 2, axis=spatial_axes[0])
            output = np.repeat(output, 2, axis=spatial_axes[1])
            return [output]

    processor = AIProcessor(
        model_path,
        input_layout=layout,
        output_layout=layout,
        scale=2,
        tile=5,
        tile_overlap=1,
        session_factory=TwoXSession,
    )
    processor.initialize()
    frame = np.random.default_rng(41).integers(
        0,
        256,
        size=(7, 9, 3),
        dtype=np.uint8,
    )

    output = processor.process(frame)

    expected = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
    assert np.array_equal(output, expected)
    assert output.shape == (14, 18, 3)
    assert processor.detected_scale == 2
    assert processor.output_dimensions == (18, 14)
    assert processor.selected_tile_size == 5
    assert processor.tiles_processed == 4


def test_tile_overlap_is_feather_blended_instead_of_hard_cut(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "blend.onnx"
    model_path.touch()

    class TileValueSession(_FakeSession):
        def run(self, output_names, input_feed):
            self.feeds.append((output_names, input_feed))
            value = np.float32(len(self.feeds) - 1)
            return [np.full_like(input_feed["input"], value)]

    processor = AIProcessor(
        model_path,
        color_order="bgr",
        tile=5,
        tile_overlap=2,
        session_factory=TileValueSession,
    )
    processor.initialize()

    output = processor.process(np.zeros((1, 8, 3), dtype=np.uint8))

    assert output[0, :, 0].tolist() == [0, 0, 0, 85, 170, 255, 255, 255]
    assert processor.tiles_processed == 2


def test_tiling_handles_smaller_edge_tiles_without_uncovered_pixels(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "edges.onnx"
    model_path.touch()
    sessions = []

    def create_session(path, sess_options, providers):
        session = _FakeSession(path, sess_options, providers)
        sessions.append(session)
        return session

    processor = AIProcessor(
        model_path,
        tile=4,
        tile_overlap=1,
        session_factory=create_session,
    )
    processor.initialize()
    frame = np.random.default_rng(43).integers(
        0,
        256,
        size=(3, 11, 3),
        dtype=np.uint8,
    )

    output = processor.process(frame)

    feed_widths = [feed[1]["input"].shape[3] for feed in sessions[0].feeds]
    assert feed_widths == [4, 4, 4, 2]
    assert np.array_equal(output, frame)
    assert processor.tiles_processed == 4


def test_auto_tiling_selects_largest_safe_candidate_and_exposes_metadata(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "auto_two_x.onnx"
    model_path.touch()

    class TwoXSession(_FakeSession):
        def run(self, output_names, input_feed):
            self.feeds.append((output_names, input_feed))
            tensor = input_feed["input"]
            return [np.repeat(np.repeat(tensor, 2, axis=2), 2, axis=3)]

    processor = AIProcessor(
        model_path,
        scale=2,
        tile="auto",
        tile_overlap=16,
        session_factory=TwoXSession,
    )
    processor.initialize()

    output = processor.process(np.zeros((360, 640, 3), dtype=np.uint8))

    assert output.shape == (720, 1280, 3)
    assert processor.selected_tile_size == 256
    assert processor.tiles_processed == 6
    assert processor.estimated_peak_tile_bytes is not None
    assert (
        processor.estimated_peak_tile_bytes
        <= ai_processor.AI_AUTO_TILE_MEMORY_BUDGET_BYTES
    )
    assert processor.original_capture_dimensions == (640, 360)
    assert processor.ai_input_dimensions == (640, 360)
    assert processor.output_dimensions == (1280, 720)


@pytest.mark.parametrize(
    ("kwargs", "error_pattern"),
    [
        ({"tile": 0}, "greater than zero"),
        ({"tile": "invalid"}, "auto.*off.*positive integer"),
        ({"tile": 32, "tile_overlap": -1}, "zero or greater"),
        ({"tile": 32, "tile_overlap": 16}, "greater than twice"),
        ({"tile": "off", "tile_size": 32}, "tiling is off"),
        ({"tile": 64, "tile_size": 96}, "Conflicting"),
    ],
)
def test_invalid_tile_configuration_is_rejected(kwargs, error_pattern: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error_pattern):
        AIProcessor("model.onnx", **kwargs)


def test_shutdown_clears_tiling_metadata_and_remains_idempotent(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "identity.onnx"
    model_path.touch()
    processor = AIProcessor(
        model_path,
        tile=4,
        tile_overlap=1,
        session_factory=_FakeSession,
    )
    processor.initialize()
    processor.process(np.zeros((6, 8, 3), dtype=np.uint8))
    assert processor.tiles_processed > 1

    processor.shutdown()
    processor.shutdown()

    assert processor.selected_tile_size is None
    assert processor.tiles_processed == 0
    assert processor.estimated_peak_tile_bytes is None


def test_ai_processor_exposes_preprocess_inference_and_postprocess_timings(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "identity.onnx"
    model_path.touch()
    processor = AIProcessor(model_path, session_factory=_FakeSession)
    processor.initialize()

    processor.process(np.zeros((4, 6, 3), dtype=np.uint8))

    assert processor.last_preprocessing_ms is not None
    assert processor.last_preprocessing_ms >= 0.0
    assert processor.last_inference_ms is not None
    assert processor.last_inference_ms >= 0.0
    assert processor.last_postprocessing_ms is not None
    assert processor.last_postprocessing_ms >= 0.0

    processor.shutdown()
    assert processor.last_preprocessing_ms is None
    assert processor.last_inference_ms is None
    assert processor.last_postprocessing_ms is None
