"""Strict ONNX Runtime adapter for simple image-to-image models."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
import onnxruntime as ort

from resource_validation import ResourceLease


SUPPORTED_IMAGE_LAYOUTS = ("nchw", "nhwc")
SUPPORTED_COLOR_ORDERS = ("rgb", "bgr")
SUPPORTED_EXECUTION_PROVIDERS = ("cpu", "directml")
SUPPORTED_SCALE_SETTINGS = ("auto", "1", "2", "3", "4")
AI_LARGE_INPUT_PIXEL_THRESHOLD = 640 * 360
DEFAULT_AI_TILE_SIZE = 256
DEFAULT_AI_TILE_OVERLAP = 16
AI_AUTO_TILE_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024
AI_AUTO_TILE_ACTIVATION_SAFETY_FACTOR = 64
AI_AUTO_TILE_CANDIDATES = (512, 384, 320, 256, 192, 160, 128, 96, 64)

logger = logging.getLogger(__name__)


class AIProcessorError(RuntimeError):
    """Raised when the ONNX image processor cannot be initialized or used."""


@dataclass(frozen=True, slots=True)
class ImageTensorMetadata:
    """Public description of one validated ONNX image tensor."""

    name: str
    dtype: str
    shape: tuple[object, ...]
    layout: str


class _SessionValue(Protocol):
    name: str
    shape: list[object]
    type: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> list[_SessionValue]: ...

    def get_outputs(self) -> list[_SessionValue]: ...

    def get_providers(self) -> list[str]: ...

    def run(self, output_names, input_feed): ...


SessionFactory = Callable[..., _InferenceSession]


def _normalize_adapter_choice(
    value: str,
    field_name: str,
    supported_values: tuple[str, ...],
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized_value = value.strip().lower()
    if normalized_value not in supported_values:
        supported = ", ".join(supported_values)
        raise ValueError(
            f"Unsupported {field_name}: {normalized_value}. Expected one of: {supported}."
        )

    return normalized_value


def _static_dimension(value: object, tensor_name: str) -> int | None:
    """Return a fixed positive dimension, or None for a dynamic dimension."""

    if isinstance(value, (int, np.integer)):
        dimension = int(value)
        if dimension <= 0:
            raise AIProcessorError(
                f"{tensor_name} contains unsupported non-positive dimension {dimension}."
            )
        return dimension

    if value is None or isinstance(value, str):
        return None

    raise AIProcessorError(
        f"{tensor_name} contains unsupported dimension metadata: {value!r}."
    )


def _normalize_scale_setting(value: str | int) -> int | None:
    if isinstance(value, bool):
        raise TypeError("AI scale must be 'auto' or an integer from 1 to 4.")
    if isinstance(value, int):
        normalized_value = str(value)
    elif isinstance(value, str):
        normalized_value = value.strip().lower()
    else:
        raise TypeError("AI scale must be 'auto' or an integer from 1 to 4.")

    if normalized_value not in SUPPORTED_SCALE_SETTINGS:
        raise ValueError("AI scale must be one of: auto, 1, 2, 3, 4.")

    return None if normalized_value == "auto" else int(normalized_value)


def _normalize_tile_setting(value: str | int) -> tuple[str, int | None]:
    if isinstance(value, bool):
        raise TypeError("AI tile mode must be 'auto', 'off', or a positive integer.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("AI tile size must be greater than zero.")
        return "fixed", value
    if not isinstance(value, str):
        raise TypeError("AI tile mode must be 'auto', 'off', or a positive integer.")

    normalized_value = value.strip().lower()
    if normalized_value in ("auto", "off"):
        return normalized_value, None
    try:
        tile_size = int(normalized_value)
    except ValueError as error:
        raise ValueError(
            "AI tile mode must be 'auto', 'off', or a positive integer."
        ) from error
    if tile_size <= 0:
        raise ValueError("AI tile size must be greater than zero.")
    return "fixed", tile_size


class AIProcessor:
    """Adapt OpenCV BGR frames to one strict image-to-image ONNX model.

    Only one float32 rank-4 image input and one float32 rank-4 image output are
    supported. Layout and color interpretation are explicit; the adapter does
    not infer arbitrary model preprocessing or postprocessing behavior.
    """

    backend_name = "onnx_ai"
    display_name = "ONNX Runtime AI"

    def __init__(
        self,
        model_path: str | Path | None = None,
        input_layout: str = "nchw",
        output_layout: str = "nchw",
        color_order: str = "rgb",
        provider: str = "cpu",
        device_id: int = 0,
        scale: str | int = "auto",
        input_width: int | None = None,
        input_height: int | None = None,
        tile: str | int = "off",
        tile_size: int | None = None,
        tile_overlap: int = DEFAULT_AI_TILE_OVERLAP,
        session_factory: SessionFactory | None = None,
        reuse_static_tiles: bool = False,
    ) -> None:
        self.model_path = model_path
        self.reuse_static_tiles = reuse_static_tiles
        self._tile_cache = {}
        self._tile_cache_signature = None
        self._cached_static_frame = None
        self._cached_static_output = None
        self._reuse_log_time = 0.0
        self.tiles_reused = 0
        self.last_compositing_ms = 0.0
        self.input_layout = _normalize_adapter_choice(
            input_layout,
            "input layout",
            SUPPORTED_IMAGE_LAYOUTS,
        )
        self.output_layout = _normalize_adapter_choice(
            output_layout,
            "output layout",
            SUPPORTED_IMAGE_LAYOUTS,
        )
        self.color_order = _normalize_adapter_choice(
            color_order,
            "color order",
            SUPPORTED_COLOR_ORDERS,
        )
        self.provider = _normalize_adapter_choice(
            provider,
            "execution provider",
            SUPPORTED_EXECUTION_PROVIDERS,
        )
        if isinstance(device_id, bool) or not isinstance(device_id, int):
            raise TypeError("DirectML device ID must be an integer.")
        if device_id < 0:
            raise ValueError("DirectML device ID must be zero or greater.")
        self.device_id = device_id
        self.expected_scale = _normalize_scale_setting(scale)
        self.scale_setting = "auto" if self.expected_scale is None else str(self.expected_scale)
        if (input_width is None) != (input_height is None):
            raise ValueError("AI input width and height must be supplied together.")
        for value, field_name in (
            (input_width, "AI input width"),
            (input_height, "AI input height"),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero.")
        self.configured_input_width = input_width
        self.configured_input_height = input_height
        tile_mode, configured_tile_size = _normalize_tile_setting(tile)
        if tile_size is not None:
            if isinstance(tile_size, bool) or not isinstance(tile_size, int):
                raise TypeError("AI tile size must be an integer.")
            if tile_size <= 0:
                raise ValueError("AI tile size must be greater than zero.")
            if tile_mode == "off":
                raise ValueError("AI tile size cannot be supplied when tiling is off.")
            if configured_tile_size is not None and configured_tile_size != tile_size:
                raise ValueError("Conflicting AI tile sizes were supplied.")
            tile_mode = "fixed"
            configured_tile_size = tile_size
        if isinstance(tile_overlap, bool) or not isinstance(tile_overlap, int):
            raise TypeError("AI tile overlap must be an integer.")
        if tile_overlap < 0:
            raise ValueError("AI tile overlap must be zero or greater.")
        if (
            configured_tile_size is not None
            and tile_overlap * 2 >= configured_tile_size
        ):
            raise ValueError(
                "AI tile size must be greater than twice the tile overlap."
            )
        self.tile_mode = tile_mode
        self.configured_tile_size = configured_tile_size
        self.tile_overlap = tile_overlap
        self._session_factory = session_factory or ort.InferenceSession
        self._session: _InferenceSession | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._input_shape: tuple[object, ...] | None = None
        self._output_shape: tuple[object, ...] | None = None
        self.available_providers: tuple[str, ...] = ()
        self.active_providers: tuple[str, ...] = ()
        self.input_metadata: ImageTensorMetadata | None = None
        self.output_metadata: ImageTensorMetadata | None = None
        self.original_capture_dimensions: tuple[int, int] | None = None
        self.ai_input_dimensions: tuple[int, int] | None = None
        self.detected_scale: int | None = None
        self.output_width: int | None = None
        self.output_height: int | None = None
        self.output_dimensions: tuple[int, int] | None = None
        self.last_inference_ms: float | None = None
        self.last_preprocessing_ms: float | None = None
        self.last_postprocessing_ms: float | None = None
        self._current_preprocessing_ms = 0.0
        self.selected_tile_size: int | None = None
        self.tiles_processed = 0
        self.estimated_peak_tile_bytes: int | None = None
        self._large_input_warning_emitted = False
        self._validation_session_lease = ResourceLease("onnx_sessions")

    @property
    def initialized(self) -> bool:
        """Return whether this processor currently owns a loaded model session."""

        return self._session is not None

    @staticmethod
    def _validate_image_metadata(
        value: _SessionValue,
        layout: str,
        tensor_name: str,
    ) -> tuple[object, ...]:
        if getattr(value, "type", None) != "tensor(float)":
            raise AIProcessorError(
                f"{tensor_name} must use float32 tensor data; got {getattr(value, 'type', None)!r}."
            )

        shape = getattr(value, "shape", None)
        if not isinstance(shape, (list, tuple)) or len(shape) != 4:
            rank = len(shape) if isinstance(shape, (list, tuple)) else "unknown"
            raise AIProcessorError(
                f"{tensor_name} must be a rank-4 {layout.upper()} image tensor; got rank {rank}."
            )

        shape = tuple(shape)
        static_shape = tuple(
            _static_dimension(dimension, tensor_name) for dimension in shape
        )
        batch_size = static_shape[0]
        if batch_size is not None and batch_size != 1:
            raise AIProcessorError(
                f"{tensor_name} must have batch size 1 or a dynamic batch dimension."
            )

        channel_index = 1 if layout == "nchw" else 3
        channel_count = static_shape[channel_index]
        if channel_count is not None and channel_count != 3:
            raise AIProcessorError(
                f"{tensor_name} must have exactly 3 image channels for {layout.upper()}; "
                f"got {channel_count}."
            )

        return shape

    @staticmethod
    def _validate_runtime_shape(
        actual_shape: tuple[int, ...],
        metadata_shape: tuple[object, ...],
        tensor_name: str,
    ) -> None:
        if len(actual_shape) != 4:
            raise AIProcessorError(
                f"{tensor_name} must be rank 4 at runtime; got shape {actual_shape}."
            )

        for index, (actual, expected) in enumerate(zip(actual_shape, metadata_shape)):
            fixed_dimension = _static_dimension(expected, tensor_name)
            if fixed_dimension is not None and actual != fixed_dimension:
                raise AIProcessorError(
                    f"{tensor_name} shape {actual_shape} does not match model metadata "
                    f"at dimension {index}: expected {fixed_dimension}, got {actual}."
                )

    def initialize(self) -> None:
        """Load and validate one image-to-image ONNX model on the requested provider."""

        if self._session is not None:
            return

        if self.model_path is None or not str(self.model_path).strip():
            raise AIProcessorError(
                "AI processing requires an ONNX model; supply a model path."
            )

        resolved_path = Path(self.model_path).expanduser()
        if not resolved_path.exists():
            raise AIProcessorError(f"ONNX model does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise AIProcessorError(f"ONNX model path is not a file: {resolved_path}")

        try:
            available_providers = tuple(ort.get_available_providers())
        except Exception as error:
            raise AIProcessorError(
                f"Unable to query ONNX Runtime execution providers: {error}"
            ) from error

        self.available_providers = available_providers
        requested_provider_name = (
            "CPUExecutionProvider"
            if self.provider == "cpu"
            else "DmlExecutionProvider"
        )
        if requested_provider_name not in available_providers:
            raise AIProcessorError(
                f"Requested {requested_provider_name} is unavailable. "
                f"Available ONNX Runtime providers: {', '.join(available_providers) or 'none'}."
            )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if self.provider == "directml":
            session_options.enable_mem_pattern = False
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            requested_providers = [
                ("DmlExecutionProvider", {"device_id": self.device_id})
            ]
        else:
            requested_providers = ["CPUExecutionProvider"]

        try:
            session = self._session_factory(
                str(resolved_path),
                sess_options=session_options,
                providers=requested_providers,
            )
            active_providers = tuple(session.get_providers())
            inputs = session.get_inputs()
            outputs = session.get_outputs()
        except Exception as error:
            raise AIProcessorError(
                f"Unable to load ONNX model '{resolved_path}': {error}"
            ) from error

        if requested_provider_name not in active_providers:
            raise AIProcessorError(
                f"Requested {requested_provider_name} is not active in the created session. "
                f"Active providers: {', '.join(active_providers) or 'none'}. "
                "CPU fallback is not accepted for this request."
            )

        if len(inputs) != 1:
            raise AIProcessorError(
                "AIProcessor supports exactly one image input; "
                f"the model exposes {len(inputs)} inputs."
            )
        if len(outputs) != 1:
            raise AIProcessorError(
                "AIProcessor supports exactly one image output; "
                f"the model exposes {len(outputs)} outputs."
            )

        input_shape = self._validate_image_metadata(
            inputs[0],
            self.input_layout,
            "Model input",
        )
        output_shape = self._validate_image_metadata(
            outputs[0],
            self.output_layout,
            "Model output",
        )

        self._session = session
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.active_providers = active_providers
        self.input_metadata = ImageTensorMetadata(
            name=inputs[0].name,
            dtype=inputs[0].type,
            shape=input_shape,
            layout=self.input_layout,
        )
        self.output_metadata = ImageTensorMetadata(
            name=outputs[0].name,
            dtype=outputs[0].type,
            shape=output_shape,
            layout=self.output_layout,
        )
        self._validation_session_lease.acquire()

    @staticmethod
    def _validate_input_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise AIProcessorError("Input frame must be a NumPy array.")
        if frame.ndim != 3:
            raise AIProcessorError(
                f"Input frame must have rank 3 with shape HxWx3; got {frame.shape}."
            )
        if frame.shape[2] != 3:
            raise AIProcessorError(
                f"Input frame must have exactly 3 BGR channels; got shape {frame.shape}."
            )
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise AIProcessorError(f"Input frame has invalid spatial shape {frame.shape}.")
        if frame.dtype != np.uint8:
            raise AIProcessorError(
                f"Input frame must use uint8 pixels; got {frame.dtype}."
            )

    def _prepare_inference_frame(self, frame: np.ndarray) -> np.ndarray:
        self._validate_input_frame(frame)
        original_width = frame.shape[1]
        original_height = frame.shape[0]
        self.original_capture_dimensions = (original_width, original_height)

        if self.configured_input_width is not None:
            inference_frame = cv2.resize(
                frame,
                (self.configured_input_width, self.configured_input_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            inference_frame = frame
            input_pixels = original_width * original_height
            if (
                input_pixels > AI_LARGE_INPUT_PIXEL_THRESHOLD
                and not self._large_input_warning_emitted
            ):
                logger.warning(
                    "Large AI input %sx%s (%s pixels) will be sent to ONNX inference "
                    "without an internal resolution limit; use --ai-input-width and "
                    "--ai-input-height for safer live processing.",
                    original_width,
                    original_height,
                    input_pixels,
                )
                self._large_input_warning_emitted = True

        self.ai_input_dimensions = (
            inference_frame.shape[1],
            inference_frame.shape[0],
        )
        return inference_frame

    def _prepare_input(self, frame: np.ndarray) -> np.ndarray:
        self._validate_input_frame(frame)

        image = frame[..., ::-1] if self.color_order == "rgb" else frame
        tensor = image.astype(np.float32) / np.float32(255.0)
        if self.input_layout == "nchw":
            tensor = np.transpose(tensor, (2, 0, 1))

        tensor = np.ascontiguousarray(tensor[np.newaxis, ...])
        if self._input_shape is None:
            raise AIProcessorError("AIProcessor input metadata is unavailable.")
        self._validate_runtime_shape(tensor.shape, self._input_shape, "Model input")
        return tensor

    def _prepare_output(self, output: object) -> np.ndarray:
        tensor = np.asarray(output)
        if tensor.dtype != np.float32:
            raise AIProcessorError(
                f"Model output must use float32 tensor data; got {tensor.dtype}."
            )
        if tensor.ndim != 4:
            raise AIProcessorError(
                f"Model output must be rank 4 at runtime; got shape {tensor.shape}."
            )
        if tensor.shape[0] != 1:
            raise AIProcessorError(
                f"Model output must contain exactly one image; got batch size {tensor.shape[0]}."
            )

        channel_index = 1 if self.output_layout == "nchw" else 3
        if tensor.shape[channel_index] != 3:
            raise AIProcessorError(
                f"Model output must have exactly 3 channels for {self.output_layout.upper()}; "
                f"got shape {tensor.shape}."
            )
        spatial_indices = (2, 3) if self.output_layout == "nchw" else (1, 2)
        if any(tensor.shape[index] <= 0 for index in spatial_indices):
            raise AIProcessorError(
                f"Model output has invalid spatial shape {tensor.shape}."
            )
        if self._output_shape is None:
            raise AIProcessorError("AIProcessor output metadata is unavailable.")

        self._validate_runtime_shape(tensor.shape, self._output_shape, "Model output")
        if not np.isfinite(tensor).all():
            raise AIProcessorError("Model output contains NaN or infinite pixel values.")

        image = tensor[0]
        if self.output_layout == "nchw":
            image = np.transpose(image, (1, 2, 0))

        image = np.rint(np.clip(image, 0.0, 1.0) * np.float32(255.0))
        image = image.astype(np.uint8)
        if self.color_order == "rgb":
            image = image[..., ::-1]

        return np.ascontiguousarray(image)

    def _calculate_output_scale(
        self,
        input_width: int,
        input_height: int,
        output: np.ndarray,
    ) -> int:
        output_height, output_width = output.shape[:2]
        if output_width <= 0 or output_height <= 0:
            raise AIProcessorError(
                f"Model output has invalid dimensions {output_width}x{output_height}."
            )
        if output_width % input_width != 0 or output_height % input_height != 0:
            raise AIProcessorError(
                "Model output scale must be an integer in both dimensions; "
                f"input is {input_width}x{input_height}, output is "
                f"{output_width}x{output_height}."
            )

        horizontal_scale = output_width // input_width
        vertical_scale = output_height // input_height
        if horizontal_scale != vertical_scale:
            raise AIProcessorError(
                "Model output uses mismatched horizontal and vertical scales: "
                f"{horizontal_scale}x horizontally and {vertical_scale}x vertically."
            )
        if horizontal_scale not in (1, 2, 3, 4):
            raise AIProcessorError(
                f"Unsupported model output scale {horizontal_scale}x; supported scales are 1x, 2x, 3x, and 4x."
            )
        if self.expected_scale is not None and horizontal_scale != self.expected_scale:
            raise AIProcessorError(
                f"Model output scale is {horizontal_scale}x, but {self.expected_scale}x was requested."
            )

        return horizontal_scale

    def _record_output_scale(
        self,
        input_width: int,
        input_height: int,
        output: np.ndarray,
    ) -> None:
        horizontal_scale = self._calculate_output_scale(
            input_width,
            input_height,
            output,
        )
        output_height, output_width = output.shape[:2]

        self.detected_scale = horizontal_scale
        self.output_width = output_width
        self.output_height = output_height
        self.output_dimensions = (output_width, output_height)

    def _run_model_once(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        if self._session is None or self._input_name is None or self._output_name is None:
            raise AIProcessorError("AIProcessor is not initialized.")

        preprocessing_start = time.perf_counter()
        input_tensor = self._prepare_input(frame)
        self._current_preprocessing_ms += (
            time.perf_counter() - preprocessing_start
        ) * 1000.0
        try:
            inference_start = time.perf_counter()
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: input_tensor},
            )
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
        except Exception as error:
            raise AIProcessorError(f"ONNX model processing failed: {error}") from error

        if len(outputs) != 1:
            raise AIProcessorError(
                f"ONNX model must return exactly one output; got {len(outputs)}."
            )
        return self._prepare_output(outputs[0]), inference_ms

    def _scale_for_memory_estimate(self) -> int:
        return self.expected_scale if self.expected_scale is not None else 4

    def _estimate_tile_bytes(self, width: int, height: int) -> int:
        scale = self._scale_for_memory_estimate()
        tensor_bytes = 3 * np.dtype(np.float32).itemsize
        input_and_output_bytes = width * height * tensor_bytes * (1 + scale * scale)
        return input_and_output_bytes * AI_AUTO_TILE_ACTIVATION_SAFETY_FACTOR

    def _select_auto_tile_size(self, width: int, height: int) -> int:
        scale = self._scale_for_memory_estimate()
        bytes_per_pixel = (
            3
            * np.dtype(np.float32).itemsize
            * (1 + scale * scale)
            * AI_AUTO_TILE_ACTIVATION_SAFETY_FACTOR
        )
        safe_pixels = AI_AUTO_TILE_MEMORY_BUDGET_BYTES // bytes_per_pixel
        maximum_safe_dimension = math.isqrt(safe_pixels)
        minimum_tile_size = self.tile_overlap * 2 + 1
        candidates = [
            candidate
            for candidate in AI_AUTO_TILE_CANDIDATES
            if minimum_tile_size <= candidate <= maximum_safe_dimension
        ]
        if not candidates:
            raise AIProcessorError(
                "Automatic AI tiling cannot choose a safe tile larger than twice "
                f"the configured {self.tile_overlap}-pixel overlap."
            )

        safe_tile_size = max(candidates)
        return min(max(width, height), safe_tile_size)

    @staticmethod
    def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = tile_size - overlap
        starts = [0]
        while starts[-1] + tile_size < length:
            starts.append(starts[-1] + stride)
        return starts

    @staticmethod
    def _blend_axis_weights(
        length: int,
        leading_overlap: int,
        trailing_overlap: int,
    ) -> np.ndarray:
        weights = np.ones(length, dtype=np.float32)
        if leading_overlap:
            ramp = np.arange(1, leading_overlap + 1, dtype=np.float32)
            weights[:leading_overlap] = ramp / np.float32(leading_overlap + 1)
        if trailing_overlap:
            ramp = np.arange(trailing_overlap, 0, -1, dtype=np.float32)
            weights[-trailing_overlap:] = ramp / np.float32(trailing_overlap + 1)
        return weights

    def _process_tiled(self, frame: np.ndarray, tile_size: int) -> np.ndarray:
        signature = (frame.shape, tile_size, self.tile_overlap)
        if signature != self._tile_cache_signature:
            self._tile_cache.clear()
            self._tile_cache_signature = signature
            self._cached_static_frame = None
            self._cached_static_output = None
        self.tiles_reused = 0
        if (self.reuse_static_tiles and self._cached_static_frame is not None
                and all(entry[2] < 120 for entry in self._tile_cache.values())
                and np.array_equal(frame, self._cached_static_frame)):
            self._tile_cache = {key: (source, output, age + 1)
                                for key, (source, output, age) in self._tile_cache.items()}
            self.tiles_reused = len(self._tile_cache)
            self.tiles_processed = 0
            self.last_inference_ms = 0.0
            if time.perf_counter() - self._reuse_log_time >= 5.0:
                logger.info("AI tile reuse: inferred 0, reused %d; unchanged frame", self.tiles_reused)
                self._reuse_log_time = time.perf_counter()
            return self._cached_static_output.copy()
        input_height, input_width = frame.shape[:2]
        x_starts = self._tile_starts(input_width, tile_size, self.tile_overlap)
        y_starts = self._tile_starts(input_height, tile_size, self.tile_overlap)
        output_accumulator: np.ndarray | None = None
        weight_accumulator: np.ndarray | None = None
        detected_tile_scale: int | None = None
        total_inference_ms = 0.0
        tile_count = 0
        entries = []
        dirty_regions = []

        for y_index, y_start in enumerate(y_starts):
            y_end = min(y_start + tile_size, input_height)
            for x_index, x_start in enumerate(x_starts):
                x_end = min(x_start + tile_size, input_width)
                tile = np.ascontiguousarray(frame[y_start:y_end, x_start:x_end])
                key = (y_start, x_start)
                cached = self._tile_cache.get(key) if self.reuse_static_tiles else None
                reused = False
                # Compare every pixel, including overlap. No motion threshold:
                # subtle text, colour changes and gradual movement must refresh.
                if cached is not None and cached[2] < 120 and np.array_equal(tile, cached[0]):
                    tile_output, inference_ms = cached[1], 0.0
                    self._tile_cache[key] = (cached[0], cached[1], cached[2] + 1)
                    self.tiles_reused += 1
                    reused = True
                else:
                    tile_output, inference_ms = self._run_model_once(tile)
                    if self.reuse_static_tiles:
                        self._tile_cache[key] = (tile.copy(), tile_output.copy(), 0)
                total_inference_ms += inference_ms
                tile_count += 1
                tile_scale = self._calculate_output_scale(
                    tile.shape[1],
                    tile.shape[0],
                    tile_output,
                )
                if detected_tile_scale is None:
                    detected_tile_scale = tile_scale
                elif tile_scale != detected_tile_scale:
                    raise AIProcessorError("Tiled model output scale changed between tiles.")
                if output_accumulator is None and not self.reuse_static_tiles:
                    output_accumulator = np.zeros(
                        (input_height * tile_scale, input_width * tile_scale, 3),
                        dtype=np.float32,
                    )
                    weight_accumulator = np.zeros(
                        (input_height * tile_scale, input_width * tile_scale),
                        dtype=np.float32,
                    )
                elif tile_scale != detected_tile_scale:
                    raise AIProcessorError(
                        "Tiled model output scale changed between tiles: "
                        f"expected {detected_tile_scale}x, got {tile_scale}x."
                    )

                scale = detected_tile_scale
                previous_x_end = (
                    min(x_starts[x_index - 1] + tile_size, input_width)
                    if x_index > 0
                    else x_start
                )
                next_x_start = (
                    x_starts[x_index + 1]
                    if x_index + 1 < len(x_starts)
                    else x_end
                )
                previous_y_end = (
                    min(y_starts[y_index - 1] + tile_size, input_height)
                    if y_index > 0
                    else y_start
                )
                next_y_start = (
                    y_starts[y_index + 1]
                    if y_index + 1 < len(y_starts)
                    else y_end
                )
                left_overlap = max(0, previous_x_end - x_start) * scale
                right_overlap = max(0, x_end - next_x_start) * scale
                top_overlap = max(0, previous_y_end - y_start) * scale
                bottom_overlap = max(0, y_end - next_y_start) * scale
                x_weights = self._blend_axis_weights(
                    tile_output.shape[1],
                    left_overlap,
                    right_overlap,
                )
                y_weights = self._blend_axis_weights(
                    tile_output.shape[0],
                    top_overlap,
                    bottom_overlap,
                )
                tile_weights = y_weights[:, np.newaxis] * x_weights[np.newaxis, :]
                output_x_start = x_start * scale
                output_y_start = y_start * scale
                output_x_end = output_x_start + tile_output.shape[1]
                output_y_end = output_y_start + tile_output.shape[0]
                if self.reuse_static_tiles:
                    region = (output_y_start, output_y_end, output_x_start, output_x_end)
                    entries.append((region, tile_output, tile_weights))
                    if not reused:
                        dirty_regions.append(region)
                    continue
                if output_accumulator is None or weight_accumulator is None:
                    raise AIProcessorError("Tiled output accumulators were not initialized.")
                output_accumulator[
                    output_y_start:output_y_end,
                    output_x_start:output_x_end,
                ] += tile_output.astype(np.float32) * tile_weights[..., np.newaxis]
                weight_accumulator[
                    output_y_start:output_y_end,
                    output_x_start:output_x_end,
                ] += tile_weights

        if self.reuse_static_tiles:
            composite_start = time.perf_counter()
            if self._cached_static_output is None:
                self._cached_static_output = np.zeros(
                    (input_height * detected_tile_scale, input_width * detected_tile_scale, 3), dtype=np.uint8)
            # Rebuild only dirty output rectangles, including every neighbouring
            # tile's overlap contribution. Re-summing avoids incremental drift.
            for y0, y1, x0, x1 in dirty_regions:
                sums = np.zeros((y1-y0, x1-x0, 3), dtype=np.float32)
                weights = np.zeros((y1-y0, x1-x0), dtype=np.float32)
                for (ty0, ty1, tx0, tx1), pixels, blend in entries:
                    ay0, ay1 = max(y0, ty0), min(y1, ty1)
                    ax0, ax1 = max(x0, tx0), min(x1, tx1)
                    if ay0 >= ay1 or ax0 >= ax1:
                        continue
                    source = np.s_[ay0-ty0:ay1-ty0, ax0-tx0:ax1-tx0]
                    target = np.s_[ay0-y0:ay1-y0, ax0-x0:ax1-x0]
                    sums[target] += pixels[source].astype(np.float32) * blend[source][..., None]
                    weights[target] += blend[source]
                if np.any(weights <= 0):
                    raise AIProcessorError("Cached tile composition left uncovered pixels.")
                self._cached_static_output[y0:y1, x0:x1] = np.rint(
                    np.clip(sums / weights[..., None], 0, 255)).astype(np.uint8)
            self.last_compositing_ms = (time.perf_counter() - composite_start) * 1000
            self._cached_static_frame = frame.copy()
            self.last_inference_ms = total_inference_ms
            self.tiles_processed = tile_count - self.tiles_reused
            return self._cached_static_output.copy()

        if (
            output_accumulator is None
            or weight_accumulator is None
            or detected_tile_scale is None
        ):
            raise AIProcessorError("Tiled inference produced no output tiles.")
        if np.any(weight_accumulator <= 0.0):
            raise AIProcessorError("Tiled inference left uncovered output pixels.")

        self.last_inference_ms = total_inference_ms
        self.tiles_processed = tile_count - self.tiles_reused
        if self.reuse_static_tiles and time.perf_counter() - self._reuse_log_time >= 5.0:
            logger.info("AI tile reuse: inferred %d, reused %d, tile %d px; exact pixel comparison", self.tiles_processed, self.tiles_reused, tile_size)
            self._reuse_log_time = time.perf_counter()
        blended = output_accumulator / weight_accumulator[..., np.newaxis]
        result = np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)
        if self.reuse_static_tiles:
            self._cached_static_frame = frame.copy()
            self._cached_static_output = result.copy()
        return result

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess, infer, and postprocess one OpenCV-style BGR frame."""

        if (
            self._session is None
            or self._input_name is None
            or self._output_name is None
        ):
            raise AIProcessorError("AIProcessor is not initialized.")

        processing_start = time.perf_counter()
        self.last_compositing_ms = 0.0
        self.detected_scale = None
        self.output_width = None
        self.output_height = None
        self.output_dimensions = None
        self.original_capture_dimensions = None
        self.ai_input_dimensions = None
        self.last_inference_ms = None
        self.last_preprocessing_ms = None
        self.last_postprocessing_ms = None
        self._current_preprocessing_ms = 0.0
        self.selected_tile_size = None
        self.tiles_processed = 0
        self.estimated_peak_tile_bytes = None
        preprocessing_start = time.perf_counter()
        inference_frame = self._prepare_inference_frame(frame)
        self._current_preprocessing_ms += (
            time.perf_counter() - preprocessing_start
        ) * 1000.0
        input_height, input_width = inference_frame.shape[:2]
        if self.tile_mode == "auto":
            tile_size = self._select_auto_tile_size(input_width, input_height)
            self.selected_tile_size = tile_size
        elif self.tile_mode == "fixed":
            if self.configured_tile_size is None:
                raise AIProcessorError("Configured AI tile size is unavailable.")
            tile_size = self.configured_tile_size
            self.selected_tile_size = tile_size
        else:
            tile_size = None

        if self.reuse_static_tiles:
            # Bound cache granularity even if Auto selects a full-frame tile.
            tile_size = max(256, self.tile_overlap * 2 + 1) if tile_size is None else min(tile_size, max(256, self.tile_overlap * 2 + 1))
            self.selected_tile_size = tile_size
        if tile_size is not None and (self.reuse_static_tiles or
            input_width > tile_size or input_height > tile_size
        ):
            output = self._process_tiled(inference_frame, tile_size)
            peak_width = min(tile_size, input_width)
            peak_height = min(tile_size, input_height)
        else:
            output, inference_ms = self._run_model_once(inference_frame)
            self.last_inference_ms = inference_ms
            self.tiles_processed = 1
            peak_width = input_width
            peak_height = input_height
        self.estimated_peak_tile_bytes = self._estimate_tile_bytes(
            peak_width,
            peak_height,
        )
        self._record_output_scale(
            input_width=input_width,
            input_height=input_height,
            output=output,
        )
        processing_ms = (time.perf_counter() - processing_start) * 1000.0
        inference_ms = self.last_inference_ms or 0.0
        self.last_preprocessing_ms = self._current_preprocessing_ms
        self.last_postprocessing_ms = max(
            0.0,
            processing_ms - self.last_preprocessing_ms - inference_ms,
        )
        if self.reuse_static_tiles and time.perf_counter() - self._reuse_log_time >= 5.0:
            logger.info("AI cached stages: inferred %d, reused %d | inference %.2f ms | dirty composition %.2f ms | total %.2f ms",
                        self.tiles_processed, self.tiles_reused, inference_ms, self.last_compositing_ms, processing_ms)
            self._reuse_log_time = time.perf_counter()
        return output

    def shutdown(self) -> None:
        self._tile_cache.clear()
        self._tile_cache_signature = None
        self._cached_static_frame = None
        self._cached_static_output = None
        """Release the single loaded model session. Safe to call repeatedly."""

        self._validation_session_lease.release()
        self._input_name = None
        self._output_name = None
        self._input_shape = None
        self._output_shape = None
        self.available_providers = ()
        self.active_providers = ()
        self.input_metadata = None
        self.output_metadata = None
        self.original_capture_dimensions = None
        self.ai_input_dimensions = None
        self.detected_scale = None
        self.output_width = None
        self.output_height = None
        self.output_dimensions = None
        self.last_inference_ms = None
        self.last_preprocessing_ms = None
        self.last_postprocessing_ms = None
        self._current_preprocessing_ms = 0.0
        self.selected_tile_size = None
        self.tiles_processed = 0
        self.estimated_peak_tile_bytes = None
        self._large_input_warning_emitted = False
        self._session = None

    def close(self) -> None:
        """Provide the project's conventional resource-release alias."""

        self.shutdown()
