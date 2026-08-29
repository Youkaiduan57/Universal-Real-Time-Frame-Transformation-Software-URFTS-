"""Processing backend abstraction, implementations, and benchmarks."""

from __future__ import annotations

import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from config import (
    FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
    FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    normalize_processing_backend,
    normalize_upscaling_method,
    validate_fsr1_like_sharpening_enabled,
    validate_fsr1_like_strength,
)
from fsr1_like import fsr1_like_upscale

try:  # pragma: no cover - optional dependency in some environments
    import torch
    import torch.nn.functional as torch_functional
except ImportError:  # pragma: no cover - depends on local environment
    torch = None
    torch_functional = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessingBenchmarkResult:
    """Aggregated timing results for one processing backend."""

    backend_name: str
    display_name: str
    median_ms: float
    p95_ms: float
    score_ms: float


class ProcessingBackend(ABC):
    """Abstract processing backend contract."""

    backend_name = "processing"
    display_name = "Processing"

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the backend can be constructed in this environment."""

        return True

    @classmethod
    def availability_reason(cls) -> str | None:
        """Return a short explanation when the backend is unavailable."""

        return None

    def synchronize_for_timing(self) -> None:
        """Synchronize the backend when accurate benchmark timing needs it."""

        return None

    def close(self) -> None:
        """Release backend resources if any exist."""

        return None

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process one BGR frame and return a resized BGR frame."""


class OpenCVProcessingBackend(ProcessingBackend):
    """CPU processing backend using OpenCV resize."""

    backend_name = "opencv_cpu"
    display_name = "OpenCV CPU"

    def __init__(
        self,
        output_width: int = OUTPUT_WIDTH,
        output_height: int = OUTPUT_HEIGHT,
        upscaling_method: str = "bicubic",
        fsr1_like_sharpening_enabled: bool = FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
        fsr1_like_sharpening_strength: float = FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
        fsr1_like_edge_strength: float = FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    ) -> None:
        self.output_width = _validate_positive_int(output_width, "output_width")
        self.output_height = _validate_positive_int(output_height, "output_height")
        self.upscaling_method = normalize_upscaling_method(upscaling_method)
        self.fsr1_like_sharpening_enabled = validate_fsr1_like_sharpening_enabled(
            fsr1_like_sharpening_enabled
        )
        self.fsr1_like_sharpening_strength = validate_fsr1_like_strength(
            fsr1_like_sharpening_strength,
            "fsr1_like_sharpening_strength",
        )
        self.fsr1_like_edge_strength = validate_fsr1_like_strength(
            fsr1_like_edge_strength,
            "fsr1_like_edge_strength",
        )

        self.interpolation_methods = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.upscaling_method == "fsr1_like":
            return fsr1_like_upscale(
                frame=frame,
                output_width=self.output_width,
                output_height=self.output_height,
                edge_strength=self.fsr1_like_edge_strength,
                sharpening_strength=self.fsr1_like_sharpening_strength,
                sharpening_enabled=self.fsr1_like_sharpening_enabled,
            )

        return cv2.resize(
            frame,
            (self.output_width, self.output_height),
            interpolation=self.interpolation_methods[self.upscaling_method],
        )


class TorchCudaProcessingBackend(ProcessingBackend):
    """PyTorch CUDA processing backend using tensor-based interpolation."""

    backend_name = "torch_cuda"
    display_name = "PyTorch CUDA"

    def __init__(
        self,
        output_width: int = OUTPUT_WIDTH,
        output_height: int = OUTPUT_HEIGHT,
        upscaling_method: str = "bicubic",
        device_index: int = 0,
    ) -> None:
        normalized_upscaling_method = normalize_upscaling_method(upscaling_method)

        if normalized_upscaling_method == "fsr1_like":
            raise ValueError(
                "PyTorch CUDA processing does not support fsr1_like; "
                "select or fall back to the OpenCV CPU backend."
            )

        if torch is None:
            raise RuntimeError("PyTorch is not installed in this environment.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for PyTorch processing.")

        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("No CUDA devices are available.")

        if device_index < 0 or device_index >= device_count:
            raise RuntimeError(
                f"Requested CUDA device {device_index} is out of range for {device_count} device(s)."
            )

        self.output_width = _validate_positive_int(output_width, "output_width")
        self.output_height = _validate_positive_int(output_height, "output_height")
        self.upscaling_method = normalized_upscaling_method
        self.device_index = device_index
        self.device = torch.device(f"cuda:{device_index}")

        if self.upscaling_method == "lanczos":
            raise ValueError("PyTorch CUDA processing does not support lanczos interpolation.")

    @classmethod
    def is_available(cls) -> bool:
        return torch is not None and bool(torch.cuda.is_available())

    @classmethod
    def availability_reason(cls) -> str | None:
        if torch is None:
            return "PyTorch is not installed."

        if not torch.cuda.is_available():
            return "CUDA is not available."

        return None

    def synchronize_for_timing(self) -> None:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR frame with shape HxWx3.")

        if frame.dtype != np.uint8:
            raise ValueError("Expected a uint8 frame for PyTorch CUDA processing.")

        if torch_functional is None:
            raise RuntimeError("PyTorch functional interpolation is not available.")

        try:
            with torch.inference_mode():
                frame_array = np.ascontiguousarray(frame)
                input_tensor = torch.from_numpy(frame_array)
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
                input_tensor = input_tensor.to(self.device, dtype=torch.float32)
                input_tensor.div_(255.0)

                if self.upscaling_method == "nearest":
                    resized_tensor = torch_functional.interpolate(
                        input_tensor,
                        size=(self.output_height, self.output_width),
                        mode="nearest",
                    )
                else:
                    resized_tensor = torch_functional.interpolate(
                        input_tensor,
                        size=(self.output_height, self.output_width),
                        mode=self.upscaling_method,
                        align_corners=False,
                    )

                output_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)
                output_tensor = output_tensor.clamp_(0.0, 1.0)
                output_tensor = output_tensor.mul_(255.0).round_().to(torch.uint8)
                return output_tensor.cpu().numpy()
        except RuntimeError as error:
            if _looks_like_cuda_oom(error):
                torch.cuda.empty_cache()
                raise RuntimeError("PyTorch CUDA processing ran out of memory.") from error

            raise


@dataclass(frozen=True, slots=True)
class _BackendSpec:
    backend_name: str
    display_name: str
    backend_type: type[ProcessingBackend]


_BACKEND_SPECS: tuple[_BackendSpec, ...] = (
    _BackendSpec(
        backend_name=OpenCVProcessingBackend.backend_name,
        display_name=OpenCVProcessingBackend.display_name,
        backend_type=OpenCVProcessingBackend,
    ),
    _BackendSpec(
        backend_name=TorchCudaProcessingBackend.backend_name,
        display_name=TorchCudaProcessingBackend.display_name,
        backend_type=TorchCudaProcessingBackend,
    ),
)


def _validate_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")

    return value


def _looks_like_cuda_oom(error: RuntimeError) -> bool:
    error_message = str(error).lower()
    return "out of memory" in error_message or "cuda" in error_message and "memory" in error_message


def available_processing_backend_names() -> tuple[str, ...]:
    """Return the backend names that can currently be constructed."""

    available_names = []

    for spec in _BACKEND_SPECS:
        if spec.backend_type.is_available():
            available_names.append(spec.backend_name)

    return tuple(available_names)


def available_processing_backend_display_names() -> tuple[str, ...]:
    """Return human-readable names for the available processing backends."""

    display_names = []

    for spec in _BACKEND_SPECS:
        if spec.backend_type.is_available():
            display_names.append(spec.display_name)

    return tuple(display_names)


def processing_backend_display_name(backend_name: str) -> str:
    """Return a readable display name for a backend identifier."""

    normalized_backend_name = normalize_processing_backend(backend_name)

    if normalized_backend_name == "auto":
        return "Auto"

    for spec in _BACKEND_SPECS:
        if spec.backend_name == normalized_backend_name:
            return spec.display_name

    return normalized_backend_name


def create_processing_backend(
    backend_name: str,
    output_width: int = OUTPUT_WIDTH,
    output_height: int = OUTPUT_HEIGHT,
    upscaling_method: str = "bicubic",
    device_index: int = 0,
    fsr1_like_sharpening_enabled: bool = FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
    fsr1_like_sharpening_strength: float = FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
    fsr1_like_edge_strength: float = FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
) -> ProcessingBackend:
    """Instantiate one concrete processing backend by name."""

    normalized_backend_name = normalize_processing_backend(backend_name)

    if normalized_backend_name == "auto":
        raise ValueError("Auto is not a concrete processing backend.")

    if normalized_backend_name == OpenCVProcessingBackend.backend_name:
        return OpenCVProcessingBackend(
            output_width=output_width,
            output_height=output_height,
            upscaling_method=upscaling_method,
            fsr1_like_sharpening_enabled=fsr1_like_sharpening_enabled,
            fsr1_like_sharpening_strength=fsr1_like_sharpening_strength,
            fsr1_like_edge_strength=fsr1_like_edge_strength,
        )

    if normalized_backend_name == TorchCudaProcessingBackend.backend_name:
        return TorchCudaProcessingBackend(
            output_width=output_width,
            output_height=output_height,
            upscaling_method=upscaling_method,
            device_index=device_index,
        )

    raise ValueError(f"Unsupported processing backend: {normalized_backend_name}")


class ProcessingBackendTuner:
    """Benchmark available processing backends and choose the fastest stable one."""

    def __init__(
        self,
        warmup_runs: int = 16,
        benchmark_runs: int = 64,
        rounds: int = 3,
    ) -> None:
        self.warmup_runs = _validate_positive_int(warmup_runs, "warmup_runs")
        self.benchmark_runs = _validate_positive_int(benchmark_runs, "benchmark_runs")
        self.rounds = _validate_positive_int(rounds, "rounds")

    @staticmethod
    def _percentile(values: list[float], percentile_value: float) -> float:
        ordered_values = sorted(values)
        index = round(
            (percentile_value / 100.0) * (len(ordered_values) - 1)
        )
        return ordered_values[index]

    def _benchmark_backend(
        self,
        backend: ProcessingBackend,
        test_frame: np.ndarray,
    ) -> ProcessingBenchmarkResult:
        round_medians: list[float] = []
        round_p95s: list[float] = []
        round_scores: list[float] = []

        for _ in range(self.rounds):
            for _ in range(self.warmup_runs):
                backend.process(test_frame)

            timings: list[float] = []

            for _ in range(self.benchmark_runs):
                backend.synchronize_for_timing()
                start = time.perf_counter()
                backend.process(test_frame)
                backend.synchronize_for_timing()
                end = time.perf_counter()
                timings.append((end - start) * 1000.0)

            median_ms = statistics.median(timings)
            p95_ms = self._percentile(timings, 95)
            round_score = median_ms + (0.5 * p95_ms)

            round_medians.append(median_ms)
            round_p95s.append(p95_ms)
            round_scores.append(round_score)

        return ProcessingBenchmarkResult(
            backend_name=backend.backend_name,
            display_name=backend.display_name,
            median_ms=statistics.median(round_medians),
            p95_ms=statistics.median(round_p95s),
            score_ms=statistics.median(round_scores),
        )

    def tune(
        self,
        test_frame: np.ndarray,
        output_width: int = OUTPUT_WIDTH,
        output_height: int = OUTPUT_HEIGHT,
        upscaling_method: str = "bicubic",
        candidate_backend_names: Iterable[str] | None = None,
        fsr1_like_sharpening_enabled: bool = FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
        fsr1_like_sharpening_strength: float = FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
        fsr1_like_edge_strength: float = FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    ) -> ProcessingBackend:
        """Benchmark available backends and return the best usable instance."""

        if candidate_backend_names is None:
            candidate_backend_names = available_processing_backend_names()

        candidate_backend_names = tuple(candidate_backend_names)
        available_display_names = available_processing_backend_display_names()
        logger.info(
            "Available processing backends: %s",
            ", ".join(available_display_names) if available_display_names else "none",
        )

        if not candidate_backend_names:
            raise RuntimeError("No processing backends are available to benchmark.")

        results: list[tuple[ProcessingBackend, ProcessingBenchmarkResult]] = []
        failures: list[str] = []

        for backend_name in candidate_backend_names:
            try:
                backend = create_processing_backend(
                    backend_name=backend_name,
                    output_width=output_width,
                    output_height=output_height,
                    upscaling_method=upscaling_method,
                    fsr1_like_sharpening_enabled=fsr1_like_sharpening_enabled,
                    fsr1_like_sharpening_strength=fsr1_like_sharpening_strength,
                    fsr1_like_edge_strength=fsr1_like_edge_strength,
                )
            except (RuntimeError, ValueError) as error:
                failures.append(f"{backend_name}: {error}")
                logger.warning("Skipping processing backend %s: %s", backend_name, error)
                continue

            try:
                result = self._benchmark_backend(
                    backend=backend,
                    test_frame=test_frame,
                )
            except Exception as error:
                failures.append(f"{backend.display_name}: {error}")
                logger.warning(
                    "Processing backend %s failed during benchmarking: %s",
                    backend.display_name,
                    error,
                )
                backend.close()
                continue

            results.append((backend, result))
            logger.info(
                "%s: median %.3f ms, 95th percentile %.3f ms, score %.3f ms",
                result.display_name,
                result.median_ms,
                result.p95_ms,
                result.score_ms,
            )

        if not results:
            failure_summary = "; ".join(failures) if failures else "no usable backends"
            raise RuntimeError(f"Unable to benchmark any processing backend: {failure_summary}")

        selected_backend, selected_result = min(
            results,
            key=lambda item: item[1].score_ms,
        )

        for backend, _ in results:
            if backend is not selected_backend:
                backend.close()

        logger.info("Selected processing backend: %s", selected_result.display_name)

        return selected_backend
