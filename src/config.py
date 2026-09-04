"""Application configuration models and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from resource_paths import is_frozen, resource_path, user_data_dir

SUPPORTED_CAPTURE_BACKENDS: Final[tuple[str, ...]] = (
	"auto",
	"wgc",
	"obs",
	"dxcam",
	"mss",
)

SUPPORTED_UPSCALING_METHODS: Final[tuple[str, ...]] = (
	"nearest",
	"bilinear",
	"bicubic",
	"lanczos",
	"fsr1_like",
)

FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH: Final[float] = 0.2
FSR1_LIKE_DEFAULT_EDGE_STRENGTH: Final[float] = 0.35
FSR1_LIKE_DEFAULT_SHARPENING_ENABLED: Final[bool] = True
FSR1_LIKE_MIN_STRENGTH: Final[float] = 0.0
FSR1_LIKE_MAX_STRENGTH: Final[float] = 1.0

SUPPORTED_PROCESSING_BACKENDS: Final[tuple[str, ...]] = (
	"auto",
	"opencv_cpu",
	"torch_cuda",
)


def _normalize_choice(value: str, field_name: str, allowed: tuple[str, ...]) -> str:
	if not isinstance(value, str):
		raise TypeError(f"{field_name} must be a string.")

	normalized_value = value.strip().lower()

	if normalized_value not in allowed:
		raise ValueError(f"Unsupported {field_name}: {normalized_value}")

	return normalized_value


def _validate_positive_int(value: int, field_name: str) -> int:
	if not isinstance(value, int):
		raise TypeError(f"{field_name} must be an integer.")

	if value <= 0:
		raise ValueError(f"{field_name} must be greater than zero.")

	return value


def _validate_non_negative_int(value: int, field_name: str) -> int:
	if not isinstance(value, int):
		raise TypeError(f"{field_name} must be an integer.")

	if value < 0:
		raise ValueError(f"{field_name} must be zero or greater.")

	return value


def _validate_positive_float(value: float, field_name: str) -> float:
	if not isinstance(value, (int, float)):
		raise TypeError(f"{field_name} must be numeric.")

	if float(value) <= 0.0:
		raise ValueError(f"{field_name} must be greater than zero.")

	return float(value)


def validate_fsr1_like_strength(value: float, field_name: str) -> float:
	"""Validate a normalized FSR1-like stage strength."""

	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise TypeError(f"{field_name} must be numeric.")

	normalized_value = float(value)

	if not FSR1_LIKE_MIN_STRENGTH <= normalized_value <= FSR1_LIKE_MAX_STRENGTH:
		raise ValueError(
			f"{field_name} must be between "
			f"{FSR1_LIKE_MIN_STRENGTH:.1f} and {FSR1_LIKE_MAX_STRENGTH:.1f}."
		)

	return normalized_value


def validate_fsr1_like_sharpening_enabled(value: bool) -> bool:
	"""Validate the FSR1-like sharpening toggle without accepting truthy values."""

	if not isinstance(value, bool):
		raise TypeError("fsr1_like_sharpening_enabled must be a boolean.")

	return value


def _validate_thread_candidates(
	candidates: tuple[int, ...],
) -> tuple[int, ...]:
	if not candidates:
		raise ValueError("opencv_thread_candidates must not be empty.")

	normalized_candidates = []
	seen_candidates = set()

	for candidate in candidates:
		candidate = _validate_positive_int(
			candidate,
			"opencv_thread_candidates entry",
		)

		if candidate in seen_candidates:
			continue

		seen_candidates.add(candidate)
		normalized_candidates.append(candidate)

	return tuple(normalized_candidates)


@dataclass(frozen=True, slots=True)
class CaptureRegion:
	"""Capture rectangle for both MSS and DXcam backends."""

	left: int = 0
	top: int = 0
	width: int = 1280
	height: int = 720

	def __post_init__(self) -> None:
		object.__setattr__(self, "left", _validate_non_negative_int(self.left, "left"))
		object.__setattr__(self, "top", _validate_non_negative_int(self.top, "top"))
		object.__setattr__(self, "width", _validate_positive_int(self.width, "width"))
		object.__setattr__(self, "height", _validate_positive_int(self.height, "height"))


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
	"""Static application configuration that does not belong in the runtime profile."""

	capture_region: CaptureRegion = field(default_factory=CaptureRegion)
	output_width: int = 1920
	output_height: int = 1080
	metrics_update_interval: float = 2.0
	opencv_thread_candidates: tuple[int, ...] = (1, 2, 4, 8)

	def __post_init__(self) -> None:
		object.__setattr__(self, "output_width", _validate_positive_int(self.output_width, "output_width"))
		object.__setattr__(self, "output_height", _validate_positive_int(self.output_height, "output_height"))
		object.__setattr__(
			self,
			"metrics_update_interval",
			_validate_positive_float(
				self.metrics_update_interval,
				"metrics_update_interval",
			),
		)
		object.__setattr__(
			self,
			"opencv_thread_candidates",
			_validate_thread_candidates(self.opencv_thread_candidates),
		)

	@property
	def output_size(self) -> tuple[int, int]:
		"""Return the upscale output size as a width/height pair."""

		return self.output_width, self.output_height


DEFAULT_APPLICATION_CONFIG = ApplicationConfig()

CAPTURE_LEFT = DEFAULT_APPLICATION_CONFIG.capture_region.left
CAPTURE_TOP = DEFAULT_APPLICATION_CONFIG.capture_region.top
CAPTURE_WIDTH = DEFAULT_APPLICATION_CONFIG.capture_region.width
CAPTURE_HEIGHT = DEFAULT_APPLICATION_CONFIG.capture_region.height

OUTPUT_WIDTH = DEFAULT_APPLICATION_CONFIG.output_width
OUTPUT_HEIGHT = DEFAULT_APPLICATION_CONFIG.output_height

METRICS_UPDATE_INTERVAL = DEFAULT_APPLICATION_CONFIG.metrics_update_interval

OPENCV_THREAD_CANDIDATES = DEFAULT_APPLICATION_CONFIG.opencv_thread_candidates


def runtime_profile_path() -> Path:
	"""Return the persisted runtime-profile location."""

	if is_frozen():
		return user_data_dir("configs") / "runtime_profile.json"
	return resource_path("configs", "runtime_profile.json")


def normalize_capture_backend(backend_name: str) -> str:
	"""Validate and normalize a capture backend name."""

	return _normalize_choice(
		backend_name,
		"capture backend",
		SUPPORTED_CAPTURE_BACKENDS,
	)


def normalize_upscaling_method(method_name: str) -> str:
	"""Validate and normalize an upscaling method name."""

	return _normalize_choice(
		method_name,
		"upscaling method",
		SUPPORTED_UPSCALING_METHODS,
	)


def upscaling_method_display_name(method_name: str) -> str:
	"""Return a readable, non-misleading label for an upscaling method."""

	normalized_method = normalize_upscaling_method(method_name)

	if normalized_method == "fsr1_like":
		return "FSR1-like"

	return normalized_method.title()


def normalize_processing_backend(backend_name: str) -> str:
	"""Validate and normalize a processing backend name."""

	return _normalize_choice(
		backend_name,
		"processing backend",
		SUPPORTED_PROCESSING_BACKENDS,
	)
