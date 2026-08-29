"""Persisted runtime selection profile."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from config import (
    FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
    FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
    normalize_capture_backend,
    normalize_processing_backend,
    normalize_upscaling_method,
    runtime_profile_path,
    upscaling_method_display_name,
    validate_fsr1_like_sharpening_enabled,
    validate_fsr1_like_strength,
)
from processing_backend import processing_backend_display_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Runtime choices that should persist across launches."""

    capture_backend: str = "auto"
    opencv_threads: int = 1
    upscaling_method: str = "bicubic"
    processing_backend: str = "auto"
    fsr1_like_sharpening_enabled: bool = FSR1_LIKE_DEFAULT_SHARPENING_ENABLED
    fsr1_like_sharpening_strength: float = FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH
    fsr1_like_edge_strength: float = FSR1_LIKE_DEFAULT_EDGE_STRENGTH

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capture_backend",
            normalize_capture_backend(self.capture_backend),
        )

        if not isinstance(self.opencv_threads, int):
            raise TypeError("opencv_threads must be an integer.")

        if self.opencv_threads <= 0:
            raise ValueError("opencv_threads must be greater than zero.")

        object.__setattr__(
            self,
            "upscaling_method",
            normalize_upscaling_method(self.upscaling_method),
        )

        object.__setattr__(
            self,
            "processing_backend",
            normalize_processing_backend(self.processing_backend),
        )
        object.__setattr__(
            self,
            "fsr1_like_sharpening_enabled",
            validate_fsr1_like_sharpening_enabled(
                self.fsr1_like_sharpening_enabled
            ),
        )
        object.__setattr__(
            self,
            "fsr1_like_sharpening_strength",
            validate_fsr1_like_strength(
                self.fsr1_like_sharpening_strength,
                "fsr1_like_sharpening_strength",
            ),
        )
        object.__setattr__(
            self,
            "fsr1_like_edge_strength",
            validate_fsr1_like_strength(
                self.fsr1_like_edge_strength,
                "fsr1_like_edge_strength",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the profile to a JSON-friendly mapping."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RuntimeProfile":
        """Build a runtime profile from decoded JSON data."""

        if not isinstance(payload, dict):
            raise TypeError("Runtime profile payload must be a dictionary.")

        return cls(
            capture_backend=str(payload.get("capture_backend", "auto")),
            opencv_threads=int(payload.get("opencv_threads", 1)),
            upscaling_method=str(payload.get("upscaling_method", "bicubic")),
            processing_backend=str(payload.get("processing_backend", "auto")),
            fsr1_like_sharpening_enabled=payload.get(
                "fsr1_like_sharpening_enabled",
                FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
            ),
            fsr1_like_sharpening_strength=payload.get(
                "fsr1_like_sharpening_strength",
                FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
            ),
            fsr1_like_edge_strength=payload.get(
                "fsr1_like_edge_strength",
                FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
            ),
        )

    @classmethod
    def load(cls, file_path: Path | None = None) -> "RuntimeProfile":
        """Load a runtime profile from disk."""

        resolved_path = file_path or runtime_profile_path()

        with resolved_path.open("r", encoding="utf-8") as profile_file:
            payload = json.load(profile_file)

        return cls.from_dict(payload)

    def save(self, file_path: Path | None = None) -> None:
        """Persist the profile atomically so an interrupted write keeps the old file."""

        resolved_path = file_path or runtime_profile_path()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp")

        try:
            with temporary_path.open("w", encoding="utf-8") as profile_file:
                json.dump(
                    self.to_dict(),
                    profile_file,
                    indent=2,
                    sort_keys=True,
                )
                profile_file.write("\n")

            temporary_path.replace(resolved_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def display(self, logger_instance: logging.Logger | None = None) -> None:
        """Log the active runtime profile in a readable format."""

        active_logger = logger_instance or logger
        active_logger.info("=" * 50)
        active_logger.info("Active Runtime Profile")
        active_logger.info("=" * 50)
        active_logger.info("Capture backend: %s", self.capture_backend.upper())
        active_logger.info("OpenCV threads: %s", self.opencv_threads)
        active_logger.info(
            "Upscaling method: %s",
            upscaling_method_display_name(self.upscaling_method),
        )
        active_logger.info(
            "FSR1-like sharpening: %s (strength %.2f)",
            "enabled" if self.fsr1_like_sharpening_enabled else "disabled",
            self.fsr1_like_sharpening_strength,
        )
        active_logger.info(
            "FSR1-like edge strength: %.2f",
            self.fsr1_like_edge_strength,
        )
        active_logger.info(
            "Processing backend: %s",
            processing_backend_display_name(self.processing_backend),
        )
        active_logger.info("=" * 50)


DEFAULT_RUNTIME_PROFILE = RuntimeProfile()
