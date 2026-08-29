"""Frame upscaling helpers."""

from __future__ import annotations

from config import (
    FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
    FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
)
from processing_backend import OpenCVProcessingBackend


class Upscaler(OpenCVProcessingBackend):
    """Compatibility wrapper for the OpenCV processing backend."""

    def __init__(
        self,
        output_width: int = OUTPUT_WIDTH,
        output_height: int = OUTPUT_HEIGHT,
        method: str = "bicubic",
        fsr1_like_sharpening_enabled: bool = FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
        fsr1_like_sharpening_strength: float = FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
        fsr1_like_edge_strength: float = FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    ) -> None:
        super().__init__(
            output_width=output_width,
            output_height=output_height,
            upscaling_method=method,
            fsr1_like_sharpening_enabled=fsr1_like_sharpening_enabled,
            fsr1_like_sharpening_strength=fsr1_like_sharpening_strength,
            fsr1_like_edge_strength=fsr1_like_edge_strength,
        )

    def upscale(self, frame):
        """Preserve the older Upscaler API."""

        return self.process(frame)
