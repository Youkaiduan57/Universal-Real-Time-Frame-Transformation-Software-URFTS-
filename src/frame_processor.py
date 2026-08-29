"""Frame processing pipeline wrappers."""

from __future__ import annotations

import time

from config import OUTPUT_HEIGHT, OUTPUT_WIDTH
from processing_backend import ProcessingBackend
from upscaler import Upscaler


class FrameProcessor:
    """Process captured frames through the active upscaler."""

    def __init__(
        self,
        processing_backend: ProcessingBackend | None = None,
        output_width: int = OUTPUT_WIDTH,
        output_height: int = OUTPUT_HEIGHT,
        method: str = "bicubic",
    ):
        self.processing_backend = processing_backend or Upscaler(
            output_width=output_width,
            output_height=output_height,
            method=method,
        )

    def process(self, frame):
        """Return a processed frame."""

        started = time.perf_counter()
        processed_frame = self.processing_backend.process(frame)
        self.last_postprocessing_ms = (time.perf_counter() - started) * 1000.0
        self.last_preprocessing_ms = None
        self.last_inference_ms = None
        return processed_frame

def refine_output(frame, strength=0.0):
    """Mild, identical detail enhancement for real and generated output.

    Does not reconstruct missing detail. Zero bypasses all work and allocation.
    """
    if not 0.0 <= strength <= 0.25:
        raise ValueError("Output refinement strength must be between 0 and 0.25.")
    if strength == 0:
        return frame
    import cv2
    import numpy as np
    kernel = np.array([[0, -strength, 0],
                       [-strength, 1 + 4 * strength, -strength],
                       [0, -strength, 0]], dtype=np.float32)
    return cv2.filter2D(frame, -1, kernel, borderType=cv2.BORDER_REPLICATE)
