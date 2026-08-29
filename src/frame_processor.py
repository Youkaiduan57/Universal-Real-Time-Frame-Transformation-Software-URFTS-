"""Frame processing pipeline wrappers."""

from __future__ import annotations

import time

import cv2

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
    """Apply mild Gaussian unsharp masking to every presented frame.

    The same deterministic spatial pass is used for real and generated frames,
    so it cannot introduce alternating filter behavior. GaussianBlur is
    separable and substantially cheaper at 1080p than the previous generic
    2-D convolution. It does not reconstruct detail; zero remains a true
    allocation-free bypass.
    """
    if not 0.0 <= strength <= 0.25:
        raise ValueError("Output refinement strength must be between 0 and 0.25.")
    if strength == 0:
        return frame
    blurred = cv2.GaussianBlur(
        frame,
        (3, 3),
        sigmaX=0.0,
        borderType=cv2.BORDER_REFLECT101,
    )
    return cv2.addWeighted(
        frame,
        1.0 + strength,
        blurred,
        -strength,
        0.0,
    )
