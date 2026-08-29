"""DXcam capture backend implementation."""

from __future__ import annotations

import cv2

try:
    import dxcam
except ImportError:  # pragma: no cover - depends on the local environment
    dxcam = None

from capture_backend import CaptureBackend
from config import CaptureRegion


class DXCamCaptureBackend(CaptureBackend):
    """Capture frames through DXcam when the backend is available."""

    def __init__(self, region: CaptureRegion | None = None):
        if dxcam is None:
            raise RuntimeError("DXcam is not installed in this environment.")

        self.capture_region = region or CaptureRegion()

        self.region = (
            self.capture_region.left,
            self.capture_region.top,
            self.capture_region.left + self.capture_region.width,
            self.capture_region.top + self.capture_region.height,
        )

        self.camera = dxcam.create(
            output_color="BGRA",
        )

    def set_capture_region(self, region: CaptureRegion) -> None:
        """Update the region passed to DXcam for subsequent grabs."""

        self.capture_region = region
        self.region = (
            region.left,
            region.top,
            region.left + region.width,
            region.top + region.height,
        )

    def grab_frame(self):
        """Capture one frame and convert it to BGR."""

        frame_bgra = self.camera.grab(
            region=self.region,
        )

        if frame_bgra is None:
            raise RuntimeError(
                "DXcam did not return a frame."
            )

        return cv2.cvtColor(
            frame_bgra,
            cv2.COLOR_BGRA2BGR,
        )

    def close(self) -> None:
        """Release DXcam resources."""

        if self.camera is not None:
            self.camera.release()
