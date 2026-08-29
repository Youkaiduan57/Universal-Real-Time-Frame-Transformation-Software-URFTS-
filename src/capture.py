"""MSS capture backend implementation."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from mss import MSS
except ImportError:  # pragma: no cover - depends on the local environment
    MSS = None

from capture_backend import CaptureBackend
from config import CaptureRegion


class MSSCaptureBackend(CaptureBackend):
    """Capture frames through the MSS screen-grab backend."""

    def __init__(self, region: CaptureRegion | None = None):
        if MSS is None:
            raise RuntimeError("MSS is not installed in this environment.")

        self.sct = MSS()
        self.capture_region = region or CaptureRegion()

        self.region = {
            "left": self.capture_region.left,
            "top": self.capture_region.top,
            "width": self.capture_region.width,
            "height": self.capture_region.height,
        }

    def set_capture_region(self, region: CaptureRegion) -> None:
        """Update the screen region used by subsequent MSS grabs."""

        self.capture_region = region
        self.region = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }

    def grab_frame(self):
        """Capture one frame and convert it to BGR."""

        raw_frame = self.sct.grab(self.region)
        frame_bgra = np.array(raw_frame)

        return cv2.cvtColor(
            frame_bgra,
            cv2.COLOR_BGRA2BGR,
        )

    def close(self) -> None:
        """Release MSS resources."""

        self.sct.close()


ScreenCapture = MSSCaptureBackend
