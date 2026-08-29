"""Capture backend selection and fallback logic."""

from __future__ import annotations

import logging

from capture import MSSCaptureBackend
from config import CaptureRegion, normalize_capture_backend
from dxcam_capture import DXCamCaptureBackend
from wgc_capture import WGCCaptureBackend
from window_capture import crop_window_client

logger = logging.getLogger(__name__)


class CaptureManager:
    """Select a capture backend with target-aware, verified fallback."""

    def __init__(
        self,
        backend: str = "auto",
        capture_region: CaptureRegion | None = None,
        window_hwnd: int | None = None,
        fallback_on_explicit_failure: bool = False,
    ):
        self.requested_backend = normalize_capture_backend(backend)
        self.capture_region = capture_region or CaptureRegion()
        self.window_hwnd = window_hwnd
        self._window_client_region = capture_region if window_hwnd is not None else None
        self.fallback_on_explicit_failure = fallback_on_explicit_failure
        self.backend_name: str | None = None
        self._wgc_client_crop_logged = False
        self.backend = self._create_backend()

    def _create_backend(self):
        if self.requested_backend == "auto":
            return self._create_automatic_backend()

        if self.requested_backend == "wgc":
            try:
                return self._create_verified_wgc()
            except Exception as error:
                if not self.fallback_on_explicit_failure:
                    raise RuntimeError(f"Requested WGC backend is unavailable: {error}") from error
                logger.warning("Saved WGC backend is unavailable: %s", error)
                return self._create_automatic_backend(exclude={"wgc"})

        if self.requested_backend == "dxcam":
            self.backend_name = "dxcam"
            return DXCamCaptureBackend(region=self.capture_region)

        if self.requested_backend == "mss":
            self.backend_name = "mss"
            return MSSCaptureBackend(region=self.capture_region)

        raise ValueError(
            f"Unsupported capture backend: "
            f"{self.requested_backend}"
        )

    def automatic_candidate_names(self) -> tuple[str, ...]:
        """Return the deliberate candidate order for the current target type."""

        if self.window_hwnd is not None:
            return "wgc", "dxcam", "mss"
        return "dxcam", "mss"

    def _create_verified_wgc(self):
        if self.window_hwnd is None:
            raise RuntimeError("WGC requires --window-title or --window-hwnd.")
        available, reason = WGCCaptureBackend.availability()
        logger.info("WGC support: %s (%s)", "available" if available else "unavailable", reason)
        if not available:
            raise RuntimeError(reason)
        backend = WGCCaptureBackend(hwnd=self.window_hwnd)
        try:
            backend.grab_frame()
        except Exception:
            backend.close()
            raise
        self.backend_name = "wgc"
        self.capture_region = backend.capture_region
        logger.info("Selected capture backend: Windows Graphics Capture")
        return backend

    def _create_automatic_backend(self, exclude: set[str] | None = None):
        excluded = exclude or set()
        for candidate in self.automatic_candidate_names():
            if candidate in excluded:
                continue
            logger.info("Testing capture backend: %s", candidate.upper())
            backend = None
            try:
                if candidate == "wgc":
                    return self._create_verified_wgc()
                if candidate == "dxcam":
                    backend = DXCamCaptureBackend(region=self.capture_region)
                    backend.grab_frame()
                else:
                    backend = MSSCaptureBackend(region=self.capture_region)
                    backend.grab_frame()
                self.backend_name = candidate
                logger.info("Selected capture backend: %s", candidate.upper())
                return backend
            except Exception as error:
                logger.warning(
                    "%s unavailable: %s: %s", candidate.upper(), type(error).__name__, error
                )
                if backend is not None:
                    try:
                        backend.close()
                    except Exception:
                        pass
        raise RuntimeError("No usable capture backend is available.")

    def grab_frame(self):
        frame = self.backend.grab_frame()
        if self.backend_name == "wgc" and self.window_hwnd is not None:
            surface_height, surface_width = frame.shape[:2]
            frame = crop_window_client(frame, self.window_hwnd)
            client_height, client_width = frame.shape[:2]
            if (
                not self._wgc_client_crop_logged
                and (client_width, client_height) != (surface_width, surface_height)
            ):
                logger.info(
                    "WGC client crop: %sx%s window surface -> %sx%s game client",
                    surface_width,
                    surface_height,
                    client_width,
                    client_height,
                )
                self._wgc_client_crop_logged = True
            elif not self._wgc_client_crop_logged and self._window_client_region is not None:
                expected_size = (
                    self._window_client_region.width,
                    self._window_client_region.height,
                )
                if expected_size != (surface_width, surface_height):
                    logger.warning(
                        "WGC could not safely crop %sx%s surface to expected "
                        "%sx%s client; retaining the complete surface",
                        surface_width,
                        surface_height,
                        *expected_size,
                    )
                    self._wgc_client_crop_logged = True
            client_region = self._window_client_region or self.capture_region
            self.capture_region = CaptureRegion(
                left=client_region.left,
                top=client_region.top,
                width=client_width,
                height=client_height,
            )
            return frame
        backend_region = getattr(self.backend, "capture_region", None)
        if backend_region is not None:
            self.capture_region = backend_region
        return frame

    def set_capture_region(self, capture_region: CaptureRegion) -> None:
        """Update the active backend without changing the selected backend type."""

        self.capture_region = capture_region
        self.backend.set_capture_region(capture_region)

    def close(self) -> None:
        self.backend.close()
