"""Tests for capture backend selection validation."""

from __future__ import annotations

import numpy as np
import pytest

import capture_manager
from capture_manager import CaptureManager
from config import CaptureRegion


def test_capture_manager_rejects_unsupported_backend() -> None:
    with pytest.raises(ValueError):
        CaptureManager(backend="unsupported-backend")


def test_wgc_frames_are_cropped_to_the_selected_game_client(monkeypatch) -> None:
    surface = np.zeros((759, 1282, 3), dtype=np.uint8)

    class FakeWGC:
        capture_region = CaptureRegion(width=1282, height=759)

        def grab_frame(self):
            return surface

    manager = CaptureManager.__new__(CaptureManager)
    manager.backend = FakeWGC()
    manager.backend_name = "wgc"
    manager.window_hwnd = 7
    manager.capture_region = CaptureRegion(left=320, top=194, width=1280, height=720)
    manager._window_client_region = manager.capture_region
    manager._wgc_client_crop_logged = False
    observed = []

    def crop(frame, hwnd):
        observed.append((frame.shape, hwnd))
        return frame[38:758, 1:1281].copy()

    monkeypatch.setattr(capture_manager, "crop_window_client", crop)

    result = manager.grab_frame()

    assert observed == [((759, 1282, 3), 7)]
    assert result.shape == (720, 1280, 3)
    assert manager.capture_region == CaptureRegion(
        left=320, top=194, width=1280, height=720
    )
    assert manager._wgc_client_crop_logged is True


def test_fixed_region_capture_does_not_apply_window_client_crop(monkeypatch) -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    class FakeDXCam:
        capture_region = CaptureRegion(width=1280, height=720)

        def grab_frame(self):
            return frame

    manager = CaptureManager.__new__(CaptureManager)
    manager.backend = FakeDXCam()
    manager.backend_name = "dxcam"
    manager.window_hwnd = 7
    manager.capture_region = FakeDXCam.capture_region
    monkeypatch.setattr(
        capture_manager,
        "crop_window_client",
        lambda *_args: pytest.fail("Non-WGC capture must not be client-cropped."),
    )

    assert manager.grab_frame() is frame
