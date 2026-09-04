"""Architecture tests for WGC without requiring a working graphics session."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

import capture_manager
import wgc_capture
from capture_manager import CaptureManager
from config import CaptureRegion, normalize_capture_backend
from wgc_capture import (
    WGCCaptureBackend,
    WGCFrameNotReadyError,
    WGCWindowClosedError,
    _NativeFrame,
)


class _FakeRuntime:
    def __init__(self, frames=(), size=(4, 3)) -> None:
        self.size = size
        self.frames = deque(frames)
        self.closed = False
        self.recreated = []
        self.released = []

    def try_get_frame(self):
        return self.frames.popleft() if self.frames else None

    def frame_to_bgr(self, frame):
        output = np.zeros((frame.height, frame.width, 3), dtype=np.uint8)
        output[:, :, 0] = 11
        output[:, :, 1] = 22
        output[:, :, 2] = 33
        return output

    def release_frame(self, frame):
        self.released.append(frame)

    def recreate(self, width, height):
        self.size = (width, height)
        self.recreated.append((width, height))

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _valid_window(monkeypatch):
    monkeypatch.setattr(wgc_capture.win32gui, "IsWindow", lambda hwnd: True)
    monkeypatch.setattr(wgc_capture.win32gui, "IsIconic", lambda hwnd: False)


def _frame(width=4, height=3):
    return _NativeFrame(pointer=None, width=width, height=height)


def test_idle_reuse_keeps_capture_open_and_resumes(monkeypatch):
    runtime = _FakeRuntime([_frame()])
    backend = WGCCaptureBackend(7, runtime=runtime, frame_timeout=0.001, poll_interval=0)
    backend.reuse_idle_frames = True
    image = backend.grab_frame()
    image[:] = 0
    assert backend.grab_frame()[0, 0].tolist() == [11, 22, 33]
    assert not runtime.closed and not runtime.recreated
    runtime.frames.extend([_frame(6, 5), _frame(6, 5)])
    assert backend.grab_frame().shape == (5, 6, 3)
    monkeypatch.setattr(wgc_capture.win32gui, "IsIconic", lambda hwnd: True)
    with pytest.raises(WGCWindowClosedError):
        backend.grab_frame()
    backend.close()


def test_wgc_identifier_is_supported() -> None:
    assert normalize_capture_backend("WGC") == "wgc"


def test_first_frame_not_ready_is_clear() -> None:
    backend = WGCCaptureBackend(
        7, runtime=_FakeRuntime(), first_frame_timeout=0.001, poll_interval=0
    )
    with pytest.raises(WGCFrameNotReadyError, match="first frame"):
        backend.grab_frame()


def test_wgc_output_is_bgr_uint8_with_current_dimensions() -> None:
    backend = WGCCaptureBackend(7, runtime=_FakeRuntime([_frame()]))
    output = backend.grab_frame()
    assert output.shape == (3, 4, 3)
    assert output.dtype == np.uint8
    assert output[0, 0].tolist() == [11, 22, 33]
    assert backend.capture_region == CaptureRegion(left=0, top=0, width=4, height=3)


def test_resize_recreates_pool_and_returns_new_size() -> None:
    runtime = _FakeRuntime([_frame(6, 5), _frame(6, 5)])
    backend = WGCCaptureBackend(7, runtime=runtime)
    output = backend.grab_frame()
    assert runtime.recreated == [(6, 5)]
    assert output.shape == (5, 6, 3)
    assert len(runtime.released) == 2


def test_closed_window_and_idempotent_close(monkeypatch) -> None:
    runtime = _FakeRuntime([_frame()])
    backend = WGCCaptureBackend(7, runtime=runtime)
    monkeypatch.setattr(wgc_capture.win32gui, "IsWindow", lambda hwnd: False)
    with pytest.raises(WGCWindowClosedError, match="closed"):
        backend.grab_frame()
    monkeypatch.setattr(wgc_capture.win32gui, "IsWindow", lambda hwnd: True)
    backend.close()
    backend.close()
    assert runtime.closed is True


def test_target_aware_candidate_ordering() -> None:
    selected = CaptureManager.__new__(CaptureManager)
    selected.window_hwnd = 7
    fixed = CaptureManager.__new__(CaptureManager)
    fixed.window_hwnd = None
    assert selected.automatic_candidate_names() == ("wgc", "dxcam", "mss")
    assert fixed.automatic_candidate_names() == ("dxcam", "mss")


def test_explicit_wgc_failure_is_clear(monkeypatch) -> None:
    monkeypatch.setattr(
        capture_manager.WGCCaptureBackend,
        "availability",
        staticmethod(lambda: (False, "unsupported test system")),
    )
    with pytest.raises(RuntimeError, match="Requested WGC backend is unavailable"):
        CaptureManager(backend="wgc", window_hwnd=7)


def test_auto_and_saved_wgc_fall_back_to_mss(monkeypatch) -> None:
    class _MSS:
        def __init__(self, region):
            self.capture_region = region
        def grab_frame(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)
        def set_capture_region(self, region):
            self.capture_region = region
        def close(self):
            pass

    monkeypatch.setattr(
        capture_manager.WGCCaptureBackend,
        "availability",
        staticmethod(lambda: (False, "unsupported test system")),
    )
    monkeypatch.setattr(
        capture_manager,
        "DXCamCaptureBackend",
        lambda region: (_ for _ in ()).throw(RuntimeError("dxcam unavailable")),
    )
    monkeypatch.setattr(capture_manager, "MSSCaptureBackend", _MSS)

    automatic = CaptureManager(backend="auto", window_hwnd=7)
    saved = CaptureManager(
        backend="wgc", window_hwnd=7, fallback_on_explicit_failure=True
    )
    assert automatic.backend_name == "mss"
    assert saved.backend_name == "mss"

