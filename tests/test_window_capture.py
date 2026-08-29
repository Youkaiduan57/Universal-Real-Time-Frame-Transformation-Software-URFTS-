"""Tests for Win32 window selection and client-area tracking without real windows."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import main
import window_capture
from capture_manager import CaptureManager
from config import CaptureRegion
from window_capture import WindowCaptureError, WindowInfo, WindowRegionTracker


def _fake_gui(monkeypatch, *, windows, rect=(0, 0, 100, 80), origin=(10, 20), iconic=False):
    visible = {hwnd: item[0] for hwnd, item in windows.items()}
    titles = {hwnd: item[1] for hwnd, item in windows.items()}
    classes = {hwnd: item[2] for hwnd, item in windows.items()}
    monkeypatch.setattr(window_capture.win32gui, "EnumWindows", lambda callback, extra: [callback(hwnd, extra) for hwnd in windows])
    monkeypatch.setattr(window_capture.win32gui, "IsWindowVisible", lambda hwnd: visible[hwnd])
    monkeypatch.setattr(window_capture.win32gui, "GetWindowText", lambda hwnd: titles[hwnd])
    monkeypatch.setattr(window_capture.win32gui, "GetClassName", lambda hwnd: classes[hwnd])
    monkeypatch.setattr(window_capture.win32gui, "IsWindow", lambda hwnd: hwnd in windows)
    monkeypatch.setattr(window_capture.win32gui, "IsIconic", lambda hwnd: iconic)
    monkeypatch.setattr(window_capture.win32gui, "GetClientRect", lambda hwnd: rect)
    monkeypatch.setattr(window_capture.win32gui, "ClientToScreen", lambda hwnd, point: (origin[0] + point[0], origin[1] + point[1]))
    monkeypatch.setattr(window_capture.win32api, "GetSystemMetrics", lambda metric: 1920 if metric == window_capture.win32con.SM_CXSCREEN else 1080)


def test_window_enumeration_filters_empty_hidden_system_and_preview(monkeypatch) -> None:
    _fake_gui(monkeypatch, windows={1: (True, "Game", "GameClass"), 2: (True, "", "App"), 3: (False, "Hidden", "App"), 4: (True, "Desktop", "Progman"), 5: (True, "UniversalUpscaler Preview", "App")})

    assert window_capture.list_visible_windows() == [WindowInfo(hwnd=1, title="Game")]


def test_title_matching_supports_exact_and_unique_partial_titles(monkeypatch) -> None:
    _fake_gui(monkeypatch, windows={1: (True, "My Game", "App")})

    assert window_capture.select_window(title="my game").hwnd == 1
    assert window_capture.select_window(title="game").hwnd == 1


def test_invalid_hwnd_is_rejected(monkeypatch) -> None:
    _fake_gui(monkeypatch, windows={1: (True, "Game", "App")})

    with pytest.raises(WindowCaptureError, match="Invalid window handle"):
        window_capture.select_window(hwnd=99)


def test_client_area_coordinates_are_converted_and_clipped(monkeypatch) -> None:
    _fake_gui(monkeypatch, windows={1: (True, "Game", "App")}, rect=(0, 0, 100, 80), origin=(-20, 30))

    assert window_capture.get_client_capture_region(1) == CaptureRegion(left=0, top=30, width=80, height=80)


@pytest.mark.parametrize(("iconic", "rect", "message"), [(True, (0, 0, 100, 80), "minimized"), (False, (0, 0, 0, 80), "zero-size")])
def test_minimized_and_zero_size_windows_are_rejected(monkeypatch, iconic, rect, message) -> None:
    _fake_gui(monkeypatch, windows={1: (True, "Game", "App")}, iconic=iconic, rect=rect)

    with pytest.raises(WindowCaptureError, match=message):
        window_capture.get_client_capture_region(1)


def test_tracker_reports_moving_and_resizing_regions(monkeypatch) -> None:
    _fake_gui(monkeypatch, windows={1: (True, "Game", "App")})
    # Each refresh calls ClientToScreen twice, so preserve one origin per refresh.
    origins = [(10, 20), (10, 20), (25, 30)]
    def client_to_screen(hwnd, point):
        index = min(client_to_screen.calls // 2, len(origins) - 1)
        client_to_screen.calls += 1
        return origins[index][0] + point[0], origins[index][1] + point[1]
    client_to_screen.calls = 0
    monkeypatch.setattr(window_capture.win32gui, "ClientToScreen", client_to_screen)
    tracker = WindowRegionTracker(WindowInfo(hwnd=1, title="Game"))

    assert tracker.refresh() == CaptureRegion(left=10, top=20, width=100, height=80)
    assert tracker.refresh() is None
    assert tracker.refresh() == CaptureRegion(left=25, top=30, width=100, height=80)


def test_capture_manager_keeps_fixed_region_without_window_tracking() -> None:
    manager = CaptureManager.__new__(CaptureManager)
    manager.capture_region = CaptureRegion(left=1, top=2, width=3, height=4)
    manager.backend = SimpleNamespace(set_capture_region=lambda region: setattr(manager, "observed", region))

    manager.set_capture_region(manager.capture_region)

    assert manager.observed == CaptureRegion(left=1, top=2, width=3, height=4)


def test_list_windows_exits_before_capture(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "parse_args", lambda: SimpleNamespace(list_windows=True))
    monkeypatch.setattr(main, "_configure_logging", lambda: None)
    monkeypatch.setattr(main, "list_visible_windows", lambda: [WindowInfo(hwnd=7, title="Test Window")])
    monkeypatch.setattr(main, "CaptureManager", lambda **kwargs: pytest.fail("capture must not start"))

    main.main()

    assert capsys.readouterr().out == "1. HWND 7 | Test Window\n"


def test_wgc_client_crop_removes_chrome(monkeypatch):
    import numpy as np
    _fake_gui(monkeypatch, windows={1: (True,"Game","App")},
              rect=(0,0,1280,720), origin=(320,194))
    monkeypatch.setattr(window_capture,"_visible_frame_bounds",lambda h:(319,157,1601,916))
    source=np.zeros((759,1282,3),dtype=np.uint8)
    source[37:757,1:1281]=123
    result=window_capture.crop_window_client(source,1)
    assert result.shape==(720,1280,3)
    assert np.all(result==123)
    assert result.flags.c_contiguous


def test_wgc_crop_does_not_guess_during_resize(monkeypatch):
    import numpy as np
    _fake_gui(monkeypatch,windows={1:(True,"Game","App")})
    monkeypatch.setattr(window_capture,"_visible_frame_bounds",lambda h:(0,0,102,110))
    source=np.zeros((120,140,3),dtype=np.uint8)
    assert window_capture.crop_window_client(source,1) is source
    client=np.zeros((80,100,3),dtype=np.uint8)
    assert window_capture.crop_window_client(client,1) is client
