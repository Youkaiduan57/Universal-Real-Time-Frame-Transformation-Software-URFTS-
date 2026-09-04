"""Windows top-level window discovery and client-area tracking."""

from __future__ import annotations

from dataclasses import dataclass

import win32con
import win32api
import win32gui

from config import CaptureRegion


PREVIEW_WINDOW_TITLE = "UniversalUpscaler Preview"
_UNSUITABLE_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd"}


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """A selectable visible top-level window."""

    hwnd: int
    title: str


class WindowCaptureError(RuntimeError):
    """Raised when a selected window cannot provide a capturable client area."""


def list_visible_windows() -> list[WindowInfo]:
    """Return user-facing, visible top-level windows in a stable order."""

    windows: list[WindowInfo] = []

    def collect(hwnd: int, _extra: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return

        title = win32gui.GetWindowText(hwnd).strip()
        if not title or title == PREVIEW_WINDOW_TITLE:
            return

        if win32gui.GetClassName(hwnd) in _UNSUITABLE_CLASSES:
            return

        windows.append(WindowInfo(hwnd=hwnd, title=title))

    win32gui.EnumWindows(collect, None)
    return sorted(windows, key=lambda item: (item.title.casefold(), item.hwnd))


def select_window(
    *,
    title: str | None = None,
    hwnd: int | None = None,
) -> WindowInfo:
    """Select a window by explicit handle, exact title, or unique partial title."""

    if hwnd is not None:
        if not win32gui.IsWindow(hwnd):
            raise WindowCaptureError(f"Invalid window handle: {hwnd}")

        window_title = win32gui.GetWindowText(hwnd).strip()
        if not window_title:
            raise WindowCaptureError(f"Window handle {hwnd} does not have a selectable title.")

        return WindowInfo(hwnd=hwnd, title=window_title)

    if not title or not title.strip():
        raise WindowCaptureError("A window title or window handle is required.")

    requested_title = title.strip()
    windows = list_visible_windows()
    exact_matches = [item for item in windows if item.title.casefold() == requested_title.casefold()]
    matches = exact_matches or [
        item for item in windows if requested_title.casefold() in item.title.casefold()
    ]

    if not matches:
        raise WindowCaptureError(f"No visible window matches title: {requested_title!r}")

    if len(matches) > 1:
        titles = ", ".join(f"{item.title!r} ({item.hwnd})" for item in matches)
        raise WindowCaptureError(
            f"Window title {requested_title!r} is ambiguous; use --window-hwnd. Matches: {titles}"
        )

    return matches[0]


def get_client_capture_region(hwnd: int) -> CaptureRegion:
    """Convert a window client rectangle to its visible primary-screen region."""

    if not win32gui.IsWindow(hwnd):
        raise WindowCaptureError(f"Selected window closed or has invalid handle: {hwnd}")

    if win32gui.IsIconic(hwnd):
        raise WindowCaptureError("Selected window is minimized; capture is paused.")

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
    screen_right, screen_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))

    visible_left = max(screen_left, 0)
    visible_top = max(screen_top, 0)
    visible_right = min(screen_right, win32api.GetSystemMetrics(win32con.SM_CXSCREEN))
    visible_bottom = min(screen_bottom, win32api.GetSystemMetrics(win32con.SM_CYSCREEN))

    if visible_right <= visible_left or visible_bottom <= visible_top:
        raise WindowCaptureError("Selected window has a zero-size or off-screen client area.")

    return CaptureRegion(
        left=visible_left,
        top=visible_top,
        width=visible_right - visible_left,
        height=visible_bottom - visible_top,
    )


class WindowRegionTracker:
    """Re-read a selected window's client-area region as it moves or resizes."""

    def __init__(self, window: WindowInfo) -> None:
        self.window = window
        self._last_region: CaptureRegion | None = None

    def refresh(self) -> CaptureRegion | None:
        """Return a new region only when the selected client area changed."""

        region = get_client_capture_region(self.window.hwnd)
        if region == self._last_region:
            return None

        self._last_region = region
        return region


def _visible_frame_bounds(hwnd):
    """DWM bounds exclude the invisible resize margins in GetWindowRect."""
    import ctypes
    from ctypes import wintypes
    rect = wintypes.RECT()
    hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd), 9, ctypes.byref(rect), ctypes.sizeof(rect))
    if hr != 0:
        return win32gui.GetWindowRect(hwnd)
    return rect.left, rect.top, rect.right, rect.bottom


def crop_window_client(frame, hwnd):
    """Crop WGC chrome only when frame and native bounds agree exactly.

    During resize/DPI transitions, retain the frame rather than guessing offsets.
    """
    box = window_client_crop_box(frame.shape[1], frame.shape[0], hwnd)
    if box is None:
        return frame
    x, y, width, height = box
    return frame[y:y+height, x:x+width].copy()


def window_client_crop_box(surface_width, surface_height, hwnd):
    """Verified client rectangle shared by CPU cropping and GPU texture copies."""
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width, height = right-left, bottom-top
    if width <= 0 or height <= 0 or (surface_height, surface_width) == (height, width):
        return None
    bounds = _visible_frame_bounds(hwnd)
    if (surface_height, surface_width) != (bounds[3]-bounds[1], bounds[2]-bounds[0]):
        return None
    screen_x, screen_y = win32gui.ClientToScreen(hwnd, (left, top))
    x, y = screen_x-bounds[0], screen_y-bounds[1]
    if x < 0 or y < 0 or x+width > surface_width or y+height > surface_height:
        return None
    return x, y, width, height
