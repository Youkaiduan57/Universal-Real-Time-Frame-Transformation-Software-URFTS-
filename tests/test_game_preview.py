from types import SimpleNamespace
import game_preview


def test_preview_is_click_through_hides_on_alt_tab_and_restores(monkeypatch):
    calls = []
    state = {"foreground": 20, "exists": True}
    gui = SimpleNamespace(
        FindWindow=lambda *a: 10, GetWindowLong=lambda *a: 0,
        SetWindowLong=lambda *a: calls.append(("style", a)),
        SetLayeredWindowAttributes=lambda *a: calls.append(("alpha", a)),
        SetForegroundWindow=lambda h: state.update(foreground=h),
        IsWindow=lambda h: state["exists"],
        GetForegroundWindow=lambda: state["foreground"], IsIconic=lambda h: False,
        SetWindowPos=lambda *a: calls.append(("position", a)),
        ShowWindow=lambda *a: calls.append(("hide", a)))
    monkeypatch.setattr(game_preview, "win32gui", gui)
    monkeypatch.setattr(game_preview.win32api, "MonitorFromWindow", lambda *a: 1)
    monkeypatch.setattr(game_preview.win32api, "GetMonitorInfo", lambda h: {"Monitor": (0,0,1920,1080)})
    preview = game_preview.GamePreview("Preview", 20)
    flags = calls[0][1][-1]
    c = game_preview.win32con
    assert flags & c.WS_EX_TRANSPARENT and flags & c.WS_EX_LAYERED and flags & c.WS_EX_NOACTIVATE
    assert calls[-1][0] == "position"
    state["foreground"] = 30
    preview.update()
    assert calls[-1][0] == "hide"
    state["foreground"] = 20
    preview.update()
    assert calls[-1][0] == "position"
    state["exists"] = False
    assert not preview.update()


def test_stop_shortcut_requires_all_three_keys(monkeypatch):
    keys = set()
    monkeypatch.setattr(game_preview.win32api, "GetAsyncKeyState", lambda k: 0x8000 if k in keys else 0)
    keys.add(ord("Q"))
    assert not game_preview.GamePreview.stop_requested()
    keys.update((game_preview.win32con.VK_CONTROL, game_preview.win32con.VK_MENU))
    assert game_preview.GamePreview.stop_requested()
