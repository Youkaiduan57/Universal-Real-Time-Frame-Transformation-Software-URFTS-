"""Brief 64x64 native window test; never captures or injects game input."""
import ctypes as ct
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import win32con
import win32gui
from d3d11_gpu_pipeline import D3D11SwapChainPresenter
from wgc_capture import _release

device,context=ct.c_void_p(),ct.c_void_p()
create=ct.WinDLL('d3d11').D3D11CreateDevice
create.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_uint,ct.c_void_p,ct.c_uint,
                 ct.c_uint,ct.POINTER(ct.c_void_p),ct.c_void_p,ct.POINTER(ct.c_void_p)]
create.restype=ct.c_long
result=create(None,1,None,0x20,None,0,7,ct.byref(device),None,ct.byref(context))
assert result>=0
presenter=None
try:
    foreground=win32gui.GetForegroundWindow()
    presenter=D3D11SwapChainPresenter(device,context,width=64,height=64)
    style=win32gui.GetWindowLong(presenter.hwnd,win32con.GWL_EXSTYLE)
    for flag in (win32con.WS_EX_NOACTIVATE,win32con.WS_EX_LAYERED,win32con.WS_EX_TRANSPARENT):
        assert style & flag
    assert win32gui.GetForegroundWindow()==foreground, 'Preview stole foreground focus'
    assert win32gui.SendMessage(presenter.hwnd,0x0084,0,0)==-1
    assert win32gui.SendMessage(presenter.hwnd,0x0021,0,0)==3
    presenter.present()
    print('Native layered swap-chain present, no-activation, and transparent hit-test checks PASS')
finally:
    if presenter: presenter.close()
    _release(context); _release(device)
