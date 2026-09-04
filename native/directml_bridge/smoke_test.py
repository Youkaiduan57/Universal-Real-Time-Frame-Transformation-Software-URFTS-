"""Synthetic GPU smoke test; does not capture or alter any user window.

Run in a separate process: py -3.12 native/directml_bridge/smoke_test.py
"""
import ctypes as ct
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wgc_capture import (
    _D3D11Texture2DDesc, _DXGISampleDesc, _D3D11MappedSubresource, _release, _vtable_function,
)
import onnxruntime
import _urfts_directml as bridge
import numpy as np


def verify_pixels(device, context, output):
    desc = _D3D11Texture2DDesc()
    _vtable_function(output, 10, None, ct.POINTER(_D3D11Texture2DDesc))(output, ct.byref(desc))
    desc.Usage, desc.BindFlags, desc.CPUAccessFlags, desc.MiscFlags = 3, 0, 0x20000, 0
    staging = ct.c_void_p()
    try:
        hr = _vtable_function(device, 5, ct.c_long, ct.POINTER(_D3D11Texture2DDesc),
                             ct.c_void_p, ct.POINTER(ct.c_void_p))(
            device, ct.byref(desc), None, ct.byref(staging))
        if hr < 0:
            raise RuntimeError(f"Readback allocation: {hr:#x}")
        _vtable_function(context, 47, None, ct.c_void_p, ct.c_void_p)(context, staging, output)
        mapped = _D3D11MappedSubresource()
        hr = _vtable_function(context, 14, ct.c_long, ct.c_void_p, ct.c_uint,
                             ct.c_int, ct.c_uint, ct.POINTER(_D3D11MappedSubresource))(
            context, staging, 0, 1, 0, ct.byref(mapped))
        if hr < 0:
            raise RuntimeError(f"Readback Map: {hr:#x}")
        try:
            raw = np.ctypeslib.as_array((ct.c_ubyte * (mapped.RowPitch * desc.Height)).from_address(mapped.pData))
            rgb = raw.reshape(desc.Height, mapped.RowPitch)[:, :desc.Width*4].reshape(desc.Height, desc.Width, 4)[..., :3]
            error = np.abs(rgb.astype(float) - np.array([192, 128, 64])).mean()
            print(f"constant_colour_mae={error:.3f}", flush=True)
            assert error < 5.0, f"Incorrect colour or tensor data: MAE={error}"
        finally:
            _vtable_function(context, 15, None, ct.c_void_p, ct.c_uint)(context, staging, 0)
    finally:
        _release(staging)


def main():
    device, context = ct.c_void_p(), ct.c_void_p()
    create = ct.WinDLL("d3d11").D3D11CreateDevice
    create.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_uint,
                       ct.c_void_p, ct.c_uint, ct.c_uint,
                       ct.POINTER(ct.c_void_p), ct.c_void_p, ct.POINTER(ct.c_void_p)]
    create.restype = ct.c_long
    hr = create(None, 1, None, 0x20, None, 0, 7, ct.byref(device), None, ct.byref(context))
    if hr < 0:
        raise RuntimeError(f"D3D11CreateDevice: {hr:#x}")
    texture = ct.c_void_p()
    generator = None
    try:
        desc = _D3D11Texture2DDesc(
            Width=160, Height=96, MipLevels=1, ArraySize=1, Format=87,
            SampleDesc=_DXGISampleDesc(1, 0), Usage=0, BindFlags=8,
            CPUAccessFlags=0, MiscFlags=0,
        )
        class InitialData(ct.Structure):
            _fields_ = [("data", ct.c_void_p), ("pitch", ct.c_uint), ("slice", ct.c_uint)]
        pixels = (ct.c_ubyte * (160 * 96 * 4))(*([64, 128, 192, 255] * (160 * 96)))
        initial = InitialData(ct.cast(pixels, ct.c_void_p), 160 * 4, 0)
        hr = _vtable_function(device, 5, ct.c_long, ct.POINTER(_D3D11Texture2DDesc),
                             ct.c_void_p, ct.POINTER(ct.c_void_p))(
            device, ct.byref(desc), ct.byref(initial), ct.byref(texture))
        if hr < 0:
            raise RuntimeError(f"CreateTexture2D: {hr:#x}")
        generator = bridge.create_frame_generator(
            str(ROOT / "models/IFRNet_S_Vimeo90K.onnx"), -1, device.value, 160, 96)
        for i in range(3):
            start = time.perf_counter()
            output = ct.c_void_p(bridge.interpolate_d3d11(
                generator, texture.value, texture.value, 160, 96))
            try:
                verify_pixels(device, context, output)
            finally:
                _release(output)
            print(f"iteration={i} elapsed_ms={(time.perf_counter()-start)*1000:.2f}", flush=True)
        print("CONSTANT-COLOUR TEXTURE TEST PASS (motion quality not tested)", flush=True)
    finally:
        if generator is not None:
            bridge.close_frame_generator(generator)
            generator = None
        _release(texture)
        _release(context)
        _release(device)


if __name__ == "__main__":
    main()
