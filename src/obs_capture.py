"""Experimental OBS Spout output receiver. GPU textures, not Virtual Camera.

OBS owns game selection and timestamps are local receipt times (not game render
timestamps). An explicit sender prevents accidentally selecting another app.
"""
from __future__ import annotations
import ctypes
import importlib
import threading
import time

from config import CaptureRegion
from wgc_capture import D3D11Frame


class OBSCaptureBackend:
    def __init__(self, sender_name="URFTS", *, loader=importlib.import_module,
                 timeout=2.0):
        try:
            native = loader("_urfts_obs_spout")
        except (ImportError, OSError) as error:
            raise RuntimeError("OBS capture needs the native Spout receiver. Run native/obs_spout/build.ps1; "
                               "install the OBS Spout output plugin and enable sender URFTS.") from error
        if getattr(native, "ABI_VERSION", 0) != 1:
            raise RuntimeError("Incompatible OBS Spout receiver ABI")
        self._receiver = native.Receiver(sender_name)
        self._owner = threading.get_ident()
        self._closed = False
        self._sequence = 0
        self.timeout = timeout
        self.capture_region = CaptureRegion()
        self.gpu_replaced_frames = 0  # Sender-side drops are not exposed by this protocol.

    @property
    def d3d11_device_pointer(self):
        return ctypes.c_void_p(self._receiver.device)

    @property
    def d3d11_context_pointer(self):
        return ctypes.c_void_p(self._receiver.context)

    def grab_gpu_frame(self):
        if self._closed or threading.get_ident() != self._owner:
            raise RuntimeError("OBS capture must run on its owning thread and remain open")
        deadline = time.perf_counter() + self.timeout
        while time.perf_counter() < deadline:
            item = self._receiver.receive()
            if item is not None:
                pointer, width, height, format_ = item
                self._sequence += 1
                self.capture_region = CaptureRegion(width=width, height=height)
                return D3D11Frame(pointer, width=width, height=height,
                                  dxgi_format=format_, sequence=self._sequence,
                                  captured_at=time.perf_counter())
            time.sleep(0.001)
        raise RuntimeError("No new OBS Spout frame within timeout. Enable output named URFTS; "
                           "keep OBS and URFTS on the same GPU with SDR output.")

    def close(self):
        if not self._closed:
            self._receiver.close()
            self._closed = True
