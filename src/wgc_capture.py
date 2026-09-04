"""Windows Graphics Capture backend using the native WinRT and D3D11 ABIs.

The public boundary remains a three-channel BGR ``numpy.uint8`` array. WGC
delivers BGRA textures in GPU memory, so each returned frame necessarily uses a
D3D11 staging texture and a GPU-to-CPU copy before dropping the alpha channel.

All WinRT, COM, and D3D11 objects are created, used, and released on the thread
that constructs :class:`WGCCaptureBackend`. ``CreateFreeThreaded`` removes the
need for a UI dispatcher, but this backend deliberately polls its bounded
two-frame pool from ``grab_frame`` rather than introducing a project-wide
asynchronous pipeline.

The optional GPU boundary returns an owned ``ID3D11Texture2D`` reference.  It
does not stage, map, or create a NumPy pixel array.  Closing the returned
``D3D11Frame`` releases that reference; the transient WGC frame and WinRT
surface are released before ``grab_gpu_frame`` returns.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import win32gui

from capture_backend import CaptureBackend
from config import CaptureRegion

logger = logging.getLogger(__name__)

_MINIMUM_WGC_BUILD = 18362
_DXGI_FORMAT_B8G8R8A8_UNORM = 87
_D3D11_USAGE_STAGING = 3
_D3D11_CPU_ACCESS_READ = 0x20000
_D3D11_MAP_READ = 1
_D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
_D3D_DRIVER_TYPE_HARDWARE = 1
_D3D_DRIVER_TYPE_WARP = 5
_RPC_E_CHANGED_MODE = 0x80010106


class WGCError(RuntimeError):
    """Base exception for Windows Graphics Capture failures."""


class WGCUnavailableError(WGCError):
    """Raised when Windows Graphics Capture is unsupported or unavailable."""


class WGCFrameNotReadyError(WGCError):
    """Raised when no WGC frame arrives before the configured deadline."""


class WGCWindowClosedError(WGCError):
    """Raised when the selected HWND is no longer capturable."""


class WGCUnsupportedTextureError(WGCError):
    """Raised when WGC supplies a texture the D3D11 GPU path cannot consume."""


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> "_GUID":
        import uuid

        parsed = uuid.UUID(value)
        raw = parsed.bytes_le
        return cls.from_buffer_copy(raw)


class _SizeInt32(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32)]


class _DXGISampleDesc(ctypes.Structure):
    _fields_ = [("Count", wintypes.UINT), ("Quality", wintypes.UINT)]


class _D3D11Texture2DDesc(ctypes.Structure):
    _fields_ = [
        ("Width", wintypes.UINT),
        ("Height", wintypes.UINT),
        ("MipLevels", wintypes.UINT),
        ("ArraySize", wintypes.UINT),
        ("Format", wintypes.UINT),
        ("SampleDesc", _DXGISampleDesc),
        ("Usage", wintypes.UINT),
        ("BindFlags", wintypes.UINT),
        ("CPUAccessFlags", wintypes.UINT),
        ("MiscFlags", wintypes.UINT),
    ]


class _D3D11MappedSubresource(ctypes.Structure):
    _fields_ = [
        ("pData", ctypes.c_void_p),
        ("RowPitch", wintypes.UINT),
        ("DepthPitch", wintypes.UINT),
    ]


@dataclass(frozen=True, slots=True)
class _NativeFrame:
    pointer: ctypes.c_void_p
    width: int
    height: int


class D3D11Frame:
    """Owned WGC texture reference plus the metadata needed by GPU consumers.

    ``texture_pointer`` must be an already-owned COM reference.  The frame
    takes ownership of that reference and releases it exactly once from
    :meth:`close`.  No arbitrary application state or CPU pixel data is kept
    here.
    """

    __slots__ = (
        "_texture",
        "width",
        "height",
        "dxgi_format",
        "sequence",
        "captured_at",
        "_closed",
    )

    def __init__(
        self,
        texture_pointer: ctypes.c_void_p | int,
        *,
        width: int,
        height: int,
        dxgi_format: int,
        sequence: int,
        captured_at: float,
    ) -> None:
        pointer_value = (
            texture_pointer.value
            if isinstance(texture_pointer, ctypes.c_void_p)
            else int(texture_pointer)
        )
        if not pointer_value:
            raise ValueError("D3D11Frame requires a non-null texture pointer.")
        if width <= 0 or height <= 0:
            raise ValueError("D3D11Frame dimensions must be positive.")
        if dxgi_format not in (_DXGI_FORMAT_B8G8R8A8_UNORM, 28):
            raise ValueError(
                "D3D11Frame requires SDR BGRA8 (87) or RGBA8 (28); "
                f"received {dxgi_format}."
            )
        if sequence < 0:
            raise ValueError("D3D11Frame sequence must be non-negative.")
        if captured_at < 0.0:
            raise ValueError("D3D11Frame capture timestamp must be non-negative.")

        self._texture = ctypes.c_void_p(pointer_value)
        self.width = int(width)
        self.height = int(height)
        self.dxgi_format = int(dxgi_format)
        self.sequence = int(sequence)
        self.captured_at = float(captured_at)
        self._closed = False

    @property
    def texture_pointer(self) -> ctypes.c_void_p:
        if self._closed or not self._texture.value:
            raise WGCError("D3D11Frame texture was accessed after close.")
        return ctypes.c_void_p(self._texture.value)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def format_name(self) -> str:
        return "DXGI_FORMAT_B8G8R8A8_UNORM" if self.dxgi_format == 87 else "DXGI_FORMAT_R8G8B8A8_UNORM"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _release(self._texture)

    def __enter__(self) -> "D3D11Frame":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def __del__(self) -> None:
        # Deterministic callers use close/context-manager semantics.  This is a
        # last-resort guard for exceptional interpreter paths.
        try:
            self.close()
        except Exception:
            pass


class _Runtime(Protocol):
    size: tuple[int, int]

    def try_get_frame(self) -> _NativeFrame | None: ...
    def frame_to_bgr(self, frame: _NativeFrame) -> np.ndarray: ...
    def release_frame(self, frame: _NativeFrame) -> None: ...
    def recreate(self, width: int, height: int) -> None: ...
    def close(self) -> None: ...


def _hresult_failed(result: int) -> bool:
    return bool(ctypes.c_uint32(result).value & 0x80000000)


def _check_hresult(result: int, operation: str) -> None:
    if _hresult_failed(result):
        code = ctypes.c_uint32(result).value
        raise WGCError(f"{operation} failed with HRESULT 0x{code:08X}.")


def _vtable_function(pointer: ctypes.c_void_p, index: int, restype, *argtypes):
    if not pointer or not pointer.value:
        raise WGCError("Attempted to call a null COM interface.")
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    address = table[index]
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(address)


def _release(pointer: ctypes.c_void_p | None) -> None:
    if pointer is not None and pointer.value:
        _vtable_function(pointer, 2, wintypes.ULONG)(pointer)
        pointer.value = None


def _add_ref(pointer: ctypes.c_void_p) -> ctypes.c_void_p:
    if not pointer or not pointer.value:
        raise WGCError("Attempted to retain a null COM interface.")
    _vtable_function(pointer, 1, wintypes.ULONG)(pointer)
    return ctypes.c_void_p(pointer.value)


def _query_interface(pointer: ctypes.c_void_p, iid: _GUID, operation: str) -> ctypes.c_void_p:
    output = ctypes.c_void_p()
    result = _vtable_function(
        pointer, 0, ctypes.c_long, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)
    )(pointer, ctypes.byref(iid), ctypes.byref(output))
    _check_hresult(result, operation)
    return output


class _HString:
    def __init__(self, combase, value: str) -> None:
        self._combase = combase
        self.value = ctypes.c_void_p()
        combase.WindowsCreateString.argtypes = [
            wintypes.LPCWSTR,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        combase.WindowsCreateString.restype = ctypes.c_long
        _check_hresult(
            combase.WindowsCreateString(value, len(value), ctypes.byref(self.value)),
            f"WindowsCreateString({value})",
        )

    def close(self) -> None:
        if self.value.value:
            self._combase.WindowsDeleteString(self.value)
            self.value.value = None


class _NativeWGCRuntime:
    """Small, explicit ABI layer over Windows.Graphics.Capture and D3D11."""

    _IID_GRAPHICS_CAPTURE_ITEM = _GUID.from_string("79C3F95B-31F7-4EC2-A464-632EF5D30760")
    _IID_CAPTURE_ITEM_INTEROP = _GUID.from_string("3628E81B-3CAC-4C60-B7F4-23CE0E0C3356")
    _IID_FRAME_POOL_STATICS2 = _GUID.from_string("589B103F-6BBC-5DF5-A991-02E28B3B66D5")
    _IID_SESSION_STATICS = _GUID.from_string("2224A540-5974-49AA-B232-0882536F4CB5")
    _IID_IDXGI_DEVICE = _GUID.from_string("54EC77FA-1377-44E6-8C32-88FD5F44C84C")
    _IID_DXGI_INTERFACE_ACCESS = _GUID.from_string("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1")
    _IID_D3D11_TEXTURE2D = _GUID.from_string("6F15AAF2-D208-4E89-9AB4-489535D34F9C")
    _IID_CLOSABLE = _GUID.from_string("30D5A829-7FA4-4026-83BB-D75BAE4EA99E")

    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd
        self.size = (0, 0)
        self._ro_initialized = False
        self._closed = False
        self._device = ctypes.c_void_p()
        self._context = ctypes.c_void_p()
        self._winrt_device = ctypes.c_void_p()
        self._item = ctypes.c_void_p()
        self._frame_pool = ctypes.c_void_p()
        self._session = ctypes.c_void_p()
        self._staging = ctypes.c_void_p()
        self._staging_desc: tuple[int, int, int] | None = None
        self._combase = ctypes.WinDLL("combase")
        self._d3d11 = ctypes.WinDLL("d3d11")

        try:
            self._initialize()
        except Exception:
            self.close()
            raise

    @classmethod
    def support_status(cls) -> tuple[bool, str]:
        if platform.system() != "Windows":
            return False, "Windows Graphics Capture is only available on Windows."
        try:
            build = int(platform.version().split(".")[-1])
        except (TypeError, ValueError):
            build = 0
        if build < _MINIMUM_WGC_BUILD:
            return False, f"WGC window interop requires Windows 10 build {_MINIMUM_WGC_BUILD} or newer."
        try:
            combase = ctypes.WinDLL("combase")
            d3d11 = ctypes.WinDLL("d3d11")
            for library, symbol in (
                (combase, "RoInitialize"),
                (combase, "RoGetActivationFactory"),
                (d3d11, "D3D11CreateDevice"),
                (d3d11, "CreateDirect3D11DeviceFromDXGIDevice"),
            ):
                getattr(library, symbol)
        except (OSError, AttributeError) as error:
            return False, f"Required Windows API is unavailable: {error}"
        return True, "Native Windows.Graphics.Capture and D3D11 APIs are present."

    @classmethod
    def is_supported(cls) -> tuple[bool, str]:
        available, reason = cls.support_status()
        if not available:
            return available, reason

        combase = ctypes.WinDLL("combase")
        ro_initialized = False
        factory = ctypes.c_void_p()
        name = None
        try:
            combase.RoInitialize.argtypes = [wintypes.UINT]
            combase.RoInitialize.restype = ctypes.c_long
            result = combase.RoInitialize(1)
            code = ctypes.c_uint32(result).value
            if _hresult_failed(result) and code != _RPC_E_CHANGED_MODE:
                _check_hresult(result, "RoInitialize")
            ro_initialized = code != _RPC_E_CHANGED_MODE
            name = _HString(combase, "Windows.Graphics.Capture.GraphicsCaptureSession")
            factory = cls._activation_factory(combase, name, cls._IID_SESSION_STATICS)
            supported = wintypes.BOOL()
            result = _vtable_function(
                factory, 6, ctypes.c_long, ctypes.POINTER(wintypes.BOOL)
            )(factory, ctypes.byref(supported))
            _check_hresult(result, "GraphicsCaptureSession.IsSupported")
            if not supported.value:
                return False, "GraphicsCaptureSession.IsSupported returned false."
            return True, reason
        except Exception as error:
            return False, str(error)
        finally:
            _release(factory)
            if name is not None:
                name.close()
            if ro_initialized:
                combase.RoUninitialize()

    @staticmethod
    def _activation_factory(combase, class_name: _HString, iid: _GUID) -> ctypes.c_void_p:
        output = ctypes.c_void_p()
        combase.RoGetActivationFactory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        combase.RoGetActivationFactory.restype = ctypes.c_long
        _check_hresult(
            combase.RoGetActivationFactory(class_name.value, ctypes.byref(iid), ctypes.byref(output)),
            "RoGetActivationFactory",
        )
        return output

    def _initialize(self) -> None:
        supported, reason = self.support_status()
        if not supported:
            raise WGCUnavailableError(reason)
        if not win32gui.IsWindow(self.hwnd):
            raise WGCWindowClosedError(f"Invalid or closed window handle: {self.hwnd}")

        self._combase.RoInitialize.argtypes = [wintypes.UINT]
        self._combase.RoInitialize.restype = ctypes.c_long
        result = self._combase.RoInitialize(1)
        code = ctypes.c_uint32(result).value
        if _hresult_failed(result) and code != _RPC_E_CHANGED_MODE:
            _check_hresult(result, "RoInitialize")
        self._ro_initialized = code != _RPC_E_CHANGED_MODE

        self._create_d3d_device()
        item_class = _HString(self._combase, "Windows.Graphics.Capture.GraphicsCaptureItem")
        pool_class = _HString(self._combase, "Windows.Graphics.Capture.Direct3D11CaptureFramePool")
        item_factory = ctypes.c_void_p()
        pool_factory = ctypes.c_void_p()
        try:
            item_factory = self._activation_factory(
                self._combase, item_class, self._IID_CAPTURE_ITEM_INTEROP
            )
            result = _vtable_function(
                item_factory,
                3,
                ctypes.c_long,
                wintypes.HWND,
                ctypes.POINTER(_GUID),
                ctypes.POINTER(ctypes.c_void_p),
            )(
                item_factory,
                self.hwnd,
                ctypes.byref(self._IID_GRAPHICS_CAPTURE_ITEM),
                ctypes.byref(self._item),
            )
            _check_hresult(result, "IGraphicsCaptureItemInterop.CreateForWindow")

            size = _SizeInt32()
            result = _vtable_function(
                self._item, 7, ctypes.c_long, ctypes.POINTER(_SizeInt32)
            )(self._item, ctypes.byref(size))
            _check_hresult(result, "GraphicsCaptureItem.Size")
            if size.width <= 0 or size.height <= 0:
                raise WGCWindowClosedError("Selected window has zero-size WGC content.")
            self.size = (size.width, size.height)

            pool_factory = self._activation_factory(
                self._combase, pool_class, self._IID_FRAME_POOL_STATICS2
            )
            result = _vtable_function(
                pool_factory,
                6,
                ctypes.c_long,
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_int32,
                _SizeInt32,
                ctypes.POINTER(ctypes.c_void_p),
            )(
                pool_factory,
                self._winrt_device,
                _DXGI_FORMAT_B8G8R8A8_UNORM,
                2,
                size,
                ctypes.byref(self._frame_pool),
            )
            _check_hresult(result, "Direct3D11CaptureFramePool.CreateFreeThreaded")

            result = _vtable_function(
                self._frame_pool, 10, ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )(self._frame_pool, self._item, ctypes.byref(self._session))
            _check_hresult(result, "Direct3D11CaptureFramePool.CreateCaptureSession")
            _check_hresult(
                _vtable_function(self._session, 6, ctypes.c_long)(self._session),
                "GraphicsCaptureSession.StartCapture",
            )
        finally:
            _release(pool_factory)
            _release(item_factory)
            pool_class.close()
            item_class.close()

    def _create_d3d_device(self) -> None:
        create = self._d3d11.D3D11CreateDevice
        create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.UINT,
            ctypes.c_void_p,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.UINT),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create.restype = ctypes.c_long
        feature_level = wintypes.UINT()
        last_result = 0
        for driver_type in (_D3D_DRIVER_TYPE_HARDWARE, _D3D_DRIVER_TYPE_WARP):
            last_result = create(
                None,
                driver_type,
                None,
                _D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                0,
                7,
                ctypes.byref(self._device),
                ctypes.byref(feature_level),
                ctypes.byref(self._context),
            )
            if not _hresult_failed(last_result):
                break
        _check_hresult(last_result, "D3D11CreateDevice")

        dxgi_device = _query_interface(self._device, self._IID_IDXGI_DEVICE, "QueryInterface(IDXGIDevice)")
        try:
            create_winrt = self._d3d11.CreateDirect3D11DeviceFromDXGIDevice
            create_winrt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            create_winrt.restype = ctypes.c_long
            _check_hresult(
                create_winrt(dxgi_device, ctypes.byref(self._winrt_device)),
                "CreateDirect3D11DeviceFromDXGIDevice",
            )
        finally:
            _release(dxgi_device)

    def try_get_frame(self) -> _NativeFrame | None:
        pointer = ctypes.c_void_p()
        result = _vtable_function(
            self._frame_pool, 7, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)
        )(self._frame_pool, ctypes.byref(pointer))
        _check_hresult(result, "Direct3D11CaptureFramePool.TryGetNextFrame")
        if not pointer.value:
            return None
        size = _SizeInt32()
        try:
            result = _vtable_function(
                pointer, 8, ctypes.c_long, ctypes.POINTER(_SizeInt32)
            )(pointer, ctypes.byref(size))
            _check_hresult(result, "Direct3D11CaptureFrame.ContentSize")
            return _NativeFrame(pointer=pointer, width=size.width, height=size.height)
        except Exception:
            _release(pointer)
            raise

    def release_frame(self, frame: _NativeFrame) -> None:
        _release(frame.pointer)

    def recreate(self, width: int, height: int) -> None:
        size = _SizeInt32(width, height)
        result = _vtable_function(
            self._frame_pool,
            6,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            _SizeInt32,
        )(self._frame_pool, self._winrt_device, _DXGI_FORMAT_B8G8R8A8_UNORM, 2, size)
        _check_hresult(result, "Direct3D11CaptureFramePool.Recreate")
        self.size = (width, height)
        _release(self._staging)
        self._staging_desc = None

    def frame_to_bgr(self, frame: _NativeFrame) -> np.ndarray:
        surface = ctypes.c_void_p()
        access = ctypes.c_void_p()
        texture = ctypes.c_void_p()
        try:
            result = _vtable_function(
                frame.pointer, 6, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)
            )(frame.pointer, ctypes.byref(surface))
            _check_hresult(result, "Direct3D11CaptureFrame.Surface")
            access = _query_interface(
                surface, self._IID_DXGI_INTERFACE_ACCESS, "QueryInterface(IDirect3DDxgiInterfaceAccess)"
            )
            result = _vtable_function(
                access, 3, ctypes.c_long, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)
            )(access, ctypes.byref(self._IID_D3D11_TEXTURE2D), ctypes.byref(texture))
            _check_hresult(result, "IDirect3DDxgiInterfaceAccess.GetInterface")
            desc = _D3D11Texture2DDesc()
            _vtable_function(texture, 10, None, ctypes.POINTER(_D3D11Texture2DDesc))(
                texture, ctypes.byref(desc)
            )
            self._ensure_staging(desc)

            # ID3D11DeviceContext::CopyResource, Map, and Unmap are slots 47,
            # 14, and 15 in the documented ID3D11DeviceContext vtable.
            _vtable_function(self._context, 47, None, ctypes.c_void_p, ctypes.c_void_p)(
                self._context, self._staging, texture
            )
            mapped = _D3D11MappedSubresource()
            result = _vtable_function(
                self._context,
                14,
                ctypes.c_long,
                ctypes.c_void_p,
                wintypes.UINT,
                ctypes.c_int,
                wintypes.UINT,
                ctypes.POINTER(_D3D11MappedSubresource),
            )(
                self._context,
                self._staging,
                0,
                _D3D11_MAP_READ,
                0,
                ctypes.byref(mapped),
            )
            _check_hresult(result, "ID3D11DeviceContext.Map")
            try:
                height = min(frame.height, int(desc.Height))
                width = min(frame.width, int(desc.Width))
                if height <= 0 or width <= 0:
                    raise WGCWindowClosedError("WGC returned zero-size frame content.")
                byte_count = mapped.RowPitch * height
                raw_type = ctypes.c_ubyte * byte_count
                raw = np.ctypeslib.as_array(raw_type.from_address(mapped.pData))
                bgra = raw.reshape(height, mapped.RowPitch)[:, : width * 4].reshape(height, width, 4)
                return bgra[:, :, :3].copy()
            finally:
                _vtable_function(
                    self._context, 15, None, ctypes.c_void_p, wintypes.UINT
                )(self._context, self._staging, 0)
        finally:
            _release(texture)
            _release(access)
            _release(surface)

    def frame_to_gpu(
        self,
        frame: _NativeFrame,
        *,
        sequence: int,
        captured_at: float,
    ) -> D3D11Frame:
        """Snapshot the pool surface on the GPU before releasing the WGC frame.

        AddRef alone does not prevent the capture pool from recycling its pixels.
        Interpolation retains endpoints across acquisitions, so it needs a copy.
        """

        surface = ctypes.c_void_p()
        access = ctypes.c_void_p()
        texture = ctypes.c_void_p()
        output: D3D11Frame | None = None
        try:
            result = _vtable_function(
                frame.pointer, 6, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)
            )(frame.pointer, ctypes.byref(surface))
            _check_hresult(result, "Direct3D11CaptureFrame.Surface")
            access = _query_interface(
                surface,
                self._IID_DXGI_INTERFACE_ACCESS,
                "QueryInterface(IDirect3DDxgiInterfaceAccess)",
            )
            result = _vtable_function(
                access,
                3,
                ctypes.c_long,
                ctypes.POINTER(_GUID),
                ctypes.POINTER(ctypes.c_void_p),
            )(access, ctypes.byref(self._IID_D3D11_TEXTURE2D), ctypes.byref(texture))
            _check_hresult(result, "IDirect3DDxgiInterfaceAccess.GetInterface")

            desc = _D3D11Texture2DDesc()
            _vtable_function(texture, 10, None, ctypes.POINTER(_D3D11Texture2DDesc))(
                texture, ctypes.byref(desc)
            )
            if int(desc.Format) != _DXGI_FORMAT_B8G8R8A8_UNORM:
                raise WGCUnsupportedTextureError(
                    "D3D11 GPU capture requires DXGI_FORMAT_B8G8R8A8_UNORM (87); "
                    f"WGC supplied format {int(desc.Format)}."
                )
            if int(desc.SampleDesc.Count) != 1:
                raise WGCUnsupportedTextureError(
                    "D3D11 GPU capture requires a single-sampled WGC texture."
                )
            width = min(frame.width, int(desc.Width))
            height = min(frame.height, int(desc.Height))
            from window_capture import window_client_crop_box
            crop = window_client_crop_box(width, height, self.hwnd)
            x, y = 0, 0
            if crop is not None:
                x, y, width, height = crop
            owned = ctypes.c_void_p()
            desc.Width, desc.Height = width, height
            desc.Usage, desc.BindFlags, desc.CPUAccessFlags, desc.MiscFlags = 0, 8, 0, 0
            result = _vtable_function(
                self._device, 5, ctypes.c_long, ctypes.POINTER(_D3D11Texture2DDesc),
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            )(self._device, ctypes.byref(desc), None, ctypes.byref(owned))
            _check_hresult(result, "CreateTexture2D(owned WGC snapshot)")
            try:
                box = (wintypes.UINT * 6)(x, y, 0, x+width, y+height, 1)
                _vtable_function(
                    self._context, 46, None, ctypes.c_void_p, wintypes.UINT,
                    wintypes.UINT, wintypes.UINT, wintypes.UINT,
                    ctypes.c_void_p, wintypes.UINT, ctypes.c_void_p,
                )(self._context, owned, 0, 0, 0, 0, texture, 0, ctypes.byref(box))
            except Exception:
                _release(owned)
                raise
            _release(texture)
            texture = owned
            output = D3D11Frame(
                texture,
                width=width,
                height=height,
                dxgi_format=int(desc.Format),
                sequence=sequence,
                captured_at=captured_at,
            )
            # D3D11Frame now owns the independent GPU snapshot.
            texture = ctypes.c_void_p()
            return output
        finally:
            if output is None:
                _release(texture)
            _release(access)
            _release(surface)

    @property
    def device_pointer(self) -> ctypes.c_void_p:
        """Borrowed device pointer, valid until this runtime is closed."""

        return ctypes.c_void_p(self._device.value)

    @property
    def context_pointer(self) -> ctypes.c_void_p:
        """Borrowed immediate-context pointer, valid until runtime close."""

        return ctypes.c_void_p(self._context.value)

    def _ensure_staging(self, source: _D3D11Texture2DDesc) -> None:
        signature = (int(source.Width), int(source.Height), int(source.Format))
        if self._staging.value and signature == self._staging_desc:
            return
        _release(self._staging)
        staging_desc = _D3D11Texture2DDesc(
            Width=source.Width,
            Height=source.Height,
            MipLevels=1,
            ArraySize=1,
            Format=source.Format,
            SampleDesc=_DXGISampleDesc(1, 0),
            Usage=_D3D11_USAGE_STAGING,
            BindFlags=0,
            CPUAccessFlags=_D3D11_CPU_ACCESS_READ,
            MiscFlags=0,
        )
        result = _vtable_function(
            self._device,
            5,
            ctypes.c_long,
            ctypes.POINTER(_D3D11Texture2DDesc),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(self._device, ctypes.byref(staging_desc), None, ctypes.byref(self._staging))
        _check_hresult(result, "ID3D11Device.CreateTexture2D(staging)")
        self._staging_desc = signature

    @staticmethod
    def _close_winrt(pointer: ctypes.c_void_p) -> None:
        if not pointer.value:
            return
        closable = ctypes.c_void_p()
        try:
            closable = _query_interface(pointer, _NativeWGCRuntime._IID_CLOSABLE, "QueryInterface(IClosable)")
            _check_hresult(
                _vtable_function(closable, 6, ctypes.c_long)(closable),
                "IClosable.Close",
            )
        except WGCError:
            logger.debug("WGC resource did not close cleanly", exc_info=True)
        finally:
            _release(closable)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_winrt(self._session)
        self._close_winrt(self._frame_pool)
        self._close_winrt(self._winrt_device)
        _release(self._staging)
        _release(self._session)
        _release(self._frame_pool)
        _release(self._item)
        _release(self._winrt_device)
        _release(self._context)
        _release(self._device)
        if self._ro_initialized:
            self._combase.RoUninitialize()
            self._ro_initialized = False


class WGCCaptureBackend(CaptureBackend):
    """Capture a selected HWND through the real Windows.Graphics.Capture API."""

    backend_name = "wgc"
    display_name = "Windows Graphics Capture"

    def __init__(
        self,
        hwnd: int,
        *,
        first_frame_timeout: float = 3.0,
        frame_timeout: float = 1.0,
        poll_interval: float = 0.001,
        runtime: _Runtime | None = None,
    ) -> None:
        if hwnd is None:
            raise WGCUnavailableError("WGC requires a selected window HWND.")
        self.hwnd = int(hwnd)
        self.first_frame_timeout = float(first_frame_timeout)
        self.frame_timeout = float(frame_timeout)
        self.poll_interval = float(poll_interval)
        self._owner_thread = threading.get_ident()
        self._runtime = runtime or _NativeWGCRuntime(self.hwnd)
        self._first_frame_received = False
        self._gpu_sequence = 0
        self._gpu_replaced_frames = 0
        self._closed = False
        self.reuse_idle_frames = False
        self._idle_frame = None
        self._idle_logged = False
        width, height = self._runtime.size
        self._surface_size = (width, height)
        self.capture_region = CaptureRegion(left=0, top=0, width=width, height=height)
        logger.info("WGC initialized for HWND %s at %sx%s", self.hwnd, width, height)

    @staticmethod
    def availability() -> tuple[bool, str]:
        return _NativeWGCRuntime.is_supported()

    def _check_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise WGCError("WGC resources must be used and closed on their creating thread.")

    def set_capture_region(self, region: CaptureRegion) -> None:
        """Ignore screen-coordinate tracking; WGC follows its capture item."""

        del region

    def grab_frame(self) -> np.ndarray:
        self._check_owner()
        if self._closed:
            raise WGCError("WGC backend is closed.")
        if not win32gui.IsWindow(self.hwnd):
            raise WGCWindowClosedError(f"Selected WGC window closed: {self.hwnd}")
        if win32gui.IsIconic(self.hwnd):
            raise WGCWindowClosedError("Selected WGC window is minimized; capture is paused.")

        timeout = self.frame_timeout if self._first_frame_received else self.first_frame_timeout
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            frame = self._runtime.try_get_frame()
            if frame is None:
                time.sleep(self.poll_interval)
                continue
            try:
                if frame.width <= 0 or frame.height <= 0:
                    raise WGCWindowClosedError("WGC capture item returned zero-size content.")
                current_size = self._surface_size
                frame_size = (frame.width, frame.height)
                if frame_size != current_size:
                    self._idle_frame = None
                    logger.info(
                        "WGC content resized from %sx%s to %sx%s; recreating frame pool",
                        *current_size,
                        *frame_size,
                    )
                    self._runtime.recreate(*frame_size)
                    self._surface_size = frame_size
                    self.capture_region = CaptureRegion(
                        left=0, top=0, width=frame.width, height=frame.height
                    )
                    continue
                output = self._runtime.frame_to_bgr(frame)
            finally:
                self._runtime.release_frame(frame)

            if output.dtype != np.uint8 or output.ndim != 3 or output.shape[2] != 3:
                raise WGCError("WGC readback did not produce an HxWx3 uint8 BGR frame.")
            if not self._first_frame_received:
                logger.info("WGC acquired first frame at %sx%s", output.shape[1], output.shape[0])
                self._first_frame_received = True
            self.capture_region = CaptureRegion(
                left=0, top=0, width=output.shape[1], height=output.shape[0]
            )
            if self.reuse_idle_frames:
                self._idle_frame = output.copy()
                self._idle_logged = False
            return output

        if (self.reuse_idle_frames and self._idle_frame is not None
                and win32gui.IsWindow(self.hwnd) and not win32gui.IsIconic(self.hwnd)):
            if not self._idle_logged:
                logger.info("WGC idle: retaining last image, no new source frame; capture remains open.")
                self._idle_logged = True
            return self._idle_frame.copy()
        raise WGCFrameNotReadyError(
            f"WGC did not deliver {'the first ' if not self._first_frame_received else 'a '}frame "
            f"within {timeout:.1f} seconds. The window may be closed, minimized, or not updating."
        )

    def grab_gpu_frame(self) -> D3D11Frame:
        """Acquire the newest available WGC texture without staging readback."""

        self._check_owner()
        if self._closed:
            raise WGCError("WGC backend is closed.")
        if not win32gui.IsWindow(self.hwnd):
            raise WGCWindowClosedError(f"Selected WGC window closed: {self.hwnd}")
        if win32gui.IsIconic(self.hwnd):
            raise WGCWindowClosedError("Selected WGC window is minimized; capture is paused.")

        convert = getattr(self._runtime, "frame_to_gpu", None)
        if convert is None:
            raise WGCUnavailableError("The active WGC runtime does not expose D3D11 GPU frames.")

        timeout = self.frame_timeout if self._first_frame_received else self.first_frame_timeout
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            frame = self._runtime.try_get_frame()
            if frame is None:
                time.sleep(self.poll_interval)
                continue

            # Drain the two-frame WGC pool and keep only the newest available
            # surface.  Replaced frames are explicit and measured rather than
            # becoming hidden latency.
            while True:
                newer = self._runtime.try_get_frame()
                if newer is None:
                    break
                self._runtime.release_frame(frame)
                self._gpu_replaced_frames += 1
                frame = newer

            try:
                if frame.width <= 0 or frame.height <= 0:
                    raise WGCWindowClosedError("WGC capture item returned zero-size content.")
                current_size = self._surface_size
                frame_size = (frame.width, frame.height)
                if frame_size != current_size:
                    logger.info(
                        "WGC content resized from %sx%s to %sx%s; recreating frame pool",
                        *current_size,
                        *frame_size,
                    )
                    self._runtime.recreate(*frame_size)
                    self._surface_size = frame_size
                    self.capture_region = CaptureRegion(
                        left=0, top=0, width=frame.width, height=frame.height
                    )
                    continue

                next_sequence = self._gpu_sequence + 1
                output = convert(
                    frame,
                    sequence=next_sequence,
                    captured_at=time.perf_counter(),
                )
            finally:
                self._runtime.release_frame(frame)

            self._gpu_sequence = next_sequence
            self._first_frame_received = True
            self.capture_region = CaptureRegion(
                left=0,
                top=0,
                width=output.width,
                height=output.height,
            )
            return output

        raise WGCFrameNotReadyError(
            f"WGC did not deliver {'the first ' if not self._first_frame_received else 'a '}GPU "
            f"frame within {timeout:.1f} seconds. The window may be closed, minimized, or not updating."
        )

    @property
    def d3d11_device_pointer(self) -> ctypes.c_void_p:
        pointer = getattr(self._runtime, "device_pointer", None)
        if pointer is None or not pointer.value:
            raise WGCUnavailableError("The active WGC runtime does not expose its D3D11 device.")
        return ctypes.c_void_p(pointer.value)

    @property
    def d3d11_context_pointer(self) -> ctypes.c_void_p:
        pointer = getattr(self._runtime, "context_pointer", None)
        if pointer is None or not pointer.value:
            raise WGCUnavailableError("The active WGC runtime does not expose its D3D11 context.")
        return ctypes.c_void_p(pointer.value)

    @property
    def gpu_replaced_frames(self) -> int:
        return self._gpu_replaced_frames

    def close(self) -> None:
        if self._closed:
            return
        self._check_owner()
        self._closed = True
        self._runtime.close()
        self._idle_frame = None
        logger.info("WGC capture closed for HWND %s", self.hwnd)
