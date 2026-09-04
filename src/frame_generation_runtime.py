"""Frame-generation device selection and optional native GPU-I/O capability hooks."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True, slots=True)
class DeviceBenchmarkResult:
    device_id: int
    total_ms: float
    calls: int

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.calls


def benchmark_directml_devices(
    factory: Callable[[int], Any],
    *,
    candidates: Iterable[int] = (0, 1, 2, 3),
    iterations: int = 3,
) -> tuple[int, tuple[DeviceBenchmarkResult, ...]]:
    """Probe DirectML adapters and return the fastest successful model warmup."""

    if iterations <= 0:
        raise ValueError("Device benchmark iterations must be positive.")
    results: list[DeviceBenchmarkResult] = []
    for device_id in candidates:
        interpolator = None
        try:
            interpolator = factory(int(device_id))
            elapsed_ms = float(interpolator.warmup(iterations=iterations))
            if math.isfinite(elapsed_ms) and elapsed_ms > 0.0:
                results.append(DeviceBenchmarkResult(int(device_id), elapsed_ms, iterations))
        except Exception:
            continue
        finally:
            if interpolator is not None:
                shutdown = getattr(interpolator, "shutdown", None)
                if callable(shutdown):
                    shutdown()
    if not results:
        raise RuntimeError("No DirectML adapter completed the frame-generation benchmark.")
    ordered = tuple(sorted(results, key=lambda result: result.average_ms))
    return ordered[0].device_id, ordered


@dataclass(frozen=True, slots=True)
class GpuResidentCapability:
    available: bool
    backend: str
    reason: str
    abi_version: int = 1


def probe_gpu_resident_backend(
    loader: Callable[[str], Any] = importlib.import_module,
    *, allow_experimental: bool = False,
) -> GpuResidentCapability:
    """Probe the optional native D3D11/D3D12 DirectML texture bridge."""

    try:
        # Load the Python package's matching ORT DLL before Windows resolves the
        # extension's onnxruntime.dll dependency from PATH.
        if loader is importlib.import_module:
            loader("onnxruntime")
        module = loader("_urfts_directml")
    except (ImportError, OSError) as error:
        return GpuResidentCapability(
            False,
            "unavailable",
            f"Native DirectML texture bridge is not installed: {error}",
        )
    abi_version = int(getattr(module, "ABI_VERSION", 0))
    required = ("create_frame_generator", "interpolate_d3d11", "close_frame_generator")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    gpu_io = bool(getattr(module, "GPU_RESIDENT_IO", False))
    if abi_version != 1 or missing or not gpu_io:
        detail = f"ABI {abi_version}" if abi_version != 1 else f"missing {', '.join(missing)}"
        if not missing and abi_version == 1 and not gpu_io:
            detail = "GPU-resident I/O capability is disabled"
        return GpuResidentCapability(False, "unavailable", f"Incompatible native bridge: {detail}")
    if not bool(getattr(module, "RUNTIME_VALIDATED", False)):
        if (allow_experimental and bool(getattr(module, "STABILIZATION_GPU", False))
                and bool(getattr(module, "PADDED_INPUT", False))):
            return GpuResidentCapability(True, "DirectML native texture bridge (experimental)",
                                         "Synthetic tests only; gameplay quality and pacing are unvalidated.")
        return GpuResidentCapability(
            False, "unavailable",
            "Native bridge is built but not validated for live presentation; use the CPU-frame pipeline.",
        )
    return GpuResidentCapability(True, "DirectML native texture bridge", "available")


class NativeDirectMLTextureInterpolator:
    """Thin owner for the optional native D3D11 texture interpolation ABI."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        device_id: int,
        d3d11_device_pointer: Any,
        inference_width: int,
        inference_height: int,
        loader: Callable[[str], Any] = importlib.import_module,
        allow_experimental: bool = False,
    ) -> None:
        capability = probe_gpu_resident_backend(loader, allow_experimental=allow_experimental)
        if not capability.available:
            raise RuntimeError(capability.reason)
        self._module = loader("_urfts_directml")
        pointer_value = int(getattr(d3d11_device_pointer, "value", d3d11_device_pointer))
        self._handle = self._module.create_frame_generator(
            str(Path(model_path)),
            int(device_id),
            pointer_value,
            int(inference_width),
            int(inference_height),
        )
        if not self._handle:
            raise RuntimeError("Native DirectML texture bridge returned a null generator handle.")

    def interpolate(self, previous: Any, current: Any):
        from wgc_capture import D3D11Frame

        pointer = self._module.interpolate_d3d11(
            self._handle,
            int(previous.texture_pointer.value),
            int(current.texture_pointer.value),
            int(current.width),
            int(current.height),
        )
        return D3D11Frame(
            pointer,
            width=current.width,
            height=current.height,
            dxgi_format=int(
                getattr(self._module, "OUTPUT_DXGI_FORMAT", current.dxgi_format)
            ),
            sequence=current.sequence,
            captured_at=current.captured_at,
        )

    def close(self) -> None:
        if self._handle:
            self._module.close_frame_generator(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()
