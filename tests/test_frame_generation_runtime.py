from __future__ import annotations

from types import SimpleNamespace

import pytest

from frame_generation_runtime import (
    benchmark_directml_devices,
    probe_gpu_resident_backend,
)


class FakeInterpolator:
    def __init__(self, device_id: int, elapsed_ms: float) -> None:
        self.device_id = device_id
        self.elapsed_ms = elapsed_ms
        self.closed = False

    def warmup(self, *, iterations: int) -> float:
        return self.elapsed_ms * iterations

    def shutdown(self) -> None:
        self.closed = True


def test_adapter_benchmark_selects_fastest_successful_device_and_closes_all() -> None:
    created = []

    def factory(device_id: int):
        if device_id == 2:
            raise RuntimeError("unavailable")
        value = FakeInterpolator(device_id, {0: 7.0, 1: 3.0, 3: 8.0}[device_id])
        created.append(value)
        return value

    selected, results = benchmark_directml_devices(factory)

    assert selected == 1
    assert [result.device_id for result in results] == [1, 0, 3]
    assert all(value.closed for value in created)


def test_adapter_benchmark_rejects_no_working_device() -> None:
    with pytest.raises(RuntimeError, match="No DirectML adapter"):
        benchmark_directml_devices(lambda _device_id: (_ for _ in ()).throw(RuntimeError()))


def test_gpu_resident_probe_reports_missing_native_bridge() -> None:
    capability = probe_gpu_resident_backend(
        lambda _name: (_ for _ in ()).throw(ImportError("not built"))
    )
    assert capability.available is False
    assert "not installed" in capability.reason


def test_gpu_resident_probe_accepts_matching_native_abi() -> None:
    module = SimpleNamespace(
        ABI_VERSION=1,
        GPU_RESIDENT_IO=True,
        RUNTIME_VALIDATED=True,
        create_frame_generator=lambda: None,
        interpolate_d3d11=lambda: None,
        close_frame_generator=lambda: None,
    )
    capability = probe_gpu_resident_backend(lambda _name: module)
    assert capability.available is True
    assert capability.backend == "DirectML native texture bridge"


def test_gpu_resident_probe_rejects_unvalidated_native_build():
    module = SimpleNamespace(
        ABI_VERSION=1, GPU_RESIDENT_IO=True,
        create_frame_generator=lambda: None,
        interpolate_d3d11=lambda: None,
        close_frame_generator=lambda: None,
    )
    capability = probe_gpu_resident_backend(lambda _name: module)
    assert not capability.available
    assert "not validated" in capability.reason
