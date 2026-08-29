from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

import ai_processor
import frame_interpolator
from ai_processor import AIProcessor
from async_pipeline import AsyncFramePipeline
from frame_interpolator import RIFEInterpolator
from resource_validation import (
    GLOBAL_RESOURCE_REGISTRY,
    QueueSizes,
    ResourceLease,
    ResourceRegistry,
    ResourceSample,
    ResourceSampler,
    summarize_resources,
)


def sample(**updates) -> ResourceSample:
    values = {
        "timestamp": 1.0,
        "cpu_memory_bytes": 100,
        "gpu_memory_bytes": 200,
        "gpu_memory_source": "fake",
        "thread_count": 2,
        "python_thread_count": 1,
        "onnx_sessions": 0,
        "d3d_resources": 0,
        "queue_input": 0,
        "queue_output": 0,
    }
    values.update(updates)
    return ResourceSample(**values)


def test_resource_lease_is_counted_and_released_idempotently() -> None:
    registry = ResourceRegistry()
    lease = ResourceLease("onnx_sessions", registry)
    lease.acquire()
    lease.acquire()
    assert registry.count("onnx_sessions") == 1
    lease.release()
    lease.release()
    assert registry.count("onnx_sessions") == 0


def test_resource_sampler_collects_registry_gpu_threads_and_queues(monkeypatch) -> None:
    registry = ResourceRegistry()
    registry.acquire("onnx_sessions")
    registry.acquire("d3d_resources")
    monkeypatch.setattr("resource_validation._process_metrics", lambda: (1234, 7))
    sampler = ResourceSampler(
        registry=registry,
        queue_probes=(lambda: QueueSizes(2, 3),),
        gpu_probe=lambda: (4096, "fake-gpu"),
        clock=lambda: 9.0,
    )
    result = sampler.sample()
    assert result.cpu_memory_bytes == 1234
    assert result.gpu_memory_bytes == 4096
    assert result.thread_count == 7
    assert result.onnx_sessions == 1
    assert result.d3d_resources == 1
    assert result.queue_total == 5


def test_cleanup_sampler_waits_for_native_thread_teardown(monkeypatch) -> None:
    samples = iter([(100, 3), (100, 3), (100, 2)])
    monkeypatch.setattr("resource_validation._process_metrics", lambda: next(samples))
    sampler = ResourceSampler(
        gpu_probe=lambda: (None, "unavailable"),
    )
    reference = sample(thread_count=2, python_thread_count=len(threading.enumerate()))
    settled = sampler.sample_after_cleanup(
        reference,
        timeout_seconds=0.2,
        interval_seconds=0.001,
    )
    assert settled.thread_count == 2


def test_resource_summary_accepts_cleanup_and_detects_leaks() -> None:
    before = sample()
    during = [
        sample(
            cpu_memory_bytes=1000,
            onnx_sessions=2,
            d3d_resources=1,
            queue_input=2,
        )
    ]
    clean = summarize_resources(before, during, sample(cpu_memory_bytes=110))
    assert clean.passed
    assert clean.peak_onnx_sessions == 2
    assert clean.peak_d3d_resources == 1
    assert clean.peak_queue_size == 2

    leaked = summarize_resources(
        before,
        during,
        sample(onnx_sessions=1, thread_count=3),
        cpu_tolerance_bytes=0,
    )
    assert leaked.passed is False
    assert leaked.checks["onnx_sessions_released"] is False
    assert leaked.checks["threads_released"] is False


class _Session:
    def __init__(self, inputs: int) -> None:
        self.inputs = inputs

    def get_inputs(self):
        return [
            SimpleNamespace(
                name=f"input_{index}",
                shape=[1, 3, "height", "width"],
                type="tensor(float)",
            )
            for index in range(self.inputs)
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(
                name="output",
                shape=[1, 3, "height", "width"],
                type="tensor(float)",
            )
        ]

    def get_providers(self):
        return ["CPUExecutionProvider"]


def test_repeated_ai_and_interpolator_recreation_releases_onnx_sessions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    model.touch()
    monkeypatch.setattr(
        ai_processor.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    baseline = GLOBAL_RESOURCE_REGISTRY.count("onnx_sessions")
    for _ in range(4):
        processor = AIProcessor(
            model,
            session_factory=lambda *args, **kwargs: _Session(1),
        )
        processor.initialize()
        assert GLOBAL_RESOURCE_REGISTRY.count("onnx_sessions") == baseline + 1
        processor.shutdown()

        interpolator = RIFEInterpolator(
            model,
            session_factory=lambda *args, **kwargs: _Session(2),
        )
        interpolator.initialize()
        assert GLOBAL_RESOURCE_REGISTRY.count("onnx_sessions") == baseline + 1
        interpolator.shutdown()
    assert GLOBAL_RESOURCE_REGISTRY.count("onnx_sessions") == baseline


def test_async_pipeline_exposes_and_clears_queue_sizes() -> None:
    class Identity:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(Identity())
    pipeline.start()
    try:
        pipeline.submit("frame")
        deadline = time.perf_counter() + 1.0
        while pipeline.queue_sizes().output == 0 and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert pipeline.queue_sizes().output == 1
        assert pipeline.take_latest(timeout=1.0) is not None
        assert pipeline.queue_sizes() == QueueSizes()
    finally:
        pipeline.stop()


def test_repeated_pipeline_startup_shutdown_has_no_thread_leak() -> None:
    class Identity:
        def process(self, frame):
            return frame

    baseline = {thread.ident for thread in threading.enumerate()}
    for cycle in range(5):
        pipeline = AsyncFramePipeline(Identity())
        pipeline.start()
        pipeline.submit(cycle)
        assert pipeline.take_latest(timeout=1.0) is not None
        pipeline.stop(timeout=1.0)
        pipeline.stop(timeout=1.0)
        assert pipeline.worker_threads_alive is False

    leaked = [
        thread.name
        for thread in threading.enumerate()
        if thread.ident not in baseline and thread.is_alive()
    ]
    assert leaked == []
