"""Process and runtime-resource snapshots used by validation tooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import subprocess
import threading
import time
from typing import Callable, Iterable


class ResourceRegistry:
    """Thread-safe counts for resources that the OS cannot expose portably."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def acquire(self, kind: str) -> None:
        with self._lock:
            self._counts[kind] = self._counts.get(kind, 0) + 1

    def release(self, kind: str) -> None:
        with self._lock:
            current = self._counts.get(kind, 0)
            if current <= 0:
                return
            self._counts[kind] = current - 1

    def count(self, kind: str) -> int:
        with self._lock:
            return self._counts.get(kind, 0)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


GLOBAL_RESOURCE_REGISTRY = ResourceRegistry()


class ResourceLease:
    """Idempotent validation counter lease for one owned resource."""

    def __init__(
        self,
        kind: str,
        registry: ResourceRegistry = GLOBAL_RESOURCE_REGISTRY,
    ) -> None:
        self.kind = kind
        self.registry = registry
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def acquire(self) -> None:
        if self._active:
            return
        self.registry.acquire(self.kind)
        self._active = True

    def release(self) -> None:
        if not self._active:
            return
        self._active = False
        self.registry.release(self.kind)

    def __enter__(self) -> "ResourceLease":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.release()

    def __del__(self) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class QueueSizes:
    input: int = 0
    output: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output


@dataclass(frozen=True, slots=True)
class ResourceSample:
    timestamp: float
    cpu_memory_bytes: int
    gpu_memory_bytes: int | None
    gpu_memory_source: str
    thread_count: int
    python_thread_count: int
    onnx_sessions: int
    d3d_resources: int
    queue_input: int
    queue_output: int

    @property
    def queue_total(self) -> int:
        return self.queue_input + self.queue_output

    def to_dict(self) -> dict[str, int | float | str | None]:
        result = asdict(self)
        result["queue_total"] = self.queue_total
        return result


@dataclass(frozen=True, slots=True)
class ResourceValidationSummary:
    passed: bool
    checks: dict[str, bool]
    deltas: dict[str, int | None]
    peak_cpu_memory_bytes: int
    peak_gpu_memory_bytes: int | None
    peak_thread_count: int
    peak_onnx_sessions: int
    peak_d3d_resources: int
    peak_queue_size: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _process_metrics() -> tuple[int, int]:
    try:
        import psutil

        process = psutil.Process()
        return int(process.memory_info().rss), int(process.num_threads())
    except (ImportError, OSError):
        # The validation framework remains usable without optional psutil.
        return 0, len(threading.enumerate())


def _nvidia_process_memory() -> tuple[int | None, str]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError):
        return None, "unavailable"
    if completed.returncode != 0:
        return None, "unavailable"
    total_mib = 0
    found = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid, memory_mib = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if pid == os.getpid():
            total_mib += memory_mib
            found = True
    return (total_mib * 1024 * 1024 if found else 0), "nvidia-smi"


class ResourceSampler:
    def __init__(
        self,
        *,
        registry: ResourceRegistry = GLOBAL_RESOURCE_REGISTRY,
        queue_probes: Iterable[Callable[[], QueueSizes]] = (),
        gpu_probe: Callable[[], tuple[int | None, str]] = _nvidia_process_memory,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.registry = registry
        self.queue_probes = tuple(queue_probes)
        self.gpu_probe = gpu_probe
        self.clock = clock

    def sample(self) -> ResourceSample:
        cpu_memory, thread_count = _process_metrics()
        gpu_memory, gpu_source = self.gpu_probe()
        queue_input = 0
        queue_output = 0
        for probe in self.queue_probes:
            sizes = probe()
            queue_input += sizes.input
            queue_output += sizes.output
        return ResourceSample(
            timestamp=self.clock(),
            cpu_memory_bytes=cpu_memory,
            gpu_memory_bytes=gpu_memory,
            gpu_memory_source=gpu_source,
            thread_count=thread_count,
            python_thread_count=len(threading.enumerate()),
            onnx_sessions=self.registry.count("onnx_sessions"),
            d3d_resources=self.registry.count("d3d_resources"),
            queue_input=queue_input,
            queue_output=queue_output,
        )

    def sample_after_cleanup(
        self,
        reference: ResourceSample,
        *,
        timeout_seconds: float = 0.5,
        interval_seconds: float = 0.02,
    ) -> ResourceSample:
        """Wait briefly for asynchronous native teardown, never indefinitely."""

        if timeout_seconds < 0.0 or interval_seconds <= 0.0:
            raise ValueError("Cleanup sampling intervals must be positive.")
        deadline = time.monotonic() + timeout_seconds
        sample = self.sample()
        while not self._released_to(reference, sample) and time.monotonic() < deadline:
            time.sleep(interval_seconds)
            sample = self.sample()
        return sample

    @staticmethod
    def _released_to(reference: ResourceSample, sample: ResourceSample) -> bool:
        return (
            sample.thread_count <= reference.thread_count
            and sample.python_thread_count <= reference.python_thread_count
            and sample.onnx_sessions <= reference.onnx_sessions
            and sample.d3d_resources <= reference.d3d_resources
            and sample.queue_total <= reference.queue_total
        )


def summarize_resources(
    before: ResourceSample,
    during: Iterable[ResourceSample],
    after: ResourceSample,
    *,
    cpu_tolerance_bytes: int = 16 * 1024 * 1024,
    gpu_tolerance_bytes: int = 16 * 1024 * 1024,
) -> ResourceValidationSummary:
    samples = [before, *during, after]
    available_gpu = [
        sample.gpu_memory_bytes
        for sample in samples
        if sample.gpu_memory_bytes is not None
    ]
    cpu_delta = after.cpu_memory_bytes - before.cpu_memory_bytes
    gpu_delta = (
        None
        if before.gpu_memory_bytes is None or after.gpu_memory_bytes is None
        else after.gpu_memory_bytes - before.gpu_memory_bytes
    )
    checks = {
        "cpu_memory_released": cpu_delta <= cpu_tolerance_bytes,
        "gpu_memory_released": (
            True if gpu_delta is None else gpu_delta <= gpu_tolerance_bytes
        ),
        "threads_released": after.thread_count <= before.thread_count,
        "python_threads_released": (
            after.python_thread_count <= before.python_thread_count
        ),
        "onnx_sessions_released": after.onnx_sessions <= before.onnx_sessions,
        "d3d_resources_released": after.d3d_resources <= before.d3d_resources,
        "queues_cleared": after.queue_total <= before.queue_total,
    }
    return ResourceValidationSummary(
        passed=all(checks.values()),
        checks=checks,
        deltas={
            "cpu_memory_bytes": cpu_delta,
            "gpu_memory_bytes": gpu_delta,
            "thread_count": after.thread_count - before.thread_count,
            "python_thread_count": (
                after.python_thread_count - before.python_thread_count
            ),
            "onnx_sessions": after.onnx_sessions - before.onnx_sessions,
            "d3d_resources": after.d3d_resources - before.d3d_resources,
            "queue_total": after.queue_total - before.queue_total,
        },
        peak_cpu_memory_bytes=max(sample.cpu_memory_bytes for sample in samples),
        peak_gpu_memory_bytes=max(available_gpu) if available_gpu else None,
        peak_thread_count=max(sample.thread_count for sample in samples),
        peak_onnx_sessions=max(sample.onnx_sessions for sample in samples),
        peak_d3d_resources=max(sample.d3d_resources for sample in samples),
        peak_queue_size=max(sample.queue_total for sample in samples),
    )
