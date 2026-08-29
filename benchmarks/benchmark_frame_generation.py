"""Live-style asynchronous RIFE frame-generation benchmark."""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from async_pipeline import AsyncFramePipeline
from frame_interpolator import RIFEInterpolator


DEFAULT_MODEL = PROJECT_ROOT / "models" / "RIFE_v3.6.onnx"


def _percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = round((percentile_value / 100.0) * (len(ordered) - 1))
    return ordered[index]


def _distribution(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return (
        statistics.mean(values),
        statistics.median(values),
        _percentile(values, 95.0),
    )


class _IdentityProcessor:
    display_name = "Identity benchmark processor"

    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame


class _SyntheticCapture:
    """Rate-limited generated capture source that can be woken during shutdown."""

    def __init__(self, width: int, height: int, capture_fps: float) -> None:
        self._stop_event = threading.Event()
        self._interval = 1.0 / capture_fps
        self._next_capture_at = time.perf_counter()
        x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        base = np.empty((height, width, 3), dtype=np.uint8)
        base[..., 0] = x
        base[..., 1] = y
        base[..., 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(
            np.uint8
        )
        self._frames = tuple(
            np.ascontiguousarray(np.roll(base, index * 3, axis=1))
            for index in range(8)
        )
        self._index = 0

    def grab(self) -> np.ndarray:
        delay = self._next_capture_at - time.perf_counter()
        if delay > 0.0:
            self._stop_event.wait(delay)
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        self._next_capture_at = max(
            self._next_capture_at + self._interval,
            time.perf_counter(),
        )
        return frame.copy()

    def close(self) -> None:
        self._stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark live async capture, RIFE midpoint generation, and presentation.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--provider", choices=("cpu", "directml"), default="directml")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--capture-fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if args.device_id < 0:
        parser.error("--device-id must be zero or greater")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be greater than zero")
    if args.capture_fps <= 0.0 or args.duration <= 0.0:
        parser.error("--capture-fps and --duration must be greater than zero")
    if args.queue_depth <= 0 or args.warmup < 0:
        parser.error("--queue-depth must be positive and --warmup non-negative")

    capture = _SyntheticCapture(args.width, args.height, args.capture_fps)
    interpolator = RIFEInterpolator(
        args.model,
        provider=args.provider,
        device_id=args.device_id,
    )
    interpolator.initialize()
    active_providers = interpolator.active_providers
    warmup_a = capture.grab()
    warmup_b = capture.grab()
    for _ in range(args.warmup):
        interpolator.interpolate(warmup_a, warmup_b)

    pipeline = AsyncFramePipeline(
        _IdentityProcessor(),
        capture_source=capture.grab,
        capture_shutdown=capture.close,
        frame_interpolator=interpolator,
        queue_depth=args.queue_depth,
        collect_telemetry=True,
    )
    interpolation_times: list[float] = []
    total_latencies: list[float] = []
    presented_frames = 0
    benchmark_start = time.perf_counter()
    pipeline.start()
    try:
        deadline = benchmark_start + args.duration
        while time.perf_counter() < deadline:
            remaining = max(0.0, deadline - time.perf_counter())
            result = pipeline.take_latest(timeout=min(0.1, remaining))
            if result is None:
                continue
            presented_at = time.perf_counter()
            presented_frames += 1
            total_latencies.append((presented_at - result.captured_at) * 1000.0)
            if result.frame_kind == "generated" and result.interpolation_ms is not None:
                interpolation_times.append(result.interpolation_ms)
    finally:
        pipeline.stop(timeout=10.0)
        elapsed = time.perf_counter() - benchmark_start
        captured_frames = pipeline.submitted_frames
        interpolated_frames = pipeline.interpolated_frames
        dropped_generated_frames = pipeline.dropped_generated_frames
        interpolation_failures = pipeline.interpolation_failures
        dropped_input_frames = pipeline.input_replacements
        dropped_presentation_frames = pipeline.result_replacements
        interpolator.shutdown()

    interpolation_distribution = _distribution(interpolation_times)
    latency_distribution = _distribution(total_latencies)
    print("Live RIFE frame-generation benchmark")
    print(f"Requested provider: {args.provider.upper()}")
    print(f"Active providers: {', '.join(active_providers)}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Duration: {elapsed:.3f} seconds")
    print(f"Capture: {captured_frames} frames, {captured_frames / elapsed:.3f} FPS")
    print(
        f"Interpolation: {interpolated_frames} frames, "
        f"{interpolated_frames / elapsed:.3f} FPS"
    )
    print(
        f"Presentation: {presented_frames} frames, "
        f"{presented_frames / elapsed:.3f} FPS"
    )
    print(
        "Interpolation avg/median/p95: "
        f"{interpolation_distribution[0]:.3f}/"
        f"{interpolation_distribution[1]:.3f}/"
        f"{interpolation_distribution[2]:.3f} ms"
    )
    print(
        "Total latency avg/median/p95: "
        f"{latency_distribution[0]:.3f}/"
        f"{latency_distribution[1]:.3f}/"
        f"{latency_distribution[2]:.3f} ms"
    )
    print(f"Dropped input frames: {dropped_input_frames}")
    print(f"Dropped presentation frames: {dropped_presentation_frames}")
    print(f"Dropped generated frames: {dropped_generated_frames}")
    print(f"Interpolation failures: {interpolation_failures}")


if __name__ == "__main__":
    main()
