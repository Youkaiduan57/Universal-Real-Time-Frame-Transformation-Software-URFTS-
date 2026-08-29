"""Headless synchronous-versus-asynchronous producer/consumer benchmark."""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from async_pipeline import AsyncFramePipeline


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    mode: str
    capture_fps: float
    output_fps: float
    median_latency_ms: float
    p95_latency_ms: float
    captured_frames: int
    processed_frames: int
    dropped_frames: int


def _percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = round((percentile_value / 100.0) * (len(ordered) - 1))
    return ordered[index]


class SyntheticCapture:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.frames = 0
        self._closed = threading.Event()

    def grab_frame(self) -> int:
        if self._closed.wait(timeout=self.delay_seconds):
            raise RuntimeError("capture closed")
        frame = self.frames
        self.frames += 1
        return frame

    def close(self) -> None:
        self._closed.set()


class SyntheticProcessor:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds

    def process(self, frame: int) -> int:
        time.sleep(self.delay_seconds)
        return frame


def _run_synchronous(
    duration: float,
    capture_delay: float,
    processing_delay: float,
) -> BenchmarkResult:
    capture = SyntheticCapture(capture_delay)
    processor = SyntheticProcessor(processing_delay)
    latencies = []
    frames = 0
    started = time.perf_counter()
    deadline = started + duration
    try:
        while time.perf_counter() < deadline:
            captured_at = time.perf_counter()
            frame = capture.grab_frame()
            processor.process(frame)
            latencies.append((time.perf_counter() - captured_at) * 1000.0)
            frames += 1
    finally:
        capture.close()
    elapsed = time.perf_counter() - started
    return BenchmarkResult(
        mode="synchronous",
        capture_fps=frames / elapsed,
        output_fps=frames / elapsed,
        median_latency_ms=statistics.median(latencies),
        p95_latency_ms=_percentile(latencies, 95.0),
        captured_frames=frames,
        processed_frames=frames,
        dropped_frames=0,
    )


def _run_asynchronous(
    duration: float,
    capture_delay: float,
    processing_delay: float,
    queue_depth: int,
) -> BenchmarkResult:
    capture = SyntheticCapture(capture_delay)
    processor = SyntheticProcessor(processing_delay)
    pipeline = AsyncFramePipeline(
        processor,
        capture_source=capture.grab_frame,
        capture_shutdown=capture.close,
        queue_depth=queue_depth,
    )
    latencies = []
    started = time.perf_counter()
    deadline = started + duration
    pipeline.start()
    try:
        while time.perf_counter() < deadline:
            remaining = max(0.0, deadline - time.perf_counter())
            result = pipeline.take_latest(timeout=min(0.05, remaining))
            if result is not None:
                latencies.append((time.perf_counter() - result.captured_at) * 1000.0)
    finally:
        pipeline.stop(timeout=2.0)
    elapsed = time.perf_counter() - started
    processed = pipeline.processed_frames
    return BenchmarkResult(
        mode="asynchronous",
        capture_fps=pipeline.submitted_frames / elapsed,
        output_fps=processed / elapsed,
        median_latency_ms=statistics.median(latencies) if latencies else 0.0,
        p95_latency_ms=_percentile(latencies, 95.0) if latencies else 0.0,
        captured_frames=pipeline.submitted_frames,
        processed_frames=processed,
        dropped_frames=pipeline.dropped_frames,
    )


def _print_results(results: list[BenchmarkResult], queue_depth: int) -> None:
    print("Synthetic capture/processing benchmark")
    print(f"Asynchronous queue depth: {queue_depth}")
    print(
        "mode         | capture FPS | output FPS | median latency ms | "
        "p95 latency ms | captured | processed | dropped"
    )
    print("-" * 118)
    for result in results:
        print(
            f"{result.mode:<12} | {result.capture_fps:>11.2f} | "
            f"{result.output_fps:>10.2f} | {result.median_latency_ms:>17.3f} | "
            f"{result.p95_latency_ms:>14.3f} | {result.captured_frames:>8} | "
            f"{result.processed_frames:>9} | {result.dropped_frames:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--capture-ms", type=float, default=4.0)
    parser.add_argument("--processing-ms", type=float, default=12.0)
    parser.add_argument("--queue-depth", type=int, default=2)
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be greater than zero")
    if args.capture_ms < 0.0 or args.processing_ms < 0.0:
        parser.error("synthetic delays must be zero or greater")
    if args.queue_depth <= 0:
        parser.error("--queue-depth must be greater than zero")

    results = [
        _run_synchronous(
            args.duration,
            args.capture_ms / 1000.0,
            args.processing_ms / 1000.0,
        ),
        _run_asynchronous(
            args.duration,
            args.capture_ms / 1000.0,
            args.processing_ms / 1000.0,
            args.queue_depth,
        ),
    ]
    _print_results(results, args.queue_depth)


if __name__ == "__main__":
    main()
