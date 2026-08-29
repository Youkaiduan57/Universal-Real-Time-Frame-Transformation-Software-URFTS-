"""Compare sequential and latest-frame processing on one real WGC window."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

import cv2

from async_pipeline import AsyncFramePipeline, ProcessedFrame
from capture_manager import CaptureManager
from config import ApplicationConfig, runtime_profile_path
from frame_processor import FrameProcessor
from processing_backend import OpenCVProcessingBackend
from runtime_profile import RuntimeProfile
from window_capture import select_window


@dataclass(frozen=True, slots=True)
class Distribution:
    average: float
    median: float
    p95: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    mode: str
    capture_fps: float
    processed_fps: float
    capture_latency_ms: Distribution
    processing_latency_ms: Distribution
    capture_to_processing_start_age_ms: Distribution
    capture_to_processing_end_latency_ms: Distribution
    total_captured_frames: int
    total_processed_frames: int
    dropped_replaced_frames: int
    elapsed_seconds: float


def _percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = round((percentile_value / 100.0) * (len(ordered) - 1))
    return ordered[index]


def _distribution(values: list[float]) -> Distribution:
    if not values:
        return Distribution(0.0, 0.0, 0.0)
    return Distribution(
        average=statistics.mean(values),
        median=statistics.median(values),
        p95=_percentile(values, 95.0),
    )


def _create_components(hwnd: int, config: ApplicationConfig):
    capture = CaptureManager(
        backend="wgc",
        window_hwnd=hwnd,
        fallback_on_explicit_failure=False,
    )
    backend = OpenCVProcessingBackend(
        output_width=config.output_width,
        output_height=config.output_height,
        upscaling_method="bicubic",
    )
    return capture, backend, FrameProcessor(processing_backend=backend)


def _warm_up(capture, processor, frames: int) -> None:
    for _ in range(frames):
        processor.process(capture.grab_frame())


def _run_sequential(hwnd: int, duration: float, warmup: int) -> BenchmarkResult:
    config = ApplicationConfig()
    capture, backend, processor = _create_components(hwnd, config)
    capture_times: list[float] = []
    processing_times: list[float] = []
    frame_ages: list[float] = []
    end_to_end_times: list[float] = []
    captured = 0
    processed = 0

    try:
        _warm_up(capture, processor, warmup)
        started_at = time.perf_counter()
        deadline = started_at + duration
        while time.perf_counter() < deadline:
            capture_started_at = time.perf_counter()
            frame = capture.grab_frame()
            capture_ended_at = time.perf_counter()
            processing_started_at = time.perf_counter()
            processor.process(frame)
            processing_ended_at = time.perf_counter()

            captured += 1
            processed += 1
            capture_times.append((capture_ended_at - capture_started_at) * 1000.0)
            processing_times.append(
                (processing_ended_at - processing_started_at) * 1000.0
            )
            frame_ages.append(
                (processing_started_at - capture_started_at) * 1000.0
            )
            end_to_end_times.append(
                (processing_ended_at - capture_started_at) * 1000.0
            )
        ended_at = time.perf_counter()
    finally:
        capture.close()
        backend.close()

    elapsed = ended_at - started_at
    return BenchmarkResult(
        mode="sequential",
        capture_fps=captured / elapsed,
        processed_fps=processed / elapsed,
        capture_latency_ms=_distribution(capture_times),
        processing_latency_ms=_distribution(processing_times),
        capture_to_processing_start_age_ms=_distribution(frame_ages),
        capture_to_processing_end_latency_ms=_distribution(end_to_end_times),
        total_captured_frames=captured,
        total_processed_frames=processed,
        dropped_replaced_frames=0,
        elapsed_seconds=elapsed,
    )


def _run_latest_frame(hwnd: int, duration: float, warmup: int) -> BenchmarkResult:
    config = ApplicationConfig()
    capture, backend, processor = _create_components(hwnd, config)
    pipeline = AsyncFramePipeline(processor)
    capture_times: list[float] = []
    processing_times: list[float] = []
    frame_ages: list[float] = []
    end_to_end_times: list[float] = []
    captured = 0
    processed_sequences: set[int] = set()

    def record(result: ProcessedFrame | None) -> None:
        if result is None or result.sequence_id in processed_sequences:
            return
        processed_sequences.add(result.sequence_id)
        processing_times.append(result.processing_ms)
        frame_ages.append(result.frame_age_ms)
        end_to_end_times.append(result.end_to_end_ms)

    try:
        _warm_up(capture, processor, warmup)
        pipeline.start()
        started_at = time.perf_counter()
        deadline = started_at + duration
        last_sequence = -1
        while time.perf_counter() < deadline:
            capture_started_at = time.perf_counter()
            frame = capture.grab_frame()
            capture_ended_at = time.perf_counter()
            capture_ms = (capture_ended_at - capture_started_at) * 1000.0
            capture_times.append(capture_ms)
            last_sequence = pipeline.submit(
                frame,
                captured_at=capture_started_at,
                capture_ms=capture_ms,
            )
            captured += 1
            record(pipeline.take_latest())
        capture_loop_ended_at = time.perf_counter()

        drain_deadline = time.perf_counter() + 5.0
        while time.perf_counter() < drain_deadline:
            result = pipeline.take_latest(timeout=0.1)
            record(result)
            if result is not None and result.sequence_id == last_sequence:
                break
        ended_at = time.perf_counter()
    finally:
        try:
            pipeline.stop()
        finally:
            capture.close()
            backend.close()

    capture_elapsed = capture_loop_ended_at - started_at
    total_elapsed = ended_at - started_at
    processed = len(processed_sequences)
    return BenchmarkResult(
        mode="latest_frame",
        capture_fps=captured / capture_elapsed,
        processed_fps=processed / total_elapsed,
        capture_latency_ms=_distribution(capture_times),
        processing_latency_ms=_distribution(processing_times),
        capture_to_processing_start_age_ms=_distribution(frame_ages),
        capture_to_processing_end_latency_ms=_distribution(end_to_end_times),
        total_captured_frames=captured,
        total_processed_frames=processed,
        dropped_replaced_frames=pipeline.dropped_frames,
        elapsed_seconds=total_elapsed,
    )


def _format_distribution(value: Distribution) -> str:
    return f"{value.average:.3f}/{value.median:.3f}/{value.p95:.3f}"


def _print_results(window_title: str, hwnd: int, results: list[BenchmarkResult]) -> None:
    print(f"Target: {window_title} (HWND {hwnd})")
    print("Backend: WGC | Processing: OpenCV CPU bicubic")
    print("Latency values are average/median/p95 in milliseconds.")
    headers = (
        "mode",
        "capture FPS",
        "processed FPS",
        "capture ms",
        "processing ms",
        "start age ms",
        "end latency ms",
        "captured",
        "processed",
        "dropped/replaced",
    )
    rows = [
        (
            result.mode,
            f"{result.capture_fps:.2f}",
            f"{result.processed_fps:.2f}",
            _format_distribution(result.capture_latency_ms),
            _format_distribution(result.processing_latency_ms),
            _format_distribution(result.capture_to_processing_start_age_ms),
            _format_distribution(result.capture_to_processing_end_latency_ms),
            str(result.total_captured_frames),
            str(result.total_processed_frames),
            str(result.dropped_replaced_frames),
        )
        for result in results
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--window-hwnd", type=int)
    target.add_argument("--window-title")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.duration <= 0.0:
        parser.error("--duration must be greater than zero")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be zero or greater")

    selected = select_window(title=args.window_title, hwnd=args.window_hwnd)
    profile = RuntimeProfile.load(runtime_profile_path())
    cv2.setNumThreads(profile.opencv_threads)

    results = [
        _run_sequential(selected.hwnd, args.duration, args.warmup_frames),
        _run_latest_frame(selected.hwnd, args.duration, args.warmup_frames),
    ]
    _print_results(selected.title, selected.hwnd, results)

    if args.output_json is not None:
        payload = {
            "target": {"title": selected.title, "hwnd": selected.hwnd},
            "backend": "wgc",
            "processing_backend": "opencv_cpu",
            "upscaling_method": "bicubic",
            "duration_seconds_per_mode": args.duration,
            "warmup_frames_per_mode": args.warmup_frames,
            "results": [asdict(result) for result in results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
