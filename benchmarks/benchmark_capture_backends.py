"""Compare real selected-window capture backends through CaptureManager."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from capture_manager import CaptureManager
from window_capture import get_client_capture_region, select_window


@dataclass(frozen=True, slots=True)
class Result:
    backend: str
    average_ms: float
    median_ms: float
    p95_ms: float
    throughput_fps: float
    dimensions: tuple[int, int]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--window-title")
    target.add_argument("--window-hwnd", type=int)
    parser.add_argument("--warmup-frames", type=_positive_int, default=20)
    parser.add_argument("--frames", type=_positive_int, default=120)
    parser.add_argument(
        "--backends", nargs="+", choices=("wgc", "mss", "dxcam"), default=("wgc", "mss", "dxcam")
    )
    return parser.parse_args()


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    return ordered[round((percentile_value / 100.0) * (len(ordered) - 1))]


def benchmark_backend(backend_name, window, region, warmup_frames, measured_frames):
    capture = CaptureManager(
        backend=backend_name,
        capture_region=region,
        window_hwnd=window.hwnd,
    )
    try:
        for _ in range(warmup_frames):
            capture.grab_frame()
        timings = []
        frame = None
        start_all = time.perf_counter()
        for _ in range(measured_frames):
            start = time.perf_counter()
            frame = capture.grab_frame()
            timings.append((time.perf_counter() - start) * 1000.0)
        elapsed = time.perf_counter() - start_all
    finally:
        capture.close()
    return Result(
        backend=backend_name,
        average_ms=statistics.mean(timings),
        median_ms=statistics.median(timings),
        p95_ms=percentile(timings, 95),
        throughput_fps=measured_frames / elapsed,
        dimensions=(frame.shape[1], frame.shape[0]),
    )


def main() -> None:
    args = parse_args()
    window = select_window(title=args.window_title, hwnd=args.window_hwnd)
    region = get_client_capture_region(window.hwnd)
    print("=" * 88)
    print("UniversalUpscaler selected-window capture benchmark")
    print(f"Window: {window.title} (HWND {window.hwnd})")
    print(f"MSS/DXcam client rectangle: {region.width}x{region.height} at {region.left},{region.top}")
    print(
        "Semantic note: WGC captures the window through Windows.Graphics.Capture; "
        "MSS/DXcam capture screen pixels from the current client-area rectangle."
    )
    print(f"Warm-up: {args.warmup_frames} frames; measured: {args.frames} frames")
    print("=" * 88)
    for backend_name in args.backends:
        try:
            result = benchmark_backend(
                backend_name, window, region, args.warmup_frames, args.frames
            )
        except Exception as error:
            print(f"{backend_name:<8} FAILED: {type(error).__name__}: {error}")
            continue
        print(
            f"{result.backend:<8} avg {result.average_ms:8.3f} ms | "
            f"median {result.median_ms:8.3f} ms | p95 {result.p95_ms:8.3f} ms | "
            f"throughput {result.throughput_fps:7.2f} FPS | "
            f"{result.dimensions[0]}x{result.dimensions[1]}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
