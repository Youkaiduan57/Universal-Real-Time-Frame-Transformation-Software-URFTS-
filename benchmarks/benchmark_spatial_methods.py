"""Benchmark OpenCV spatial upscalers on one real captured frame."""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from capture_manager import CaptureManager
from config import OUTPUT_HEIGHT, OUTPUT_WIDTH, runtime_profile_path
from processing_backend import OpenCVProcessingBackend
from runtime_profile import DEFAULT_RUNTIME_PROFILE, RuntimeProfile

DEFAULT_WARMUP_RUNS = 20
DEFAULT_BENCHMARK_RUNS = 100
METHODS = ("bicubic", "lanczos", "fsr1_like")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    method: str
    average_ms: float
    median_ms: float
    p95_ms: float
    operations_per_second: float


def _positive_int(value: str) -> int:
    parsed_value = int(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")

    return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark bicubic, Lanczos, and the full FSR1-like path.",
    )
    parser.add_argument("--input-image", type=Path, default=None)
    parser.add_argument("--output-width", type=_positive_int, default=OUTPUT_WIDTH)
    parser.add_argument("--output-height", type=_positive_int, default=OUTPUT_HEIGHT)
    parser.add_argument("--warmup-runs", type=_positive_int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--benchmark-runs", type=_positive_int, default=DEFAULT_BENCHMARK_RUNS)
    return parser.parse_args()


def percentile(values: list[float], percentile_value: float) -> float:
    ordered_values = sorted(values)
    index = round((percentile_value / 100.0) * (len(ordered_values) - 1))
    return ordered_values[index]


def load_runtime_profile() -> RuntimeProfile:
    profile_file = runtime_profile_path()

    if not profile_file.exists():
        return DEFAULT_RUNTIME_PROFILE

    return RuntimeProfile.load(profile_file)


def load_or_capture_frame(input_image: Path | None):
    if input_image is not None:
        frame = cv2.imread(str(input_image), cv2.IMREAD_COLOR)

        if frame is None:
            raise RuntimeError(f"Unable to load input image: {input_image}")

        return frame, f"loaded image {input_image}"

    capture = CaptureManager(backend="auto")

    try:
        frame = capture.grab_frame()
        source = f"real screen capture via {capture.backend_name.upper()}"
    finally:
        capture.close()

    return frame, source


def benchmark_methods(
    frame,
    output_width: int,
    output_height: int,
    warmup_runs: int,
    benchmark_runs: int,
    profile: RuntimeProfile,
) -> list[BenchmarkResult]:
    """Benchmark methods in a rotating order to avoid fixed-order thermal bias."""

    backends = {
        method: OpenCVProcessingBackend(
            output_width=output_width,
            output_height=output_height,
            upscaling_method=method,
            fsr1_like_sharpening_enabled=profile.fsr1_like_sharpening_enabled,
            fsr1_like_sharpening_strength=profile.fsr1_like_sharpening_strength,
            fsr1_like_edge_strength=profile.fsr1_like_edge_strength,
        )
        for method in METHODS
    }
    retained_outputs = {}

    for warmup_index in range(warmup_runs):
        method_offset = warmup_index % len(METHODS)
        method_order = METHODS[method_offset:] + METHODS[:method_offset]

        for method in method_order:
            retained_outputs[method] = backends[method].process(frame)

    timings = {method: [] for method in METHODS}

    for benchmark_index in range(benchmark_runs):
        method_offset = benchmark_index % len(METHODS)
        method_order = METHODS[method_offset:] + METHODS[:method_offset]

        for method in method_order:
            start = time.perf_counter()
            retained_outputs[method] = backends[method].process(frame)
            timings[method].append((time.perf_counter() - start) * 1000.0)

    results = []

    for method in METHODS:
        method_timings = timings[method]
        average_ms = statistics.mean(method_timings)
        results.append(
            BenchmarkResult(
                method=method,
                average_ms=average_ms,
                median_ms=statistics.median(method_timings),
                p95_ms=percentile(method_timings, 95.0),
                operations_per_second=1000.0 / average_ms,
            )
        )

    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    profile = load_runtime_profile()
    cv2.setNumThreads(profile.opencv_threads)
    frame, source = load_or_capture_frame(args.input_image)

    print("=" * 72)
    print("UniversalUpscaler Spatial Method Benchmark")
    print("=" * 72)
    print(f"Source: {source}")
    print(f"Input resolution: {frame.shape[1]} x {frame.shape[0]}")
    print(f"Output resolution: {args.output_width} x {args.output_height}")
    print(f"OpenCV threads: {profile.opencv_threads}")
    print(f"Warm-up runs per method: {args.warmup_runs}")
    print(f"Measured runs per method: {args.benchmark_runs}")
    print(
        "FSR1-like settings: "
        f"edge={profile.fsr1_like_edge_strength:.2f}, "
        f"sharpening={'on' if profile.fsr1_like_sharpening_enabled else 'off'}, "
        f"sharpening_strength={profile.fsr1_like_sharpening_strength:.2f}"
    )

    results = benchmark_methods(
        frame=frame,
        output_width=args.output_width,
        output_height=args.output_height,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        profile=profile,
    )

    print()
    print(f"{'Method':<14} {'Average':>12} {'Median':>12} {'P95':>12} {'Ops/s':>12}")
    print("-" * 66)

    for result in results:
        print(
            f"{result.method:<14} "
            f"{result.average_ms:>9.3f} ms "
            f"{result.median_ms:>9.3f} ms "
            f"{result.p95_ms:>9.3f} ms "
            f"{result.operations_per_second:>12.2f}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
