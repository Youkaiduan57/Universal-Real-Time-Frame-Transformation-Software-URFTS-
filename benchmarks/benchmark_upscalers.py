import statistics
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIRECTORY))

from capture import ScreenCapture
from config import OUTPUT_HEIGHT, OUTPUT_WIDTH


WARMUP_RUNS = 30
BENCHMARK_RUNS = 300

METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def percentile(values, percentile_value):
    ordered_values = sorted(values)

    index = round(
        (percentile_value / 100)
        * (len(ordered_values) - 1)
    )

    return ordered_values[index]


def benchmark_method(frame, method_name, interpolation):
    for _ in range(WARMUP_RUNS):
        cv2.resize(
            frame,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            interpolation=interpolation,
        )

    timings = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        cv2.resize(
            frame,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            interpolation=interpolation,
        )

        end = time.perf_counter()

        timings.append((end - start) * 1000)

    average_ms = statistics.mean(timings)
    median_ms = statistics.median(timings)
    percentile_95_ms = percentile(timings, 95)
    theoretical_fps = 1000 / average_ms

    print(f"\nMethod: {method_name}")
    print(f"Average: {average_ms:.3f} ms")
    print(f"Median: {median_ms:.3f} ms")
    print(f"95th percentile: {percentile_95_ms:.3f} ms")
    print(f"Resize-only FPS: {theoretical_fps:.1f}")


def main():
    capture = ScreenCapture()

    try:
        frame = capture.grab_frame()
    finally:
        capture.close()

    print("=" * 50)
    print("UniversalUpscaler Resize Benchmark")
    print("=" * 50)
    print(
        f"Input resolution: "
        f"{frame.shape[1]} x {frame.shape[0]}"
    )
    print(
        f"Output resolution: "
        f"{OUTPUT_WIDTH} x {OUTPUT_HEIGHT}"
    )
    print(f"Runs per method: {BENCHMARK_RUNS}")

    for method_name, interpolation in METHODS.items():
        benchmark_method(
            frame,
            method_name,
            interpolation,
        )

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()