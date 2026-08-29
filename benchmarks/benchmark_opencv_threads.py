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


WARMUP_RUNS = 50
BENCHMARK_RUNS = 500

THREAD_COUNTS = [
    1,
    2,
    4,
    8,
]


def percentile(values, percentile_value):
    ordered_values = sorted(values)

    index = round(
        (percentile_value / 100)
        * (len(ordered_values) - 1)
    )

    return ordered_values[index]


def benchmark_thread_count(frame, thread_count):
    cv2.setNumThreads(thread_count)

    for _ in range(WARMUP_RUNS):
        cv2.resize(
            frame,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )

    timings = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        cv2.resize(
            frame,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )

        end = time.perf_counter()

        timings.append(
            (end - start) * 1000
        )

    average_ms = statistics.mean(timings)
    median_ms = statistics.median(timings)
    percentile_95_ms = percentile(timings, 95)
    worst_ms = max(timings)

    print()
    print(f"OpenCV threads: {thread_count}")
    print(f"Average: {average_ms:.3f} ms")
    print(f"Median: {median_ms:.3f} ms")
    print(
        f"95th percentile: "
        f"{percentile_95_ms:.3f} ms"
    )
    print(f"Worst frame: {worst_ms:.3f} ms")


def main():
    capture = ScreenCapture()

    try:
        frame = capture.grab_frame()
    finally:
        capture.close()

    print("=" * 50)
    print("UniversalUpscaler OpenCV Thread Benchmark")
    print("=" * 50)

    print(
        f"Input resolution: "
        f"{frame.shape[1]} x {frame.shape[0]}"
    )

    print(
        f"Output resolution: "
        f"{OUTPUT_WIDTH} x {OUTPUT_HEIGHT}"
    )

    print(
        f"OpenCV optimized: "
        f"{cv2.useOptimized()}"
    )

    print(
        f"OpenCV default threads: "
        f"{cv2.getNumThreads()}"
    )

    for thread_count in THREAD_COUNTS:
        benchmark_thread_count(
            frame,
            thread_count,
        )

    cv2.setNumThreads(0)

    print()
    print("=" * 50)


if __name__ == "__main__":
    main()