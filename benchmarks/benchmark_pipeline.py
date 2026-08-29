import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIRECTORY))

from capture import ScreenCapture
from frame_processor import FrameProcessor


WARMUP_FRAMES = 30
BENCHMARK_FRAMES = 300


def milliseconds(seconds):
    return seconds * 1000.0


def percentile(values, percentile_value):
    ordered_values = sorted(values)

    index = int(
        round(
            (percentile_value / 100.0)
            * (len(ordered_values) - 1)
        )
    )

    return ordered_values[index]


def main():
    capture = ScreenCapture()
    processor = FrameProcessor()

    capture_times = []
    processing_times = []
    pipeline_times = []

    print("Warming up...")

    try:
        for _ in range(WARMUP_FRAMES):
            frame = capture.grab_frame()
            processor.process(frame)

        print(
            f"Benchmarking {BENCHMARK_FRAMES} frames..."
        )

        benchmark_start = time.perf_counter()

        for _ in range(BENCHMARK_FRAMES):
            pipeline_start = time.perf_counter()

            capture_start = time.perf_counter()
            frame = capture.grab_frame()
            capture_end = time.perf_counter()

            processing_start = time.perf_counter()
            processor.process(frame)
            processing_end = time.perf_counter()

            pipeline_end = time.perf_counter()

            capture_times.append(
                milliseconds(capture_end - capture_start)
            )

            processing_times.append(
                milliseconds(
                    processing_end - processing_start
                )
            )

            pipeline_times.append(
                milliseconds(pipeline_end - pipeline_start)
            )

        benchmark_elapsed = (
            time.perf_counter() - benchmark_start
        )

    finally:
        capture.close()

    measured_fps = BENCHMARK_FRAMES / benchmark_elapsed

    print()
    print("=" * 50)
    print("UniversalUpscaler Controlled Benchmark")
    print("=" * 50)
    print(f"Frames measured: {BENCHMARK_FRAMES}")
    print(f"Measured FPS: {measured_fps:.2f}")
    print()
    print(
        "Capture average:",
        f"{statistics.mean(capture_times):.2f} ms",
    )
    print(
        "Capture median:",
        f"{statistics.median(capture_times):.2f} ms",
    )
    print(
        "Capture 95th percentile:",
        f"{percentile(capture_times, 95):.2f} ms",
    )
    print()
    print(
        "Processing average:",
        f"{statistics.mean(processing_times):.2f} ms",
    )
    print(
        "Processing median:",
        f"{statistics.median(processing_times):.2f} ms",
    )
    print(
        "Processing 95th percentile:",
        f"{percentile(processing_times, 95):.2f} ms",
    )
    print()
    print(
        "Pipeline average:",
        f"{statistics.mean(pipeline_times):.2f} ms",
    )
    print(
        "Pipeline median:",
        f"{statistics.median(pipeline_times):.2f} ms",
    )
    print(
        "Pipeline 95th percentile:",
        f"{percentile(pipeline_times, 95):.2f} ms",
    )
    print("=" * 50)


if __name__ == "__main__":
    main()