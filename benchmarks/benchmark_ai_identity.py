"""Validate and benchmark the local ONNX identity image adapter."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from ai_processor import (
    AIProcessor,
    SUPPORTED_COLOR_ORDERS,
    SUPPORTED_EXECUTION_PROVIDERS,
    SUPPORTED_IMAGE_LAYOUTS,
)


def percentile(values: list[float], percentile_value: float) -> float:
    ordered_values = sorted(values)
    index = round((percentile_value / 100.0) * (len(ordered_values) - 1))
    return ordered_values[index]


def print_timings(label: str, values: list[float]) -> None:
    print(f"{label} average: {statistics.mean(values):.4f} ms")
    print(f"{label} median: {statistics.median(values):.4f} ms")
    print(f"{label} p95: {percentile(values, 95):.4f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a generated ONNX identity image model with one provider.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layout", choices=SUPPORTED_IMAGE_LAYOUTS, default="nchw")
    parser.add_argument(
        "--color-order",
        choices=SUPPORTED_COLOR_ORDERS,
        default="rgb",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_EXECUTION_PROVIDERS,
        default="cpu",
    )
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(
            f"Identity model not found: {args.model}. Generate it with identity_onnx_model.py first."
        )
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("Width and height must be greater than zero.")
    if args.warmup < 0 or args.frames <= 0:
        raise SystemExit("Warmup must be non-negative and frames must be greater than zero.")
    if args.device_id < 0:
        raise SystemExit("Device ID must be zero or greater.")

    rng = np.random.default_rng(2026)
    frames = [
        rng.integers(
            0,
            256,
            size=(args.height, args.width, 3),
            dtype=np.uint8,
        )
        for _ in range(max(args.warmup, args.frames))
    ]
    processor = AIProcessor(
        model_path=args.model,
        input_layout=args.layout,
        output_layout=args.layout,
        color_order=args.color_order,
        provider=args.provider,
        device_id=args.device_id,
    )
    processor.initialize()
    active_providers = processor.active_providers

    inference_times = []
    total_times = []
    maximum_pixel_error = 0
    try:
        for frame in frames[: args.warmup]:
            processor.process(frame)

        for frame in frames[: args.frames]:
            total_start = time.perf_counter()
            output = processor.process(frame)
            total_times.append((time.perf_counter() - total_start) * 1000.0)
            if processor.last_inference_ms is None:
                raise RuntimeError("Inference timing was not recorded.")
            inference_times.append(processor.last_inference_ms)

            frame_error = int(
                np.abs(output.astype(np.int16) - frame.astype(np.int16)).max()
            )
            maximum_pixel_error = max(maximum_pixel_error, frame_error)
            if frame_error > 1:
                raise RuntimeError(
                    f"Identity validation failed with maximum pixel error {frame_error}."
                )
    finally:
        processor.shutdown()

    print("=" * 56)
    print("UniversalUpscaler ONNX Identity Benchmark")
    print("=" * 56)
    print(f"Model: {args.model}")
    print(f"Requested provider: {args.provider.upper()}")
    print(f"Active session providers: {', '.join(active_providers)}")
    if args.provider == "directml":
        print(f"DirectML device ID: {args.device_id}")
    print(f"Layout: {args.layout.upper()}")
    print(f"Color order: {args.color_order.upper()}")
    print(f"Frame size: {args.width}x{args.height}")
    print(f"Warm-up frames: {args.warmup}")
    print(f"Frames measured: {args.frames}")
    print_timings("Inference", inference_times)
    print_timings("Total processing", total_times)
    print(f"Maximum pixel error: {maximum_pixel_error}")
    print("Identity validation: passed (maximum pixel error <= 1)")
    print("=" * 56)


if __name__ == "__main__":
    main()
