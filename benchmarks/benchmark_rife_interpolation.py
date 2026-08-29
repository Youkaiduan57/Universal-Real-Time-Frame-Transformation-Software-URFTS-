"""Offline midpoint interpolation and timing for one RIFE ONNX model."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from frame_interpolator import RIFEInterpolator


def _percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = round((percentile_value / 100.0) * (len(ordered) - 1))
    return ordered[index]


def _format_distribution(values: list[float]) -> str:
    return (
        f"{statistics.mean(values):.3f}/"
        f"{statistics.median(values):.3f}/"
        f"{_percentile(values, 95.0):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and time one offline RIFE midpoint frame.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frame-a", type=Path, required=True)
    parser.add_argument("--frame-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("nchw", "nhwc"), default="nchw")
    parser.add_argument("--provider", choices=("cpu", "directml"), default="cpu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if args.device_id < 0:
        parser.error("--device-id must be zero or greater")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")

    frame_a = cv2.imread(str(args.frame_a), cv2.IMREAD_COLOR)
    frame_b = cv2.imread(str(args.frame_b), cv2.IMREAD_COLOR)
    if frame_a is None:
        parser.error(f"unable to decode frame A: {args.frame_a}")
    if frame_b is None:
        parser.error(f"unable to decode frame B: {args.frame_b}")
    if frame_a.shape != frame_b.shape:
        parser.error(
            f"input image dimensions must match: {frame_a.shape} vs {frame_b.shape}"
        )

    processor = RIFEInterpolator(
        args.model,
        input_layout=args.layout,
        output_layout=args.layout,
        provider=args.provider,
        device_id=args.device_id,
    )
    initialization_start = time.perf_counter()
    processor.initialize()
    initialization_ms = (time.perf_counter() - initialization_start) * 1000.0
    active_providers = processor.active_providers
    output = None
    inference_times = []
    total_times = []
    try:
        for _ in range(args.warmup):
            processor.interpolate(frame_a, frame_b)
        for _ in range(args.iterations):
            total_start = time.perf_counter()
            output = processor.interpolate(frame_a, frame_b)
            total_times.append((time.perf_counter() - total_start) * 1000.0)
            if processor.last_inference_ms is None:
                raise RuntimeError("RIFE inference timing was not recorded")
            inference_times.append(processor.last_inference_ms)
        input_dimensions = processor.input_dimensions
        padded_dimensions = processor.padded_input_dimensions
        output_dimensions = processor.output_dimensions
    finally:
        processor.shutdown()

    if output is None or output_dimensions is None:
        raise RuntimeError("RIFE did not produce an output frame")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise RuntimeError(f"unable to save interpolated frame: {args.output}")
    saved = cv2.imread(str(args.output), cv2.IMREAD_COLOR)
    if saved is None or (saved.shape[1], saved.shape[0]) != output_dimensions:
        raise RuntimeError("saved interpolation dimensions do not match model output")

    print("RIFE offline midpoint interpolation")
    print(f"Requested provider: {args.provider.upper()}")
    print(f"Active providers: {', '.join(active_providers)}")
    print(f"Input dimensions: {input_dimensions[0]}x{input_dimensions[1]}")
    print(f"Padded model dimensions: {padded_dimensions[0]}x{padded_dimensions[1]}")
    print(f"Output dimensions: {output_dimensions[0]}x{output_dimensions[1]}")
    print(f"Initialization: {initialization_ms:.3f} ms")
    print(f"Inference avg/median/p95: {_format_distribution(inference_times)} ms")
    print(f"Total avg/median/p95: {_format_distribution(total_times)} ms")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
