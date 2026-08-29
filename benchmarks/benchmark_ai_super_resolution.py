"""Offline image validation and timing for one ONNX image-to-image model."""

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

from ai_processor import (
    AIProcessor,
    ImageTensorMetadata,
    SUPPORTED_COLOR_ORDERS,
    SUPPORTED_EXECUTION_PROVIDERS,
    SUPPORTED_IMAGE_LAYOUTS,
    SUPPORTED_SCALE_SETTINGS,
)


def percentile(values: list[float], percentile_value: float) -> float:
    ordered_values = sorted(values)
    index = round((percentile_value / 100.0) * (len(ordered_values) - 1))
    return ordered_values[index]


def print_timings(label: str, values: list[float]) -> None:
    print(f"{label} average: {statistics.mean(values):.4f} ms")
    print(f"{label} median: {statistics.median(values):.4f} ms")
    print(f"{label} p95: {percentile(values, 95):.4f} ms")


def format_metadata(metadata: ImageTensorMetadata) -> str:
    return (
        f"name={metadata.name}, dtype={metadata.dtype}, "
        f"shape={metadata.shape}, layout={metadata.layout.upper()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark one ONNX image-to-image model offline.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--layout", choices=SUPPORTED_IMAGE_LAYOUTS, default="nchw")
    parser.add_argument(
        "--color-order",
        choices=SUPPORTED_COLOR_ORDERS,
        default="rgb",
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_EXECUTION_PROVIDERS,
        default="cpu",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--scale", choices=SUPPORTED_SCALE_SETTINGS, default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.input_image.is_file():
        raise SystemExit(f"Input image not found: {args.input_image}")
    if args.device_id < 0:
        raise SystemExit("Device ID must be zero or greater.")
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("Warmup must be non-negative and iterations must be greater than zero.")

    frame = cv2.imread(str(args.input_image), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"Unable to decode input image: {args.input_image}")

    processor = AIProcessor(
        model_path=args.model,
        input_layout=args.layout,
        output_layout=args.layout,
        color_order=args.color_order,
        provider=args.provider,
        device_id=args.device_id,
        scale=args.scale,
    )
    processor.initialize()
    if processor.input_metadata is None or processor.output_metadata is None:
        raise RuntimeError("Model metadata was not recorded during initialization.")
    input_metadata = processor.input_metadata
    output_metadata = processor.output_metadata
    active_providers = processor.active_providers

    inference_times = []
    total_times = []
    output = None
    try:
        for _ in range(args.warmup):
            processor.process(frame)

        for _ in range(args.iterations):
            total_start = time.perf_counter()
            output = processor.process(frame)
            total_times.append((time.perf_counter() - total_start) * 1000.0)
            if processor.last_inference_ms is None:
                raise RuntimeError("Inference timing was not recorded.")
            inference_times.append(processor.last_inference_ms)

        detected_scale = processor.detected_scale
        output_dimensions = processor.output_dimensions
    finally:
        processor.shutdown()

    if output is None or detected_scale is None or output_dimensions is None:
        raise RuntimeError("The model did not produce validated image output.")

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_image), output):
        raise RuntimeError(f"Unable to save output image: {args.output_image}")
    saved_output = cv2.imread(str(args.output_image), cv2.IMREAD_COLOR)
    if saved_output is None:
        raise RuntimeError(f"Unable to read saved output image: {args.output_image}")
    saved_dimensions = (saved_output.shape[1], saved_output.shape[0])
    if saved_dimensions != output_dimensions:
        raise RuntimeError(
            f"Saved image dimensions {saved_dimensions} do not match model output {output_dimensions}."
        )

    print("=" * 64)
    print("UniversalUpscaler Offline ONNX Image Validation")
    print("=" * 64)
    print(f"Requested provider: {args.provider.upper()}")
    print(f"Active session providers: {', '.join(active_providers)}")
    print(f"Input dimensions: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Output dimensions: {output_dimensions[0]}x{output_dimensions[1]}")
    print(f"Detected scale: {detected_scale}x")
    print(f"Model input metadata: {format_metadata(input_metadata)}")
    print(f"Model output metadata: {format_metadata(output_metadata)}")
    print(f"Warm-up iterations: {args.warmup}")
    print(f"Measured iterations: {args.iterations}")
    print_timings("Inference", inference_times)
    print_timings("Total processing", total_times)
    print(f"Output file: {args.output_image.resolve()}")
    print("=" * 64)


if __name__ == "__main__":
    main()
