"""Generate spatial-method comparison images and objective descriptive metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from capture_manager import CaptureManager
from config import OUTPUT_HEIGHT, OUTPUT_WIDTH, runtime_profile_path
from processing_backend import OpenCVProcessingBackend
from runtime_profile import DEFAULT_RUNTIME_PROFILE, RuntimeProfile

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "output" / "spatial_comparison"
SUPPORTED_COMPARISON_METHODS = ("bicubic", "lanczos", "fsr1_like")


def _positive_int(value: str) -> int:
    parsed_value = int(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")

    return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare spatial upscalers on one captured or loaded real frame.",
    )
    parser.add_argument("--input-image", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--output-width", type=_positive_int, default=OUTPUT_WIDTH)
    parser.add_argument("--output-height", type=_positive_int, default=OUTPUT_HEIGHT)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_COMPARISON_METHODS,
        default=list(SUPPORTED_COMPARISON_METHODS),
    )
    parser.add_argument(
        "--side-by-side",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


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


def calculate_quality_metrics(frame: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = float(np.mean(cv2.magnitude(gradient_x, gradient_y)))

    local_mean = cv2.blur(gray, (5, 5))
    local_mean_squared = cv2.blur(gray * gray, (5, 5))
    local_variance = np.maximum(local_mean_squared - (local_mean * local_mean), 0.0)
    local_contrast = float(np.mean(np.sqrt(local_variance)))

    low_clipping = float(np.count_nonzero(frame == 0) / frame.size * 100.0)
    high_clipping = float(np.count_nonzero(frame == 255) / frame.size * 100.0)

    return {
        "mean_edge_strength": edge_strength,
        "mean_local_contrast": local_contrast,
        "low_clipping_percent": low_clipping,
        "high_clipping_percent": high_clipping,
        "total_clipping_percent": low_clipping + high_clipping,
    }


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = cv2.copyMakeBorder(
        frame,
        54,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(24, 24, 24),
    )
    cv2.putText(
        labeled,
        label,
        (18, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    profile = load_runtime_profile()
    cv2.setNumThreads(profile.opencv_threads)
    frame, source = load_or_capture_frame(args.input_image)
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    source_path = output_directory / "source_capture.png"

    if not cv2.imwrite(str(source_path), frame):
        raise RuntimeError(f"Unable to write source image: {source_path}")

    outputs = {}
    metrics = {}
    output_paths = {}

    for method in args.methods:
        backend = OpenCVProcessingBackend(
            output_width=args.output_width,
            output_height=args.output_height,
            upscaling_method=method,
            fsr1_like_sharpening_enabled=profile.fsr1_like_sharpening_enabled,
            fsr1_like_sharpening_strength=profile.fsr1_like_sharpening_strength,
            fsr1_like_edge_strength=profile.fsr1_like_edge_strength,
        )
        output = backend.process(frame)
        output_path = output_directory / (
            f"{method}_{args.output_width}x{args.output_height}.png"
        )

        if not cv2.imwrite(str(output_path), output):
            raise RuntimeError(f"Unable to write comparison image: {output_path}")

        outputs[method] = output
        metrics[method] = calculate_quality_metrics(output)
        output_paths[method] = str(output_path)

    side_by_side_path = None

    if args.side_by_side:
        labeled_outputs = [
            add_label(outputs[method], "FSR1-like" if method == "fsr1_like" else method.title())
            for method in args.methods
        ]
        side_by_side = np.hstack(labeled_outputs)
        side_by_side_path = output_directory / "side_by_side.png"

        if not cv2.imwrite(str(side_by_side_path), side_by_side):
            raise RuntimeError(f"Unable to write side-by-side image: {side_by_side_path}")

    report = {
        "source": source,
        "source_path": str(source_path),
        "input_resolution": [frame.shape[1], frame.shape[0]],
        "output_resolution": [args.output_width, args.output_height],
        "fsr1_like_settings": {
            "edge_strength": profile.fsr1_like_edge_strength,
            "sharpening_enabled": profile.fsr1_like_sharpening_enabled,
            "sharpening_strength": profile.fsr1_like_sharpening_strength,
        },
        "metrics": metrics,
        "outputs": output_paths,
        "side_by_side": str(side_by_side_path) if side_by_side_path else None,
        "reference_metrics": "SSIM/PSNR omitted because no meaningful ground-truth reference exists.",
    }
    metrics_path = output_directory / "quality_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(report, metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")

    print("=" * 72)
    print("UniversalUpscaler Spatial Quality Comparison")
    print("=" * 72)
    print(f"Source: {source}")
    print(f"Source image: {source_path}")
    print()
    print(
        f"{'Method':<14} {'Edge strength':>15} {'Local contrast':>16} "
        f"{'Clipping':>12}"
    )
    print("-" * 62)

    for method in args.methods:
        method_metrics = metrics[method]
        print(
            f"{method:<14} "
            f"{method_metrics['mean_edge_strength']:>15.4f} "
            f"{method_metrics['mean_local_contrast']:>16.4f} "
            f"{method_metrics['total_clipping_percent']:>10.4f}%"
        )

    print()

    for method in args.methods:
        print(f"{method}: {output_paths[method]}")

    if side_by_side_path is not None:
        print(f"side_by_side: {side_by_side_path}")

    print(f"metrics: {metrics_path}")
    print("SSIM/PSNR omitted: there is no meaningful ground-truth reference.")
    print("=" * 72)


if __name__ == "__main__":
    main()
