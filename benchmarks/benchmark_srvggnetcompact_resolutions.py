"""Benchmark SRVGGNetCompact x2 at conservative live AI input resolutions."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from ai_processor import (
    DEFAULT_AI_TILE_OVERLAP,
    DEFAULT_AI_TILE_SIZE,
    AIProcessor,
    SUPPORTED_EXECUTION_PROVIDERS,
)


DEFAULT_MODEL = PROJECT_ROOT / "models" / "SRVGGNetCompact_x2.onnx"
DEFAULT_RESOLUTIONS = ("160x90", "320x180", "480x270", "640x360")


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    input_dimensions: tuple[int, int]
    tile_mode: str
    success: bool
    output_dimensions: tuple[int, int] | None = None
    detected_scale: int | None = None
    inference_average_ms: float | None = None
    inference_median_ms: float | None = None
    inference_p95_ms: float | None = None
    total_average_ms: float | None = None
    total_median_ms: float | None = None
    total_p95_ms: float | None = None
    effective_fps: float | None = None
    active_providers: tuple[str, ...] = ()
    selected_tile_size: int | None = None
    tiles_processed: int = 0
    estimated_peak_tile_bytes: int | None = None
    output_path: Path | None = None
    error: str | None = None


def _percentile(values: list[float], percentile_value: float) -> float:
    ordered_values = sorted(values)
    index = round((percentile_value / 100.0) * (len(ordered_values) - 1))
    return ordered_values[index]


def _parse_resolution(value: str) -> tuple[int, int]:
    normalized = value.strip().lower()
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution {value!r}; expected WIDTHxHEIGHT."
        )
    try:
        width, height = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution {value!r}; expected integer WIDTHxHEIGHT."
        ) from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution {value!r}; width and height must be positive."
        )
    return width, height


def _create_source_frame(width: int, height: int) -> np.ndarray:
    """Create one deterministic BGR capture frame without downloading an image."""

    horizontal = np.linspace(0, 255, width, dtype=np.uint8)
    vertical = np.linspace(0, 255, height, dtype=np.uint8)
    blue = np.broadcast_to(horizontal, (height, width))
    green = np.broadcast_to(vertical[:, np.newaxis], (height, width))
    red = ((blue.astype(np.uint16) + green.astype(np.uint16)) // 2).astype(
        np.uint8
    )
    return np.ascontiguousarray(np.dstack((blue, green, red)))


def _benchmark_resolution(
    *,
    model_path: Path,
    source_frame: np.ndarray,
    input_dimensions: tuple[int, int],
    tile_mode: str,
    tile_size: int,
    tile_overlap: int,
    provider: str,
    device_id: int,
    warmup: int,
    iterations: int,
    output_directory: Path,
) -> ResolutionResult:
    width, height = input_dimensions
    processor: AIProcessor | None = None
    active_providers: tuple[str, ...] = ()
    try:
        processor = AIProcessor(
            model_path=model_path,
            input_layout="nchw",
            output_layout="nchw",
            color_order="rgb",
            provider=provider,
            device_id=device_id,
            scale=2,
            input_width=width,
            input_height=height,
            tile=(
                "off"
                if tile_mode == "full"
                else "auto" if tile_mode == "auto" else tile_size
            ),
            tile_overlap=tile_overlap,
        )
        processor.initialize()
        active_providers = processor.active_providers

        for _ in range(warmup):
            processor.process(source_frame)

        inference_times: list[float] = []
        total_times: list[float] = []
        output: np.ndarray | None = None
        for _ in range(iterations):
            total_start = time.perf_counter()
            output = processor.process(source_frame)
            total_times.append((time.perf_counter() - total_start) * 1000.0)
            if processor.last_inference_ms is None:
                raise RuntimeError("Inference timing was not recorded.")
            inference_times.append(processor.last_inference_ms)

        if output is None:
            raise RuntimeError("No measured output was produced.")
        if processor.detected_scale != 2:
            raise RuntimeError(
                f"Expected a 2x model output; detected {processor.detected_scale!r}."
            )
        expected_output_dimensions = (width * 2, height * 2)
        if processor.output_dimensions != expected_output_dimensions:
            raise RuntimeError(
                f"Expected output {expected_output_dimensions}, got "
                f"{processor.output_dimensions}."
            )

        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_directory
            / f"srvggnetcompact_x2_{provider}_{tile_mode}_{width}x{height}.png"
        )
        if not cv2.imwrite(str(output_path), output):
            raise RuntimeError(f"Unable to save output image: {output_path}")
        saved_output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if saved_output is None:
            raise RuntimeError(f"Unable to read saved output image: {output_path}")
        saved_dimensions = (saved_output.shape[1], saved_output.shape[0])
        if saved_dimensions != expected_output_dimensions:
            raise RuntimeError(
                f"Saved output dimensions {saved_dimensions} do not match "
                f"{expected_output_dimensions}."
            )

        inference_average = statistics.mean(inference_times)
        inference_median = statistics.median(inference_times)
        inference_p95 = _percentile(inference_times, 95)
        total_average = statistics.mean(total_times)
        total_median = statistics.median(total_times)
        total_p95 = _percentile(total_times, 95)
        return ResolutionResult(
            input_dimensions=input_dimensions,
            tile_mode=tile_mode,
            success=True,
            output_dimensions=expected_output_dimensions,
            detected_scale=processor.detected_scale,
            inference_average_ms=inference_average,
            inference_median_ms=inference_median,
            inference_p95_ms=inference_p95,
            total_average_ms=total_average,
            total_median_ms=total_median,
            total_p95_ms=total_p95,
            effective_fps=1000.0 / total_median,
            active_providers=active_providers,
            selected_tile_size=processor.selected_tile_size,
            tiles_processed=processor.tiles_processed,
            estimated_peak_tile_bytes=processor.estimated_peak_tile_bytes,
            output_path=output_path.resolve(),
        )
    except Exception as error:
        return ResolutionResult(
            input_dimensions=input_dimensions,
            tile_mode=tile_mode,
            success=False,
            active_providers=active_providers,
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        if processor is not None:
            processor.shutdown()
        del processor
        gc.collect()


def _format_timing(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _print_results(
    results: list[ResolutionResult],
    provider: str,
    device_id: int,
    warmup: int,
    iterations: int,
) -> None:
    print("=" * 132)
    print("UniversalUpscaler SRVGGNetCompact x2 Resolution Benchmark")
    print("=" * 132)
    print(f"Requested provider: {provider.upper()}")
    print(f"Device ID: {device_id}")
    print(f"Warm-up iterations per resolution: {warmup}")
    print(f"Measured iterations per resolution: {iterations}")
    print(
        "Peak memory is a conservative tensor/activation proxy, not measured VRAM or RAM."
    )
    print(
        "Input     Mode    Tile/count  Output      Scale  "
        "Infer avg/med/p95 (ms)       Total avg/med/p95 (ms)       "
        "FPS     Peak MiB  Status"
    )
    print("-" * 132)
    for result in results:
        width, height = result.input_dimensions
        output = (
            "-"
            if result.output_dimensions is None
            else f"{result.output_dimensions[0]}x{result.output_dimensions[1]}"
        )
        scale = "-" if result.detected_scale is None else f"{result.detected_scale}x"
        tile = (
            "full/1"
            if result.selected_tile_size is None
            else f"{result.selected_tile_size}/{result.tiles_processed}"
        )
        inference = "/".join(
            _format_timing(value)
            for value in (
                result.inference_average_ms,
                result.inference_median_ms,
                result.inference_p95_ms,
            )
        )
        total = "/".join(
            _format_timing(value)
            for value in (
                result.total_average_ms,
                result.total_median_ms,
                result.total_p95_ms,
            )
        )
        fps = "-" if result.effective_fps is None else f"{result.effective_fps:.2f}"
        peak_mib = (
            "-"
            if result.estimated_peak_tile_bytes is None
            else f"{result.estimated_peak_tile_bytes / (1024 * 1024):.1f}"
        )
        status = "SUCCESS" if result.success else "FAILURE"
        print(
            f"{width}x{height:<5} {result.tile_mode:<7} {tile:<11} "
            f"{output:<11} {scale:<6} {inference:<28} {total:<28} "
            f"{fps:<7} {peak_mib:<9} {status}"
        )
        print(
            "  Active providers: "
            + (", ".join(result.active_providers) or "none recorded")
        )
        if result.output_path is not None:
            print(f"  Output file: {result.output_path}")
        if result.error is not None:
            print(f"  Error: {result.error}")
    print("=" * 132)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the local SRVGGNetCompact x2 ONNX model sequentially at "
            "conservative internal input resolutions."
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=_parse_resolution,
        default=[_parse_resolution(value) for value in DEFAULT_RESOLUTIONS],
        metavar="WIDTHxHEIGHT",
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_EXECUTION_PROVIDERS,
        default="directml",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--tile-modes",
        nargs="+",
        choices=("full", "tiled", "auto"),
        default=["full"],
        help="Benchmark full-frame, fixed tiled, and/or automatic tiled inference.",
    )
    parser.add_argument("--tile-size", type=int, default=DEFAULT_AI_TILE_SIZE)
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=DEFAULT_AI_TILE_OVERLAP,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "srvggnetcompact_resolution_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if args.device_id < 0:
        raise SystemExit("Device ID must be zero or greater.")
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit(
            "Warmup must be non-negative and iterations must be greater than zero."
        )
    if args.tile_size <= 0:
        raise SystemExit("Tile size must be greater than zero.")
    if args.tile_overlap < 0:
        raise SystemExit("Tile overlap must be zero or greater.")
    if args.tile_overlap * 2 >= args.tile_size:
        raise SystemExit("Tile size must be greater than twice the tile overlap.")

    maximum_width = max(width for width, _ in args.resolutions)
    maximum_height = max(height for _, height in args.resolutions)
    source_frame = _create_source_frame(maximum_width, maximum_height)
    results = []
    for resolution in args.resolutions:
        for tile_mode in args.tile_modes:
            results.append(
                _benchmark_resolution(
                    model_path=args.model,
                    source_frame=source_frame,
                    input_dimensions=resolution,
                    tile_mode=tile_mode,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                    provider=args.provider,
                    device_id=args.device_id,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    output_directory=args.output_dir,
                )
            )
    _print_results(
        results,
        provider=args.provider,
        device_id=args.device_id,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    if any(not result.success for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
