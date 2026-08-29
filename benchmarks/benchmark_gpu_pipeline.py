"""Compare real WGC CPU bilinear processing with a chosen D3D11 GPU scaler.

The CPU mode is intentionally headless: it measures WGC staging/readback and
OpenCV bilinear processing without ``cv2.imshow``.  The GPU mode performs real
flip-model swap-chain presentation with vertical sync.  The differing
presentation semantics are printed with the results and must be considered
when comparing FPS and loop latency.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

import cv2

from capture_manager import CaptureManager
from config import ApplicationConfig, runtime_profile_path
from d3d11_gpu_pipeline import (
    D3D11GpuPipeline,
    GpuPipelineReport,
    LatencySummary,
    SUPPORTED_D3D11_SCALERS,
)
from frame_processor import FrameProcessor
from processing_backend import OpenCVProcessingBackend
from runtime_profile import DEFAULT_RUNTIME_PROFILE, RuntimeProfile
from window_capture import WindowInfo, select_window


@dataclass(frozen=True, slots=True)
class CpuBenchmarkResult:
    duration_seconds: float
    processed_frames: int
    processed_fps: float
    capture_readback: LatencySummary
    processing: LatencySummary
    cpu_loop: LatencySummary
    source_width: int
    source_height: int
    output_width: int
    output_height: int


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _summary(values: list[float]) -> LatencySummary:
    if not values:
        return LatencySummary(0.0, 0.0, 0.0)
    return LatencySummary(
        average_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        p95_ms=_percentile(values, 0.95),
    )


def _runtime_profile() -> RuntimeProfile:
    path = runtime_profile_path()
    return RuntimeProfile.load(path) if path.exists() else DEFAULT_RUNTIME_PROFILE


def _run_cpu(
    *,
    hwnd: int,
    output_width: int,
    output_height: int,
    warmup_seconds: float,
    duration_seconds: float,
) -> CpuBenchmarkResult:
    profile = _runtime_profile()
    cv2.setNumThreads(profile.opencv_threads)
    capture = CaptureManager(
        backend="wgc",
        window_hwnd=hwnd,
        fallback_on_explicit_failure=False,
    )
    backend = OpenCVProcessingBackend(
        output_width=output_width,
        output_height=output_height,
        upscaling_method="bilinear",
    )
    processor = FrameProcessor(processing_backend=backend)
    captures: list[float] = []
    processing: list[float] = []
    loops: list[float] = []
    source_width = 0
    source_height = 0

    try:
        warmup_deadline = time.perf_counter() + warmup_seconds
        while time.perf_counter() < warmup_deadline:
            processor.process(capture.grab_frame())

        started_at = time.perf_counter()
        deadline = started_at + duration_seconds
        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()
            capture_start = loop_start
            frame = capture.grab_frame()
            capture_end = time.perf_counter()
            process_start = capture_end
            processor.process(frame)
            process_end = time.perf_counter()
            source_height, source_width = frame.shape[:2]
            captures.append((capture_end - capture_start) * 1000.0)
            processing.append((process_end - process_start) * 1000.0)
            loops.append((process_end - loop_start) * 1000.0)
        ended_at = time.perf_counter()
    finally:
        capture.close()
        backend.close()

    elapsed = ended_at - started_at
    return CpuBenchmarkResult(
        duration_seconds=elapsed,
        processed_frames=len(loops),
        processed_fps=(len(loops) / elapsed) if elapsed else 0.0,
        capture_readback=_summary(captures),
        processing=_summary(processing),
        cpu_loop=_summary(loops),
        source_width=source_width,
        source_height=source_height,
        output_width=output_width,
        output_height=output_height,
    )


def _run_gpu(
    *,
    hwnd: int,
    output_width: int,
    output_height: int,
    warmup_seconds: float,
    duration_seconds: float,
    method: str,
) -> GpuPipelineReport:
    with D3D11GpuPipeline(
        hwnd=hwnd,
        output_width=output_width,
        output_height=output_height,
        method=method,
        vsync=True,
    ) as pipeline:
        return pipeline.run(
            duration_seconds=duration_seconds,
            warmup_seconds=warmup_seconds,
        )


def _latency(value: LatencySummary) -> str:
    return f"{value.average_ms:.3f}/{value.median_ms:.3f}/{value.p95_ms:.3f}"


def _print_results(
    window: WindowInfo,
    cpu: CpuBenchmarkResult,
    gpu: GpuPipelineReport,
    *,
    method: str,
) -> None:
    print(f"Target: {window.title} (HWND {window.hwnd})")
    print(f"Adapter: {gpu.adapter_description}")
    print(
        f"Source: CPU {cpu.source_width}x{cpu.source_height}; "
        f"GPU {gpu.source_width}x{gpu.source_height}; "
        f"output {gpu.output_width}x{gpu.output_height}"
    )
    print(
        f"CPU method: bilinear | GPU method: {method} | "
        "Source/swap-chain format: DXGI_FORMAT_B8G8R8A8_UNORM"
    )
    print("Latency values are average/median/p95 in milliseconds.")
    print()
    print("| Metric | CPU WGC -> NumPy -> OpenCV (headless) | WGC -> D3D11 -> Present |")
    print("|---|---:|---:|")
    print(f"| FPS | {cpu.processed_fps:.2f} | {gpu.presented_fps:.2f} |")
    print(f"| Frames | {cpu.processed_frames} | {gpu.presented_frames} |")
    print(f"| Capture/acquisition | {_latency(cpu.capture_readback)} | {_latency(gpu.acquisition)} |")
    print(f"| Processing/scale submit | {_latency(cpu.processing)} | {_latency(gpu.scale_submit)} |")
    print(f"| Present call | n/a (headless) | {_latency(gpu.present_submit)} |")
    print(f"| CPU loop | {_latency(cpu.cpu_loop)} | {_latency(gpu.cpu_loop)} |")
    print(f"| Dropped/replaced | 0 | {gpu.replaced_frames} |")
    print()
    print("CPU mode is headless and does not include OpenCV window presentation.")
    print("GPU mode uses a two-buffer flip-discard swap chain with Present(1, 0): vsync is enabled.")
    print("GPU Present timing may include synchronization/blocking; it is not shader execution time.")
    print("D3D11 GPU timestamp/disjoint queries are not implemented in this milestone.")


def parse_args() -> argparse.Namespace:
    config = ApplicationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--window-hwnd", type=int)
    target.add_argument("--window-title")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--output-width", type=int, default=config.output_width)
    parser.add_argument("--output-height", type=int, default=config.output_height)
    parser.add_argument(
        "--method",
        choices=SUPPORTED_D3D11_SCALERS,
        default="bilinear",
        help=(
            "D3D11 scaling method, including fsr1_like; "
            "the CPU comparison remains bilinear."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration < 10.0:
        raise SystemExit("--duration must be at least 10 seconds for a meaningful comparison.")
    if args.warmup < 0.0:
        raise SystemExit("--warmup cannot be negative.")
    if args.output_width <= 0 or args.output_height <= 0:
        raise SystemExit("Output dimensions must be positive.")

    window = select_window(title=args.window_title, hwnd=args.window_hwnd)
    print("Running CPU WGC/readback/OpenCV bilinear benchmark...")
    cpu = _run_cpu(
        hwnd=window.hwnd,
        output_width=args.output_width,
        output_height=args.output_height,
        warmup_seconds=args.warmup,
        duration_seconds=args.duration,
    )
    print(f"Running GPU-resident WGC/D3D11 {args.method}/presentation benchmark...")
    gpu = _run_gpu(
        hwnd=window.hwnd,
        output_width=args.output_width,
        output_height=args.output_height,
        warmup_seconds=args.warmup,
        duration_seconds=args.duration,
        method=args.method,
    )
    _print_results(window, cpu, gpu, method=args.method)


if __name__ == "__main__":
    main()
