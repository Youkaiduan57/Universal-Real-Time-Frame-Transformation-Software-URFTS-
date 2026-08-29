"""Optional moving-window runtime performance telemetry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import statistics
import time
from typing import Callable


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    presented_at: float
    capture_ms: float
    preprocessing_ms: float | None
    inference_ms: float | None
    postprocessing_ms: float | None
    total_frame_ms: float
    dropped_frames: int
    active_provider: str
    capture_dimensions: tuple[int, int] | None
    ai_input_dimensions: tuple[int, int] | None
    ai_output_dimensions: tuple[int, int] | None
    tile_mode: str
    interpolation_ms: float | None = None
    interpolation_provider: str = "none"
    frame_generation: str = "off"
    dropped_generated_frames: int = 0
    scheduled_presentation_timestamp: float = 0.0
    actual_presentation_timestamp: float = 0.0
    pacing_error_ms: float = 0.0
    late_frames: int = 0
    generated_frames_dropped_late: int = 0
    real_frames_dropped_late: int = 0
    estimated_source_fps: float = 0.0
    presentation_fps: float = 0.0
    runtime_state: str = "running"
    recovery_retry_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    fallback_activations: int = 0
    presented_real_frames: int = 0
    presented_generated_frames: int = 0
    presented_frames: int = 0
    generated_frames_requested: int = 0


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    fps: float
    capture_ms: float
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    total_frame_ms: float
    median_frame_ms: float
    p95_frame_ms: float
    dropped_frames: int
    active_provider: str
    capture_dimensions: tuple[int, int] | None
    ai_input_dimensions: tuple[int, int] | None
    ai_output_dimensions: tuple[int, int] | None
    tile_mode: str
    sample_count: int
    interpolation_ms: float = 0.0
    interpolation_provider: str = "none"
    frame_generation: str = "off"
    dropped_generated_frames: int = 0
    scheduled_presentation_timestamp: float = 0.0
    actual_presentation_timestamp: float = 0.0
    pacing_error_ms: float = 0.0
    median_pacing_error_ms: float = 0.0
    p95_pacing_error_ms: float = 0.0
    late_frames: int = 0
    generated_frames_dropped_late: int = 0
    real_frames_dropped_late: int = 0
    estimated_source_fps: float = 0.0
    presentation_fps: float = 0.0
    runtime_state: str = "running"
    recovery_retry_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    fallback_activations: int = 0
    presented_real_frames: int = 0
    presented_generated_frames: int = 0
    presented_frames: int = 0
    generated_frames_requested: int = 0


def _percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = round((percentile_value / 100.0) * (len(ordered) - 1))
    return ordered[index]


class PerformanceMetrics:
    """Maintain roughly 60 presented frames only when telemetry is enabled."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        window_size: int = 60,
        log_interval_seconds: float = 5.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if window_size <= 0:
            raise ValueError("Telemetry window size must be greater than zero.")
        if log_interval_seconds <= 0.0:
            raise ValueError("Telemetry log interval must be greater than zero.")
        self.enabled = bool(enabled)
        self._clock = clock
        self._log_interval_seconds = float(log_interval_seconds)
        self._last_log_at = clock()
        self._samples: deque[TelemetrySample] = deque(maxlen=window_size)

        # Backward-compatible latest values for existing callers.
        self.pipeline_fps = 0.0
        self.capture_ms = 0.0
        self.upscale_ms = 0.0
        self.frame_age_ms = 0.0
        self.end_to_end_ms = 0.0
        self.dropped_frames = 0

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def record(
        self,
        *,
        presented_at: float,
        capture_ms: float,
        preprocessing_ms: float | None,
        inference_ms: float | None,
        postprocessing_ms: float | None,
        total_frame_ms: float,
        dropped_frames: int,
        active_provider: str,
        capture_dimensions: tuple[int, int] | None,
        ai_input_dimensions: tuple[int, int] | None,
        ai_output_dimensions: tuple[int, int] | None,
        tile_mode: str,
        interpolation_ms: float | None = None,
        interpolation_provider: str = "none",
        frame_generation: str = "off",
        dropped_generated_frames: int = 0,
        scheduled_presentation_timestamp: float = 0.0,
        actual_presentation_timestamp: float = 0.0,
        pacing_error_ms: float = 0.0,
        late_frames: int = 0,
        generated_frames_dropped_late: int = 0,
        real_frames_dropped_late: int = 0,
        estimated_source_fps: float = 0.0,
        presentation_fps: float = 0.0,
        runtime_state: str = "running",
        recovery_retry_attempts: int = 0,
        successful_recoveries: int = 0,
        failed_recoveries: int = 0,
        fallback_activations: int = 0,
        presented_real_frames: int = 0,
        presented_generated_frames: int = 0,
        presented_frames: int = 0,
        generated_frames_requested: int = 0,
    ) -> None:
        """Record one presented frame; disabled telemetry returns immediately."""

        if not self.enabled:
            return
        self._samples.append(
            TelemetrySample(
                presented_at=presented_at,
                capture_ms=float(capture_ms),
                preprocessing_ms=preprocessing_ms,
                inference_ms=inference_ms,
                postprocessing_ms=postprocessing_ms,
                total_frame_ms=float(total_frame_ms),
                dropped_frames=int(dropped_frames),
                active_provider=active_provider,
                capture_dimensions=capture_dimensions,
                ai_input_dimensions=ai_input_dimensions,
                ai_output_dimensions=ai_output_dimensions,
                tile_mode=tile_mode,
                interpolation_ms=interpolation_ms,
                interpolation_provider=interpolation_provider,
                frame_generation=frame_generation,
                dropped_generated_frames=int(dropped_generated_frames),
                scheduled_presentation_timestamp=float(scheduled_presentation_timestamp),
                actual_presentation_timestamp=float(actual_presentation_timestamp),
                pacing_error_ms=float(pacing_error_ms),
                late_frames=int(late_frames),
                generated_frames_dropped_late=int(generated_frames_dropped_late),
                real_frames_dropped_late=int(real_frames_dropped_late),
                estimated_source_fps=float(estimated_source_fps),
                presentation_fps=float(presentation_fps),
                runtime_state=str(runtime_state),
                recovery_retry_attempts=int(recovery_retry_attempts),
                successful_recoveries=int(successful_recoveries),
                failed_recoveries=int(failed_recoveries),
                fallback_activations=int(fallback_activations),
                presented_real_frames=int(presented_real_frames),
                presented_generated_frames=int(presented_generated_frames),
                presented_frames=int(presented_frames),
                generated_frames_requested=int(generated_frames_requested),
            )
        )
        snapshot = self.snapshot()
        if snapshot is not None:
            self.pipeline_fps = snapshot.fps
            self.capture_ms = snapshot.capture_ms
            self.upscale_ms = (
                snapshot.preprocessing_ms
                + snapshot.inference_ms
                + snapshot.postprocessing_ms
            )
            self.end_to_end_ms = snapshot.total_frame_ms
            self.dropped_frames = snapshot.dropped_frames

    @staticmethod
    def _average_optional(
        samples: list[TelemetrySample],
        field_name: str,
    ) -> float:
        values = [
            value
            for sample in samples
            if (value := getattr(sample, field_name)) is not None
        ]
        return statistics.mean(values) if values else 0.0

    def snapshot(self) -> TelemetrySnapshot | None:
        if not self.enabled or not self._samples:
            return None
        samples = list(self._samples)
        total_frame_times = [sample.total_frame_ms for sample in samples]
        if len(samples) >= 2:
            elapsed = samples[-1].presented_at - samples[0].presented_at
            fps = (len(samples) - 1) / elapsed if elapsed > 0.0 else 0.0
        else:
            fps = 0.0
        latest = samples[-1]
        pacing_errors = [sample.pacing_error_ms for sample in samples]
        return TelemetrySnapshot(
            fps=fps,
            capture_ms=statistics.mean(sample.capture_ms for sample in samples),
            preprocessing_ms=self._average_optional(samples, "preprocessing_ms"),
            inference_ms=self._average_optional(samples, "inference_ms"),
            postprocessing_ms=self._average_optional(samples, "postprocessing_ms"),
            total_frame_ms=statistics.mean(total_frame_times),
            median_frame_ms=statistics.median(total_frame_times),
            p95_frame_ms=_percentile(total_frame_times, 95.0),
            dropped_frames=latest.dropped_frames,
            active_provider=latest.active_provider,
            capture_dimensions=latest.capture_dimensions,
            ai_input_dimensions=latest.ai_input_dimensions,
            ai_output_dimensions=latest.ai_output_dimensions,
            tile_mode=latest.tile_mode,
            sample_count=len(samples),
            interpolation_ms=self._average_optional(
                samples,
                "interpolation_ms",
            ),
            interpolation_provider=latest.interpolation_provider,
            frame_generation=latest.frame_generation,
            dropped_generated_frames=latest.dropped_generated_frames,
            scheduled_presentation_timestamp=latest.scheduled_presentation_timestamp,
            actual_presentation_timestamp=latest.actual_presentation_timestamp,
            pacing_error_ms=latest.pacing_error_ms,
            median_pacing_error_ms=statistics.median(pacing_errors),
            p95_pacing_error_ms=_percentile(pacing_errors, 95.0),
            late_frames=latest.late_frames,
            generated_frames_dropped_late=latest.generated_frames_dropped_late,
            real_frames_dropped_late=latest.real_frames_dropped_late,
            estimated_source_fps=latest.estimated_source_fps,
            presentation_fps=latest.presentation_fps or fps,
            runtime_state=latest.runtime_state,
            recovery_retry_attempts=latest.recovery_retry_attempts,
            successful_recoveries=latest.successful_recoveries,
            failed_recoveries=latest.failed_recoveries,
            fallback_activations=latest.fallback_activations,
            presented_real_frames=latest.presented_real_frames,
            presented_generated_frames=latest.presented_generated_frames,
            presented_frames=latest.presented_frames,
            generated_frames_requested=latest.generated_frames_requested,
        )

    def maybe_log(
        self,
        target_logger: logging.Logger,
        *,
        now: float | None = None,
    ) -> bool:
        """Log the requested five-second aggregate and return whether it logged."""

        if not self.enabled:
            return False
        current_time = self._clock() if now is None else now
        if current_time - self._last_log_at < self._log_interval_seconds:
            return False
        snapshot = self.snapshot()
        self._last_log_at = current_time
        if snapshot is None:
            return False
        target_logger.info(
            "Performance (last %s frames): average FPS %.2f, median frame %.2f ms, "
            "p95 frame %.2f ms, dropped frames %s, pacing error %.2f ms, "
            "scheduled %.6f, actual %.6f, late %s, generated-late drops %s, "
            "real-late drops %s, source %.2f FPS, presentation %.2f FPS, state %s, "
            "stages capture %.2f ms, processing %.2f ms, RIFE %.2f ms, "
            "retries %s, recoveries %s, recovery failures %s, fallbacks %s.",
            snapshot.sample_count,
            snapshot.fps,
            snapshot.median_frame_ms,
            snapshot.p95_frame_ms,
            snapshot.dropped_frames,
            snapshot.pacing_error_ms,
            snapshot.scheduled_presentation_timestamp,
            snapshot.actual_presentation_timestamp,
            snapshot.late_frames,
            snapshot.generated_frames_dropped_late,
            snapshot.real_frames_dropped_late,
            snapshot.estimated_source_fps,
            snapshot.presentation_fps,
            snapshot.runtime_state,
            snapshot.capture_ms,
            snapshot.preprocessing_ms + snapshot.inference_ms + snapshot.postprocessing_ms,
            snapshot.interpolation_ms,
            snapshot.recovery_retry_attempts,
            snapshot.successful_recoveries,
            snapshot.failed_recoveries,
            snapshot.fallback_activations,
        )
        return True

    def update(
        self,
        pipeline_fps: float,
        capture_ms: float,
        upscale_ms: float,
        frame_age_ms: float = 0.0,
        end_to_end_ms: float = 0.0,
        dropped_frames: int = 0,
    ) -> None:
        """Retain the original snapshot setter for compatibility."""

        self.pipeline_fps = pipeline_fps
        self.capture_ms = capture_ms
        self.upscale_ms = upscale_ms
        self.frame_age_ms = frame_age_ms
        self.end_to_end_ms = end_to_end_ms
        self.dropped_frames = dropped_frames
