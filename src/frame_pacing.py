"""Reusable low-latency presentation scheduling.

The pacer owns only timing policy.  Capture, processing, and rendering remain
independent, and callers retain control of the actual presentation operation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
import threading
import time
from typing import Any, Callable, Iterable


PACING_MODES = ("auto", "off", "fixed")
DEFAULT_MAX_FRAME_LATENCY_MS = 100.0


@dataclass(frozen=True, slots=True)
class PresentationFrame:
    payload: Any
    captured_at: float
    frame_kind: str = "real"


@dataclass(frozen=True, slots=True)
class PacingDecision:
    frame: PresentationFrame
    present: bool
    scheduled_at: float
    actual_at: float
    pacing_error_ms: float
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PacingSnapshot:
    scheduled_presentation_timestamp: float = 0.0
    actual_presentation_timestamp: float = 0.0
    pacing_error_ms: float = 0.0
    late_frames: int = 0
    generated_frames_dropped_late: int = 0
    real_frames_dropped_late: int = 0
    estimated_source_fps: float = 0.0
    presentation_fps: float = 0.0
    presented_frames: int = 0
    presented_real_frames: int = 0
    presented_generated_frames: int = 0


class SourceRateEstimator:
    """Estimate a stable source cadence from monotonic capture timestamps."""

    def __init__(
        self,
        *,
        window_size: int = 15,
        minimum_samples: int = 5,
        reset_multiplier: float = 6.0,
    ) -> None:
        if window_size < minimum_samples or minimum_samples < 2:
            raise ValueError("Source-rate estimator window is too small.")
        self._timestamps: deque[float] = deque(maxlen=window_size + 1)
        self._minimum_samples = minimum_samples
        self._reset_multiplier = float(reset_multiplier)

    def add(self, timestamp: float) -> None:
        if self._timestamps and timestamp <= self._timestamps[-1]:
            return
        if len(self._timestamps) >= 3:
            intervals = self._intervals()
            median = statistics.median(intervals)
            if median > 0.0 and timestamp - self._timestamps[-1] > (
                median * self._reset_multiplier
            ):
                self._timestamps.clear()
        self._timestamps.append(float(timestamp))

    @property
    def fps(self) -> float:
        intervals = self._intervals()
        if not intervals:
            return 0.0
        median = statistics.median(intervals)
        return 1.0 / median if median > 0.0 else 0.0

    @property
    def reliable(self) -> bool:
        intervals = self._intervals()
        if len(intervals) < self._minimum_samples:
            return False
        mean = statistics.mean(intervals)
        if mean <= 0.0:
            return False
        # A deliberately conservative gate keeps irregular or bursty capture
        # uncapped instead of inventing a source rate.
        return statistics.pstdev(intervals) / mean <= 0.20

    def _intervals(self) -> list[float]:
        return [
            current - previous
            for previous, current in zip(self._timestamps, list(self._timestamps)[1:])
            if current > previous
        ]


class FramePacer:
    """Schedule newest presentation batches without busy waiting."""

    def __init__(
        self,
        *,
        mode: str = "auto",
        target_fps: float | None = None,
        max_frame_latency_ms: float = DEFAULT_MAX_FRAME_LATENCY_MS,
        clock: Callable[[], float] = time.perf_counter,
        shutdown_event: threading.Event | None = None,
        waiter: Callable[[float], bool] | None = None,
        keep_latest_real: bool = False,
        timestamp_tolerance: float = 0.05,
        queue_draining_momentum: float = 0.0,
    ) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in PACING_MODES:
            raise ValueError(f"Frame pacing must be one of: {', '.join(PACING_MODES)}.")
        if target_fps is not None and (
            not math.isfinite(float(target_fps)) or float(target_fps) <= 0.0
        ):
            raise ValueError("Target FPS must be a positive finite value.")
        if not math.isfinite(float(max_frame_latency_ms)) or max_frame_latency_ms <= 0.0:
            raise ValueError("Maximum frame latency must be a positive finite value.")
        if normalized_mode == "fixed" and target_fps is None:
            raise ValueError("Fixed frame pacing requires --target-fps.")
        if not 0.0 <= float(timestamp_tolerance) <= 0.25:
            raise ValueError("Timestamp tolerance must be between 0 and 0.25.")
        if not 0.0 <= float(queue_draining_momentum) <= 0.10:
            raise ValueError("Queue-draining momentum must be between 0 and 0.10.")

        self.keep_latest_real = keep_latest_real
        self.mode = normalized_mode
        self.target_fps = None if target_fps is None else float(target_fps)
        self.max_frame_latency_ms = float(max_frame_latency_ms)
        self._max_latency_seconds = self.max_frame_latency_ms / 1000.0
        self._clock = clock
        self._shutdown_event = shutdown_event or threading.Event()
        self._waiter = waiter or self._shutdown_event.wait
        self._source_rate = SourceRateEstimator()
        self.timestamp_tolerance = float(timestamp_tolerance)
        self.queue_draining_momentum = float(queue_draining_momentum)
        self._drain_debt_seconds = 0.0
        self._last_scheduled: float | None = None
        self._presented_at: deque[float] = deque(maxlen=120)
        self._scheduled_at = 0.0
        self._actual_at = 0.0
        self._error_ms = 0.0
        self._late_frames = 0
        self._generated_drops = 0
        self._real_drops = 0
        self._presented_frames = 0
        self._presented_real_frames = 0
        self._presented_generated_frames = 0

    @property
    def shutdown_event(self) -> threading.Event:
        return self._shutdown_event

    def stop(self) -> None:
        self._shutdown_event.set()

    def _base_interval(self, *, batch_size: int) -> float | None:
        if self.mode == "off":
            return None
        if self.target_fps is not None:
            return 1.0 / self.target_fps
        if not self._source_rate.reliable:
            return None
        source_interval = 1.0 / self._source_rate.fps
        return source_interval / max(1, batch_size)

    def _interval(self, *, batch_size: int) -> tuple[float | None, float | None]:
        base = self._base_interval(batch_size=batch_size)
        if base is None:
            return None, None
        if self._drain_debt_seconds <= 0.0 or self.queue_draining_momentum <= 0.0:
            return base, base
        return base * (1.0 - self.queue_draining_momentum), base

    def pace_batch(self, frames: Iterable[PresentationFrame]) -> list[PacingDecision]:
        """Collect decisions for callers that do not render live output."""
        return list(self.iter_pace_batch(frames))

    def iter_pace_batch(
        self,
        frames: Iterable[PresentationFrame],
    ) -> Iterable[PacingDecision]:
        """Yield each decision at its presentation slot; render before advancing.

        Pace one newest ordered batch, normally ``[midpoint, real]``."""

        batch = list(frames)
        for frame in batch:
            if frame.frame_kind != "generated":
                self._source_rate.add(frame.captured_at)
        interval, base_interval = self._interval(batch_size=len(batch))
        lateness_tolerance = max(
            0.0005,
            (base_interval or 0.0) * self.timestamp_tolerance,
        )

        for index, frame in enumerate(batch):
            now = self._clock()
            if interval is None or self._last_scheduled is None:
                scheduled = now
            else:
                scheduled = self._last_scheduled + interval

            # Interpolation needs the following real frame before its generated
            # predecessors exist.  If processing completed after the old
            # cadence slot, begin this newest batch now instead of discarding
            # otherwise usable generated output.  The latency check below
            # still prevents generated frames from delaying the real frame
            # beyond its configured budget.
            if (
                index == 0
                and frame.frame_kind == "generated"
                and interval is not None
                and now - scheduled > lateness_tolerance
            ):
                self._drain_debt_seconds = min(
                    self._max_latency_seconds,
                    self._drain_debt_seconds + max(0.0, now - scheduled),
                )
                scheduled = now

            # A generated midpoint never gets to hold a ready real frame past
            # its own following presentation slot or latency budget.
            if frame.frame_kind == "generated":
                following_real = next(
                    (candidate for candidate in batch[index + 1 :] if candidate.frame_kind != "generated"),
                    None,
                )
                real_deadline = (
                    frame.captured_at + self._max_latency_seconds
                    if following_real is None
                    else following_real.captured_at + self._max_latency_seconds
                )
                next_real_slot = scheduled + (interval or 0.0)
                timing_late = interval is not None and (
                    now - scheduled > lateness_tolerance
                    or now >= scheduled + interval
                )
                if timing_late or next_real_slot >= real_deadline:
                    yield self._drop(frame, scheduled, now, "generated_late")
                    continue

            preserve_real = self.keep_latest_real and frame.frame_kind != "generated"
            age = now - frame.captured_at
            if age > self._max_latency_seconds and not preserve_real:
                yield self._drop(frame, scheduled, now, "maximum_latency")
                continue

            # Latency wins over cadence.  Present a real frame immediately when
            # waiting for its nominal slot would exceed the configured budget.
            wait_until = scheduled
            if frame.frame_kind != "generated" and scheduled - frame.captured_at > self._max_latency_seconds:
                wait_until = now
            delay = max(0.0, wait_until - now)
            if delay > 0.0 and self._waiter(delay):
                yield self._drop(frame, scheduled, self._clock(), "shutdown")
                continue

            actual = self._clock()
            if actual - frame.captured_at > self._max_latency_seconds and not preserve_real:
                yield self._drop(frame, scheduled, actual, "maximum_latency")
                continue
            error_ms = (actual - scheduled) * 1000.0
            if error_ms > 0.5:
                self._late_frames += 1
            self._record_presented(frame, scheduled, actual, error_ms)
            if interval is not None and base_interval is not None:
                self._drain_debt_seconds = max(
                    0.0,
                    self._drain_debt_seconds - max(0.0, base_interval - interval),
                )

            # Re-anchor after a seriously late real frame so the pacer never
            # emits a catch-up burst of stale presentation slots.
            if frame.frame_kind != "generated" and interval is not None and actual - scheduled > interval:
                self._last_scheduled = actual
            else:
                self._last_scheduled = scheduled

            yield PacingDecision(frame, True, scheduled, actual, error_ms)

    def _drop(
        self,
        frame: PresentationFrame,
        scheduled: float,
        actual: float,
        reason: str,
    ) -> PacingDecision:
        if frame.frame_kind == "generated":
            self._generated_drops += 1
        else:
            self._real_drops += 1
        return PacingDecision(
            frame=frame,
            present=False,
            scheduled_at=scheduled,
            actual_at=actual,
            pacing_error_ms=(actual - scheduled) * 1000.0,
            reason=reason,
        )

    def _record_presented(
        self,
        frame: PresentationFrame,
        scheduled: float,
        actual: float,
        error_ms: float,
    ) -> None:
        self._scheduled_at = scheduled
        self._actual_at = actual
        self._error_ms = error_ms
        self._presented_at.append(actual)
        self._presented_frames += 1
        if frame.frame_kind == "generated":
            self._presented_generated_frames += 1
        else:
            self._presented_real_frames += 1

    def snapshot(self) -> PacingSnapshot:
        fps = 0.0
        if len(self._presented_at) >= 2:
            elapsed = self._presented_at[-1] - self._presented_at[0]
            if elapsed > 0.0:
                fps = (len(self._presented_at) - 1) / elapsed
        return PacingSnapshot(
            scheduled_presentation_timestamp=self._scheduled_at,
            actual_presentation_timestamp=self._actual_at,
            pacing_error_ms=self._error_ms,
            late_frames=self._late_frames,
            generated_frames_dropped_late=self._generated_drops,
            real_frames_dropped_late=self._real_drops,
            estimated_source_fps=self._source_rate.fps,
            presentation_fps=fps,
            presented_frames=self._presented_frames,
            presented_real_frames=self._presented_real_frames,
            presented_generated_frames=self._presented_generated_frames,
        )
