"""Focused tests for optional moving-window performance telemetry."""

from __future__ import annotations

import logging

import pytest

from metrics import PerformanceMetrics


def _record(metrics: PerformanceMetrics, index: int, total_ms: float) -> None:
    metrics.record(
        presented_at=float(index),
        capture_ms=float(index),
        preprocessing_ms=1.0,
        inference_ms=2.0,
        postprocessing_ms=3.0,
        total_frame_ms=total_ms,
        dropped_frames=index,
        active_provider="DmlExecutionProvider",
        capture_dimensions=(640, 360),
        ai_input_dimensions=(320, 180),
        ai_output_dimensions=(640, 360),
        tile_mode="auto (256px, 2 tiles)",
    )


def test_disabled_telemetry_does_not_retain_samples_or_log(caplog) -> None:
    metrics = PerformanceMetrics(enabled=False)

    _record(metrics, 1, 10.0)

    with caplog.at_level(logging.INFO):
        logged = metrics.maybe_log(logging.getLogger("telemetry-test"), now=10.0)
    assert metrics.sample_count == 0
    assert metrics.snapshot() is None
    assert logged is False
    assert caplog.text == ""


def test_enabled_telemetry_maintains_bounded_moving_averages() -> None:
    metrics = PerformanceMetrics(
        enabled=True,
        window_size=3,
        clock=lambda: 0.0,
    )
    for index, total_ms in enumerate((10.0, 20.0, 30.0, 40.0)):
        _record(metrics, index, total_ms)

    snapshot = metrics.snapshot()

    assert snapshot is not None
    assert snapshot.sample_count == 3
    assert snapshot.fps == pytest.approx(1.0)
    assert snapshot.capture_ms == pytest.approx(2.0)
    assert snapshot.preprocessing_ms == pytest.approx(1.0)
    assert snapshot.inference_ms == pytest.approx(2.0)
    assert snapshot.postprocessing_ms == pytest.approx(3.0)
    assert snapshot.total_frame_ms == pytest.approx(30.0)
    assert snapshot.median_frame_ms == pytest.approx(30.0)
    assert snapshot.p95_frame_ms == pytest.approx(40.0)
    assert snapshot.dropped_frames == 3
    assert snapshot.active_provider == "DmlExecutionProvider"
    assert snapshot.capture_dimensions == (640, 360)
    assert snapshot.ai_input_dimensions == (320, 180)
    assert snapshot.ai_output_dimensions == (640, 360)
    assert snapshot.tile_mode == "auto (256px, 2 tiles)"


def test_enabled_telemetry_logs_five_second_summary(caplog) -> None:
    metrics = PerformanceMetrics(
        enabled=True,
        log_interval_seconds=5.0,
        clock=lambda: 0.0,
    )
    _record(metrics, 0, 10.0)
    _record(metrics, 1, 20.0)
    target_logger = logging.getLogger("telemetry-summary")

    with caplog.at_level(logging.INFO):
        assert metrics.maybe_log(target_logger, now=4.9) is False
        assert metrics.maybe_log(target_logger, now=5.0) is True

    assert "average FPS 1.00" in caplog.text
    assert "median frame 15.00 ms" in caplog.text
    assert "p95 frame 20.00 ms" in caplog.text
    assert "dropped frames 1" in caplog.text


def test_frame_generation_telemetry_tracks_interpolation_and_drops() -> None:
    metrics = PerformanceMetrics(enabled=True, clock=lambda: 0.0)
    metrics.record(
        presented_at=1.0,
        capture_ms=1.0,
        preprocessing_ms=2.0,
        inference_ms=3.0,
        postprocessing_ms=4.0,
        total_frame_ms=20.0,
        dropped_frames=2,
        active_provider="OpenCV CPU",
        capture_dimensions=(128, 72),
        ai_input_dimensions=(128, 72),
        ai_output_dimensions=(128, 72),
        tile_mode="off",
        interpolation_ms=7.5,
        interpolation_provider="DmlExecutionProvider",
        frame_generation="rife",
        dropped_generated_frames=1,
        presented_real_frames=30,
        presented_generated_frames=120,
        presented_frames=150,
        generated_frames_requested=4,
    )

    snapshot = metrics.snapshot()

    assert snapshot is not None
    assert snapshot.interpolation_ms == pytest.approx(7.5)
    assert snapshot.interpolation_provider == "DmlExecutionProvider"
    assert snapshot.frame_generation == "rife"
    assert snapshot.dropped_generated_frames == 1
    assert snapshot.presented_real_frames == 30
    assert snapshot.presented_generated_frames == 120
    assert snapshot.presented_frames == 150
    assert snapshot.generated_frames_requested == 4
