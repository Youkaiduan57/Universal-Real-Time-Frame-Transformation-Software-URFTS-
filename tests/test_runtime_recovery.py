from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from runtime_recovery import (
    RecoveringCapture,
    RecoveringComponent,
    RecoveringInterpolator,
    RecoveringProcessor,
    RecoveryController,
    RecoveryExhausted,
    RecoveryInterrupted,
    RetryPolicy,
    RuntimeState,
)


def policy(attempts: int = 3) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=attempts,
        initial_backoff_seconds=0.0,
        maximum_backoff_seconds=0.0,
    )


class Capture:
    backend_name = "fake"
    capture_region = None

    def __init__(self, values) -> None:
        self.values = iter(values)
        self.closed = 0

    def grab_frame(self):
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self):
        self.closed += 1


def test_temporary_capture_failure_recreates_and_recovers() -> None:
    good = np.zeros((2, 3, 3), dtype=np.uint8)
    initial = Capture([RuntimeError("temporary")])
    created = []
    recovering = RecoveringCapture(
        initial,
        lambda: created.append(Capture([good])) or created[-1],
        RecoveryController(policy=policy()),
    )
    assert recovering.grab_frame() is good
    assert initial.closed == 1
    assert recovering.controller.snapshot().successful_recoveries == 1


def test_capture_none_and_minimized_zero_size_are_source_unavailable_then_recover() -> None:
    good = np.zeros((2, 3, 3), dtype=np.uint8)
    replacements = iter(
        [Capture([np.zeros((0, 0, 3), dtype=np.uint8)]), Capture([good])]
    )
    recovering = RecoveringCapture(
        Capture([None]),
        lambda: next(replacements),
        RecoveryController(policy=policy()),
    )
    assert recovering.grab_frame() is good
    assert recovering.controller.snapshot().retry_attempts == 2


def test_invalid_closed_source_hits_retry_limit() -> None:
    controller = RecoveryController(policy=policy(2))
    recovering = RecoveringCapture(
        Capture([RuntimeError("window closed")]),
        lambda: Capture([RuntimeError("window closed")]),
        controller,
    )
    with pytest.raises(RecoveryExhausted):
        recovering.grab_frame()
    assert controller.state is RuntimeState.FAILED


def test_resolution_change_is_detected_without_restart() -> None:
    capture = Capture(
        [
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.zeros((4, 5, 3), dtype=np.uint8),
        ]
    )
    recovering = RecoveringCapture(
        capture,
        lambda: capture,
        RecoveryController(policy=policy()),
    )
    recovering.grab_frame()
    recovering.grab_frame()
    assert recovering.resolution_changes == 1


class Processor:
    active_providers = ("DmlExecutionProvider",)

    def __init__(self, *, error=None, value="processed") -> None:
        self.error = error
        self.value = value
        self.closed = 0

    def process(self, frame):
        if self.error:
            raise self.error
        return self.value

    def shutdown(self):
        self.closed += 1


def test_temporary_inference_failure_recreates_session() -> None:
    initial = Processor(error=RuntimeError("ORT inference failed"))
    recovering = RecoveringProcessor(
        initial,
        lambda: Processor(value="retried"),
        RecoveryController(policy=policy()),
        category="inference",
    )
    assert recovering.process(object()) == "retried"
    assert initial.closed == 1


def test_repeated_directml_failure_without_fallback_is_visible() -> None:
    recovering = RecoveringProcessor(
        Processor(error=RuntimeError("DML device failure")),
        lambda: Processor(error=RuntimeError("DML device failure")),
        RecoveryController(policy=policy(2)),
        category="inference_provider",
    )
    with pytest.raises(RecoveryExhausted):
        recovering.process(object())
    assert recovering.controller.snapshot().fallback_activations == 0


def test_explicit_provider_fallback_activates_cpu() -> None:
    recovering = RecoveringProcessor(
        Processor(error=RuntimeError("DML device failure")),
        lambda: Processor(error=RuntimeError("DML device failure")),
        RecoveryController(policy=policy(2)),
        category="inference_provider",
        fallback_factory=lambda: Processor(value="cpu"),
    )
    assert recovering.process(object()) == "cpu"
    snapshot = recovering.controller.snapshot()
    assert snapshot.fallback_activations == 1
    assert snapshot.state == "degraded"


class Interpolator:
    produces_intermediate_frame = True
    active_providers = ("DmlExecutionProvider",)
    last_inference_ms = 1.0

    def __init__(self, fail=False) -> None:
        self.fail = fail

    def interpolate(self, a, b):
        if self.fail:
            raise RuntimeError("RIFE failed")
        return "middle"

    def shutdown(self):
        pass


def test_interpolation_degrades_then_retries_while_real_path_can_continue() -> None:
    now = [1.0]
    wrapper = RecoveringInterpolator(
        Interpolator(fail=True),
        lambda: Interpolator(),
        RecoveryController(policy=policy()),
        clock=lambda: now[0],
    )
    with pytest.raises(RuntimeError):
        wrapper.interpolate("A", "B")
    assert wrapper.controller.state is RuntimeState.DEGRADED
    assert wrapper.interpolate("B", "C") == "middle"


def test_renderer_component_is_recreated_after_device_failure() -> None:
    class Renderer:
        def __init__(self, fail=False):
            self.fail = fail
            self.closed = 0

        def run(self):
            if self.fail:
                raise RuntimeError("device lost")
            return "complete"

        def close(self):
            self.closed += 1

    initial = Renderer(fail=True)
    wrapper = RecoveringComponent(
        initial,
        lambda: Renderer(),
        RecoveryController(policy=policy()),
        category="renderer",
    )
    assert wrapper.call("run") == "complete"
    assert initial.closed == 1


def test_backoff_is_interruptible_and_does_not_deadlock() -> None:
    shutdown = threading.Event()
    controller = RecoveryController(
        policy=RetryPolicy(3, 5.0, 5.0),
        shutdown_event=shutdown,
    )
    errors = []
    worker = threading.Thread(
        target=lambda: _capture_error(
            errors,
            lambda: controller.recover(
                "capture",
                lambda: None,
                initial_error=RuntimeError("lost"),
            ),
        )
    )
    worker.start()
    time.sleep(0.02)
    controller.stop()
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert errors and isinstance(errors[0], RecoveryInterrupted)


def _capture_error(target, operation) -> None:
    try:
        operation()
    except BaseException as error:
        target.append(error)


def test_queue_clear_callback_runs_once_per_recovery_and_shutdown_is_idempotent() -> None:
    clears = []
    controller = RecoveryController(policy=policy(), on_recovery=lambda: clears.append(1))
    result = controller.recover(
        "capture",
        lambda: "ok",
        initial_error=RuntimeError("temporary"),
    )
    assert result == "ok"
    assert clears == [1]
    assert controller.snapshot().queue_clears == 1
    controller.stop()
    controller.stop()
    controller.mark_stopped()
    controller.mark_stopped()
    assert controller.state is RuntimeState.STOPPED


def test_nonretryable_configuration_error_is_not_restarted() -> None:
    calls = []
    controller = RecoveryController(policy=policy())
    with pytest.raises(ValueError):
        controller.recover(
            "configuration",
            lambda: calls.append(1),
            initial_error=ValueError("invalid model"),
            retryable=lambda error: not isinstance(error, ValueError),
        )
    assert calls == []
