"""Bounded, interruptible live-runtime recovery primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import threading
import time
from typing import Any, Callable, TypeVar


logger = logging.getLogger(__name__)
T = TypeVar("T")


class RuntimeState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    SOURCE_UNAVAILABLE = "source_unavailable"
    RECOVERING = "recovering"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class RecoveryError(RuntimeError):
    pass


class RecoveryInterrupted(RecoveryError):
    pass


class RecoveryExhausted(RecoveryError):
    def __init__(self, category: str, attempts: int, cause: BaseException) -> None:
        self.category = category
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"{category} recovery failed after {attempts} attempts: {cause}"
        )


class SourceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_backoff_seconds: float = 0.05
    maximum_backoff_seconds: float = 1.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("Recovery retry limit must be greater than zero.")
        if self.initial_backoff_seconds < 0.0:
            raise ValueError("Initial recovery backoff cannot be negative.")
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("Maximum recovery backoff cannot be below the initial backoff.")
        if self.multiplier < 1.0:
            raise ValueError("Recovery backoff multiplier must be at least one.")

    def backoff(self, attempt: int) -> float:
        return min(
            self.maximum_backoff_seconds,
            self.initial_backoff_seconds * self.multiplier ** max(0, attempt - 1),
        )


@dataclass(frozen=True, slots=True)
class RecoverySnapshot:
    state: str
    retry_attempts: int
    successful_recoveries: int
    failed_recoveries: int
    fallback_activations: int
    queue_clears: int
    last_failure_category: str


class RecoveryController:
    """Coordinate state, counters, bounded retries, and shutdown wakeups."""

    def __init__(
        self,
        *,
        policy: RetryPolicy | None = None,
        shutdown_event: threading.Event | None = None,
        on_recovery: Callable[[], None] | None = None,
        state_callback: Callable[[RuntimeState], None] | None = None,
        target_logger: logging.Logger = logger,
    ) -> None:
        self.policy = policy or RetryPolicy()
        self.shutdown_event = shutdown_event or threading.Event()
        self._on_recovery = on_recovery
        self._state_callback = state_callback
        self._logger = target_logger
        self._lock = threading.RLock()
        self._state = RuntimeState.STARTING
        self._retry_attempts = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0
        self._fallback_activations = 0
        self._queue_clears = 0
        self._last_failure_category = "none"

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    def set_on_recovery(self, callback: Callable[[], None] | None) -> None:
        with self._lock:
            self._on_recovery = callback

    def transition(self, state: RuntimeState) -> None:
        with self._lock:
            if self._state == RuntimeState.STOPPED:
                return
            self._state = state
        self._notify_state(state)

    def _notify_state(self, state: RuntimeState) -> None:
        if self._state_callback is not None:
            self._state_callback(state)

    def mark_running(self) -> None:
        self.transition(RuntimeState.RUNNING)

    def mark_degraded(self, category: str, error: BaseException) -> None:
        with self._lock:
            self._last_failure_category = category
            self._state = RuntimeState.DEGRADED
        self._notify_state(RuntimeState.DEGRADED)
        self._logger.warning(
            "runtime_failure category=%s state=degraded error=%s",
            category,
            error,
        )

    def clear_stale_queue(self) -> None:
        with self._lock:
            callback = self._on_recovery
        if callback is not None:
            callback()
            with self._lock:
                self._queue_clears += 1

    def mark_fallback(self, category: str) -> None:
        with self._lock:
            self._fallback_activations += 1
            self._successful_recoveries += 1
            self._last_failure_category = category
            self._state = RuntimeState.DEGRADED
        self._notify_state(RuntimeState.DEGRADED)
        self._logger.warning(
            "runtime_recovery category=%s fallback_activated=true",
            category,
        )

    def recover(
        self,
        category: str,
        operation: Callable[[], T],
        *,
        initial_error: BaseException,
        retryable: Callable[[BaseException], bool] | None = None,
        fallback: Callable[[], T] | None = None,
        source_unavailable: bool = False,
    ) -> T:
        """Run bounded retries, optionally activating one explicit fallback."""

        predicate = retryable or (lambda error: True)
        if not predicate(initial_error):
            with self._lock:
                self._state = RuntimeState.FAILED
                self._failed_recoveries += 1
                self._last_failure_category = category
            self._notify_state(RuntimeState.FAILED)
            self._logger.error(
                "runtime_failure category=%s retryable=false final=true error=%s",
                category,
                initial_error,
            )
            raise initial_error

        with self._lock:
            self._state = (
                RuntimeState.SOURCE_UNAVAILABLE
                if source_unavailable
                else RuntimeState.RECOVERING
            )
            self._last_failure_category = category
            callback = self._on_recovery
            recovering_state = self._state
        self._notify_state(recovering_state)
        if callback is not None:
            self.clear_stale_queue()

        last_error: BaseException = initial_error
        for attempt in range(1, self.policy.max_attempts + 1):
            delay = self.policy.backoff(attempt)
            with self._lock:
                self._retry_attempts += 1
            self._logger.warning(
                "runtime_recovery category=%s attempt=%s/%s backoff_seconds=%.3f error=%s",
                category,
                attempt,
                self.policy.max_attempts,
                delay,
                last_error,
            )
            if delay > 0.0 and self.shutdown_event.wait(delay):
                raise RecoveryInterrupted(
                    f"Shutdown interrupted {category} recovery backoff."
                )
            if self.shutdown_event.is_set():
                raise RecoveryInterrupted(f"Shutdown interrupted {category} recovery.")
            try:
                result = operation()
            except Exception as error:
                last_error = error
                if not predicate(error):
                    break
            else:
                with self._lock:
                    self._successful_recoveries += 1
                    self._state = RuntimeState.RUNNING
                self._notify_state(RuntimeState.RUNNING)
                self._logger.info(
                    "runtime_recovery category=%s success=true attempt=%s",
                    category,
                    attempt,
                )
                return result

        if fallback is not None and not self.shutdown_event.is_set():
            try:
                result = fallback()
            except Exception as error:
                last_error = error
            else:
                with self._lock:
                    self._fallback_activations += 1
                    self._successful_recoveries += 1
                    self._state = RuntimeState.DEGRADED
                self._notify_state(RuntimeState.DEGRADED)
                self._logger.warning(
                    "runtime_recovery category=%s fallback_activated=true",
                    category,
                )
                return result

        with self._lock:
            self._failed_recoveries += 1
            self._state = RuntimeState.FAILED
        self._notify_state(RuntimeState.FAILED)
        self._logger.error(
            "runtime_recovery category=%s final=true attempts=%s error=%s",
            category,
            self.policy.max_attempts,
            last_error,
        )
        raise RecoveryExhausted(category, self.policy.max_attempts, last_error)

    def stop(self) -> None:
        with self._lock:
            if self._state == RuntimeState.STOPPED:
                return
            self._state = RuntimeState.STOPPING
        self._notify_state(RuntimeState.STOPPING)
        self.shutdown_event.set()

    def mark_stopped(self) -> None:
        self.shutdown_event.set()
        with self._lock:
            self._state = RuntimeState.STOPPED
        self._notify_state(RuntimeState.STOPPED)

    def snapshot(self) -> RecoverySnapshot:
        with self._lock:
            return RecoverySnapshot(
                state=self._state.value,
                retry_attempts=self._retry_attempts,
                successful_recoveries=self._successful_recoveries,
                failed_recoveries=self._failed_recoveries,
                fallback_activations=self._fallback_activations,
                queue_clears=self._queue_clears,
                last_failure_category=self._last_failure_category,
            )


def _close_component(component: Any) -> None:
    closer = getattr(component, "shutdown", None) or getattr(component, "close", None)
    if callable(closer):
        closer()


class RecoveringComponent:
    """Recreate a failed capture, processor, or renderer within one boundary."""

    def __init__(
        self,
        component: Any,
        factory: Callable[[], Any],
        controller: RecoveryController,
        *,
        category: str,
        fallback_factory: Callable[[], Any] | None = None,
        retryable: Callable[[BaseException], bool] | None = None,
    ) -> None:
        self.component = component
        self.factory = factory
        self.controller = controller
        self.category = category
        self.fallback_factory = fallback_factory
        self.retryable = retryable
        self._closed = False
        self._lock = threading.RLock()

    def _replace(self, factory: Callable[[], Any]) -> Any:
        with self._lock:
            old = self.component
            _close_component(old)
            self.component = factory()
            return self.component

    def call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self.component, method_name)(*args, **kwargs)
        except Exception as initial_error:
            def retry_operation() -> Any:
                component = self._replace(self.factory)
                return getattr(component, method_name)(*args, **kwargs)

            fallback = None
            if self.fallback_factory is not None:
                def fallback_operation() -> Any:
                    component = self._replace(self.fallback_factory)
                    self.factory = self.fallback_factory
                    return getattr(component, method_name)(*args, **kwargs)
                fallback = fallback_operation

            return self.controller.recover(
                self.category,
                retry_operation,
                initial_error=initial_error,
                retryable=self.retryable,
                fallback=fallback,
                source_unavailable=self.category == "capture",
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            _close_component(self.component)


class RecoveringCapture(RecoveringComponent):
    """Capture boundary with temporary-unavailability and resize detection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._pre_grab = kwargs.pop("pre_grab", None)
        super().__init__(*args, category="capture", **kwargs)
        self._backend_name = str(getattr(self.component, "backend_name", "unknown"))
        self._capture_region = getattr(self.component, "capture_region", None)
        self._dimensions: tuple[int, int] | None = None
        self.resolution_changes = 0

    @property
    def backend_name(self) -> str:
        return str(getattr(self.component, "backend_name", self._backend_name))

    @property
    def capture_region(self) -> Any:
        return getattr(self.component, "capture_region", self._capture_region)

    def set_capture_region(self, region: Any) -> None:
        self._capture_region = region
        if self.component is not None:
            self.component.set_capture_region(region)

    def prepare_thread_handoff(self) -> None:
        """Release an initial probe so a worker can recreate thread-bound capture."""

        with self._lock:
            if self.component is None:
                return
            self._backend_name = self.backend_name
            self._capture_region = self.capture_region
            _close_component(self.component)
            self.component = None

    def _ensure_component(self) -> Any:
        with self._lock:
            if self.component is None:
                self.component = self.factory()
                self._backend_name = str(getattr(self.component, "backend_name", self._backend_name))
            return self.component

    @staticmethod
    def _validate_frame(frame: Any) -> Any:
        if frame is None:
            raise SourceUnavailableError("Capture temporarily returned no frame.")
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2 and (shape[0] <= 0 or shape[1] <= 0):
            raise SourceUnavailableError("Capture source is minimized or zero-sized.")
        return frame

    def grab_frame(self) -> Any:
        try:
            frame = self._grab_component(self._ensure_component())
        except Exception as initial_error:
            def retry_grab() -> Any:
                return self._grab_component(self._replace(self.factory))
            frame = self.controller.recover(
                "capture",
                retry_grab,
                initial_error=initial_error,
                retryable=self.retryable,
                source_unavailable=True,
            )
        shape = getattr(frame, "shape", None)
        dimensions = None if shape is None or len(shape) < 2 else (int(shape[1]), int(shape[0]))
        if dimensions is not None and self._dimensions is not None and dimensions != self._dimensions:
            self.resolution_changes += 1
            self.controller.clear_stale_queue()
            logger.info(
                "runtime_recovery category=capture resolution_change=%sx%s->%sx%s",
                *self._dimensions,
                *dimensions,
            )
        self._dimensions = dimensions
        return frame

    def _grab_component(self, component: Any) -> Any:
        if self._pre_grab is not None:
            self._pre_grab(component)
        return self._validate_frame(component.grab_frame())


class RecoveringProcessor(RecoveringComponent):
    def process(self, frame: Any) -> Any:
        return self.call("process", frame)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.component, name)


class RecoveringInterpolator:
    """Degrade to real frames and retry interpolation without blocking them."""

    produces_intermediate_frame = True

    def __init__(
        self,
        component: Any,
        factory: Callable[[], Any],
        controller: RecoveryController,
        *,
        clock: Callable[[], float] = time.perf_counter,
        fallback_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.component = component
        self.factory = factory
        self.controller = controller
        self._clock = clock
        self._fallback_factory = fallback_factory
        self._attempt = 0
        self._next_retry_at = 0.0
        self._degraded = False
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.component, name)

    @property
    def active_providers(self) -> tuple[str, ...]:
        return tuple(getattr(self.component, "active_providers", ()))

    @property
    def last_inference_ms(self) -> float | None:
        return getattr(self.component, "last_inference_ms", None)

    def interpolate(self, frame_a: Any, frame_b: Any) -> Any:
        fallback_activated = False
        if self._degraded:
            if self._attempt >= self.controller.policy.max_attempts:
                if self._fallback_factory is None:
                    raise RecoveryError("Interpolation recovery retry limit reached.")
                try:
                    _close_component(self.component)
                    self.component = self._fallback_factory()
                except Exception as error:
                    self.controller.mark_degraded("interpolation", error)
                    raise RecoveryError("Interpolation provider fallback failed.") from error
                self._fallback_factory = None
                self._degraded = False
                self._attempt = 0
                self.controller.mark_fallback("interpolation")
                fallback_activated = True
            if self._degraded and self._clock() < self._next_retry_at:
                raise RecoveryError("Interpolation is temporarily degraded.")
            if self._degraded:
                self._attempt += 1
            try:
                if self._degraded:
                    _close_component(self.component)
                    self.component = self.factory()
            except Exception as error:
                self._next_retry_at = self._clock() + self.controller.policy.backoff(self._attempt)
                self.controller.mark_degraded("interpolation", error)
                raise RecoveryError("Interpolation recovery is pending.") from error
            self._degraded = False
            if not fallback_activated:
                self.controller.mark_running()
            logger.info("runtime_recovery category=interpolation success=true attempt=%s", self._attempt)
        try:
            result = self.component.interpolate(frame_a, frame_b)
            self._attempt = 0
            return result
        except Exception as error:
            self._degraded = True
            self._attempt = max(1, self._attempt)
            self._next_retry_at = self._clock() + self.controller.policy.backoff(self._attempt)
            self.controller.mark_degraded("interpolation", error)
            raise

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        _close_component(self.component)
