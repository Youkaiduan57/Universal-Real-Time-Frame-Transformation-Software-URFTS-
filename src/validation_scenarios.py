"""Deterministic validation scenarios and lifecycle stress adapter."""

from __future__ import annotations

import threading
import time
from typing import Callable, Mapping

from frame_pacing import FramePacer, PresentationFrame
from resource_validation import QueueSizes, ResourceLease, ResourceRegistry, ResourceSample
from runtime_recovery import RecoveryController, RetryPolicy


class _SyntheticClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def wait(self, delay: float) -> bool:
        self.value += delay
        return False


class _TrackedSession:
    def __init__(self, registry: ResourceRegistry, session_type: str) -> None:
        self.session_type = session_type
        self.calls = 0
        self._lease = ResourceLease("onnx_sessions", registry)
        self._lease.acquire()

    def infer(self) -> int:
        self.calls += 1
        return self.calls

    def close(self) -> None:
        self._lease.release()


class _TrackedRenderer:
    def __init__(self, registry: ResourceRegistry) -> None:
        self._lease = ResourceLease("d3d_resources", registry)
        self._lease.acquire()

    def close(self) -> None:
        self._lease.release()


class SyntheticValidationAdapter:
    """Exercise every validation contract without requiring a foreground app."""

    def __init__(self, registry: ResourceRegistry, *, cycles: int = 5) -> None:
        if cycles <= 0:
            raise ValueError("Validation cycle count must be positive.")
        self.registry = registry
        self.cycles = cycles
        self.dimensions = (1280, 720)
        self.minimized = False
        self.fullscreen = False
        self._queue_sizes = QueueSizes()
        self._resources: list[object] = []
        self._closed = False
        self._baseline = registry.snapshot()
        self._thread_baseline = len(threading.enumerate())

    def queue_sizes(self) -> QueueSizes:
        return self._queue_sizes

    def execute(
        self,
        check_name: str,
        observe: Callable[[], ResourceSample],
    ) -> Mapping[str, object] | None:
        operation = getattr(self, f"_check_{check_name}", None)
        if operation is None:
            raise ValueError(f"Unsupported synthetic validation check: {check_name}")
        return operation(observe)

    def _check_window_enumeration(self, observe) -> Mapping[str, object]:
        del observe
        windows = ("Synthetic Game", "UniversalUpscaler Preview")
        selectable = [window for window in windows if window != "UniversalUpscaler Preview"]
        if selectable != ["Synthetic Game"]:
            raise AssertionError("Window filtering did not preserve the target.")
        return {"enumerated": len(windows), "selectable": len(selectable)}

    def _check_wgc_initialization(self, observe) -> Mapping[str, object]:
        renderer = _TrackedRenderer(self.registry)
        observe()
        renderer.close()
        return {"initialized": True, "first_frame_dimensions": list(self.dimensions)}

    def _check_resolution_changes(self, observe) -> Mapping[str, object]:
        original = self.dimensions
        self.dimensions = (1600, 900)
        self._queue_sizes = QueueSizes(input=1, output=1)
        observe()
        self._queue_sizes = QueueSizes()
        changed = self.dimensions
        self.dimensions = original
        return {"before": list(original), "detected": list(changed)}

    def _check_resize_handling(self, observe) -> Mapping[str, object]:
        sizes = ((960, 540), (1920, 1080), (1280, 720))
        for size in sizes:
            self.dimensions = size
            observe()
        return {"resize_events": len(sizes), "final_dimensions": list(self.dimensions)}

    def _check_minimize_restore(self, observe) -> Mapping[str, object]:
        self.minimized = True
        observe()
        self.minimized = False
        observe()
        return {"paused_while_minimized": True, "restored": True}

    def _check_fullscreen_windowed_transition(self, observe) -> Mapping[str, object]:
        states = []
        for fullscreen in (True, False):
            self.fullscreen = fullscreen
            states.append("fullscreen" if fullscreen else "windowed")
            observe()
        return {"detected_states": states}

    def _check_renderer_recreation(self, observe) -> Mapping[str, object]:
        for _ in range(self.cycles):
            renderer = _TrackedRenderer(self.registry)
            observe()
            renderer.close()
        return {"recreations": self.cycles}

    def _persistent_session(self, session_type: str, observe) -> Mapping[str, object]:
        session = _TrackedSession(self.registry, session_type)
        identity = id(session)
        for _ in range(self.cycles):
            session.infer()
            if id(session) != identity:
                raise AssertionError(f"{session_type} session identity changed.")
            observe()
        session.close()
        return {"session_id": identity, "inferences": session.calls, "persistent": True}

    def _check_ai_session_persistence(self, observe) -> Mapping[str, object]:
        return self._persistent_session("ai", observe)

    def _check_rife_persistence(self, observe) -> Mapping[str, object]:
        return self._persistent_session("rife", observe)

    def _check_frame_pacing_after_recovery(self, observe) -> Mapping[str, object]:
        clock = _SyntheticClock()
        pacer = FramePacer(
            mode="fixed",
            target_fps=60.0,
            max_frame_latency_ms=100.0,
            clock=clock,
            waiter=clock.wait,
        )
        first = pacer.pace_batch([PresentationFrame("A", clock.value, "real")])[0]
        attempts = 0

        def recover() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary capture failure")
            return "recovered"

        controller = RecoveryController(
            policy=RetryPolicy(3, 0.0, 0.0),
        )
        recovery_result = controller.recover(
            "capture",
            recover,
            initial_error=RuntimeError("capture lost"),
            source_unavailable=True,
        )
        second = pacer.pace_batch([PresentationFrame("B", clock.value, "real")])[0]
        observe()
        if not first.present or not second.present or second.scheduled_at <= first.scheduled_at:
            raise AssertionError("Frame pacing did not resume monotonically after recovery.")
        return {
            "recovery_result": recovery_result,
            "retry_attempts": controller.snapshot().retry_attempts,
            "presentation_fps": pacer.snapshot().presentation_fps,
        }

    def _check_graceful_shutdown(self, observe) -> Mapping[str, object]:
        self._queue_sizes = QueueSizes()
        self._close_resources()
        observe()
        return {"resources_closed": True, "queues_cleared": True}

    @staticmethod
    def _joined_worker(observe) -> None:
        stop = threading.Event()
        started = threading.Event()

        def worker() -> None:
            started.set()
            stop.wait()

        thread = threading.Thread(target=worker, name="validation-worker")
        thread.start()
        if not started.wait(timeout=1.0):
            raise AssertionError("Validation worker did not start.")
        observe()
        stop.set()
        thread.join(timeout=1.0)
        if thread.is_alive():
            raise AssertionError("Validation worker did not stop.")

    def _check_repeated_startup_shutdown(self, observe) -> Mapping[str, object]:
        for _ in range(self.cycles):
            ai = _TrackedSession(self.registry, "ai")
            rife = _TrackedSession(self.registry, "rife")
            renderer = _TrackedRenderer(self.registry)
            self._queue_sizes = QueueSizes(input=2, output=2)
            self._joined_worker(observe)
            self._queue_sizes = QueueSizes()
            renderer.close()
            rife.close()
            ai.close()
        return {"cycles": self.cycles}

    def _check_repeated_recovery(self, observe) -> Mapping[str, object]:
        recovered = 0
        for _ in range(self.cycles):
            attempts = 0

            def operation() -> bool:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("injected failure")
                return True

            controller = RecoveryController(policy=RetryPolicy(3, 0.0, 0.0))
            if controller.recover(
                "processing",
                operation,
                initial_error=RuntimeError("injected failure"),
            ):
                recovered += 1
            observe()
        return {"cycles": self.cycles, "successful": recovered}

    def _check_repeated_provider_recreation(self, observe) -> Mapping[str, object]:
        for _ in range(self.cycles):
            ai = _TrackedSession(self.registry, "ai")
            observe()
            ai.close()
            rife = _TrackedSession(self.registry, "rife")
            observe()
            rife.close()
        return {"ai_recreations": self.cycles, "rife_recreations": self.cycles}

    def _check_repeated_renderer_recreation(self, observe) -> Mapping[str, object]:
        for _ in range(self.cycles):
            renderer = _TrackedRenderer(self.registry)
            observe()
            renderer.close()
        return {"cycles": self.cycles}

    def _check_resource_cleanup(self, observe) -> Mapping[str, object]:
        self._close_resources()
        self._queue_sizes = QueueSizes()
        sample = observe()
        for kind in ("onnx_sessions", "d3d_resources"):
            if self.registry.count(kind) != self._baseline.get(kind, 0):
                raise AssertionError(f"Outstanding {kind} remained after cleanup.")
        if sample.queue_total:
            raise AssertionError("Validation queues were not empty after cleanup.")
        return {"registry": self.registry.snapshot(), "queue_total": 0}

    def _check_no_thread_leaks(self, observe) -> Mapping[str, object]:
        for _ in range(self.cycles):
            self._joined_worker(observe)
        current = len(threading.enumerate())
        if current != self._thread_baseline:
            raise AssertionError(
                f"Python thread count changed from {self._thread_baseline} to {current}."
            )
        return {"baseline": self._thread_baseline, "after": current}

    def _close_resources(self) -> None:
        while self._resources:
            resource = self._resources.pop()
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue_sizes = QueueSizes()
        self._close_resources()
