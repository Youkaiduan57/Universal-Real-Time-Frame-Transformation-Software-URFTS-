"""Deterministic synthetic benchmark for pacing and recovery policy."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frame_pacing import FramePacer, PresentationFrame  # noqa: E402
from runtime_recovery import RecoveryController, RetryPolicy  # noqa: E402


class SyntheticClock:
    def __init__(self) -> None:
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value

    def wait(self, delay: float) -> bool:
        self.value += delay
        return False


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    return ordered[round((percent / 100.0) * (len(ordered) - 1))]


def run_scenario(
    name: str,
    *,
    mode: str,
    target_fps: float | None,
    generated: bool = False,
    slow_interpolation: bool = False,
    count: int = 120,
) -> dict[str, float | int | str]:
    clock = SyntheticClock()
    pacer = FramePacer(
        mode=mode,
        target_fps=target_fps,
        max_frame_latency_ms=100.0,
        clock=clock,
        waiter=clock.wait,
    )
    decisions = []
    first_capture = clock.value
    decisions += pacer.pace_batch([PresentationFrame("A", first_capture, "real")])
    real_count = 1
    for index in range(1, count):
        if mode == "off":
            clock.value += 1 / 240.0
        captured_at = clock.value
        if generated:
            if slow_interpolation:
                clock.value += 0.040
            batch = [
                PresentationFrame(f"M{index}", captured_at, "generated"),
                PresentationFrame(f"R{index}", captured_at, "real"),
            ]
        else:
            batch = [PresentationFrame(f"R{index}", captured_at, "real")]
        decisions += pacer.pace_batch(batch)
        real_count += 1

    presented = [decision for decision in decisions if decision.present]
    errors = [abs(decision.pacing_error_ms) for decision in presented]
    latencies = [
        (decision.actual_at - decision.frame.captured_at) * 1000.0
        for decision in presented
    ]
    snapshot = pacer.snapshot()
    generated_presented = sum(
        decision.frame.frame_kind == "generated" for decision in presented
    )
    real_presented = len(presented) - generated_presented
    return {
        "scenario": name,
        "requested_fps": "uncapped" if target_fps is None else target_fps,
        "presentation_fps": round(snapshot.presentation_fps, 3),
        "median_pacing_error_ms": round(statistics.median(errors), 3),
        "p95_pacing_error_ms": round(percentile(errors, 95.0), 3),
        "median_latency_ms": round(statistics.median(latencies), 3),
        "real_frames_presented": real_presented,
        "generated_frames_presented": generated_presented,
        "generated_frames_dropped": snapshot.generated_frames_dropped_late,
        "late_frames": snapshot.late_frames,
    }


def recovery_benchmark() -> dict[str, int | str]:
    attempts = 0
    clears = 0

    def clear_queue() -> None:
        nonlocal clears
        clears += 1

    def recover_capture() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic temporary capture failure")
        return "frame"

    controller = RecoveryController(
        policy=RetryPolicy(3, 0.0, 0.0),
        on_recovery=clear_queue,
    )
    result = controller.recover(
        "capture",
        recover_capture,
        initial_error=RuntimeError("synthetic capture failure"),
        source_unavailable=True,
    )
    snapshot = controller.snapshot()
    return {
        "result": result,
        "retry_attempts": snapshot.retry_attempts,
        "successful_recoveries": snapshot.successful_recoveries,
        "failed_recoveries": snapshot.failed_recoveries,
        "queue_clears": clears,
        "final_state": snapshot.state,
    }


def main() -> None:
    rows = [
        run_scenario("pacing_off", mode="off", target_fps=None),
        run_scenario("fixed_30", mode="fixed", target_fps=30.0),
        run_scenario("fixed_60", mode="fixed", target_fps=60.0),
        run_scenario("rife_midpoint", mode="fixed", target_fps=60.0, generated=True),
        run_scenario(
            "slow_interpolation",
            mode="fixed",
            target_fps=60.0,
            generated=True,
            slow_interpolation=True,
        ),
    ]
    report = {"pacing": rows, "recovery": recovery_benchmark()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
