"""Scripted compatibility and resource-validation framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Protocol, Sequence

from resource_validation import (
    ResourceSample,
    ResourceSampler,
    ResourceValidationSummary,
    summarize_resources,
)


COMPATIBILITY_CHECKS = (
    "window_enumeration",
    "wgc_initialization",
    "resolution_changes",
    "resize_handling",
    "minimize_restore",
    "fullscreen_windowed_transition",
    "renderer_recreation",
    "ai_session_persistence",
    "rife_persistence",
    "frame_pacing_after_recovery",
    "graceful_shutdown",
)

RESOURCE_CHECKS = (
    "repeated_startup_shutdown",
    "repeated_recovery",
    "repeated_provider_recreation",
    "repeated_renderer_recreation",
    "resource_cleanup",
    "no_thread_leaks",
)

DEFAULT_VALIDATION_CHECKS = COMPATIBILITY_CHECKS + RESOURCE_CHECKS
OPTIONAL_DETECTION_CHECKS = frozenset({"fullscreen_windowed_transition"})


class ValidationSkipped(RuntimeError):
    """Raised when a live target cannot expose an optional detectable state."""


class ValidationAdapter(Protocol):
    def execute(
        self,
        check_name: str,
        observe: Callable[[], ResourceSample],
    ) -> Mapping[str, object] | None: ...

    def close(self) -> None: ...


class CallbackValidationAdapter:
    """Adapter for real applications supplied as named scripted callbacks."""

    def __init__(
        self,
        callbacks: Mapping[
            str,
            Callable[[Callable[[], ResourceSample]], Mapping[str, object] | None],
        ],
        *,
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        self.callbacks = dict(callbacks)
        self.close_callback = close_callback
        self._closed = False

    def execute(
        self,
        check_name: str,
        observe: Callable[[], ResourceSample],
    ) -> Mapping[str, object] | None:
        callback = self.callbacks.get(check_name)
        if callback is None:
            raise ValidationSkipped(f"No callback was supplied for {check_name}.")
        return callback(observe)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.close_callback is not None:
            self.close_callback()


@dataclass(frozen=True, slots=True)
class ValidationScript:
    checks: tuple[str, ...] = DEFAULT_VALIDATION_CHECKS
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ValidationScript":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        checks = payload.get("checks", DEFAULT_VALIDATION_CHECKS)
        if not isinstance(checks, list) or not all(
            isinstance(check, str) and check for check in checks
        ):
            raise ValueError("Validation script checks must be a list of names.")
        unknown = sorted(set(checks) - set(DEFAULT_VALIDATION_CHECKS))
        if unknown:
            raise ValueError(
                "Unknown validation checks: " + ", ".join(unknown)
            )
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Validation script metadata must be an object.")
        return cls(tuple(checks), metadata)


@dataclass(frozen=True, slots=True)
class ValidationCheckResult:
    name: str
    status: str
    duration_ms: float
    details: dict[str, object]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    schema_version: int
    started_at: str
    duration_ms: float
    passed: bool
    checks: tuple[ValidationCheckResult, ...]
    resource_before: ResourceSample
    resource_during: tuple[ResourceSample, ...]
    resource_after: ResourceSample
    resource_summary: ResourceValidationSummary
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
            "resources": {
                "before": self.resource_before.to_dict(),
                "during": [sample.to_dict() for sample in self.resource_during],
                "after": self.resource_after.to_dict(),
                "summary": self.resource_summary.to_dict(),
            },
            "metadata": self.metadata,
        }

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class ScriptedValidationRunner:
    def __init__(
        self,
        adapter: ValidationAdapter,
        *,
        sampler: ResourceSampler | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.adapter = adapter
        self.sampler = sampler or ResourceSampler()
        self.clock = clock

    def run(self, script: ValidationScript | None = None) -> ValidationReport:
        active_script = script or ValidationScript()
        started_at = datetime.now(timezone.utc).isoformat()
        run_started = self.clock()
        before = self.sampler.sample()
        observations: list[ResourceSample] = []
        results: list[ValidationCheckResult] = []

        def observe() -> ResourceSample:
            sample = self.sampler.sample()
            observations.append(sample)
            return sample

        cleanup_error: Exception | None = None
        try:
            for check_name in active_script.checks:
                check_started = self.clock()
                try:
                    details = self.adapter.execute(check_name, observe) or {}
                except ValidationSkipped as error:
                    result = ValidationCheckResult(
                        name=check_name,
                        status="skipped",
                        duration_ms=(self.clock() - check_started) * 1000.0,
                        details={},
                        error=str(error),
                    )
                except Exception as error:
                    result = ValidationCheckResult(
                        name=check_name,
                        status="failed",
                        duration_ms=(self.clock() - check_started) * 1000.0,
                        details={},
                        error=f"{type(error).__name__}: {error}",
                    )
                else:
                    result = ValidationCheckResult(
                        name=check_name,
                        status="passed",
                        duration_ms=(self.clock() - check_started) * 1000.0,
                        details=dict(details),
                    )
                results.append(result)
                observe()
        finally:
            try:
                self.adapter.close()
            except Exception as error:
                cleanup_error = error

        if cleanup_error is not None:
            results.append(
                ValidationCheckResult(
                    name="adapter_cleanup",
                    status="failed",
                    duration_ms=0.0,
                    details={},
                    error=f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
            )

        gc.collect()
        after = self.sampler.sample_after_cleanup(before)
        resource_summary = summarize_resources(before, observations, after)
        checks_passed = all(
            result.status == "passed"
            or (
                result.status == "skipped"
                and result.name in OPTIONAL_DETECTION_CHECKS
            )
            for result in results
        )
        return ValidationReport(
            schema_version=1,
            started_at=started_at,
            duration_ms=(self.clock() - run_started) * 1000.0,
            passed=checks_passed and resource_summary.passed,
            checks=tuple(results),
            resource_before=before,
            resource_during=tuple(observations),
            resource_after=after,
            resource_summary=resource_summary,
            metadata=dict(active_script.metadata),
        )
