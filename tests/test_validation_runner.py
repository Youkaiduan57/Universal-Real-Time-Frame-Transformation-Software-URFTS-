from __future__ import annotations

import json
from pathlib import Path

import pytest

from resource_validation import ResourceRegistry, ResourceSampler
from validation_runner import (
    CallbackValidationAdapter,
    DEFAULT_VALIDATION_CHECKS,
    ScriptedValidationRunner,
    ValidationScript,
    ValidationSkipped,
)
from validation_scenarios import SyntheticValidationAdapter


def synthetic_runner(cycles: int = 3):
    registry = ResourceRegistry()
    adapter = SyntheticValidationAdapter(registry, cycles=cycles)
    sampler = ResourceSampler(
        registry=registry,
        queue_probes=(adapter.queue_sizes,),
        gpu_probe=lambda: (None, "unavailable"),
    )
    return ScriptedValidationRunner(adapter, sampler=sampler)


def test_full_scripted_validation_generates_structured_passing_report(
    tmp_path: Path,
) -> None:
    report = synthetic_runner().run()
    assert report.passed
    assert len(report.checks) == len(DEFAULT_VALIDATION_CHECKS)
    assert all(check.status == "passed" for check in report.checks)
    assert report.resource_summary.peak_onnx_sessions >= 2
    assert report.resource_summary.peak_d3d_resources >= 1
    assert report.resource_summary.peak_queue_size >= 4
    assert report.resource_after.onnx_sessions == 0
    assert report.resource_after.d3d_resources == 0

    output = tmp_path / "validation.json"
    report.write(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["passed"] is True
    assert set(payload["resources"]) == {"before", "during", "after", "summary"}


@pytest.mark.parametrize(
    "check_name",
    [
        "repeated_startup_shutdown",
        "repeated_recovery",
        "repeated_provider_recreation",
        "repeated_renderer_recreation",
        "resource_cleanup",
        "no_thread_leaks",
    ],
)
def test_repeated_resource_validation_checks_clean_up(check_name: str) -> None:
    report = synthetic_runner(cycles=4).run(ValidationScript((check_name,)))
    assert report.passed
    assert report.checks[0].status == "passed"
    assert report.resource_after.onnx_sessions == 0
    assert report.resource_after.d3d_resources == 0
    assert report.resource_after.queue_total == 0


def test_callback_adapter_records_failure_skip_and_always_closes() -> None:
    closed = []

    def fail(observe):
        observe()
        raise RuntimeError("injected")

    def skip(observe):
        del observe
        raise ValidationSkipped("not detectable")

    adapter = CallbackValidationAdapter(
        {"window_enumeration": fail, "wgc_initialization": skip},
        close_callback=lambda: closed.append(True),
    )
    sampler = ResourceSampler(gpu_probe=lambda: (None, "unavailable"))
    report = ScriptedValidationRunner(adapter, sampler=sampler).run(
        ValidationScript(("window_enumeration", "wgc_initialization"))
    )
    assert report.passed is False
    assert [check.status for check in report.checks] == ["failed", "skipped"]
    assert closed == [True]


def test_required_skip_fails_but_optional_detection_skip_is_allowed() -> None:
    def skip(observe):
        del observe
        raise ValidationSkipped("not detectable")

    sampler = ResourceSampler(gpu_probe=lambda: (None, "unavailable"))
    required = ScriptedValidationRunner(
        CallbackValidationAdapter({"wgc_initialization": skip}),
        sampler=sampler,
    ).run(ValidationScript(("wgc_initialization",)))
    assert required.passed is False

    optional = ScriptedValidationRunner(
        CallbackValidationAdapter({"fullscreen_windowed_transition": skip}),
        sampler=sampler,
    ).run(ValidationScript(("fullscreen_windowed_transition",)))
    assert optional.passed is True


def test_adapter_cleanup_failure_is_structured_instead_of_escaping() -> None:
    def close_failure() -> None:
        raise RuntimeError("cleanup failed")

    adapter = CallbackValidationAdapter(
        {"graceful_shutdown": lambda observe: {"ok": bool(observe())}},
        close_callback=close_failure,
    )
    report = ScriptedValidationRunner(
        adapter,
        sampler=ResourceSampler(gpu_probe=lambda: (None, "unavailable")),
    ).run(ValidationScript(("graceful_shutdown",)))
    assert report.passed is False
    assert report.checks[-1].name == "adapter_cleanup"
    assert report.checks[-1].status == "failed"
    assert "cleanup failed" in (report.checks[-1].error or "")


def test_validation_script_loads_selection_and_rejects_unknown_checks(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "script.json"
    script_path.write_text(
        json.dumps(
            {
                "checks": ["window_enumeration", "graceful_shutdown"],
                "metadata": {"application": "test"},
            }
        ),
        encoding="utf-8",
    )
    script = ValidationScript.load(script_path)
    assert script.checks == ("window_enumeration", "graceful_shutdown")
    assert script.metadata == {"application": "test"}

    script_path.write_text(json.dumps({"checks": ["unknown"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown validation checks"):
        ValidationScript.load(script_path)
