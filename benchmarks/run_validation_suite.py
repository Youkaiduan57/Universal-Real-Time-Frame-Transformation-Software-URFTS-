"""Run the scripted compatibility and resource validation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resource_validation import ResourceRegistry, ResourceSampler  # noqa: E402
from validation_runner import (  # noqa: E402
    ScriptedValidationRunner,
    ValidationScript,
)
from validation_scenarios import SyntheticValidationAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UniversalUpscaler compatibility and resource validation.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Optional JSON script selecting validation checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "validation_report.json",
        help="Structured JSON report path.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Positive lifecycle/recovery repetitions.",
    )
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be greater than zero.")
    return args


def main() -> int:
    args = parse_args()
    script = ValidationScript.load(args.script) if args.script else ValidationScript()
    script = ValidationScript(
        script.checks,
        {
            **script.metadata,
            "adapter": "synthetic",
            "cycles": args.cycles,
        },
    )
    registry = ResourceRegistry()
    adapter = SyntheticValidationAdapter(registry, cycles=args.cycles)
    sampler = ResourceSampler(
        registry=registry,
        queue_probes=(adapter.queue_sizes,),
    )
    report = ScriptedValidationRunner(adapter, sampler=sampler).run(script)
    report.write(args.output)
    summary = {
        "passed": report.passed,
        "checks_passed": sum(check.status == "passed" for check in report.checks),
        "checks_skipped": sum(check.status == "skipped" for check in report.checks),
        "checks_failed": sum(check.status == "failed" for check in report.checks),
        "resource_summary": report.resource_summary.to_dict(),
        "report": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
