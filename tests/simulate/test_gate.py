"""The CI gate: every planned call has a result, enough are scored, and enough pass."""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.simulate.calls import write_call
from unmute_core.testing.simulate import Runner, RunOptions, evaluate
from unmute_core.testing.simulate._gate import render, write
from unmute_core.testing.simulate._report import summarize

PLANNED = [("booking", 0), ("booking", 1), ("intake", 0), ("intake", 1)]


@pytest.mark.parametrize(
    ("calls", "options", "passed", "reasons"),
    [
        ({}, {}, True, []),
        (
            {("booking", 1): {"passed": False, "failed": ("no_errors",)}},
            {},
            False,
            ["pass rate 75% is below 100%"],
        ),
        ({("booking", 1): {"passed": False}}, {"min_pass": 0.5}, True, []),
        (
            {("booking", 1): {"left_out": "environment"}, ("intake", 1): {"left_out": "caller"}},
            {},
            False,
            [
                "only 2/4 calls are scored (need 75%); the environment, the judge or the "
                "caller broke the rest"
            ],
        ),
        ({("intake", 1): {"left_out": "judge"}}, {}, True, []),
        ({("intake", 1): None}, {}, False, ["1 planned call(s) have no result"]),
    ],
    ids=[
        "all_pass",
        "one_failure",
        "one_failure_under_a_lower_bar",
        "half_left_out",
        "one_left_out_is_fine",
        "a_call_with_no_result",
    ],
)
def test_the_gate_judges_the_planned_calls(
    tmp_path: Path,
    calls: dict[tuple[str, int], dict[str, Any] | None],
    options: dict[str, float],
    passed: bool,
    reasons: list[str],
) -> None:
    for scenario, index in PLANNED:
        kind = calls.get((scenario, index), {})
        if kind is not None:
            write_call(tmp_path, scenario, index, **kind)
    verdict = evaluate(tmp_path, planned=PLANNED, **options)
    assert (verdict.passed, verdict.reasons) == (passed, reasons)


def test_a_left_out_call_counts_neither_as_passed_nor_failed(tmp_path: Path) -> None:
    write_call(tmp_path, "booking", 0)
    write_call(tmp_path, "booking", 1, passed=False, left_out="caller")
    verdict = evaluate(tmp_path, planned=PLANNED[:2])
    assert (verdict.scored, verdict.passed_calls, verdict.pass_rate) == (1, 1, 1.0)
    summary = summarize(tmp_path)
    assert (summary["calls"], summary["calls_run"], summary["left_out"]) == (1, 2, 1)
    assert summary["left_out_reasons"] == {"caller": 1}


def test_a_call_that_passed_on_a_retry_is_flaky(tmp_path: Path) -> None:
    path = write_call(tmp_path, "booking", 0)
    path.with_name("00.attempt1.json").write_text('{"type": "attempt", "crashed": true}\n')
    write_call(tmp_path, "booking", 1)
    verdict = evaluate(tmp_path, planned=PLANNED[:2])
    assert verdict.passed
    assert verdict.flaky == ["booking #00"]


def test_a_failing_call_is_named_with_its_failed_core_checks(tmp_path: Path) -> None:
    write_call(tmp_path, "booking", 0, passed=False, failed=("saved_values_match",))
    verdict = evaluate(tmp_path, planned=PLANNED[:1])
    text = render(verdict, framework="livekit")
    assert text.startswith("### Call simulator gate (livekit): FAIL\n")
    assert "| booking #00 | saved_values_match |" in text


def test_the_verdict_goes_to_the_job_summary_and_gate_json(tmp_path: Path) -> None:
    write_call(tmp_path, "booking", 0)
    (tmp_path / "summary.md").write_text("the run summary")
    step_summary = tmp_path / "step-summary.md"
    verdict = evaluate(tmp_path, planned=PLANNED[:1])
    text = write(tmp_path, verdict, framework="pipecat", step_summary=step_summary)
    assert "PASS" in text.splitlines()[0]
    assert step_summary.read_text().startswith(text)
    assert "the run summary" in step_summary.read_text()
    assert '"passed": true' in (tmp_path / "gate.json").read_text()


@pytest.mark.parametrize(
    ("named", "meta", "planned"),
    [
        (
            [],
            {"scenarios": ["booking_happy", "intake_happy"], "calls": 2},
            [("booking_happy", 0), ("booking_happy", 1), ("intake_happy", 0), ("intake_happy", 1)],
        ),
        (
            ["intake_happy"],
            {"scenarios": ["booking_happy", "intake_happy"], "calls": 1},
            [("intake_happy", 0)],
        ),
        (["intake_happy", "booking_happy"], None, [("intake_happy", 0), ("booking_happy", 0)]),
    ],
    ids=["the_runs_own_plan", "only_the_named_scenarios", "no_run_yet"],
)
def test_a_recheck_plans_the_calls_the_run_planned(
    tmp_path: Path, named: list[str], meta: dict[str, Any] | None, planned: list[tuple[str, int]]
) -> None:
    pytest.importorskip("pipecat")
    if meta is not None:
        (tmp_path / "run.json").write_text(json.dumps(meta))
    options = RunOptions(framework="pipecat", scenarios=named, out=tmp_path)
    assert Runner(options).run_plan() == planned
