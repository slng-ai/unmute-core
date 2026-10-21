"""A pass or fail verdict on a run directory, for CI.

The verdict reads the same call files as the summary. The gate passes when:

* every planned call has a result (a call the simulator broke for good has none),
* at least ``min_scored`` of the planned calls are scored, so a run the environment broke
  is never reported as a pass, and
* the scored calls' pass rate is at least ``min_pass``.

A call that failed and ran again (``--retry-failed``) keeps its last attempt. Earlier
attempts stay beside it as ``<nn>.attempt<k>.json``, and a call that passed on a retry is
reported as flaky.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from unmute_core.testing.simulate._report import load_calls, scored, wrong_saves

DEFAULT_MIN_PASS = 1.0
"""The pass rate the gate wants by default: every scored call."""
DEFAULT_MIN_SCORED = 0.75
"""The share of planned calls that must be scored by default."""


class Verdict(BaseModel):
    """The gate's verdict on a run.

    Attributes:
        passed: True when the run meets the gate.
        reasons: Why it does not, in words.
        planned: How many calls the run planned.
        scored: How many of them are scored.
        passed_calls: How many scored calls passed.
        pass_rate: The scored calls' pass rate.
        min_pass: The pass rate it needed.
        min_scored: The share of planned calls that had to be scored.
        missing: The planned calls with no result.
        failing: Each failed scored call, with the core checks it failed.
        flaky: The calls that passed only on a retry.
        wrong_saves: The saves made as the scored calls ended that should not have been.
    """

    passed: bool
    reasons: list[str]
    planned: int
    scored: int
    passed_calls: int
    pass_rate: float
    min_pass: float
    min_scored: float
    missing: list[str]
    failing: list[dict[str, Any]]
    flaky: list[str]
    wrong_saves: int


def call_index(path: str | Path) -> int | None:
    """Return the index of a call file, like 3 for ``03.jsonl``, or None for another file."""
    stem = Path(path).stem
    return int(stem) if stem.isdigit() else None


def attempts(path: Path) -> list[Path]:
    """Return the earlier attempts kept beside a call file."""
    return sorted(path.parent.glob(f"{path.stem}.attempt*.json"))


def evaluate(
    out: Path,
    *,
    planned: Sequence[tuple[str, int]],
    min_pass: float = DEFAULT_MIN_PASS,
    min_scored: float = DEFAULT_MIN_SCORED,
) -> Verdict:
    """Return the gate's verdict on the planned calls of a run directory.

    Args:
        out: The run directory.
        planned: Each planned call: its scenario and index.
        min_pass: The pass rate the scored calls need.
        min_scored: The share of planned calls that must be scored.
    """
    by_key = {
        (c["scenario"], index): c
        for c in load_calls(out)
        if (index := call_index(c["path"])) is not None
    }
    missing = [f"{s} #{i:02d}" for s, i in planned if (s, i) not in by_key]
    chosen = [by_key[key] for key in planned if key in by_key]
    scored_calls = [c for c in chosen if scored(c)]
    passed = [c for c in scored_calls if c["checks"]["passed"]]

    def name(call: dict[str, Any]) -> str:
        """Return a call as the verdict names it."""
        return f"{call['scenario']} #{call_index(call['path']):02d}"

    failing = [
        {
            "call": name(c),
            "path": c["path"],
            "failed": [r["id"] for r in c["checks"]["results"] if r["core"] and not r["passed"]],
        }
        for c in scored_calls
        if not c["checks"]["passed"]
    ]
    flaky = [name(c) for c in passed if attempts(Path(c["path"]))]
    pass_rate = len(passed) / len(scored_calls) if scored_calls else 0.0
    scored_share = len(scored_calls) / len(planned) if planned else 0.0
    reasons = []
    if missing:
        reasons.append(f"{len(missing)} planned call(s) have no result")
    if scored_share < min_scored:
        reasons.append(
            f"only {len(scored_calls)}/{len(planned)} calls are scored "
            f"(need {min_scored:.0%}); the environment, the judge or the caller broke the rest"
        )
    if pass_rate < min_pass:
        reasons.append(f"pass rate {pass_rate:.0%} is below {min_pass:.0%}")
    return Verdict(
        passed=not reasons,
        reasons=reasons,
        planned=len(planned),
        scored=len(scored_calls),
        passed_calls=len(passed),
        pass_rate=pass_rate,
        min_pass=min_pass,
        min_scored=min_scored,
        missing=missing,
        failing=failing,
        flaky=flaky,
        wrong_saves=wrong_saves(scored_calls),
    )


def render(verdict: Verdict, *, framework: str) -> str:
    """Return the verdict as Markdown."""
    lines = [
        f"### Call simulator gate ({framework}): {'PASS' if verdict.passed else 'FAIL'}",
        "",
        f"{verdict.passed_calls}/{verdict.scored} scored calls passed "
        f"({verdict.pass_rate:.0%}, gate {verdict.min_pass:.0%}); "
        f"{verdict.scored}/{verdict.planned} planned calls scored.",
    ]
    lines += [f"- {reason}" for reason in verdict.reasons]
    if verdict.wrong_saves:
        lines.append(f"- {verdict.wrong_saves} wrong save(s) at call end")
    if verdict.failing:
        lines += ["", "| failing call | core checks failed |", "|---|---|"]
        lines += [
            f"| {row['call']} | {', '.join(row['failed']) or '-'} |" for row in verdict.failing
        ]
    if verdict.missing:
        lines += ["", "No result: " + ", ".join(verdict.missing)]
    if verdict.flaky:
        lines += ["", "Passed on retry (flaky): " + ", ".join(verdict.flaky)]
    return "\n".join(lines) + "\n"


def write(out: Path, verdict: Verdict, *, framework: str, step_summary: Path | None) -> str:
    """Write ``gate.json``, add the verdict to a CI job's summary, and return it as Markdown.

    Args:
        out: The run directory.
        verdict: The verdict.
        framework: The framework the run used.
        step_summary: A GitHub Actions job's summary file, or None.
    """
    (out / "gate.json").write_text(json.dumps(verdict.model_dump(), indent=1))
    text = render(verdict, framework=framework)
    if step_summary is not None:
        with step_summary.open("a") as handle:
            handle.write(text + "\n")
            if (summary := out / "summary.md").exists():
                handle.write(
                    "<details><summary>Run summary</summary>\n\n"
                    + summary.read_text()
                    + "\n</details>\n\n"
                )
    return text
