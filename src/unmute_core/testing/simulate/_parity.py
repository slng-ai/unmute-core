"""Two runs of the same scenarios, call by call: usually LiveKit against Pipecat.

Calls pair by scenario and index (``calls/<scenario>/<nn>.jsonl`` in both runs). Pass
counts are the summary's: scored calls only. A pair of scored calls is compared on what
should match across frameworks even though the words do not:

* the task lifecycle: the ordered task events with their group, task and step;
* the saved values: what each value was left holding, coarse enough to recur across
  calls (a value that meets the scenario is that, free text is not its words);
* how the call ended, and which core checks failed (shown, not flagged).

The callers are simulated, so two calls of a scenario never say the same things, and a
caller who says something else can lead either agent elsewhere. A difference is
explained when each side's outcome also occurred on the other side in some scored call of
the same scenario (*varies*), or, for a saved value, when each caller said the value their
agent saved (*caller*). A pair **diverges** when its task sequences or a saved value
differ in a way neither explains: one side did something the other never did in that
scenario.

Whether a value meets the scenario is the verdict the run's own ``saved_values_match``
recorded, not matched again: a scenario's expected dates move with the day it ran.
"""

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unmute_core.testing.simulate._checks import (
    PHONE_VALUES,
    caller_said_it,
    digits,
    spoken_digits,
)
from unmute_core.testing.simulate._record import parse_json
from unmute_core.testing.simulate._report import load_calls, run_meta, scored

TaskToken = tuple[str, str, str, str]
"""One task event as both sides should agree on it: stage, group, task and step."""
_RANK = {"same": 0, "caller": 1, "varies": 2, "diverges": 3}


@dataclass(frozen=True)
class CallView:
    """One scored call, reduced to what should match across frameworks.

    Attributes:
        path: The call file.
        passed: True when it passed every core check.
        tasks: Its task events, in order.
        variables: Each saved value as the call left it.
        ended_by: How it ended.
        core_failed: The core checks it failed.
        expectation_failed: The expected values ``saved_values_match`` found wrong.
        caller_lines: What the caller said.
    """

    path: Path
    passed: bool
    tasks: tuple[TaskToken, ...]
    variables: dict[str, dict[str, Any]]
    ended_by: str
    core_failed: tuple[str, ...]
    expectation_failed: frozenset[str]
    caller_lines: tuple[str, ...]


@dataclass(frozen=True)
class Run:
    """One run directory, read for parity.

    Attributes:
        framework: The framework it ran on.
        counts: Scenario to its passed, scored and left-out calls.
        views: Scenario to call index to the scored call.
        expect: Scenario to expected value to its matcher's kind.
    """

    framework: str
    counts: dict[str, tuple[int, int, int]]
    views: dict[str, dict[str, CallView]]
    expect: dict[str, dict[str, str]]


@dataclass(frozen=True)
class ValueRow:
    """How one saved value compares across a pair.

    Attributes:
        var: The value's name.
        verdict: ``same``, ``caller``, ``varies`` or ``diverges``.
        reason: Why, in words.
    """

    var: str
    verdict: str
    reason: str


@dataclass(frozen=True)
class Pair:
    """Two scored calls of one scenario with the same index, one per run.

    Attributes:
        scenario: The scenario.
        index: The call index.
        a: The first run's call.
        b: The second run's call.
        tasks: How the task sequences compare: ``same``, ``varies`` or ``diverges``.
        values: How each saved value compares.
    """

    scenario: str
    index: str
    a: CallView
    b: CallView
    tasks: str
    values: list[ValueRow]

    @property
    def values_verdict(self) -> str:
        """The worst verdict of the pair's values."""
        return max((row.verdict for row in self.values), key=_RANK.__getitem__, default="same")

    @property
    def divergent(self) -> bool:
        """True when the tasks or a value differ in a way nothing explains."""
        return self.tasks == "diverges" or self.values_verdict == "diverges"

    @property
    def explained(self) -> bool:
        """True when the two differ in tasks or values, and the callers explain it."""
        return not self.divergent and (self.tasks != "same" or self.values_verdict != "same")

    @property
    def differs(self) -> bool:
        """True when the two differ in anything compared."""
        return (
            self.tasks != "same"
            or self.values_verdict != "same"
            or self.a.ended_by != self.b.ended_by
            or self.a.core_failed != self.b.core_failed
        )

    def weight(self) -> int:
        """Return how far apart the two calls are, to rank the divergent pairs."""
        moved = 0
        if self.tasks == "diverges":
            matcher = difflib.SequenceMatcher(a=self.a.tasks, b=self.b.tasks, autojunk=False)
            moved = sum(
                (i2 - i1) + (j2 - j1)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes()
                if tag != "equal"
            )
        diverging = sum(1 for row in self.values if row.verdict == "diverges")
        return 5 * (self.a.passed != self.b.passed) + 3 * diverging + moved


# -- reading ----------------------------------------------------------------------


def _token(event: dict[str, Any]) -> TaskToken:
    """Return one task event as both sides should agree on it."""
    data = event.get("data") or {}
    step = data.get("step")
    return (
        str(event.get("stage") or ""),
        str(data.get("group") or ""),
        str(data.get("task") or ""),
        "" if step is None else str(step),
    )


def _expectation_failed(checks: dict[str, Any], expected: set[str]) -> frozenset[str]:
    """Return the expected values the run's own ``saved_values_match`` found wrong."""
    detail = next(
        (
            r.get("detail") or ""
            for r in checks.get("results") or []
            if r.get("id") == "saved_values_match"
        ),
        "",
    )
    return frozenset(var for var in expected if re.search(rf"(^|; ){re.escape(var)}: ", detail))


def load_run(out: Path) -> Run:
    """Read a run directory for parity."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    views: dict[str, dict[str, CallView]] = defaultdict(dict)
    expect: dict[str, dict[str, str]] = {}
    for call in load_calls(out):
        row = counts[call["scenario"]]
        checks = call["checks"]
        expect[call["scenario"]] = dict(checks.get("expect_kinds") or {})
        if not scored(call):
            row[2] += 1
            continue
        row[0] += bool(checks["passed"])
        row[1] += 1
        record = call["call"]
        path = Path(call["path"])
        views[call["scenario"]][path.stem] = CallView(
            path=path,
            passed=bool(checks["passed"]),
            tasks=tuple(_token(e) for e in call["events"] if e["kind"] == "task_event"),
            variables=record.get("variables") or {},
            ended_by=str(record.get("ended_by")),
            core_failed=tuple(
                sorted(r["id"] for r in checks["results"] if r["core"] and not r["passed"])
            ),
            expectation_failed=_expectation_failed(checks, set(expect[call["scenario"]])),
            caller_lines=tuple(
                str(e.get("text") or "") for e in call["events"] if e["kind"] == "caller_said"
            ),
        )
    return Run(
        framework=str(run_meta(out).get("framework") or "?"),
        counts={k: (v[0], v[1], v[2]) for k, v in counts.items()},
        views=dict(views),
        expect=expect,
    )


# -- comparing ----------------------------------------------------------------------


def _alnum(text: str) -> str:
    """Return the letters and digits of a text, lower-cased."""
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _free_text(var: str, value: Any) -> bool:
    """Return whether a saved value is free text: words, not a number, flag or object."""
    text = str(value).strip()
    return (
        var not in PHONE_VALUES
        and not isinstance(parse_json(value), dict | list | bool)
        and text.lower() not in ("true", "false")
        and " " in text
    )


def _key(var: str, value: Any) -> str:
    """Return a saved value as both sides should agree on it: case and punctuation aside."""
    if var in PHONE_VALUES:
        return digits(value)
    got = parse_json(value)
    if isinstance(got, dict):
        return ", ".join(f"{k}={_alnum(str(v))}" for k, v in sorted(got.items()))
    return _alnum(str(value))


def _outcome(var: str, view: CallView, kind: str | None) -> str:
    """Return what one call left in a value, coarse enough to recur across calls."""
    row = view.variables.get(var) or {}
    stored = row.get("stored")
    if stored is None:
        return "not saved"
    unconfirmed = "" if row.get("confirmed") is not False else ", unconfirmed"
    # "any" accepts alternatives, so meeting it does not pin the value.
    if kind and kind != "any" and var not in view.expectation_failed:
        return "meets the scenario" + unconfirmed
    if _free_text(var, stored):
        return ("text off the scenario" if kind else "text") + unconfirmed
    return _key(var, stored) + unconfirmed


def _said(var: str, value: Any, lines: tuple[str, ...]) -> bool:
    """Return whether this call's caller said the value its agent saved."""
    if caller_said_it(var, value, lines):
        return True
    text = str(value).strip()
    if var in PHONE_VALUES or isinstance(parse_json(value), dict) or not _alnum(text):
        return False
    if text.lower() in ("true", "false"):
        return False
    said = " ".join(lines)
    return _alnum(text) in _alnum(said) or (
        bool(digits(text)) and digits(text) in spoken_digits(said)
    )


def _value_row(
    var: str, kind: str | None, x: CallView, y: CallView, seen_a: set[str], seen_b: set[str]
) -> ValueRow:
    """Return how one saved value compares across a pair."""
    ox, oy = _outcome(var, x, kind), _outcome(var, y, kind)
    sx = (x.variables.get(var) or {}).get("stored")
    sy = (y.variables.get(var) or {}).get("stored")
    if ox == oy:
        if ox.startswith("text") and _key(var, sx) != _key(var, sy):
            return ValueRow(var, "caller", "free text, each caller's words")
        return ValueRow(var, "same", "")
    if ox in seen_b and oy in seen_a:
        return ValueRow(var, "varies", "each also occurs on the other side")
    if (
        sx is not None
        and sy is not None
        and _said(var, sx, x.caller_lines)
        and _said(var, sy, y.caller_lines)
    ):
        return ValueRow(var, "caller", "each caller said the value saved")
    meets_x, meets_y = ox.startswith("meets"), oy.startswith("meets")
    if kind and kind != "any" and meets_x != meets_y:
        reason = "meets the scenario on one side only"
    elif (sx is None) != (sy is None):
        reason = "saved on one side only"
    elif ox.split(", unconfirmed")[0] == oy.split(", unconfirmed")[0]:
        reason = "confirmed on one side only"
    else:
        reason = "differs"
    return ValueRow(var, "diverges", reason)


def pairs(a: Run, b: Run) -> tuple[list[Pair], int]:
    """Return the paired scored calls, and how many scored calls had no scored partner."""
    found: list[Pair] = []
    unpaired = 0
    for scenario in sorted(set(a.views) | set(b.views)):
        left, right = a.views.get(scenario, {}), b.views.get(scenario, {})
        unpaired += len(set(left) ^ set(right))
        kinds = a.expect.get(scenario) or b.expect.get(scenario) or {}
        views = [*left.values(), *right.values()]
        names = sorted(set(kinds) | {var for view in views for var in view.variables})
        seen_tasks_a = {v.tasks for v in left.values()}
        seen_tasks_b = {v.tasks for v in right.values()}
        seen_a = {n: {_outcome(n, v, kinds.get(n)) for v in left.values()} for n in names}
        seen_b = {n: {_outcome(n, v, kinds.get(n)) for v in right.values()} for n in names}
        for index in sorted(set(left) & set(right)):
            x, y = left[index], right[index]
            if x.tasks == y.tasks:
                tasks = "same"
            elif x.tasks in seen_tasks_b and y.tasks in seen_tasks_a:
                tasks = "varies"
            else:
                tasks = "diverges"
            values = [
                _value_row(n, kinds.get(n), x, y, seen_a[n], seen_b[n])
                for n in names
                if n in kinds or n in x.variables or n in y.variables
            ]
            found.append(Pair(scenario, index, x, y, tasks, values))
    return found, unpaired


# -- rendering ----------------------------------------------------------------------


def _cell(text: Any, limit: int = 80) -> str:
    """Return text for a table cell, cut to ``limit``."""
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _event(token: TaskToken) -> str:
    """Return a task event in words."""
    stage, group, task, step = token
    if task:
        return f"{stage} {task}" + (f" ({group} step {step})" if group else "")
    return f"{stage} {group}" + (f" step {step}" if step else "")


def _value(row: dict[str, Any] | None) -> str:
    """Return a saved value for a table cell."""
    if not row or row.get("stored") is None:
        return "(not saved)"
    got = parse_json(row["stored"])
    text = ", ".join(f"{k}={v}" for k, v in got.items()) if isinstance(got, dict) else str(got)
    return _cell(text, 70) + (" (unconfirmed)" if row.get("confirmed") is False else "")


def _failed(view: CallView) -> str:
    """Return a call's failed core checks for a table cell."""
    return ", ".join(view.core_failed) or "-"


def _side_by_side(a: tuple[TaskToken, ...], b: tuple[TaskToken, ...]) -> list[str]:
    """Return table rows of two task sequences aligned, the matching stretches folded."""
    rows: list[str] = []
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            same = list(a[i1:i2])
            if len(same) > 2:
                rows.append(f"| | *{len(same) - 1} matching events* | |")
                same = same[-1:]
            rows += [f"| = | {_cell(_event(t))} | {_cell(_event(t))} |" for t in same]
            continue
        left, right = list(a[i1:i2]), list(b[j1:j2])
        for k in range(max(len(left), len(right))):
            x = _cell(_event(left[k])) if k < len(left) else ""
            y = _cell(_event(right[k])) if k < len(right) else ""
            rows.append(f"| ≠ | {x} | {y} |")
    return rows


def _rate(passed: int, total: int) -> str:
    """Return a pass rate with its counts."""
    return f"{passed / total * 100:.0f}% ({passed}/{total})" if total else "-"


def _totals(counts: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    """Return the passed, scored and left-out calls of a run."""
    return (
        sum(v[0] for v in counts.values()),
        sum(v[1] for v in counts.values()),
        sum(v[2] for v in counts.values()),
    )


def _top(pair: Pair, n: int, la: str, lb: str) -> list[str]:
    """Return the section of one divergent pair."""
    diverging = [r for r in pair.values if r.verdict == "diverges"]
    reasons = " (" + "; ".join(f"{r.var}: {r.reason}" for r in diverging) + ")"
    lines = [
        "",
        f"### {n}. {pair.scenario} #{pair.index}",
        "",
        f"Tasks: **{pair.tasks}**. Values: **{pair.values_verdict}**"
        + (reasons if diverging else "")
        + f". Ended: {pair.a.ended_by} / {pair.b.ended_by}. "
        f"Passed: {'yes' if pair.a.passed else 'no'} / {'yes' if pair.b.passed else 'no'}. "
        f"Core checks failed: {_failed(pair.a)} / {_failed(pair.b)}. "
        f"Call: `calls/{pair.scenario}/{pair.index}.jsonl` in both runs.",
    ]
    if pair.tasks != "same":
        lines += [
            "",
            f"| | {la} task events | {lb} task events |",
            "|---|---|---|",
            *_side_by_side(pair.a.tasks, pair.b.tasks),
        ]
    return [
        *lines,
        "",
        f"| value | {la} | {lb} | |",
        "|---|---|---|---|",
        *(
            f"| {r.var} | {_value(pair.a.variables.get(r.var))} "
            f"| {_value(pair.b.variables.get(r.var))} "
            f"| {r.verdict}{f': {r.reason}' if r.reason else ''} |"
            for r in pair.values
        ),
    ]


def _differing(found: list[Pair], la: str, lb: str) -> list[str]:
    """Return the table of every pair that differs in anything compared."""
    differing = [p for p in found if p.differs]
    lines = [
        "",
        "## Paired calls that differ",
        "",
        f"{len(found) - len(differing)} of {len(found)} pairs match on task sequence, saved "
        "values, ending and failed core checks, and are not listed.",
        "",
        f"| scenario | # | tasks | values | ended | {la} core failures | {lb} core failures |",
        "|---|---|---|---|---|---|---|",
    ]
    for pair in differing:
        verdict = pair.values_verdict
        named = [r.var for r in pair.values if r.verdict == verdict]
        values_cell = verdict if verdict == "same" else f"{verdict}: {', '.join(named)}"
        tasks_cell = f"**{pair.tasks}**" if pair.tasks == "diverges" else pair.tasks
        if verdict == "diverges":
            values_cell = f"**{values_cell}**"
        ended = (
            pair.a.ended_by
            if pair.a.ended_by == pair.b.ended_by
            else f"{pair.a.ended_by} / {pair.b.ended_by}"
        )
        lines.append(
            f"| {pair.scenario} | {pair.index} | {tasks_cell} | {values_cell} | {ended} "
            f"| {_failed(pair.a)} | {_failed(pair.b)} |"
        )
    return lines


def _scenario_rows(
    scenarios: list[str],
    by_scenario: dict[str, list[Pair]],
    counts_a: dict[str, tuple[int, int, int]],
    counts_b: dict[str, tuple[int, int, int]],
    apart: list[str],
) -> list[str]:
    """Return one table row per scenario: pass counts and how its pairs compare."""
    lines = []
    for scenario in scenarios:
        rows = by_scenario.get(scenario, [])
        ca, cb = counts_a.get(scenario), counts_b.get(scenario)
        mark = "**" if scenario in apart else ""
        lines.append(
            f"| {mark}{scenario}{mark} "
            f"| {f'{ca[0]}/{ca[1]}' if ca else '-'} | {f'{cb[0]}/{cb[1]}' if cb else '-'} "
            f"| {len(rows)} | {sum(1 for p in rows if p.explained)} "
            f"| {sum(1 for p in rows if p.divergent)} "
            f"| {sum(1 for p in rows if p.tasks == 'diverges')} "
            f"| {sum(1 for p in rows if p.values_verdict == 'diverges')} "
            f"| {sum(1 for p in rows if p.a.ended_by != p.b.ended_by)} |"
        )
    return lines


def parity(a_out: Path, b_out: Path, *, top: int = 10) -> str:
    """Return Markdown: two runs of the same scenarios, call by call.

    Args:
        a_out: The first run directory, usually LiveKit's.
        b_out: The second, usually Pipecat's.
        top: How many divergent pairs to show in full.
    """
    run_a, run_b = load_run(a_out), load_run(b_out)
    la, lb = run_a.framework, run_b.framework
    if la == lb:
        la, lb = (a_out.name, b_out.name) if a_out.name != b_out.name else ("A", "B")
    counts_a, counts_b = run_a.counts, run_b.counts
    found, unpaired = pairs(run_a, run_b)
    by_scenario: dict[str, list[Pair]] = defaultdict(list)
    for pair in found:
        by_scenario[pair.scenario].append(pair)
    scenarios = sorted(set(counts_a) | set(counts_b))

    def gap(scenario: str) -> float:
        """Return the pass-rate gap of a scenario, in calls."""
        pa, na, _ = counts_a.get(scenario, (0, 0, 0))
        pb, nb, _ = counts_b.get(scenario, (0, 0, 0))
        return abs(pa / na - pb / nb) * max(na, nb) if na and nb else 0.0

    apart = [s for s in scenarios if gap(s) >= 2 - 1e-9]
    divergent = [p for p in found if p.divergent]
    pa, na, xa = _totals(counts_a)
    pb, nb, xb = _totals(counts_b)
    lines = [
        f"# Parity: {la} vs {lb}",
        "",
        f"- **{la}**: `{a_out}` ({run_a.framework})",
        f"- **{lb}**: `{b_out}` ({run_b.framework})",
        "",
        "## Headline",
        "",
        f"- Scored calls: {la} {na}, {lb} {nb} (left out: {xa} and {xb}).",
        f"- Pass rate: {la} {_rate(pa, na)}, {lb} {_rate(pb, nb)}.",
        f"- Scenarios whose pass rates differ by 2+ calls: {len(apart)}"
        + (f" ({', '.join(apart)})" if apart else "")
        + ".",
        f"- Divergent pairs: {len(divergent)} of {len(found)} compared "
        f"(task sequence {sum(1 for p in divergent if p.tasks == 'diverges')}, "
        f"saved values {sum(1 for p in divergent if p.values_verdict == 'diverges')})"
        + (f"; {unpaired} scored calls had no scored partner." if unpaired else "."),
        "",
        "## Per scenario",
        "",
        "Pass counts are scored calls; a pair is the two runs' scored calls with the same "
        "index. *Explained*: the pair's task sequences or saved values differ, but each "
        "side's also occurred on the other side, or each caller said what was saved. "
        "*Divergent*: a difference neither explains.",
        "",
        f"| scenario | {la} passed | {lb} passed | pairs | explained | divergent "
        "| of which tasks | of which values | ended differently |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += _scenario_rows(scenarios, by_scenario, counts_a, counts_b, apart)
    ranked = sorted(divergent, key=lambda p: (-p.weight(), p.scenario, p.index))
    lines += ["", f"## Top divergent pairs ({min(top, len(ranked))} of {len(ranked)})"]
    if not ranked:
        lines += ["", "None."]
    for n, pair in enumerate(ranked[:top], 1):
        lines += _top(pair, n, la, lb)
    lines += _differing(found, la, lb)
    return "\n".join(lines) + "\n"
