"""The deterministic checks on one simulated call.

Each check returns a ``CheckResult``. A ``core`` check decides whether the call passed;
the others are signals reported beside them. The judge (``_judge``) adds what code cannot
see.

The checks read the call's records as ``run_call`` writes them: the call record (how it
ended, the saved values, the errors), the timeline (what each side said, task events,
saves, tool calls, model requests) and every mock request.
"""

import datetime as dt
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from unmute_core.tasks import EMPTY_REPLY_GAVE_UP, EMPTY_REPLY_GAVE_UP_AFTER_TOOL, FINISH_TOOL_NAME
from unmute_core.testing.simulate._mocks import AVAILABLE_TIMES
from unmute_core.testing.simulate._record import RUNTIME_TOOLS, parse_json
from unmute_core.testing.simulate._scenario import Scenario

Records = list[dict[str, Any]]

FALLBACK_LINES = (EMPTY_REPLY_GAVE_UP, EMPTY_REPLY_GAVE_UP_AFTER_TOOL)
"""What a step says when its model keeps answering with nothing."""
CALL_VALUE = "call"
"""The State value the runtime keeps the call's own facts in."""
PHONE_VALUES = ("phone_number", "callback_number", "caller_phone")
"""The State values that hold a phone number in the bundled agents."""
_END_STAGES = {
    "task.completed": "completed",
    "task.unserved": "handed back",
    "task.failed": "failed",
}
_STATUS = {"completed": "completed", "handed back": "unserved", "failed": "failed"}
_NUMBER_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}  # fmt: skip
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
)  # fmt: skip
_CHANGE_WORDS = ("change", "update", "instead", "correct", "cancel", "different")
"""A hand-back worded with one of these asks to change what was served: a new request."""


class CheckResult(BaseModel):
    """One check's verdict on one call.

    Attributes:
        id: The check. The judge's start with ``judge.``.
        passed: True when the call passed it.
        core: True when it decides whether the call passed.
        category: Why it failed, as labels split by commas, to count failures by.
        detail: What failed, in words.
        evidence: The events or turns it failed on.
    """

    id: str
    passed: bool
    core: bool
    category: str = ""
    detail: str = ""
    evidence: list[Any] = Field(default_factory=list)


# -- reading values -------------------------------------------------------------


def as_text(value: object) -> str:
    """Return a saved value as text: a string as it is, other JSON data as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def digits(value: object) -> str:
    """Return the digits in a value."""
    return "".join(ch for ch in as_text(value or "") if ch.isdigit())


def _norm_id(value: object) -> str:
    """Return an id with case, spaces and separators aside."""
    return re.sub(r"[\s\-_.]", "", as_text(value or "")).upper()


def spoken_digits(text: str) -> str:
    """Return the digits a line says, written or spoken ("double six" is 66)."""
    out: list[str] = []
    repeat = 1
    for token in re.findall(r"[a-z]+|\d", text.lower()):
        if token in ("double", "triple"):
            repeat = 2 if token == "double" else 3
            continue
        digit = token if token.isdigit() else _NUMBER_WORDS.get(token)
        if digit is not None:
            out.append(digit * repeat)
        repeat = 1
    return "".join(out)


def match_value(matcher: Mapping[str, Any], stored: object) -> tuple[bool, str]:
    """Return whether a saved value meets an ``expect`` matcher, and why not.

    A matcher is one of: ``absent``, ``any`` (a list of matchers), ``phone``, ``bool``,
    ``exact``, ``exact_ci`` (an id, case and separators aside), ``present`` (a regular
    expression), ``text`` (groups of words, one of each group in the value), ``fields``
    (a matcher per field of an object) and ``appointment`` (service words, day and time).

    Raises:
        ValueError: If the matcher is none of these.
    """
    if matcher.get("absent"):
        return stored is None, f"expected nothing saved, found {stored!r}"
    if "any" in matcher:
        reasons = []
        for option in matcher["any"]:
            ok, why = match_value(option, stored)
            if ok:
                return True, ""
            reasons.append(why)
        return False, " / ".join(reasons)
    if stored is None:
        return False, "nothing saved"
    if "phone" in matcher:
        want, got = digits(matcher["phone"]), digits(stored)
        if got == want:
            return True, ""
        if got.endswith(want[-9:]):
            return False, f"right number, other country code: {stored}"
        return False, f"expected {matcher['phone']}, saved {stored}"
    if "bool" in matcher:
        want = "true" if matcher["bool"] else "false"
        return as_text(stored).lower() == want, f"expected {want}, saved {stored}"
    if "exact" in matcher:
        ok = stored == matcher["exact"] or as_text(stored) == as_text(matcher["exact"])
        return ok, f"expected {matcher['exact']!r}, saved {stored!r}"
    if "present" in matcher:
        ok = bool(re.fullmatch(matcher["present"], as_text(stored)))
        return ok, f"saved {stored!r} does not match {matcher['present']!r}"
    if "fields" in matcher:
        return _match_fields(matcher["fields"], stored)
    if "exact_ci" in matcher:
        ok = _norm_id(stored) == _norm_id(matcher["exact_ci"])
        return ok, f"expected {matcher['exact_ci']}, saved {stored}"
    if "text" in matcher:
        low = as_text(stored).lower()
        missing = [group for group in matcher["text"] if not any(w in low for w in group)]
        return not missing, f"saved {stored!r} lacks {missing}"
    if "appointment" in matcher:
        return _match_appointment(matcher["appointment"], stored)
    raise ValueError(f"unknown matcher {dict(matcher)}")


def _match_fields(fields: Mapping[str, Any], stored: object) -> tuple[bool, str]:
    """Return whether each field of a saved object meets its matcher, and why not."""
    got = parse_json(stored)
    if not isinstance(got, dict):
        return False, f"not an object: {stored!r}"
    problems = []
    for name, sub in fields.items():
        ok, why = match_value(sub, got.get(name))
        if not ok:
            problems.append(f"{name}: {why}")
    return not problems, "; ".join(problems)


def _match_appointment(want: Mapping[str, Any], stored: object) -> tuple[bool, str]:
    """Return whether a saved appointment has the service, day and time, and why not."""
    got = parse_json(stored)
    if not isinstance(got, dict):
        return False, f"not an appointment: {stored!r}"
    problems = []
    service = str(got.get("service") or "").lower()
    if not any(word in service for word in want["service"]):
        problems.append(f"service {got.get('service')!r}")
    if got.get("day") != want["day"]:
        problems.append(f"day {got.get('day')} (want {want['day']})")
    if got.get("time") != want["time"]:
        problems.append(f"time {got.get('time')} (want {want['time']})")
    return not problems, "; ".join(problems)


def caller_said_it(name: str, value: object, caller_lines: Iterable[str]) -> bool:
    """Return whether the caller said a value themselves, though the scenario never gave it.

    A mismatch is then the simulated caller leaving its brief, not the agent inventing
    something.
    """
    text = " ".join(caller_lines).lower()
    if name in PHONE_VALUES:
        tail = digits(value)[-9:]
        return bool(tail) and tail in spoken_digits(text)
    if name == "appointment":
        got = parse_json(value)
        if not isinstance(got, dict) or not got.get("day"):
            return False
        try:
            day = dt.date.fromisoformat(str(got["day"]))
        except ValueError:
            return False
        return str(day.year) in text or _MONTH_NAMES[day.month - 1] in text
    return False


def _from_tool(value: object, outputs: list[str]) -> bool:
    """Return whether a saved value, every field of it, is in an earlier tool result."""
    if value is None or not outputs:
        return False
    text = " ".join(outputs)
    got = parse_json(value)
    if isinstance(got, dict):
        parts = [as_text(v) for v in got.values() if v not in (None, "")]
        return bool(parts) and all(part in text for part in parts)
    return as_text(value) in text


def _fabricated(name: str, value: object, scenario: Scenario) -> list[str]:
    """Return what in a saved value matches nothing the caller could have given."""
    known = scenario.known
    if value is None:
        return []
    problems: list[str] = []
    if name in PHONE_VALUES:
        phones = {digits(p) for key in PHONE_VALUES for p in known.get(key, [])}
        got = digits(value)
        if got not in phones and not any(got.endswith(p[-9:]) for p in phones if p):
            problems.append(f"{name}={value}")
    elif name == "appointment":
        got = parse_json(value)
        if not isinstance(got, dict):
            return [f"appointment={value!r}"]
        service = str(got.get("service") or "").lower()
        if not any(word in service for word in known.get("service", [])):
            problems.append(f"service={got.get('service')!r}")
        if got.get("day") not in known.get("day", []):
            problems.append(f"day={got.get('day')}")
        if got.get("time") not in known.get("time", []):
            problems.append(f"time={got.get('time')}")
    elif name == "order_id" and _norm_id(value) not in {
        _norm_id(v) for v in known.get("order_id", [])
    }:
        problems.append(f"order_id={value}")
    return problems


def _leaves(value: object) -> list[str]:
    """Return the scalar texts inside a saved value."""
    value = parse_json(value)
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _leaves(item)]
    if value is None:
        return []
    return [value if isinstance(value, str) else json.dumps(value)]


# -- reading the timeline -------------------------------------------------------


def events_of(records: Records) -> Records:
    """Return the timeline of a call's records."""
    return [r for r in records if r.get("type") == "event"]


def call_of(records: Records) -> dict[str, Any]:
    """Return the call record of a call's records."""
    return next(r for r in records if r.get("type") == "call")


def mock_requests(records: Records) -> Records:
    """Return every request the mocks answered during the call."""
    return [
        request
        for record in records
        if record.get("type") == "mock_requests"
        for request in record.get("requests") or []
    ]


def _task_events(events: Records) -> list[tuple[int, dict[str, Any]]]:
    """Return each task event with its place in the timeline."""
    return [(i, e) for i, e in enumerate(events) if e["kind"] == "task_event"]


def _world_tool(event: dict[str, Any]) -> bool:
    """Return whether a timeline event is a call of a tool that returns something of the world."""
    return event["kind"] == "tool_call" and event.get("name") not in RUNTIME_TOOLS


# -- the checks -------------------------------------------------------------------


def check_no_errors(call: dict[str, Any]) -> CheckResult:
    """No error of any kind. The endpoint's and the simulator's also leave the call unscored."""
    errors = call.get("errors") or []
    return CheckResult(
        id="no_errors",
        core=True,
        passed=not errors,
        category=",".join(sorted({str(e.get("kind", "?")) for e in errors})),
        detail="; ".join(f"{e.get('kind')}: {str(e.get('error') or '')[:200]}" for e in errors),
        evidence=errors[:5],
    )


def check_values(call: dict[str, Any], scenario: Scenario) -> CheckResult:
    """Every value the scenario expects was left saved as it expects."""
    variables = call.get("variables") or {}
    problems = []
    for name, matcher in scenario.expect.items():
        ok, why = match_value(matcher, (variables.get(name) or {}).get("stored"))
        if not ok:
            problems.append(f"{name}: {why}")
    return CheckResult(
        id="saved_values_match",
        core=True,
        passed=not problems,
        category="value_mismatch" if problems else "",
        detail="; ".join(problems),
    )


def check_fabrication(events: Records, scenario: Scenario) -> CheckResult:
    """No saved value the caller never gave.

    A value outside the scenario's facts that the caller did say is the simulated
    caller's deviation, reported under its own category. A value read off an earlier tool
    result is not invented either.
    """
    found: list[dict[str, Any]] = []
    caller_lines: list[str] = []
    tool_outputs: list[str] = []
    for e in events:
        if e["kind"] == "caller_said":
            caller_lines.append(e["text"])
        elif _world_tool(e):
            tool_outputs.append(as_text(e.get("output") or ""))
        elif e["kind"] == "state_write":
            for name, value in (e.get("applied") or {}).items():
                if _from_tool(value, tool_outputs):
                    continue
                if problems := _fabricated(name, value, scenario):
                    found.append(
                        {
                            "t": e["t"],
                            "writer": e.get("writer") or e.get("origin") or "main agent",
                            "values": problems,
                            "caller_said_it": caller_said_it(name, value, caller_lines),
                        }
                    )
    invented = [f for f in found if not f["caller_said_it"]]
    category = "fabricated_value" if invented else "caller_off_script_value" if found else ""
    return CheckResult(
        id="no_fabricated_save",
        core=True,
        passed=not invented,
        category=category,
        detail="; ".join(
            f"{f['writer']} saved {', '.join(f['values'])}"
            + (" (the caller said it)" if f["caller_said_it"] else "")
            for f in found
        ),
        evidence=found,
    )


def _answers(saved: Mapping[str, Any], line: str) -> bool:
    """Return whether a caller line plainly gave the saved values, heard before the step."""
    low = line.lower()
    for name, value in saved.items():
        if name == "wants_reminder":
            if "remind" not in low:
                return False
        elif name in PHONE_VALUES:
            if digits(value)[-9:] not in spoken_digits(low):
                return False
        elif name == "order_id":
            if _norm_id(value) not in _norm_id(line):
                return False
        elif name == "appointment":
            words = r"\d|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|noon"
            if not re.search(words, low):
                return False
        else:
            return False
    return True


def _opening_save(events: Records, i: int, task: str, saved: dict[str, Any]) -> dict[str, Any]:
    """Return a save a step made before the caller spoke in it, and what it came from.

    It came from a tool result that ended the step, from the caller's line just before the
    step, or from older turns.
    """
    last_caller = next((x["text"] for x in reversed(events[:i]) if x["kind"] == "caller_said"), "")
    by_tool = next(
        (
            x["stage"] == "task.ended_on_tool"
            for x in events[i + 1 :]
            if x["kind"] == "task_event" and x["data"].get("task") == task
        ),
        False,
    )
    answered = _answers(saved, last_caller)
    return {
        "t": events[i]["t"],
        "task": task,
        "saved": saved,
        "last_caller_line": last_caller,
        "category": "opening_save_by_tool_result"
        if by_tool
        else "opening_save_answered_just_before"
        if answered
        else "opening_save_from_older_turns",
    }


def check_save_after_caller_spoke(events: Records) -> CheckResult:
    """A step's save needs a caller turn inside that step.

    Unless every value it saves came back from a tool the step ran: a step whose whole job
    is an action has nothing to ask the caller. Saves held for the call's end are the
    caller's earlier answers, and are left to ``check_no_wrong_call_end_save``.
    """
    violations: list[dict[str, Any]] = []
    started_at: dict[str, int] = {}
    caller_since: dict[str, int] = {}
    tool_output: dict[str, str] = {}
    for i, e in enumerate(events):
        if e["kind"] == "call_end_saves":
            break
        if e["kind"] == "task_event" and e["stage"] == "task.started":
            task = e["data"].get("task")
            started_at[task] = i
            caller_since[task] = 0
            tool_output[task] = ""
        elif e["kind"] == "caller_said":
            for task in caller_since:
                caller_since[task] += 1
        elif _world_tool(e):
            for task in tool_output:
                tool_output[task] += "\n" + as_text(e.get("output") or "")
        elif e["kind"] == "state_write" and (task := e.get("writer")):
            if task not in started_at or caller_since.get(task, 0) > 0:
                continue
            saved = {k: v for k, v in (e.get("applied") or {}).items() if v is not None}
            returned = tool_output.get(task, "")
            if not saved or all(
                leaf in returned for value in saved.values() for leaf in _leaves(value)
            ):
                continue
            violations.append(_opening_save(events, i, task, saved))
    return CheckResult(
        id="no_save_before_caller_spoke_in_step",
        core=True,
        passed=not violations,
        category=",".join(sorted({v["category"] for v in violations})),
        detail="; ".join(
            f"{v['task']} saved {', '.join(v['saved'])} before the caller spoke in it"
            for v in violations
        ),
        evidence=violations,
    )


def check_no_wrong_call_end_save(
    call: dict[str, Any], events: Records, scenario: Scenario
) -> CheckResult:
    """No save made as the call ends that should not have been.

    A held save is the caller's early answer, made at the call's end only if they did not
    take it back. Wrong is a value the scenario says must not be saved or is not what it
    expects, and any one reported as confirmed for a value a task confirms: a save at the
    call's end is never the caller's agreement. It is core only for the scenarios tagged
    ``early_answer``; elsewhere it is counted as ``wrong_saves`` in the summary.
    """
    variables = call.get("variables") or {}
    wrong = []
    at_end = False
    for e in events:
        if e["kind"] == "call_end_saves":
            at_end = True
            continue
        if not at_end or e["kind"] != "state_write":
            continue
        for name, value in (e.get("applied") or {}).items():
            if name == CALL_VALUE:
                continue
            problems = []
            if name in scenario.expect:
                ok, why = match_value(scenario.expect[name], value)
                if not ok:
                    problems.append(why)
            confirmed = (e.get("confirmed") or {}).get(name)
            if confirmed and (variables.get(name) or {}).get("confirming_task"):
                problems.append("reported as confirmed")
            if problems:
                wrong.append({"t": e["t"], "var": name, "value": value, "why": problems})
    return CheckResult(
        id="no_wrong_call_end_save",
        core="early_answer" in scenario.tags,
        passed=not wrong,
        category="wrong_call_end_save" if wrong else "",
        detail="; ".join(
            f"{w['var']}={w['value']!r} saved at call end: {', '.join(w['why'])}" for w in wrong
        ),
        evidence=wrong,
    )


def _runs(events: Records) -> dict[str, dict[str, Any]]:
    """Return the task events grouped by run id, with the delegate each run belongs to."""
    runs: dict[str, dict[str, Any]] = {}
    for i, e in _task_events(events):
        data = e["data"]
        run_id = data.get("task_run_id")
        if run_id is None:
            continue
        run = runs.setdefault(run_id, {"first": i, "events": []})
        run["events"].append(e)
        if "delegate" not in run:
            if data.get("group"):
                run["delegate"] = data["group"]
            elif e["stage"] in (
                "task.requires",
                "task.entered_via",
                "task.refused_rerun",
                "task.started",
            ):
                run["delegate"] = data.get("task")
    return runs


def _expected_record(
    run: dict[str, Any], providers: Mapping[str, str]
) -> tuple[str | None, list[tuple[str, str, tuple[str, ...]]]]:
    """Return the status and the step rows a run's record should have, from its events."""
    status = None
    steps: list[tuple[str, str, tuple[str, ...]]] = []
    delegate = run.get("delegate")
    stages = {e["stage"] for e in run["events"]}
    provider = providers.get(delegate or "")
    if provider and "group.started" not in stages and "task.requires" not in stages:
        # A provider whose values were already set is listed as skipped, with no event.
        steps.append((provider, "skipped", ()))
    for e in run["events"]:
        stage, data = e["stage"], e["data"]
        if stage in _END_STAGES:
            steps.append((data.get("task"), _END_STAGES[stage], tuple(data.get("saved") or ())))
            if not data.get("group") and data.get("task") == delegate:
                status = _STATUS[_END_STAGES[stage]]
        elif stage == "group.step_skipped":
            steps.append((data.get("task"), "skipped", ()))
        elif stage == "group.completed":
            status = "completed"
        elif stage == "group.stopped":
            last = next((s for s in reversed(steps) if s[0] == data.get("task")), None)
            status = {"handed back": "unserved", "failed": "failed"}.get(last[1] if last else "")
    if status is None and steps and not any(e["data"].get("group") for e in run["events"]):
        # A task whose prerequisite ended it: the provider's outcome.
        last = steps[-1]
        status = _STATUS[last[1]] if last[1] != "skipped" else None
    return status, steps


def _finished(run: dict[str, Any]) -> bool:
    """Return whether a run returned. One still running when the caller hung up did not."""
    delegate = run.get("delegate")
    for e in run["events"]:
        if e["stage"] in ("group.completed", "group.stopped"):
            return True
        data = e["data"]
        if (
            e["stage"] in _END_STAGES
            and not data.get("group")
            and (data.get("task") == delegate or e["stage"] != "task.completed")
        ):
            return True
    return False


def check_run_records(
    events: Records,
    delegates: Iterable[str],
    providers: Mapping[str, str] | None = None,
    via_tools: Mapping[str, str] | None = None,
) -> CheckResult:
    """Each run record a delegate or a via tool returned matches its task events.

    A via tool's record belongs to its task's run entered through it (``task.entered_via``);
    a delegate's to the task's or group's other runs. A record no model ever read, because
    the caller hung up first, is not in the timeline, so a run may lack its record.
    """
    via_tools = via_tools or {}
    delegates = set(delegates)
    runs = sorted(_runs(events).values(), key=lambda r: r["first"])
    outputs: dict[tuple[str, bool], list[Any]] = {}
    for e in events:
        if e["kind"] != "tool_call":
            continue
        name = str(e.get("name") or "")
        out = parse_json(e.get("output"))
        if name in via_tools:
            if out is None or (isinstance(out, dict) and "status" in out and "steps" in out):
                outputs.setdefault((via_tools[name], True), []).append(out)
        elif name in delegates:
            outputs.setdefault((name, False), []).append(out)
    by_delegate: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for run in runs:
        via = any(e["stage"] == "task.entered_via" for e in run["events"])
        by_delegate.setdefault((run.get("delegate") or "?", via), []).append(run)
    problems = []
    for (name, via), outs in outputs.items():
        runs_of = by_delegate.get((name, via), [])
        problems.extend(_record_problems(name, via, outs, runs_of, providers or {}))
    return CheckResult(
        id="run_record_matches_events",
        core=True,
        passed=not problems,
        category="run_record_mismatch" if problems else "",
        detail="; ".join(problems),
    )


def _record_problems(
    name: str,
    via: bool,
    outs: list[Any],
    runs: list[dict[str, Any]],
    providers: Mapping[str, str],
) -> list[str]:
    """Return how one delegate's returned records differ from its runs' events."""
    label = f"{name} (via)" if via else name
    records = [o for o in outs if isinstance(o, dict) and "status" in o]
    refusals = [o for o in outs if isinstance(o, dict) and "refused" in o]
    unread = sum(1 for o in outs if o is None)
    refused = [r for r in runs if any(e["stage"] == "task.refused_rerun" for e in r["events"])]
    started = [r for r in runs if r not in refused and _finished(r)]
    problems = []
    if not len(started) - unread <= len(records) <= len(started):
        problems.append(f"{label}: {len(records)} run records returned, {len(started)} runs")
    if len(refused) > len(refusals) + unread:
        problems.append(f"{label}: refused_rerun event without a refusal result")
    for record, run in zip(records, started, strict=False):
        status, steps = _expected_record(run, providers)
        if record.get("task") != name:
            problems.append(f"{label}: record names {record.get('task')}")
        if status is not None and record.get("status") != status:
            problems.append(f"{label}: record status {record.get('status')}, events say {status}")
        got = sorted(
            (
                (row.get("step"), row.get("outcome"), tuple(sorted((row.get("saved") or {}))))
                for row in record.get("steps") or []
            ),
            key=str,
        )
        want = sorted(((s, o, tuple(sorted(saved))) for s, o, saved in steps), key=str)
        if got != want:
            problems.append(f"{label}: record steps {got} vs events {want}")
    return problems


def check_ended_naturally(call: dict[str, Any], scenario: Scenario) -> CheckResult:
    """One side hung up within the turn cap."""
    ended = call.get("ended_by")
    ok = ended in ("caller_hangup", "agent_hangup") and call.get("turns", 0) <= scenario.turn_cap
    return CheckResult(
        id="ended_naturally",
        core=True,
        passed=ok,
        category="" if ok else f"ended_by_{ended}",
        detail=f"ended by {ended} after {call.get('turns')} caller turns (cap {scenario.turn_cap})",
    )


def _started(events: Records, task: str, after: int = -1) -> bool:
    """Return whether a task started after the given place in the timeline."""
    return any(
        i > after
        and e["data"].get("task") == task
        and e["stage"] in ("task.started", "task.requires")
        for i, e in _task_events(events)
    )


def _any_of(
    alternatives: list[dict[str, Any]], events: Records, variables: Mapping[str, Any]
) -> bool:
    """Return whether one of the alternatives held.

    An alternative is a saved ``value``, like ``{"appointment.time": "17:00"}``, or a task
    started after the first group ended (``task_started_after_first_group``).
    """
    first_group_end = next(
        (i for i, e in _task_events(events) if e["stage"] in ("group.completed", "group.stopped")),
        None,
    )
    for alt in alternatives:
        if "value" in alt:
            ((path, want),) = alt["value"].items()
            name, _, field = path.partition(".")
            got = parse_json((variables.get(name) or {}).get("stored"))
            if isinstance(got, dict) and got.get(field) == want:
                return True
        later = alt.get("task_started_after_first_group")
        if later and first_group_end is not None and _started(events, later, first_group_end):
            return True
    return False


def check_requirements(
    call: dict[str, Any], records: Records, scenario: Scenario
) -> CheckResult | None:
    """What the scenario requires happened on the way, or None when it requires nothing.

    ``requires`` takes:

    * ``task_started``: a task that must start.
    * ``mock_called``: a mock that must answer a request.
    * ``tools_called``: tools that must run.
    * ``time_in_availability``: the saved appointment is at a free time.
    * ``any_of``: alternatives, one of which must hold: a saved ``value``
      (``{"appointment.time": "17:00"}``) or a task started after the first group ended
      (``task_started_after_first_group``).
    * ``injected``: see ``check_injected_arguments``.
    """
    req = scenario.requires
    if not req.keys() - {"injected"}:
        return None
    events = events_of(records)
    requests = mock_requests(records)
    variables = call.get("variables") or {}
    called = {e.get("name") for e in events if e["kind"] == "tool_call"}
    called |= {r.get("tool") for r in requests}

    problems = []
    if (task := req.get("task_started")) and not _started(events, task):
        problems.append(f"{task} never started")
    if (mock := req.get("mock_called")) and not any(r.get("mock") == mock for r in requests):
        problems.append(f"the {mock} mock was never called")
    problems.extend(
        f"{name} was never called" for name in req.get("tools_called") or () if name not in called
    )
    if req.get("time_in_availability"):
        booked = parse_json((variables.get("appointment") or {}).get("stored"))
        if isinstance(booked, dict) and booked.get("day"):
            free = AVAILABLE_TIMES[dt.date.fromisoformat(booked["day"]).weekday()]
            if booked.get("time") not in free:
                problems.append(f"booked {booked.get('time')} on {booked['day']}, free: {free}")
    if (alternatives := req.get("any_of")) and not _any_of(alternatives, events, variables):
        problems.append(f"none of {alternatives} happened")
    return CheckResult(
        id="scenario_requirements",
        core=True,
        passed=not problems,
        category="requirement_unmet" if problems else "",
        detail="; ".join(problems),
    )


def check_served_not_handed_back(events: Records, scenario: Scenario) -> CheckResult | None:
    """No step hands back a request an earlier task already served.

    ``scenario.served`` names the words each task's request is said in. Once that task
    saved, a finish whose ``unserved_request`` has one of them hands the served request
    back. A request to change it is not counted. "Saved" is ``task.completed``, or
    ``task.unserved`` carrying ``saved``: a hand-back that saved what the caller confirmed
    with it.

    Returns:
        The result, or None when the scenario names no served requests.
    """
    if not scenario.served:
        return None
    done: set[str] = set()
    found = []
    for e in events:
        if e["kind"] == "task_event":
            data = e["data"]
            if e["stage"] == "task.completed" or (
                e["stage"] == "task.unserved" and data.get("saved")
            ):
                done.add(data.get("task"))
            continue
        if e["kind"] != "tool_call" or e.get("name") != FINISH_TOOL_NAME:
            continue
        arguments = parse_json(e.get("arguments"))
        request = (
            str(arguments.get("unserved_request") or "").lower()
            if isinstance(arguments, dict)
            else ""
        )
        if any(word in request for word in _CHANGE_WORDS):
            continue
        found.extend(
            f"{task} was done, then handed back: {request!r}"
            for task in sorted(done & set(scenario.served))
            if any(word in request for word in scenario.served[task])
        )
    return CheckResult(
        id="no_served_request_handed_back",
        core=True,
        passed=not found,
        category="served_request_handed_back" if found else "",
        detail="; ".join(found),
    )


def check_owner_rewrites(events: Records, task_values: Iterable[str] = ()) -> CheckResult:
    """Extra: the main agent writes a value a task owns, through a tool.

    After the task saved it (a rewrite, which withdraws the value's confirmation), or before
    any task did (the value is set without the task that asks for it ever agreeing it).
    Writes by the prefetch, an event or the call itself are not the main agent's.
    """
    owned = set(task_values)
    task_saved: dict[str, str] = {}
    found = []
    for e in events:
        if e["kind"] != "state_write":
            continue
        writer = e.get("writer")
        if not writer and e.get("origin") != "tool":
            continue
        for name, value in (e.get("applied") or {}).items():
            if writer:
                task_saved[name] = writer
            elif name in task_saved or name in owned:
                found.append(
                    {
                        "t": e["t"],
                        "var": name,
                        "value": value,
                        "saved_by": task_saved.get(name),
                        "confirmed_after": (e.get("confirmed") or {}).get(name),
                    }
                )
    categories = sorted(
        {
            "owner_rewrote_task_value" if f["saved_by"] else "owner_wrote_task_value_first"
            for f in found
        }
    )
    return CheckResult(
        id="owner_does_not_rewrite_task_values",
        core=False,
        passed=not found,
        category=",".join(categories),
        detail="; ".join(
            (
                f"main agent rewrote {f['var']} (saved by {f['saved_by']})"
                if f["saved_by"]
                else f"main agent wrote {f['var']} before any task saved it"
            )
            + (" and unconfirmed it" if f["confirmed_after"] is False else "")
            for f in found
        ),
        evidence=found,
    )


def check_fallback_lines(events: Records) -> CheckResult:
    """Extra: no step gave up on a model that kept answering with nothing."""
    found = [
        e["text"]
        for e in events
        if e["kind"] == "agent_said" and any(line in e["text"] for line in FALLBACK_LINES)
    ]
    return CheckResult(
        id="no_empty_response_fallback",
        core=False,
        passed=not found,
        category="empty_response_fallback" if found else "",
        detail="; ".join(found),
    )


def check_injected_arguments(
    call: dict[str, Any], records: Records, scenario: Scenario
) -> CheckResult | None:
    """A tool's hidden arguments are the values saved by the end of the call.

    ``requires["injected"]`` is ``{mock: {argument: value[.field]}}``: every request that
    mock answered carried, for each argument, what the call ended up saving there. The
    model never sees these arguments; the mock log keeps them.

    Returns:
        The result, or None when the scenario names no injected arguments.
    """
    injected = scenario.requires.get("injected")
    if not injected:
        return None
    variables = call.get("variables") or {}
    requests = mock_requests(records)
    problems = []
    for mock, arguments in injected.items():
        made = [r for r in requests if r.get("mock") == mock]
        if not made:
            problems.append(f"the {mock} mock was never called")
        for request in made:
            for argument, source in arguments.items():
                root, _, field = source.partition(".")
                want: Any = (variables.get(root) or {}).get("stored")
                if field:
                    parent = parse_json(want)
                    want = parent.get(field) if isinstance(parent, dict) else None
                got = (request.get("params") or {}).get(argument)
                if got != want:
                    problems.append(f"{mock} {argument}={got!r}, saved {source}={want!r}")
    return CheckResult(
        id="injected_arguments_match",
        core=True,
        passed=not problems,
        category="injected_mismatch" if problems else "",
        detail="; ".join(problems),
    )


def run_checks(
    records: Records,
    scenario: Scenario,
    *,
    delegates: Iterable[str] = (),
    providers: Mapping[str, str] | None = None,
    via_tools: Mapping[str, str] | None = None,
    task_values: Iterable[str] = (),
) -> list[CheckResult]:
    """Run every deterministic check on one call.

    Args:
        records: The call's records.
        scenario: Its scenario.
        delegates: The tools of the main agent that start a task or a group.
        providers: Each task to the one task that saves all it requires.
        via_tools: Each tool whose success enters a task, to that task.
        task_values: The State values some task saves.

    Returns:
        Each check's result. A check the scenario has nothing for is left out.
    """
    call = call_of(records)
    events = events_of(records)
    results = [
        check_no_errors(call),
        check_values(call, scenario),
        check_fabrication(events, scenario),
        check_save_after_caller_spoke(events),
        check_no_wrong_call_end_save(call, events, scenario),
        check_run_records(events, delegates, providers, via_tools),
        check_ended_naturally(call, scenario),
    ]
    optional = (
        check_requirements(call, records, scenario),
        check_served_not_handed_back(events, scenario),
        check_injected_arguments(call, records, scenario),
    )
    results.extend(result for result in optional if result is not None)
    results.extend([check_owner_rewrites(events, task_values), check_fallback_lines(events)])
    return results
