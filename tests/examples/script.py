"""What both conversation runners share: the scenario format, the checks and the report.

A scenario is a JSON file in an example's ``scenarios/`` folder: a whole call, typed by a
scripted caller, run against the example's real agent and real model::

    {
      "name": "a whole claim",
      "call_start": {"call": {"direction": "inbound", "calling_number": "+447700900123"}},
      "turns": [
        {"caller": "KP482913, born 5 June 1979.", "expect": ["Is your policy number"]},
        {"caller": "Yes.", "expect": ["Thank you, Daniel"], "max_requests": 2}
      ],
      "state": {"claim_number": "KC100231"}
    }

1. ``greeting`` (optional) is text the greeting must hold.
2. Each turn's ``expect`` holds text that must appear in what the agent said that turn,
   ignoring case. ``reject`` holds text that must not. Code-spoken lines are word for word,
   so they make exact checks; the model's own words are checked loosely, if at all.
3. ``max_requests`` caps the model requests the turn may make. An audio run reports the
   requests but does not cap them: a spoken answer the turn detector splits in two costs
   one more request, by design.
4. ``language`` is the language a caller line is spoken in on an audio run, ``en`` when
   left out.
5. ``state`` maps a state path to the value it must hold when the call ends.
6. ``call_start`` is what ``UNMUTE_CALL_START`` simulates: a phone caller's numbers, and
   the values a call starts with, like ``{"account_id": "WA-1001"}``.
7. ``dispatch`` is a ``call_start`` object sent the way a real dispatch sends it: in the
   LiveKit job's metadata, or in the body of the Pipecat runner's request. It wins over
   ``call_start``.
8. ``refused`` is text the voice stack must log when the call does not start, like
   ``Missing call_start fields: account_id``. The call must then have no greeting and no
   session. Give such a scenario no turns.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unmute_core.state import CallState


class Refusals(logging.Handler):
    """Keeps every error the voice stack logs: why a call did not start."""

    def __init__(self) -> None:
        """Start with none."""
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the message."""
        self.messages.append(record.getMessage())


REFUSALS = Refusals()
logging.getLogger("voice_stack").addHandler(REFUSALS)

STATES: list[CallState] = []
"""Every call state built while a scenario runs. The last one is the call's."""

_init = CallState.__init__


def _remembered(self: CallState, *args: Any, **kwargs: Any) -> None:
    """Build the state, then remember it, so a runner can read the call's values."""
    _init(self, *args, **kwargs)
    STATES.append(self)


setattr(CallState, "__init__", _remembered)  # noqa: B010


@dataclass
class Turn:
    """One caller line, and what the agent did with it.

    Attributes:
        caller: What the caller said.
        said: Every line the agent said after it, in order.
        tools: The tools the agent called, in order.
        requests: The model requests the turn made.
        heard: What speech to text heard of the caller, on an audio run.
    """

    caller: str
    heard: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    requests: int = 0


@dataclass
class Call:
    """What one scenario's call did."""

    greeting: list[str] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def load(paths: list[str]) -> list[dict[str, Any]]:
    """Return the scenarios in the files named, each with the path it came from."""
    return [{**json.loads(Path(path).read_text()), "path": path} for path in paths]


def start_values(scenario: dict[str, Any]) -> None:
    """Set ``UNMUTE_CALL_START`` for the scenario's call, or clear it."""
    REFUSALS.messages.clear()
    if "call_start" in scenario:
        os.environ["UNMUTE_CALL_START"] = json.dumps(scenario["call_start"])
    else:
        os.environ.pop("UNMUTE_CALL_START", None)


def refusal(scenario: dict[str, Any], started: bool) -> list[str]:
    """Return what is wrong with how the call started, or did not.

    Args:
        scenario: The scenario, which may expect the call to be refused.
        started: Whether the call got a session.
    """
    wanted = scenario.get("refused")
    logged = " | ".join(REFUSALS.messages)
    if wanted is None:
        return [] if started else [f"the call did not start: {logged or 'no reason logged'}"]
    if started:
        return [f"the call started, but it should be refused: {wanted!r}"]
    if wanted not in logged:
        return [f"refused, but the log does not say {wanted!r}: {logged or 'nothing logged'}"]
    return []


def check(scenario: dict[str, Any], call: Call) -> list[str]:
    """Return every expectation the call missed."""
    problems = list(call.problems)
    greeting = " ".join(call.greeting).lower()
    problems.extend(
        f"greeting: missing {text!r}"
        for text in scenario.get("greeting", [])
        if text.lower() not in greeting
    )
    for index, (want, got) in enumerate(zip(scenario["turns"], call.turns, strict=False), 1):
        said = " ".join(got.said).lower()
        problems.extend(
            f"turn {index}: missing {text!r}"
            for text in want.get("expect", [])
            if text.lower() not in said
        )
        problems.extend(
            f"turn {index}: said {text!r}"
            for text in want.get("reject", [])
            if text.lower() in said
        )
        # A spoken answer the turn detector splits in two costs one more request, by design.
        audio = bool(os.getenv("CONVERSE_AUDIO"))
        if (cap := want.get("max_requests")) is not None and got.requests > cap and not audio:
            problems.append(f"turn {index}: {got.requests} model requests, at most {cap}")
    if len(call.turns) < len(scenario["turns"]):
        problems.append(f"only {len(call.turns)} of {len(scenario['turns'])} turns ran")
    state = STATES[-1] if STATES else None
    for path, value in scenario.get("state", {}).items():
        got = None if state is None else state.plain(path)
        if got != value:
            problems.append(f"state {path}: {got!r}, expected {value!r}")
    return problems


def report(framework: str, scenario: dict[str, Any], call: Call) -> bool:
    """Print the call as a transcript, then its problems. Return True when it passed.

    With ``CONVERSE_OUT`` set, also write the call as JSON into that folder.
    """
    problems = check(scenario, call)
    print(f"\n=== {framework}: {scenario['name']} ({scenario['path']})")
    for line in call.greeting:
        print(f"  agent : {line}")
    for turn in call.turns:
        print(f"  caller: {turn.caller}")
        if turn.heard:
            print(f"   heard: {' '.join(turn.heard)}")
        for line in turn.said:
            print(f"  agent : {line}")
        print(f"          [{turn.requests} model requests; tools: {', '.join(turn.tools) or '-'}]")
    total = sum(turn.requests for turn in call.turns)
    print(f"  model requests: {total} over {len(call.turns)} caller turns")
    for problem in problems:
        print(f"  FAIL  {problem}")
    print(f"  {'PASS' if not problems else 'FAIL'}")
    if out := os.getenv("CONVERSE_OUT"):
        folder = Path(out)
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{Path(scenario['path']).stem}-{framework}.json"
        state = STATES[-1].values if STATES else {}
        record = {
            "framework": framework,
            "scenario": scenario["name"],
            "greeting": call.greeting,
            "turns": [turn.__dict__ for turn in call.turns],
            "requests": total,
            "problems": problems,
            "state": json.loads(json.dumps(state, default=str)),
        }
        (folder / name).write_text(json.dumps(record, indent=2))
    return not problems


def exit_code(results: list[bool]) -> None:
    """Exit 0 when every scenario passed, else 1."""
    passed = sum(results)
    print(f"\n{passed} of {len(results)} scenarios passed")
    sys.exit(0 if passed == len(results) else 1)
