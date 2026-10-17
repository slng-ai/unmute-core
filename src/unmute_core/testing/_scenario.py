"""The scenario format, the checks and the report that both conversation runners share."""

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, StringConstraints
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.settings import EnvSettings
from unmute_core.state import CallState

Text = Annotated[str, StringConstraints(min_length=1)]


class _Scenario(BaseModel):
    """Immutable, and refuses keys it does not know, so a typo in a scenario fails."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ScenarioTurn(_Scenario):
    """One caller line, and what the agent must do with it.

    Attributes:
        caller: What the caller says.
        expect: Text that must appear in what the agent said this turn, ignoring case.
        reject: Text that must not appear.
        max_requests: The most model requests the turn may make. An audio run does not
            cap it: a spoken answer the turn detector splits in two costs one more.
        language: The language the line is spoken in on an audio run. None uses the
            run's caller language.
    """

    caller: Text
    expect: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()
    max_requests: Annotated[int, Field(ge=0)] | None = None
    language: str | None = None


class Scenario(_Scenario):
    """A whole call, typed by a scripted caller, run on an agent's real model.

    Attributes:
        name: What the call is, as a person would say it.
        path: The file it came from. ``load`` sets it.
        greeting: Text the greeting must hold.
        turns: The caller's lines, in order.
        state: A state path to the value it must hold when the call ends.
        call_start: What ``UNMUTE_CALL_START`` simulates: a phone caller's numbers, and
            the values a call starts with.
        dispatch: A ``call_start`` object sent the way a real dispatch sends it: in the
            LiveKit job's metadata, or in the body of the Pipecat runner's request. It
            wins over ``call_start``.
        refused: Text the project must log when the call does not start. The call then
            has no greeting and no session. Give such a scenario no turns.
    """

    name: Text
    path: str = ""
    greeting: tuple[str, ...] = ()
    turns: tuple[ScenarioTurn, ...] = ()
    state: dict[str, JsonValue] = Field(default_factory=dict)
    call_start: dict[str, JsonValue] | None = None
    dispatch: dict[str, JsonValue] | None = None
    refused: str | None = None


class ConverseSettings(EnvSettings):
    """How the runners run a call, read from the environment.

    Attributes:
        audio: Run the whole voice pipeline, with a caller whose lines are spoken.
        real_voice: Keep the project's own text to speech on a typed Pipecat run.
        verbose: Log the library's steps and each tool call's arguments.
        out: A folder to write each call into, as JSON.
        caller_language: The language caller lines are spoken in, unless a turn names one.
        voice_cache: The folder spoken caller lines are kept in.
        slng_api_key: The SLNG key that speaks the caller's lines on an audio run.
        slng_world_part: The SLNG region that speaks them.
    """

    audio: bool = Field(default=False, validation_alias="CONVERSE_AUDIO")
    real_voice: bool = Field(default=False, validation_alias="CONVERSE_REAL_VOICE")
    verbose: bool = Field(default=False, validation_alias="CONVERSE_VERBOSE")
    out: Path | None = Field(default=None, validation_alias="CONVERSE_OUT")
    caller_language: str = Field(default="en", validation_alias="CONVERSE_CALLER_LANGUAGE")
    voice_cache: Path | None = Field(default=None, validation_alias="CONVERSE_VOICE_CACHE")
    slng_api_key: SecretStr | None = Field(default=None, validation_alias="SLNG_API_KEY")
    slng_world_part: str = Field(default="eu-north", validation_alias="SLNG_WORLD_PART")


class Refusals(logging.Handler):
    """Keeps every error the project logs: why a call did not start."""

    def __init__(self) -> None:
        """Start with none."""
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the message."""
        self.messages.append(record.getMessage())


REFUSALS = Refusals()
STATES: list[CallState] = []
"""Every call state built while a scenario runs. The last one is the call's."""


def watch() -> None:
    """Remember every call state built from now on, and every error logged.

    A runner calls it once before the first call, so it can read the call's values and
    why a call did not start. Calling it again changes nothing.
    """
    if getattr(CallState.__init__, "_remembers", False):
        return
    init = CallState.__init__

    def remembered(self: CallState, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        STATES.append(self)

    remembered._remembers = True  # ty: ignore[unresolved-attribute]
    setattr(CallState, "__init__", remembered)  # noqa: B010
    logging.getLogger().addHandler(REFUSALS)


@original_errors
@dataclass(config=VALUE)
class Turn:
    """One caller line, and what the agent did with it.

    Attributes:
        caller: What the caller said.
        heard: What speech to text heard of the caller, on an audio run.
        said: Every line the agent said after it, in order.
        tools: The tools the agent called, in order.
        requests: The model requests the turn made.
    """

    caller: str
    heard: list[str] = dataclasses.field(default_factory=list)
    said: list[str] = dataclasses.field(default_factory=list)
    tools: list[str] = dataclasses.field(default_factory=list)
    requests: int = 0


@original_errors
@dataclass(config=VALUE)
class Call:
    """What one scenario's call did."""

    greeting: list[str] = dataclasses.field(default_factory=list)
    turns: list[Turn] = dataclasses.field(default_factory=list)
    problems: list[str] = dataclasses.field(default_factory=list)


def load(paths: list[str]) -> list[Scenario]:
    """Return the scenarios in the files named, each with the path it came from."""
    return [
        Scenario.model_validate({**json.loads(Path(path).read_text()), "path": path})
        for path in paths
    ]


def start_values(scenario: Scenario) -> None:
    """Set ``UNMUTE_CALL_START`` for the scenario's call, or clear it."""
    REFUSALS.messages.clear()
    STATES.clear()
    if scenario.call_start is not None:
        os.environ["UNMUTE_CALL_START"] = json.dumps(scenario.call_start)
    else:
        os.environ.pop("UNMUTE_CALL_START", None)


def refusal(scenario: Scenario, started: bool) -> list[str]:
    """Return what is wrong with how the call started, or did not.

    Args:
        scenario: The scenario, which may expect the call to be refused.
        started: Whether the call got a session.
    """
    wanted = scenario.refused
    logged = " | ".join(REFUSALS.messages)
    if wanted is None:
        return [] if started else [f"the call did not start: {logged or 'no reason logged'}"]
    if started:
        return [f"the call started, but it should be refused: {wanted!r}"]
    if wanted not in logged:
        return [f"refused, but the log does not say {wanted!r}: {logged or 'nothing logged'}"]
    return []


def check(scenario: Scenario, call: Call, *, audio: bool = False) -> list[str]:
    """Return every expectation the call missed.

    Args:
        scenario: What the call must do.
        call: What it did.
        audio: The call ran on audio, so ``max_requests`` is not capped.
    """
    problems = list(call.problems)
    greeting = " ".join(call.greeting).lower()
    problems.extend(
        f"greeting: missing {text!r}" for text in scenario.greeting if text.lower() not in greeting
    )
    for index, (want, got) in enumerate(zip(scenario.turns, call.turns, strict=False), 1):
        said = " ".join(got.said).lower()
        problems.extend(
            f"turn {index}: missing {text!r}" for text in want.expect if text.lower() not in said
        )
        problems.extend(
            f"turn {index}: said {text!r}" for text in want.reject if text.lower() in said
        )
        cap = want.max_requests
        if cap is not None and got.requests > cap and not audio:
            problems.append(f"turn {index}: {got.requests} model requests, at most {cap}")
    if len(call.turns) < len(scenario.turns):
        problems.append(f"only {len(call.turns)} of {len(scenario.turns)} turns ran")
    state = STATES[-1] if STATES else None
    for path, value in scenario.state.items():
        got = None if state is None else state.plain(path)
        if got != value:
            problems.append(f"state {path}: {got!r}, expected {value!r}")
    return problems


def report(framework: str, scenario: Scenario, call: Call, settings: ConverseSettings) -> bool:
    """Print the call as a transcript, then its problems. Return True when it passed.

    With ``CONVERSE_OUT`` set, also write the call as JSON into that folder.
    """
    problems = check(scenario, call, audio=settings.audio)
    print(f"\n=== {framework}: {scenario.name} ({scenario.path})")
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
    if settings.out is not None:
        settings.out.mkdir(parents=True, exist_ok=True)
        state = STATES[-1].values if STATES else {}
        record = {
            "framework": framework,
            "scenario": scenario.name,
            "greeting": call.greeting,
            "turns": [dataclasses.asdict(turn) for turn in call.turns],
            "requests": total,
            "problems": problems,
            "state": json.loads(json.dumps(state, default=str)),
        }
        name = f"{Path(scenario.path).stem}-{framework}.json"
        (settings.out / name).write_text(json.dumps(record, indent=2))
    return not problems


def exit_code(results: list[bool]) -> None:
    """Exit 0 when every scenario passed, else 1."""
    passed = sum(results)
    print(f"\n{passed} of {len(results)} scenarios passed")
    sys.exit(0 if passed == len(results) else 1)
