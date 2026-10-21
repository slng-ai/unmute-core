"""A scenario: one simulated caller, the agent they ring, and what the call must leave behind.

A scenario file is JSON. ``agent`` names a bundled agent file or a path to one. The caller
gets ``persona``, ``goal``, ``facts`` and ``moves``. What the call must leave behind:

* ``expect``: each saved value, as a matcher (``_checks.match_value``). A value that must
  never be saved is ``{"absent": true}``.
* ``known``: every value the caller can rightly give, so a saved value that matches none
  is an invented one, not a mismatch.
* ``requires``: what must happen on the way, like a task that starts or a mock that is
  called. ``judge_notes`` tell the judge what this call is about.

Dates are written as templates, so a scenario reads the same any day. They are filled in
the agent's time zone as the scenario loads:

* ``{{date:today}}``, ``{{date:+7}}``, ``{{date:next tuesday}}``: an ISO date. "Next" is
  the first such weekday at least three days ahead, so it is never ambiguous.
* ``{{say:next tuesday}}``: the same day the way a caller says it, with the month and year
  for a follow-up question.
"""

import datetime as dt
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from unmute_core.testing.simulate._agent import AgentFile, load_agent

SCENARIOS = Path(__file__).parent / "scenarios"
"""The scenario files that ship with the simulator."""

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)  # fmt: skip
_ORDINAL_WORDS = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth", "twentieth", "twenty-first", "twenty-second",
    "twenty-third", "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh",
    "twenty-eighth", "twenty-ninth", "thirtieth", "thirty-first",
)  # fmt: skip
_TEMPLATE = re.compile(r"\{\{(date|say):([a-z0-9+ ]+)\}\}")


def next_weekday(today: dt.date, weekday: int, *, min_days: int = 3) -> dt.date:
    """Return the first ``weekday`` at least ``min_days`` after ``today``."""
    day = today + dt.timedelta(days=min_days)
    return day + dt.timedelta(days=(weekday - day.weekday()) % 7)


def say_day(day: dt.date) -> str:
    """Return a day as the caller says it, with the month and year for a follow-up."""
    name = WEEKDAYS[day.weekday()].capitalize()
    return (
        f"{name} the {_ORDINAL_WORDS[day.day - 1]} (that is {name}, {MONTHS[day.month - 1]} "
        f"{day.day}, {day.year}; give the month and year only if asked)"
    )


def resolve_day(spec: str, today: dt.date) -> dt.date:
    """Return the day a template names: ``today``, ``+N`` or ``next <weekday>``.

    Raises:
        ValueError: If the template names no day.
    """
    if spec == "today":
        return today
    if spec.startswith("+") and spec[1:].isdigit():
        return today + dt.timedelta(days=int(spec[1:]))
    words = spec.split()
    if len(words) == 2 and words[0] == "next" and words[1] in WEEKDAYS:
        return next_weekday(today, WEEKDAYS.index(words[1]))
    raise ValueError(f"unknown date template {spec!r}: use today, +N or next <weekday>")


def fill_dates(value: Any, today: dt.date) -> Any:
    """Return JSON data with every date template filled."""
    if isinstance(value, str):

        def filled(match: re.Match[str]) -> str:
            day = resolve_day(match[2].strip(), today)
            return day.isoformat() if match[1] == "date" else say_day(day)

        return _TEMPLATE.sub(filled, value)
    if isinstance(value, Mapping):
        return {key: fill_dates(item, today) for key, item in value.items()}
    if isinstance(value, list):
        return [fill_dates(item, today) for item in value]
    return value


class Scenario(BaseModel):
    """One simulated caller and what their call must leave behind.

    Attributes:
        id: The scenario's name. Bundled ones are their file names.
        agent: The agent it calls: a bundled agent's name, or a path to an agent file.
        persona: Who the caller is.
        goal: What they want from the call.
        facts: Everything they know, by name. The only values they may give.
        moves: What they do, in order.
        expect: Saved value name to its matcher.
        known: Value name to every value the caller can rightly give.
        requires: What must happen on the way. See ``_checks.check_requirements``.
        served: Task to the words its request is said in, for the hand-back check.
        judge_notes: What the judge should know about this call.
        turn_cap: The most caller turns before the call is cut off.
        tags: Labels to pick scenarios by.
        line: What the line carries: ``calling_number`` and ``direction``. Empty for a
            call with no caller ID.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    agent: str
    persona: str
    goal: str
    facts: dict[str, str]
    moves: list[str]
    expect: dict[str, dict[str, Any]] = Field(default_factory=dict)
    known: dict[str, list[Any]] = Field(default_factory=dict)
    requires: dict[str, Any] = Field(default_factory=dict)
    served: dict[str, list[str]] = Field(default_factory=dict)
    judge_notes: str = ""
    turn_cap: int = Field(18, ge=1, le=60)
    tags: list[str] = Field(default_factory=list)
    line: dict[str, str] = Field(default_factory=dict)

    def caller_brief(self) -> dict[str, Any]:
        """Return what the simulated caller is told."""
        return {
            "persona": self.persona,
            "goal": self.goal,
            "facts": self.facts,
            "moves": self.moves,
        }


def _agent_path(name: str, base: Path) -> str:
    """Return an agent reference as ``load_agent`` reads it: a bundled name, or a path."""
    if not name.endswith(".json"):
        return name
    path = Path(name)
    return str(path if path.is_absolute() else base / path)


def load_scenario(path: Path) -> tuple[Scenario, AgentFile]:
    """Read one scenario file and its agent, with the dates filled for today.

    A relative agent path is read from the scenario file's folder.

    Raises:
        FileNotFoundError: If the agent is not there.
        ValueError: If the file does not fit ``Scenario``, or a date template is unknown.
    """
    raw = json.loads(path.read_text())
    agent = load_agent(_agent_path(str(raw.get("agent", "")), path.parent))
    today = dt.datetime.now(ZoneInfo(agent.timezone)).date()
    scenario = Scenario.model_validate(fill_dates(raw, today))
    return scenario.model_copy(update={"agent": agent.name}), agent


def scenario_paths(names: Sequence[str]) -> list[Path]:
    """Return the scenario files to run: every bundled one when ``names`` is empty.

    A name is a bundled scenario's id, a tag of bundled ones (``tag:booking``), or a path.

    Raises:
        FileNotFoundError: If a name matches nothing.
    """
    bundled = sorted(SCENARIOS.glob("*.json"))
    if not names:
        return bundled
    paths: list[Path] = []
    for name in names:
        if name.startswith("tag:"):
            tagged = [p for p in bundled if name[4:] in json.loads(p.read_text()).get("tags", [])]
            if not tagged:
                raise FileNotFoundError(f"no bundled scenario has the tag {name[4:]!r}")
            paths.extend(tagged)
        elif (path := Path(name)).suffix == ".json":
            if not path.is_file():
                raise FileNotFoundError(f"no scenario file {name}")
            paths.append(path)
        elif (path := SCENARIOS / f"{name}.json").is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(
                f"no scenario {name!r}: give a JSON file, tag:<tag>, or a bundled id "
                "(python -m unmute_core.testing scenarios lists them)"
            )
    return list(dict.fromkeys(paths))
