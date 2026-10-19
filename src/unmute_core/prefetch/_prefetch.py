"""What a call reads as it starts, saved into its state: before the greeting, or beside it.

Ported from unmute@d0c67f1 internal/generate/prefetch.go, which emits this as per-agent text.
Nothing here can fail a call: an entry that is skipped, times out or raises leaves its values
at their defaults, and the call greets on time. A ``Lookup`` may run beside the greeting
instead, with ``background=True``: then the first model request waits for it, unless the
lookup has ``wait=False``.

What an entry may save where is checked when it is built: a declared, single value,
of a type the reading fits, and a ``Literal`` that offers every value the reading can give.
"""

import asyncio
import inspect
import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import field
from datetime import datetime
from functools import cached_property
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, get_args
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    SkipValidation,
    ValidationInfo,
    model_validator,
)
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.state import CallState, StateOrigin
from unmute_core.state._call_info import (
    CALL,
    DIRECTIONS,
    CallChannel,
    CallInfo,
    call_model,
    reserved_name_error,
)
from unmute_core.state._paths import literal_choices, value_base
from unmute_core.tasks import CallStartResults
from unmute_core.tools import ToolRegistry, ToolSpec
from unmute_core.tools._spec import HIDDEN_RESULT, tool_succeeded

logger = logging.getLogger("unmute_core.prefetch")

BUDGET_S = 2.0
BACKGROUND_BUDGET_S = 10.0
"""Seconds each background lookup may take."""
RESULT_MAX_CHARS = 15_000
"""The longest result text a step's ``tool`` input shows."""

ClockField = Literal["date", "time", "datetime", "day_of_week", "year", "timezone"]
CLOCK_FIELDS: tuple[str, ...] = get_args(ClockField)
CallValueSource = Literal[
    "calling_number",
    "called_number",
    "remote_number",
    "direction",
    "channel",
    "carrier_call_id",
    "agent_id",
    "session_id",
]
"""The ``state.call`` fields a ``CallValue`` may copy: the ones set before the greeting."""
CALL_VALUE_SOURCES: tuple[str, ...] = get_args(CallValueSource)
_CLOSED = MappingProxyType({"direction": DIRECTIONS, "channel": get_args(CallChannel)})
FITS: dict[str, frozenset[str]] = {
    "date": frozenset({"str", "Date"}),
    "time": frozenset({"str", "Time"}),
    "datetime": frozenset({"str"}),
    "day_of_week": frozenset({"str", "Literal"}),
    "year": frozenset({"str", "int"}),
    "timezone": frozenset({"str", "Literal"}),
    "calling_number": frozenset({"str", "Phone"}),
    "called_number": frozenset({"str", "Phone"}),
    "remote_number": frozenset({"str", "Phone"}),
    "direction": frozenset({"str", "Literal"}),
    "channel": frozenset({"str", "Literal"}),
}
"""What each reading may be saved to, as the type grammar names a value's type."""

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _entry_name(info: ValidationInfo) -> str:
    """Return the name of the entry being built, for an error text."""
    return str(info.data.get("name", "?"))


def _names_not_a_string[T](assign: T, info: ValidationInfo) -> T:
    """Refuse a bare string where a list of value names belongs.

    Raises:
        TypeError: Naming the entry and the fix.
    """
    if isinstance(assign, str):
        raise TypeError(
            f"prefetch {_entry_name(info)}: assign takes a list of value names, not {assign!r}"
        )
    return assign


def _check_assign(name: str, assign: Mapping[str, object] | Sequence[str]) -> None:
    """Refuse an entry that saves nothing."""
    if not assign:  # a length of zero names no value; Field(min_length) words it differently
        raise ValueError(f"prefetch {name}: assign names no value, so the entry would save nothing")


def _known_zone(zone: str, info: ValidationInfo) -> str:
    """Return the zone once it is an IANA zone.

    Raises:
        ValueError: Naming the entry and a zone to use.
    """
    try:
        ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(
            f"prefetch {_entry_name(info)}: {zone!r} is not an IANA time zone, like 'Europe/Madrid'"
        ) from error
    return zone


def _known_clock_fields(assign: object, info: ValidationInfo) -> object:
    """Refuse a clock field the clock cannot read.

    Raises:
        ValueError: Naming the entry and the fields to use.
    """
    if isinstance(assign, Mapping) and (
        unknown := sorted({f for f in assign.values() if isinstance(f, str)} - set(CLOCK_FIELDS))
    ):
        raise ValueError(
            f"prefetch {_entry_name(info)}: {unknown[0]!r} is not a clock field. "
            f"Use one of {', '.join(CLOCK_FIELDS)}"
        )
    return assign


def _is_empty(value: object) -> bool:
    """Return True for a value with nothing in it. False and 0 are values."""
    return value is None or value == ""


@original_errors
@dataclass(frozen=True, config=VALUE)
class Reading:
    """What an entry reads into one value.

    Attributes:
        read: The clock field or ``state.call`` field, or None for a lookup's open result.
        values: Every value the reading can give, when that set is closed.
    """

    read: str | None = None
    values: tuple[str, ...] = ()


@original_errors
@dataclass(frozen=True, config=VALUE)
class Clock:
    """Read the clock in one zone, once, at the start of the call.

    The value is never refreshed, so a call that crosses midnight keeps the day it started on.
    The weekday is spelled out here rather than read from the locale.

    Attributes:
        name: The entry name, used in log lines.
        timezone: An IANA zone, like ``Europe/Madrid``. Never the container's clock.
        assign: State value name to the clock field it gets, like ``{"booking_date": "date"}``.
    """

    name: str
    timezone: Annotated[str, AfterValidator(_known_zone)]
    assign: Annotated[
        Mapping[str, ClockField],
        BeforeValidator(_names_not_a_string),
        BeforeValidator(_known_clock_fields),
    ]

    @model_validator(mode="after")
    def _saves_something(self) -> Self:
        """Refuse an entry that saves nothing."""
        _check_assign(self.name, self.assign)
        return self

    def read(self) -> dict[str, object]:
        """Read the clock once and derive every assigned field from that one reading."""
        now = datetime.now(ZoneInfo(self.timezone))
        fields = {
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M"),
            "datetime": now.isoformat(timespec="minutes"),
            "day_of_week": _DAYS[now.weekday()],
            "year": str(now.year),
            "timezone": self.timezone,
        }
        return {name: fields[field] for name, field in self.assign.items()}

    def readings(self) -> dict[str, Reading]:
        """Return what the entry reads into each value."""
        closed = {"day_of_week": _DAYS, "timezone": (self.timezone,)}
        return {name: Reading(field, closed.get(field, ())) for name, field in self.assign.items()}


@original_errors
@dataclass(frozen=True, config=VALUE)
class CallValue:
    """Copy a value the call carries into a value of your own, like the caller's number.

    ``state.call`` is read-only. Copy a number into your own value to confirm it with the
    caller, or to let a step correct it. A withheld caller ID is already absent in
    ``state.call``, so an absent value skips the entry.

    Attributes:
        name: The entry name, used in log lines.
        source: The ``state.call`` field to read, like ``calling_number``.
        assign: The state value names that get it.
    """

    name: str
    source: CallValueSource
    assign: Annotated[Sequence[str], BeforeValidator(_names_not_a_string)]

    @model_validator(mode="after")
    def _saves_something(self) -> Self:
        """Refuse an entry that saves nothing."""
        _check_assign(self.name, self.assign)
        return self

    def read(self, call: CallInfo) -> object:
        """Return the call's value, or None when the call carries none."""
        return getattr(call, self.source)

    def readings(self) -> dict[str, Reading]:
        """Return what the entry reads into each value. An id is read unchecked."""
        if self.source not in FITS:
            return dict.fromkeys(self.assign, Reading())
        return dict.fromkeys(self.assign, Reading(self.source, _CLOSED.get(self.source, ())))


class LookupFailed(Exception):  # noqa: N818 - read as "the lookup failed"
    """A tool a ``Lookup`` runs was refused or reported a failure."""


@original_errors
@dataclass(frozen=True, config=VALUE)
class Lookup:
    """Run a tool as the call starts and save fields of its result.

    Attributes:
        name: The entry name, used in log lines and as its label in ``Prefetch.results``.
        run: A ``ToolSpec``, run like any tool with its locked arguments filled from the
            state, ``{{call.<field>}}`` among them. Or a function called with the call state, returning
            the result mapping or an awaitable of it. A plain function runs in a thread, so
            a blocking one cannot delay the greeting.
        assign: State value name to the top-level result key it gets. A tool may save
            nothing: its result then only reaches a step's ``tool`` input, through
            ``Prefetch.results``.
        inputs: State value paths the lookup reads. The entry is skipped when one is empty,
            and the saved values are reset when one of them changes later.
        needs_confirmed: State values that must not be waiting for the caller.
        background: False runs it before the greeting, inside the prefetch budget. True
            runs it beside the greeting: ``Prefetch.run_background``.
        wait: True (the default) holds the first model request until a background lookup
            ends, for up to ``first_request_timeout_s``. False never holds a request, for a
            lookup that only tells another system something. It needs ``background=True``.
            The call's end waits for it, for up to ``end_timeout_s``, and neither the end
            nor a failure stops it.
    """

    name: str
    run: SkipValidation[ToolSpec | Callable[[CallState[Any]], object]]
    assign: Annotated[Mapping[str, str], BeforeValidator(_names_not_a_string)]
    inputs: Sequence[str] = ()
    needs_confirmed: Sequence[str] = ()
    background: bool = False
    wait: bool = True

    @model_validator(mode="after")
    def _runs_with_nothing_to_fill(self) -> Self:
        """Refuse an entry that saves nothing, or a tool the model would have to fill.

        Raises:
            ValueError: Naming the entry and the fix.
        """
        if not self.wait and not self.background:
            raise ValueError(
                f"prefetch entry {self.name!r}: wait=False needs background=True, because "
                "the greeting waits for a lookup that runs before it"
            )
        if not isinstance(self.run, ToolSpec):
            _check_assign(self.name, self.assign)
            return self
        if unset := self.run.unlocked_required:
            raise ValueError(
                f"prefetch entry {self.name!r}: tool argument {unset[0]} has no source. Lock it"
            )
        if self.run.announce:
            raise ValueError(
                f"prefetch entry {self.name!r}: nobody waits for its tool, so drop the "
                "announce line"
            )
        return self

    def readings(self) -> dict[str, Reading]:
        """Return what the entry reads into each value: a result field nothing checks."""
        return dict.fromkeys(self.assign, Reading())

    def can_run(self, state: CallState[Any]) -> bool:
        """Return False when an input is empty or a needed value waits for the caller."""
        return not any(_is_empty(state.plain(path)) for path in self.inputs) and not any(
            state.is_unconfirmed(name) for name in self.needs_confirmed
        )

    async def call(self, state: CallState[Any]) -> Mapping[str, object]:
        """Run the tool and return its result.

        Args:
            state: The call state.

        Raises:
            TypeError: If the tool returned something other than a mapping.
            LookupFailed: If the tool was refused or reported a failure.
            ToolError: If the tool failed.
        """
        if isinstance(self.run, ToolSpec):
            result = await ToolRegistry([self.run]).call(self.run.name, {}, state)
            if isinstance(result, str):
                result = _json_object(result)
            if not tool_succeeded(result):
                raise LookupFailed(f"the tool reported {result}")
        elif inspect.iscoroutinefunction(self.run):
            result = await self.run(state)
        else:
            result = await asyncio.to_thread(self.run, state)
            if inspect.isawaitable(result):
                result = await result
        if not isinstance(result, Mapping):
            raise TypeError(f"returned {type(result).__name__}, not a mapping of result fields")
        return result


Entry = Clock | CallValue | Lookup


def _json_object(text: str) -> object:
    """Read a tool's JSON text result, or keep the text when it is not JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _shown(entry: Lookup, result: Mapping[str, object]) -> object:
    """What a step may read of a result: ``{"ok": true}`` for a tool that hides it."""
    if isinstance(entry.run, ToolSpec) and not entry.run.show_result:
        return dict(HIDDEN_RESULT)
    return result


def _record(results: CallStartResults | None, name: str, result: object) -> None:
    """Keep a lookup's result for a step's ``tool`` input, as JSON, cut to its longest."""
    if results is not None:
        text = json.dumps(result, ensure_ascii=False, default=str)
        results.record(name, text[:RESULT_MAX_CHARS])


def _unique_names(entries: Sequence[Entry]) -> tuple[Entry, ...]:
    """Return the entries once no two share a name.

    Raises:
        ValueError: Naming the repeated entry.
    """
    if repeated := [name for name, count in Counter(e.name for e in entries).items() if count > 1]:
        raise ValueError(
            f"prefetch entry names must be unique: two entries are named {repeated[0]!r}"
        )
    return tuple(entries)


@original_errors
@dataclass(eq=False, config=VALUE)
class Prefetch:
    """Resolve a call's prefetch entries in order, inside one time budget.

    Example:
        ``await Prefetch([Clock("today", "Europe/Madrid", {"booking_date": "date"})]).run(state)``

    Attributes:
        entries: The entries, in the order they run.
        state: The State model. With it, every entry is checked when the prefetch is built
            against the types of the values it saves.
        budget: Seconds the entries before the greeting may take together. Past it the rest
            keeps its defaults.
        background_budget: Seconds each background lookup may take.

    Raises:
        ValueError: If two entries share a name, or an entry cannot save a value.
    """

    entries: Annotated[Sequence[Entry], AfterValidator(_unique_names)]
    state: type[BaseModel] | None = field(default=None, kw_only=True)
    budget: float = field(default=BUDGET_S, kw_only=True)
    background_budget: float = field(default=BACKGROUND_BUDGET_S, kw_only=True)

    @model_validator(mode="after")
    def _entries_fit_the_state(self) -> Self:
        """Check every entry against the State model, when there is one."""
        if self.state is not None:
            self._check(self.state)
        return self

    @property
    def fills(self) -> frozenset[str]:
        """The state values an entry fills before the greeting. Background lookups do not."""
        return frozenset(name for entry in self._before_greeting for name in entry.readings())

    @cached_property
    def _before_greeting(self) -> tuple[Entry, ...]:
        """The entries that run before the greeting."""
        return tuple(e for e in self.entries if not (isinstance(e, Lookup) and e.background))

    def _check(self, state: type[BaseModel]) -> None:
        """Check each saved value: declared, a single value, and of a type its reading fits.

        Raises:
            ValueError: Naming the entry, the value and the fix.
        """
        fields = {info.alias or name: info.annotation for name, info in state.model_fields.items()}
        for entry in self.entries:
            if isinstance(entry, Lookup) and isinstance(entry.run, ToolSpec):
                entry.run.check_locked(call_model(state))
            for name, reading in entry.readings().items():
                where = f"prefetch entry {entry.name!r}"
                if name == CALL:
                    raise reserved_name_error(where)
                if name not in fields:
                    raise ValueError(f"{where} assigns an undeclared runtime variable: {name!r}")
                base = value_base(fields[name])
                if base in ("shape", "list"):
                    raise ValueError(
                        f"{where} cannot fill the shape or list {name!r}: save it from a task "
                        "instead"
                    )
                if reading.read is None:
                    continue
                if base not in (fits := FITS[reading.read]):
                    raise ValueError(
                        f"{where} cannot save its {reading.read} to {name!r} ({base}): it fits "
                        + " or ".join(sorted(fits))
                    )
                offered = {str(choice) for choice in literal_choices(fields[name])}
                if base == "Literal" and (missing := sorted(set(reading.values) - offered)):
                    raise ValueError(
                        f"{where} can read a {reading.read} that {name!r} does not offer: "
                        + ", ".join(missing)
                    )

    async def run(
        self,
        state: CallState[Any],
        *,
        results: CallStartResults | None = None,
    ) -> None:
        """Resolve the entries before the greeting and save their values. Never raises.

        Background lookups wait for ``run_background``.

        Args:
            state: The call's state, filled in place.
            results: Keeps what each lookup returned, by entry name, for a step's ``tool``
                input. Give the same one to ``CallTasks(call_start=...)``.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.budget
        for entry in self._before_greeting:
            if loop.time() >= deadline:
                return
            try:
                await self._resolve(entry, state, deadline, results)
            except TimeoutError:
                logger.warning(
                    "prefetch %s: startup budget expired; %s", entry.name, _keeps_default(entry)
                )
                return
            except Exception:
                logger.exception("prefetch %s: failed; %s", entry.name, _keeps_default(entry))

    async def run_background(
        self,
        state: CallState[Any],
        *,
        results: CallStartResults | None = None,
        wait: bool | None = None,
    ) -> None:
        """Resolve the background lookups in order, each in its own budget. Never raises.

        Start it once the greeting has started. ``CallHooks`` does, and holds the first model
        request until the lookups with ``wait=True`` end.

        Args:
            state: The call's state, filled in place.
            results: Keeps what each lookup returned, as for ``run``.
            wait: True runs only the lookups with ``wait=True``, False only the ones with
                ``wait=False``. None runs them all.
        """
        loop = asyncio.get_running_loop()
        for entry in self.entries:
            if not (isinstance(entry, Lookup) and entry.background):
                continue
            if wait is not None and entry.wait != wait:
                continue
            try:
                deadline = loop.time() + self.background_budget
                await self._resolve(entry, state, deadline, results)
            except TimeoutError:
                logger.warning("prefetch %s: took too long; %s", entry.name, _keeps_default(entry))
            except Exception:
                logger.exception("prefetch %s: failed; %s", entry.name, _keeps_default(entry))

    async def _resolve(
        self,
        entry: Entry,
        state: CallState[Any],
        deadline: float,
        results: CallStartResults | None,
    ) -> None:
        """Resolve one entry and save it, or log why it was skipped."""
        inputs: tuple[str, ...] = ()
        if isinstance(entry, Clock):
            values = entry.read()
        elif isinstance(entry, CallValue):
            value = entry.read(state.call)
            if _is_empty(value):
                logger.info(
                    "prefetch %s: skipped, the call carries no %s", entry.name, entry.source
                )
                return
            values = dict.fromkeys(entry.assign, value)
        else:
            if not entry.can_run(state):
                logger.info("prefetch %s: skipped, an input is unavailable", entry.name)
                return
            try:
                result = await _wait(entry.call(state), deadline)
            except Exception as error:
                _record(results, entry.name, {"error": str(error) or type(error).__name__})
                raise
            _record(results, entry.name, _shown(entry, result))
            values = {name: result.get(field) for name, field in entry.assign.items()}
            inputs = tuple(dict.fromkeys(path.split(".")[0] for path in entry.inputs))
        state.save(
            values,
            inputs=inputs,
            origin=StateOrigin("prefetch", entry.name),
        )
        awaiting = any(name in state.confirm for name in values)
        logger.info(
            "prefetch %s: resolved %s%s",
            entry.name,
            ", ".join(values),
            ", awaiting confirmation" if awaiting else "",
        )


def _keeps_default(entry: Entry) -> str:
    """Name what a failed entry did not fill, agreeing in number."""
    names = ", ".join(entry.assign)
    return (
        f"{names} keeps its default" if len(entry.assign) == 1 else f"{names} keep their defaults"
    )


def _retrieve(task: asyncio.Future[Any]) -> None:
    """Read a finished lookup's exception, so asyncio never reports it as unread."""
    if not task.cancelled():
        task.exception()


async def _wait[T](work: Awaitable[T], deadline: float) -> T:
    """Run one lookup, giving up at the deadline without waiting for it to stop.

    Raises:
        TimeoutError: If the lookup was not done by the deadline.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(work)
    task.add_done_callback(_retrieve)
    try:
        done, _ = await asyncio.wait({task}, timeout=max(0, deadline - loop.time()))
        if not done or loop.time() >= deadline:
            raise TimeoutError
        return task.result()
    finally:
        if not task.done():
            task.cancel()
