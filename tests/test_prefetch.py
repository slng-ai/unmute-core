"""Prefetch: the three entry kinds, the one budget, and what the call carries.

Ported from unmute@d0c67f1 internal/generate/prefetch_smoke_test.go and prefetch_test.go.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ValidationError

from tests.outputs import AnyResult
from unmute_core.prefetch import CallValue, Clock, Lookup, Prefetch
from unmute_core.state import CallInfo, CallState, StateSchema, call_info, call_phone_region
from unmute_core.tasks import CallStartResults
from unmute_core.tools import ToolSpec, local_tool

MADRID = ZoneInfo("Europe/Madrid")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMBER = "+34600111222"


class SalonState(BaseModel):
    """The salon concierge's prefetched values."""

    today_date: str | None = None
    weekday: str | None = None
    year: str | None = None
    local_time: str | None = None
    stamp: str | None = None
    zone: str | None = None
    customer_phone: str | None = None
    customer_name: str = ""
    customer_id: str | None = None
    call_id: str | None = None
    customer_count: int | None = None


UNCONFIRMED = ("customer_phone", "customer_name", "customer_id")
CONFIRM = dict.fromkeys(UNCONFIRMED, "verify_caller")
DEPENDENCIES = {"customer_name": ["customer_phone"], "customer_id": ["customer_phone"]}
TODAY = Clock(
    "today",
    "Europe/Madrid",
    {
        "today_date": "date",
        "weekday": "day_of_week",
        "year": "year",
        "local_time": "time",
        "stamp": "datetime",
        "zone": "timezone",
    },
)
CALLER = CallValue("caller", "calling_number", ["customer_phone"])


def profile(run: Any, name: str = "profile") -> Lookup:
    """The customer lookup, keyed on the caller's number."""
    return Lookup(
        name,
        run=run,
        assign={"customer_name": "name", "customer_id": "id"},
        inputs=["customer_phone"],
    )


def look_up_customer(state: CallState) -> dict[str, str]:
    """A local tool, plain and blocking."""
    assert state.plain("customer_phone") == NUMBER
    return {"name": "Ana", "id": "C-1"}


def fresh(call: CallInfo | None = None) -> CallState:
    """A new call's state."""
    return CallState(SalonState, confirm=CONFIRM, dependencies=DEPENDENCIES, call=call)


def inbound(number: str) -> CallInfo:
    """What an inbound phone call from ``number`` carries."""
    return call_info("inbound", remote_number=number)


async def test_the_clock_is_read_in_the_declared_zone() -> None:
    state = fresh()
    await Prefetch([TODAY]).run(state)

    now = datetime.now(MADRID)
    assert state.values.today_date == now.date().isoformat()
    assert state.values.weekday == WEEKDAYS[now.weekday()]
    assert state.values.year == str(now.year)
    assert state.values.local_time is not None and len(state.values.local_time) == 5
    assert state.values.stamp is not None and state.values.stamp.startswith(now.date().isoformat())
    assert state.values.zone == "Europe/Madrid"


async def test_a_call_without_facts_skips_every_entry_that_needs_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, "unmute_core.prefetch")
    state = fresh()
    await Prefetch([TODAY, CALLER, profile(look_up_customer)]).run(state)

    assert state.values.today_date
    assert state.values.customer_phone is None
    assert state.values.customer_name == ""
    assert all(state.is_unconfirmed(name) for name in UNCONFIRMED)
    assert caplog.messages == [
        "prefetch today: resolved today_date, weekday, year, local_time, stamp, zone",
        "prefetch caller: skipped, the call carries no calling_number",
        "prefetch profile: skipped, an input is unavailable",
    ]


async def test_a_looked_up_caller_is_saved_and_waits_for_confirmation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, "unmute_core.prefetch")
    state = fresh(inbound(NUMBER))
    await Prefetch([CALLER, profile(look_up_customer)]).run(state)

    assert state.values.customer_phone == NUMBER
    assert (state.values.customer_name, state.values.customer_id) == ("Ana", "C-1")
    assert all(state.is_unconfirmed(name) for name in UNCONFIRMED)
    assert caplog.messages == [
        "prefetch caller: resolved customer_phone, awaiting confirmation",
        "prefetch profile: resolved customer_name, customer_id, awaiting confirmation",
    ]
    assert NUMBER not in caplog.text

    state.save({"customer_phone": NUMBER}, step="verify_caller")
    assert state.render("customer_name") == "Ana"


async def test_an_async_lookup_is_awaited() -> None:
    async def look_up(state: CallState) -> dict[str, str]:
        await asyncio.sleep(0)
        return {"name": "Ana", "id": "C-1"}

    state = fresh(inbound(NUMBER))
    await Prefetch([CALLER, profile(look_up)]).run(state)

    assert state.values.customer_name == "Ana"


@pytest.mark.parametrize(
    "withheld",
    [
        "",
        "anonymous",
        "ANONYMOUS",
        "+266696687",
        "+8628245225",
        "+34 600",
        "0034600111222",
        "+1",
        "+1234567890123456",
        "+0123456789",
        "+1234567",
    ],
)
async def test_a_withheld_caller_id_is_absent(withheld: str) -> None:
    state = fresh(inbound(withheld))
    await Prefetch([CALLER]).run(state)

    assert state.values.customer_phone is None


@pytest.mark.parametrize("real", [NUMBER, "+865669612345"])
async def test_a_real_number_is_kept_even_when_it_starts_like_a_placeholder(real: str) -> None:
    state = fresh(inbound(real))
    await Prefetch([CALLER]).run(state)

    assert state.values.customer_phone == real


async def test_an_id_is_copied_as_it_is() -> None:
    state = fresh(CallInfo(carrier_call_id="unknown"))
    await Prefetch([CallValue("call", "carrier_call_id", ["call_id"])]).run(state)

    assert state.values.call_id == "unknown"


async def test_one_budget_bounds_the_whole_block_and_late_results_are_dropped() -> None:
    calls = []

    async def slow(state: CallState) -> dict[str, str]:
        calls.append(state)
        try:
            await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            await asyncio.sleep(0.1)
        return {"name": "LATE_NAME", "id": "LATE_ID"}

    state = fresh(inbound(NUMBER))
    prefetch = Prefetch([CALLER, profile(slow), profile(slow, "second")], budget=0.12)
    started = time.monotonic()
    await prefetch.run(state)

    assert time.monotonic() - started < 0.2, "a fresh budget per entry, or waited on cancellation"
    assert len(calls) == 2
    before = (state.values.model_dump(), [state.is_unconfirmed(name) for name in UNCONFIRMED])
    await asyncio.sleep(0.15)
    assert (state.values.model_dump(), [state.is_unconfirmed(n) for n in UNCONFIRMED]) == before


async def test_a_blocking_tool_does_not_delay_the_greeting() -> None:
    def blocking(state: CallState) -> dict[str, str]:
        time.sleep(0.25)
        return {"name": "BLOCKING_NAME", "id": "BLOCKING_ID"}

    state = fresh(inbound(NUMBER))
    started = time.monotonic()
    await Prefetch([CALLER, profile(blocking)], budget=0.12).run(state)

    assert time.monotonic() - started < 0.2
    await asyncio.sleep(0.3)
    assert state.values.customer_name == ""


async def test_an_expired_budget_stops_the_block_and_says_what_kept_its_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def too_slow(state: CallState) -> dict[str, str]:
        await asyncio.sleep(1)
        return {}

    state = fresh(inbound(NUMBER))
    await Prefetch([CALLER, profile(too_slow), TODAY], budget=0.05).run(state)

    assert state.values.customer_phone == NUMBER
    assert state.values.customer_name == ""
    assert state.values.today_date is None
    assert (
        "prefetch profile: startup budget expired; customer_name, customer_id keep their defaults"
        in caplog.messages
    )


async def test_a_failed_lookup_is_logged_and_the_next_entry_still_runs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def explode(state: CallState) -> dict[str, str]:
        raise RuntimeError("the customer store is down")

    state = fresh(inbound(NUMBER))
    await Prefetch([CALLER, profile(explode), TODAY]).run(state)

    assert state.values.customer_name == ""
    assert state.values.today_date
    assert "prefetch profile: failed; customer_name, customer_id keep their defaults" in (
        caplog.messages
    )


@pytest.mark.parametrize("result", [["Ana"], {"count": "many"}])
async def test_a_result_that_does_not_fit_saves_nothing(result: object) -> None:
    count = Lookup("count", run=lambda state: result, assign={"customer_count": "count"})
    state = fresh()
    await Prefetch([count, TODAY]).run(state)

    assert state.values.customer_count is None
    assert state.values.today_date


async def test_a_long_value_is_saved_whole() -> None:
    long_name = "A" * 600
    state = fresh(inbound(NUMBER))
    await Prefetch([CALLER, profile(lambda state: {"name": long_name, "id": "C-1"})]).run(state)

    assert state.values.customer_name == long_name


async def test_a_lookup_waits_for_a_value_the_caller_has_not_confirmed() -> None:
    lookup = Lookup(
        "profile",
        run=look_up_customer,
        assign={"customer_name": "name"},
        needs_confirmed=["customer_phone"],
    )
    state = fresh(inbound(NUMBER))
    await Prefetch([CALLER, lookup]).run(state)
    assert state.values.customer_name == ""

    state.save({"customer_phone": NUMBER}, step="verify_caller")
    await Prefetch([lookup]).run(state)
    assert state.values.customer_name == "Ana"


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Clock("today", "Mars/Olympus", {"d": "date"}), "not an IANA time zone"),
        (
            lambda: Clock("today", "Europe/Madrid", {"d": "week"}),  # ty: ignore[invalid-argument-type]
            "'week' is not a clock field",
        ),
        (lambda: CallValue("caller", "calling_number", []), "assign names no value"),
        (lambda: CallValue("caller", "calling_number", "customer_phone"), "a list of value names"),
        (
            lambda: CallValue("caller", "from_number", ["customer_phone"]),  # ty: ignore[invalid-argument-type]
            "'calling_number'",
        ),
        (
            lambda: Prefetch([CALLER, CALLER]),
            "prefetch entry names must be unique: two entries are named 'caller'",
        ),
    ],
)
def test_a_bad_entry_is_refused_when_it_is_built(build: Any, message: str) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        build()


INTAKE = StateSchema(
    [
        {"name": "today", "type": "Date | None"},
        {"name": "weekday", "type": 'Literal["Monday", "Tuesday"] | None'},
        {
            "name": "days",
            "type": 'Literal["Monday", "Tuesday", "Wednesday", "Thursday", '
            '"Friday", "Saturday", "Sunday"] | None',
        },
        {"name": "year", "type": "int | None"},
        {"name": "caller", "type": "Phone | None"},
        {"name": "direction", "type": 'Literal["inbound"] | None'},
        {"name": "contact", "type": "NameEmail | None"},
        {"name": "visits", "type": "list[str]"},
    ]
).model


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (
            CallValue("c", "calling_number", ["ghost"]),
            "assigns an undeclared runtime variable: 'ghost'",
        ),
        (CallValue("c", "calling_number", ["call"]), "names 'call', which unmute-core reserves"),
        (CallValue("c", "calling_number", ["contact"]), "cannot fill the shape or list 'contact'"),
        (Lookup("l", lambda state: {}, {"visits": "visits"}), "cannot fill the shape or list"),
        (
            Clock("t", "Europe/Madrid", {"year": "date"}),
            "cannot save its date to 'year' (int): it fits Date or str",
        ),
        (CallValue("c", "calling_number", ["year"]), "cannot save its calling_number to 'year'"),
        (
            Clock("t", "Europe/Madrid", {"weekday": "day_of_week"}),
            "can read a day_of_week that 'weekday' does not offer: Friday, Saturday, Sunday",
        ),
        (
            CallValue("c", "direction", ["direction"]),
            "can read a direction that 'direction' does not offer: outbound",
        ),
    ],
)
def test_an_entry_that_cannot_save_its_value_is_refused_when_built(entry, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        Prefetch([entry], state=INTAKE)


def test_entries_that_fit_their_values_are_accepted():
    Prefetch(
        [
            Clock("t", "Europe/Madrid", {"today": "date", "year": "year", "days": "day_of_week"}),
            CallValue("c", "calling_number", ["caller"]),
            CallValue("x", "carrier_call_id", ["year"]),
            Lookup("l", lambda state: {}, {"year": "n"}),
        ],
        state=INTAKE,
    )


@pytest.mark.parametrize(
    ("call", "region"),
    [
        (call_info("inbound", remote_number=NUMBER, local_number="+14155552671"), "ES"),
        (call_info("outbound", remote_number="+14155552671", local_number=NUMBER), "US"),
        (call_info("inbound", remote_number="anonymous", local_number="+14155552671"), "US"),
        (CallInfo(), None),
    ],
)
def test_the_call_country_comes_from_the_callers_number_first(call, region):
    assert call_phone_region(call) == region


@pytest.mark.parametrize(
    ("direction", "calling", "called"),
    [("inbound", NUMBER, "+34910000000"), ("outbound", "+34910000000", NUMBER)],
)
def test_call_info_names_the_numbers_by_who_called_whom(direction, calling, called):
    call = call_info(
        direction, remote_number=NUMBER, local_number="+34910000000", carrier_call_id="c1"
    )
    assert (call.calling_number, call.called_number, call.remote_number) == (
        calling,
        called,
        NUMBER,
    )
    assert (call.channel, call.carrier_call_id) == ("phone", "c1")


def test_call_info_leaves_out_what_the_call_lacks_and_refuses_a_bad_direction():
    call = call_info("inbound", remote_number="")
    assert (call.calling_number, call.remote_number, call.direction) == (None, None, "inbound")
    with pytest.raises(ValidationError, match="'inbound' or 'outbound'"):
        call_info("sideways")  # ty: ignore[invalid-argument-type]


def call_tool(result: dict[str, Any], seen: list[dict[str, Any]]) -> ToolSpec:
    """A tool whose locked ``call`` reads the call id, answering ``result``."""

    async def lookup_call(call: str) -> dict[str, Any]:
        """Look the call up."""
        seen.append({"call": call})
        return result

    return local_tool(lookup_call, output=AnyResult, locked={"call": "{{call.carrier_call_id}}"})


async def test_a_lookup_runs_a_tool_with_its_locked_arguments_and_records_its_result() -> None:
    seen: list[dict[str, Any]] = []
    lookup = Lookup("who", call_tool({"id": "C-9"}, seen), {"call_id": "id"})
    prefetch = Prefetch([lookup], state=SalonState)
    state = fresh(CallInfo(carrier_call_id="CA1"))
    results = CallStartResults()
    await prefetch.run(state, results=results)
    assert seen == [{"call": "CA1"}]
    assert state.plain("call_id") == "C-9"
    assert results.describe("who") == '{"id": "C-9"}'


async def test_a_refused_tool_keeps_the_default(caplog) -> None:
    lookup = Lookup("who", call_tool({"id": "C-9"}, []), {"call_id": "id"})
    state = fresh()
    with caplog.at_level(logging.ERROR):
        await Prefetch([lookup]).run(state)
    assert state.plain("call_id") is None
    assert "refused" in caplog.text


async def test_a_background_lookup_waits_for_run_background_and_fills_nothing_first() -> None:
    lookup = Lookup("who", lambda _state: {"id": "C-9"}, {"call_id": "id"}, background=True)
    prefetch = Prefetch([TODAY, lookup], state=SalonState)
    assert "call_id" not in prefetch.fills
    assert "today_date" in prefetch.fills
    state = fresh()
    await prefetch.run(state)
    assert state.plain("call_id") is None
    await prefetch.run_background(state)
    assert state.plain("call_id") == "C-9"


async def test_a_slow_background_lookup_keeps_the_default(caplog) -> None:
    async def slow(_state: CallState) -> dict[str, str]:
        await asyncio.sleep(1)
        return {"id": "late"}

    prefetch = Prefetch(
        [Lookup("who", slow, {"call_id": "id"}, background=True)], background_budget=0.01
    )
    state = fresh()
    with caplog.at_level(logging.WARNING):
        await prefetch.run_background(state)
    assert state.plain("call_id") is None
    assert "took too long" in caplog.text


def test_a_lookup_tool_must_lock_every_argument_and_say_nothing() -> None:
    async def needs(order: str) -> dict[str, str]:
        """Needs an order."""
        return {}

    with pytest.raises(ValueError, match="tool argument order has no source"):
        Lookup("x", local_tool(needs, output=AnyResult), {"call_id": "id"})
    with pytest.raises(ValueError, match="drop the announce line"):
        Lookup(
            "x",
            local_tool(needs, output=AnyResult, locked={"order": "1"}, announce="Hold on."),
            {"call_id": "id"},
        )


async def test_a_step_reads_a_hidden_result_as_ok_and_a_failure_as_its_error() -> None:
    async def secret() -> dict[str, str]:
        """Answers something the model must not read."""
        return {"id": "C-9"}

    async def broken() -> dict[str, str]:
        """Fails."""
        raise RuntimeError("down")

    hidden = Lookup(
        "secret", local_tool(secret, output=AnyResult, show_result=False), {"call_id": "id"}
    )
    failed = Lookup("broken", local_tool(broken, output=AnyResult), {"customer_id": "id"})
    results = CallStartResults()
    state = fresh()
    await Prefetch([hidden, failed]).run(state, results=results)
    assert state.plain("call_id") == "C-9"
    assert results.describe("secret") == '{"ok": true}'
    assert "broken failed before completing" in results.describe("broken")


async def test_a_tool_lookup_reads_a_json_text_result() -> None:
    async def text() -> str:
        """Answers JSON text."""
        return '{"id": "C-9"}'

    state = fresh()
    await Prefetch([Lookup("text", local_tool(text, output=AnyResult), {"call_id": "id"})]).run(
        state
    )
    assert state.plain("call_id") == "C-9"


def test_a_lookup_tool_reading_an_unknown_value_is_refused_at_build() -> None:
    async def needs(who: str) -> dict[str, str]:
        """Needs a value."""
        return {}

    tool = local_tool(needs, output=AnyResult, locked={"who": "{{nobody}}"})
    with pytest.raises(ValueError, match="nobody"):
        Prefetch([Lookup("x", tool, {"call_id": "id"})], state=SalonState)
