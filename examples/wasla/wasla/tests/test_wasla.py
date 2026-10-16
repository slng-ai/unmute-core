"""Wasla's agent definition and ledger, offline: no model, no network.

The whole calls run live, on both frameworks, from ``../scenarios`` (see the README).
"""

from datetime import date, time
from decimal import Decimal

import pytest
from unmute_core.call import CallStartError
from unmute_core.state import call_info
from unmute_core.tools import SpokenLines, ToolError

from wasla import ledger as ledger_module
from wasla.agent import AGENT_ID, GREETING, WASLA, Wasla
from wasla.ledger import Ledger
from wasla.words import amount, clock, day

CALL = call_info("outbound", remote_number="+971501234567", agent_id=AGENT_ID, session_id="c-1")
LAYLA = {"account_id": "WA-1001"}


@pytest.fixture(autouse=True)
def thursday(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_module, "today", lambda: date(2026, 10, 1))
    monkeypatch.delenv("UNMUTE_CALL_START", raising=False)


@pytest.mark.parametrize(
    ("dispatched", "refusal"),
    [
        (None, "Missing call_start fields: account_id"),
        ({}, "Missing call_start fields: account_id"),
        ({"account_id": "1001"}, "call_start: account_id"),
        ({"account_id": "WA-9999"}, "no account WA-9999"),
    ],
)
async def test_a_call_starts_only_with_an_account_the_ledger_holds(
    dispatched: dict[str, str] | None, refusal: str
) -> None:
    with pytest.raises(CallStartError, match=refusal):
        await Wasla().new_call(CALL, dispatched)


async def test_the_greeting_names_the_holder_and_nothing_else_is_known() -> None:
    state = await Wasla().new_call(CALL, LAYLA)
    assert GREETING.line(state) == (
        "Hello, this is Lina calling from Wasla. Am I speaking with Layla Haddad?"
    )
    snapshot = str(state.snapshot())
    for secret in ("Falcon", "dirham", "434"):
        assert secret not in snapshot


async def test_another_account_comes_from_the_dispatch() -> None:
    state = await Wasla().new_call(CALL, {"account_id": "WA-1002"})
    assert GREETING.line(state) == (
        "Hello, this is Lina calling from Wasla. Am I speaking with Omar Khalil?"
    )


async def test_only_a_matching_date_of_birth_returns_the_account() -> None:
    state = await Wasla().new_call(CALL, LAYLA)
    tools = WASLA.plan.tools
    wrong = await tools.call("verify_holder", {"birth_date": "1990-03-15"}, state)
    assert isinstance(wrong, dict)
    assert wrong["status"] == "mismatch"
    assert wrong["merchant"] is None
    right = await tools.call("verify_holder", {"birth_date": "1990-03-14"}, state)
    assert isinstance(right, dict)
    assert right["merchant"] == "Falcon Electronics"
    assert right["amount_words"] == "four hundred thirty-four dirhams and fifty fils"


async def test_three_wrong_dates_lock_the_account() -> None:
    wasla = Wasla(Ledger())
    state = await wasla.new_call(CALL, LAYLA)
    tools = wasla.plan.tools
    for _ in range(3):
        result = await tools.call("verify_holder", {"birth_date": "2000-01-01"}, state)
    assert isinstance(result, dict)
    assert result["status"] == "locked"
    right = await tools.call("verify_holder", {"birth_date": "1990-03-14"}, state)
    assert isinstance(right, dict)
    assert right["status"] == "locked"


@pytest.mark.parametrize(
    ("tool", "given", "line"),
    [
        ("record_promise", {"day": "2026-10-20"}, "a day up to Sunday the 11th of October."),
        ("record_promise", {"day": "2026-09-30"}, "That day has already gone."),
        ("schedule_callback", {"day": "2026-10-20", "at": "17:00"}, "a day up to Sunday"),
    ],
)
async def test_a_day_outside_the_window_is_answered_in_code_before_the_read_back(
    tool: str, given: dict[str, str], line: str
) -> None:
    books = Ledger()
    wasla = Wasla(books)
    state = await wasla.new_call(CALL, LAYLA)
    tools = wasla.plan.tools
    outcome = tools.precheck(tool, given, state)
    said = tools.use_result(tool, outcome, state, SpokenLines())
    assert said is not None and line in said
    assert await tools.call(tool, given, state) == outcome
    assert books.records == {}


async def test_a_promise_is_recorded_once_per_call() -> None:
    books = Ledger()
    wasla = Wasla(books)
    state = await wasla.new_call(CALL, LAYLA)
    first = await wasla.plan.tools.call("record_promise", {"day": "2026-10-09"}, state)
    again = await wasla.plan.tools.call("record_promise", {"day": "2026-10-09"}, state)
    assert isinstance(first, dict)
    assert first == again
    assert first["when"] == "on Friday the 9th of October"
    assert books.codes("WA-1001") == ["PTP"]


async def test_a_concern_is_noted_in_their_words_without_trailing_stops() -> None:
    books = Ledger()
    wasla = Wasla(books)
    state = await wasla.new_call(CALL, LAYLA)
    noted = await wasla.plan.tools.call(
        "note_concern",
        {"kind": "purchase-dispute", "note": "I never bought anything from Falcon."},
        state,
    )
    assert isinstance(noted, dict)
    assert noted["status"] == "DISPUTE"
    assert next(iter(books.records.values()))["note"] == "I never bought anything from Falcon"


async def test_a_wrong_number_is_recorded_on_the_calls_own_account() -> None:
    books = Ledger()
    wasla = Wasla(books)
    state = await wasla.new_call(CALL, LAYLA)
    [schema] = wasla.plan.tools.only(["mark_wrong_number"]).schemas()
    assert schema["function"]["parameters"]["properties"] == {}
    await wasla.plan.tools.call("mark_wrong_number", {}, state)
    assert books.codes("WA-1001") == ["WN"]


def test_the_main_agent_sees_three_flows_and_no_writes() -> None:
    tasks = WASLA.plan
    assert {d.name for d in tasks.delegates()} >= {"account", "callback", "wrong_number"}
    for tool in ("record_promise", "note_concern", "mark_wrong_number", "verify_holder"):
        assert not tasks.visible_to_main(tool)


def test_words() -> None:
    assert amount(Decimal("434.50")) == "four hundred thirty-four dirhams and fifty fils"
    assert amount(Decimal(1249)) == "one thousand two hundred forty-nine dirhams"
    assert amount(Decimal("1.05")) == "one dirham and five fils"
    assert day(date(2026, 10, 9)) == "Friday the 9th of October"
    assert clock(time(17, 30)) == "5:30 PM"
