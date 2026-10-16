"""``save_call_start`` saves the dispatch's values into State, with the env under them."""

import logging

import pytest
from pydantic import BaseModel

from tests.outputs import AnyResult
from unmute_core.call import CallStartError, save_call_start
from unmute_core.model import render_prompt
from unmute_core.settings import SettingsError
from unmute_core.state import (
    CallInfo,
    CallStartRule,
    CallState,
    StateEvent,
    StateSchema,
    call_info,
)
from unmute_core.tools import ToolRegistry, local_tool
from unmute_core.tools._spec import fill_text


class State(BaseModel):
    customer_id: str | None = None
    party: int | None = None
    brand: str | None = None
    note: str | None = None


RULES: dict[str, CallStartRule] = {
    "customer_id": "required",
    "party": "optional",
    "brand": "optional",
}


@pytest.fixture(autouse=True)
def _no_env(monkeypatch) -> None:
    monkeypatch.delenv("UNMUTE_CALL_START", raising=False)


def test_the_dispatch_wins_over_the_env_and_undeclared_names_are_dropped(monkeypatch) -> None:
    monkeypatch.setenv("UNMUTE_CALL_START", '{"customer_id": "env", "party": 2, "note": "x"}')
    state = CallState(State, call_start=RULES)
    events: list[StateEvent] = []
    state.listen(events.append)

    saved = save_call_start(state, {"customer_id": "c_42", "nope": True})

    assert saved == {"customer_id": "c_42", "party": 2}
    assert state.plain("customer_id") == "c_42"
    assert state.plain("note") is None
    assert events[0].origin.kind == "call_start"


def test_without_the_env_only_the_dispatch_is_read(monkeypatch) -> None:
    monkeypatch.setenv("UNMUTE_CALL_START", '{"party": 2}')
    state = CallState(State, call_start=RULES)

    assert save_call_start(state, {"customer_id": "c_42"}, env=False) == {"customer_id": "c_42"}


def test_without_rules_any_state_value_may_be_set() -> None:
    state = CallState(State)
    assert save_call_start(state, {"note": "n", "other": 1}) == {"note": "n"}


def test_a_missing_required_value_refuses_the_call() -> None:
    with pytest.raises(CallStartError, match="Missing call_start fields: customer_id"):
        save_call_start(CallState(State, call_start=RULES), {"customer_id": "", "party": 2})


def test_a_value_that_does_not_fit_saves_nothing() -> None:
    state = CallState(State, call_start=RULES)
    with pytest.raises(CallStartError, match=r"^call_start: party: "):
        save_call_start(state, {"customer_id": "c_42", "party": "many"})
    assert state.plain("customer_id") is None


def test_a_none_value_keeps_the_default() -> None:
    state = CallState(State, call_start=RULES)
    assert save_call_start(state, {"customer_id": "c", "party": None}) == {"customer_id": "c"}


def test_a_bad_env_value_names_the_variable(monkeypatch) -> None:
    monkeypatch.setenv("UNMUTE_CALL_START", "[1, 2]")
    with pytest.raises(SettingsError, match="UNMUTE_CALL_START"):
        save_call_start(CallState(State), None)


def test_an_optional_value_not_given_reads_as_empty_and_never_refuses_a_tool() -> None:
    state = CallState(State, call_start=RULES)
    save_call_start(state, {"customer_id": "c_42"})
    assert render_prompt("Welcome to {{brand}}. {{note}}", state) == (
        "Welcome to . none recorded yet."
    )
    assert fill_text("/shops/{{brand}}/{{customer_id}}", state) == "/shops//c_42"
    assert fill_text("Checking {{brand}}{{customer_id}}.", state) == "Checking c_42."


def test_call_start_rules_name_state_values() -> None:
    with pytest.raises(ValueError, match="call_start names 'nope'"):
        CallState(State, call_start={"nope": "required"})
    with pytest.raises(ValueError, match="must be 'required' or 'optional'"):
        CallState(State, call_start={"brand": "maybe"})  # ty: ignore[invalid-argument-type]


def test_a_schema_row_declares_its_call_start_rule() -> None:
    schema = StateSchema([{"name": "customer_id", "call_start": "required"}, {"name": "note"}])
    assert schema.call_start == {"customer_id": "required"}


def test_a_value_cannot_both_start_the_call_and_wait_for_the_caller() -> None:
    with pytest.raises(ValueError, match="'brand' is in both call_start and confirm"):
        CallState(State, call_start={"brand": "optional"}, confirm={"brand": "verify"})


CALLER = "+34600111222"
TRUNK = "+34911111111"
OTHER = "+34922222222"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (None, ("web", "outbound", TRUNK, OTHER)),
        (call_info("inbound", remote_number=CALLER), ("phone", "inbound", CALLER, OTHER)),
    ],
)
def test_a_dispatch_call_simulates_a_web_call_and_only_fills_a_phone_calls_gaps(
    monkeypatch, line, expected
) -> None:
    monkeypatch.setenv(
        "UNMUTE_CALL_START",
        '{"call": {"direction": "outbound", "calling_number": "+34911111111", '
        '"called_number": "+34922222222", "agent_id": "salon", "session_id": "s-9"}}',
    )
    state = CallState(State, call=line)

    save_call_start(state, None)

    call = state.call
    assert (call.channel, call.direction, call.calling_number, call.called_number) == expected
    assert (call.agent_id, call.session_id) == ("salon", "s-9")


@pytest.mark.parametrize("given", [{"call": "+34600"}, {"call": {"color": "red"}}])
def test_a_dispatch_call_that_does_not_fit_stops_the_call(given) -> None:
    state = CallState(State)
    with pytest.raises(CallStartError, match="call"):
        save_call_start(state, given)
    assert state.call == CallInfo(session_id=state.call.session_id)


def test_an_undeclared_dispatch_value_is_dropped_with_a_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        save_call_start(CallState(State, call_start=RULES), {"customer_id": "c", "note": "x"})
    assert "note is no call-start value" in caplog.text


async def test_a_whole_locked_optional_value_not_given_sends_empty_text() -> None:
    seen: list[dict] = []

    async def shop(brand: str) -> dict:
        """Look the shop up."""
        seen.append({"brand": brand})
        return {}

    state = CallState(State, call_start=RULES)
    tool = local_tool(shop, output=AnyResult, locked={"brand": "{{brand}}"})
    await ToolRegistry([tool]).call("shop", {}, state)
    assert seen == [{"brand": ""}]
