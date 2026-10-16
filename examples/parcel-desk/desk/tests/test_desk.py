"""The spec builds, and the agent it makes does what the spec says."""

import asyncio
from datetime import date, timedelta

import pytest
from unmute_core.state import CallState, call_info
from unmute_core.tasks import CallTasks, DelegateRun

from desk.agent import build
from desk.spec import SPEC
from desk.tools import Depot

CALLER = "+447911123456"
CALL = call_info("inbound", remote_number=CALLER, agent_id="parcel-desk", session_id="call-1")


def _call(desk=None) -> tuple:
    desk = desk or build()
    state = asyncio.run(desk.new_call(CALL))
    return desk, state


def test_the_state_comes_from_the_specs_type_text() -> None:
    _, state = _call()
    assert state.plain("caller_phone") == CALLER
    assert date.fromisoformat(state.plain("today"))
    state.save({"parcel": {"tracking_id": 1042, "status": "held", "town": "York"}})
    assert state.plain("parcel.tracking_id") == "1042"
    with pytest.raises(Exception, match="status"):
        state.save({"parcel": {"tracking_id": "x", "status": "lost", "town": "York"}})


def test_the_greeting_is_a_fixed_line() -> None:
    desk, state = _call()
    assert desk.greeting.line(state) == "Hi, this is Parcel Desk. How can I help?"


def test_tools_carry_the_specs_options() -> None:
    tools = build().plan.tools
    assert tools.get("find_parcel").announce == ("Let me look that up.", "One moment.")
    assert tools.ends_call("end_call", {"ended": True})
    assert "tracking_id" not in str(tools.only(["reschedule_delivery"]).schemas())


def test_a_locked_argument_reads_the_state_and_the_call() -> None:
    depot = Depot()
    desk, state = _call(build(depot=depot))
    tools = desk.plan.tools
    refused = asyncio.run(tools.call("reschedule_delivery", {"day": "2099-01-01"}, state))
    assert isinstance(refused, dict)
    assert "parcel" in refused["refused"]
    state.save({"parcel": depot.parcels["PD-1042"]})
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    moved = asyncio.run(tools.call("reschedule_delivery", {"day": tomorrow}, state))
    assert moved["parcel"]["delivery_day"] == tomorrow
    asyncio.run(tools.call("save_callback_note", {"note": "a refund"}, state))
    assert depot.notes == [{"note": "a refund", "call": "call-1"}]


def test_leaving_a_callback_note_ends_the_call() -> None:
    desk, state = _call()
    tasks = CallTasks(desk.plan, state)
    [tool] = [t for t in tasks.delegate_tools() if t["name"] == "request_callback"]
    assert "when it finishes the call ends" in tool["description"]
    run = tasks.start("request_callback", caller_turns=1)
    assert isinstance(run, DelegateRun)
    step = run.next_step()
    assert step is not None
    step.opened(caller_turns=1)
    assert asyncio.run(step.call_tool("save_callback_note", {"note": "a refund"})) is None
    run.ended(step)
    assert run.next_step() is None
    run.result(caller_turns=2)
    assert run.ends_call
    assert state.plain("callback_note") == "a refund"


def test_a_spec_that_names_an_unknown_type_is_refused_by_its_row() -> None:
    spec = {**SPEC, "variables": [{"name": "parcel", "type": "Parcle | None"}]}
    with pytest.raises(ValueError, match="Parcle"):
        build(spec)


def test_the_agent_runs_the_plan_on_the_router_for_its_call(monkeypatch) -> None:
    monkeypatch.setenv("SLNG_API_KEY", "test")
    desk, state = _call()
    agent = desk.agent(state)
    assert agent.tasks is desk.plan
    assert agent.name is None
    assert agent.model.headers.session_id == "call-1"
    assert agent.model.headers.agent_id == "parcel-desk"


def test_the_router_defaults_give_way_to_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("SLNG_API_KEY", "test")
    monkeypatch.delenv("SLNG_ROUTER_URL", raising=False)
    monkeypatch.setenv("SLNG_MODEL", "")
    call = build().model_call(CALL)
    assert call.model == "slng/auto"
    assert str(call.client.base_url) == "https://eu-north.context-router.slng.ai/v1/"

    monkeypatch.setenv("SLNG_MODEL", "openai/gpt-5.6-luna")
    assert build().model_call(CALL).model == "openai/gpt-5.6-luna"


def test_every_call_starts_empty() -> None:
    desk = build()
    first: CallState = asyncio.run(desk.new_call(CALL))
    first.save({"callback_note": "x"})
    assert asyncio.run(desk.new_call(CALL)).plain("callback_note") is None
