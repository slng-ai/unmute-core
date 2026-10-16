"""The agent definition loads, and the diary only ever sees a confirmed phone number."""

import asyncio

from unmute_core.state import call_info

from salon.agent import AGENT_ID, agents, new_call, tools
from salon.store import Salon


def test_each_agent_sees_its_own_tools_and_tasks() -> None:
    concierge, specialist = (agent.tasks for agent in agents(Salon()))
    assert concierge is not None and specialist is not None
    assert {d.name for d in concierge.delegates()} == {"verify_customer", "book"}
    assert {d.name for d in specialist.delegates()} == {"verify_customer"}
    assert concierge.visible_to_main("look_up_salon_info")
    assert not concierge.visible_to_main("book_appointment")
    assert specialist.visible_to_main("record_complaint")
    assert not specialist.visible_to_main("find_or_create_customer")


def test_the_model_never_sees_the_locked_phone_number() -> None:
    [schema] = tools(Salon()).only(["find_slots"]).schemas()
    assert set(schema["function"]["parameters"]["properties"]) == {"date", "service", "intent"}
    assert schema["function"]["parameters"].get("required", []) == []


def test_every_tool_date_is_an_iso_date() -> None:
    """One date format for every tool, so code speaks each one by the same rule."""
    registry = tools(Salon())
    dates = [
        (tool, options)
        for tool in registry.names
        for name, field in registry.get(tool).parameters["properties"].items()
        if name == "date"
        for options in [[field, *field.get("anyOf", [])]]
    ]
    assert {name for name, _ in dates} >= {"find_slots", "book_appointment", "move_appointment"}
    for name, options in dates:
        assert any(option.get("format") == "date" for option in options), name


def test_an_unconfirmed_phone_number_refuses_the_diary() -> None:
    state = asyncio.run(
        new_call(call_info("inbound", remote_number="+34600111222", agent_id=AGENT_ID))
    )
    assert state.plain("customer_phone") == "+34600111222"
    assert state.plain("booking_date")
    refused = asyncio.run(tools(Salon()).call("find_slots", {"date": "", "service": "any"}, state))
    assert isinstance(refused, dict)
    assert "customer_phone not set yet" in refused["refused"]


def test_both_agents_run_their_tasks_in_record_mode() -> None:
    concierge, specialist = (agent.tasks for agent in agents(Salon()))
    assert concierge is not None and specialist is not None
    assert concierge.record and specialist.record


def test_record_complaint_says_a_line_as_it_starts() -> None:
    assert tools(Salon()).get("record_complaint").announce == ("Let me get that written down.",)
