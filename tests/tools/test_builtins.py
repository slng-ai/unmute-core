"""The tools the library ships: the date and time, and the caller's number."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from unmute_core.state import CallState, call_info
from unmute_core.tools import ToolRegistry, current_datetime_tool, user_phone_number_tool

NUMBER = "+34600111222"


async def test_the_clock_answers_in_its_zone():
    registry = ToolRegistry([current_datetime_tool("Europe/Madrid")])
    [schema] = registry.schemas()
    assert schema["function"]["name"] == "current_datetime"
    assert schema["function"]["parameters"]["properties"] == {}

    answer = await registry.call("current_datetime", {})
    assert isinstance(answer, dict)

    now = datetime.fromisoformat(answer["local_datetime"])
    assert now.utcoffset() == datetime.now(ZoneInfo("Europe/Madrid")).utcoffset()
    assert answer["timezone"] == "Europe/Madrid"
    assert answer["local_date"] == now.date().isoformat()
    assert answer["day_of_week"] == now.strftime("%A")
    assert answer["utc_offset"] in ("+01:00", "+02:00")
    assert isinstance(answer["is_dst"], bool)


def test_an_unknown_zone_fails_at_start():
    with pytest.raises(ValueError, match="not an IANA time zone"):
        current_datetime_tool("Mars/Olympus")


@pytest.mark.parametrize("direction", ["inbound", "outbound"])
async def test_the_caller_number_and_call_type_come_from_the_call(direction):
    registry = ToolRegistry([user_phone_number_tool(name="get_user_phone_number")])
    state = CallState(call=call_info(direction, remote_number=NUMBER))
    known = await registry.call("get_user_phone_number", {}, state)
    assert known == {
        "available": True,
        "phone_number": NUMBER,
        "call_type": direction,
        "reason": None,
    }
    assert isinstance(known, dict)


async def test_a_web_call_has_no_number():
    registry = ToolRegistry([user_phone_number_tool()])
    unknown = await registry.call("user_phone_number", {}, CallState())
    assert isinstance(unknown, dict)
    assert unknown["available"] is False
    assert unknown["phone_number"] is None
