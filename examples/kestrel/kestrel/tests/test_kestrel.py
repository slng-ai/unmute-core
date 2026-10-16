"""Kestrel's agent definition and claims desk, offline: no model, no network.

The whole calls run live, on both frameworks, from ``../scenarios`` (see the README).
"""

from collections.abc import Mapping
from datetime import date, timedelta

import pytest
from unmute_core.state import CallState, call_info
from unmute_core.tools import ToolError, ToolRegistry

from kestrel import claims
from kestrel.agent import AGENT_ID, KESTREL, Kestrel
from kestrel.claims import ClaimsDesk, car_for
from kestrel.state import Vehicle, policy_number
from kestrel.words import spoken_day, spoken_when

CALL = call_info("inbound", remote_number="+447700900123", agent_id=AGENT_ID, session_id="c-1")
FACTS = {
    "incident_day": "2026-09-28",
    "incident_time": "evening",
    "place": "the A40 near Oxford",
    "what_happened": "a van reversed into my front bumper",
    "injuries": False,
    "police_report": "none",
    "other_party": True,
    "vehicle_drivable": True,
}


@pytest.fixture(autouse=True)
def thursday(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claims, "today", lambda: date(2026, 10, 1))


async def found(kestrel: Kestrel, policy: str = "KP482913", born: str = "1979-06-05"):
    state = await kestrel.new_call(CALL)
    tools = kestrel.plan.tools
    result = await tools.call("lookup_policy", {"policy_number": policy, "birth_date": born}, state)
    assert isinstance(result, dict)
    state.save({"policy": result["policy"]})
    return state, tools


async def note(tools: ToolRegistry, facts: Mapping[str, object], state: CallState) -> dict:
    """Note facts as a step does: call the tool, then save what its result names."""
    result = await tools.call("note_facts", facts, state)
    tools.save_result("note_facts", result, state)
    assert isinstance(result, dict)
    return result


def test_the_greeting_asks_for_the_policy_number() -> None:
    assert KESTREL.plan is not None
    from kestrel.agent import GREETING

    assert GREETING.text == (
        "Thanks for calling Kestrel Insurance claims, this is Sam. Can I take your policy "
        "number, please?"
    )


def test_only_the_claim_flow_and_the_guide_reach_the_main_agent() -> None:
    tasks = KESTREL.plan
    for name in ("lookup_policy", "note_facts", "file_claim"):
        assert name in tasks.spec.task_only_tools
    [schema] = tasks.tools.only(["note_facts"]).schemas()
    assert not {"report", "policy"} & set(schema["function"]["parameters"]["properties"])
    [schema] = tasks.tools.only(["file_claim"]).schemas()
    assert set(schema["function"]["parameters"]["properties"]) == set()


@pytest.mark.parametrize(
    ("said", "spelling"),
    [(" kp482913 ", "KP482913"), ("KP 482 913", "KP482913"), ("KP-482913.", "KP482913")],
)
def test_a_policy_number_is_kp_and_six_digits(said: str, spelling: str) -> None:
    assert policy_number(said) == spelling


@pytest.mark.parametrize("said", ["KP48291", "KP4829133", "KX482913", "KP48291A"])
def test_anything_else_is_refused(said: str) -> None:
    with pytest.raises(ValueError, match="is not a policy number"):
        policy_number(said)


async def test_a_wrong_birth_date_finds_nothing() -> None:
    state = await KESTREL.new_call(CALL)
    result = await KESTREL.plan.tools.call(
        "lookup_policy", {"policy_number": "KP482913", "birth_date": "1980-06-05"}, state
    )
    assert isinstance(result, dict)
    assert result["found"] is False
    assert "Daniel" not in str(result)


async def test_code_asks_each_question_in_turn_then_the_claim_is_complete() -> None:
    state, tools = await found(Kestrel())
    asked = []
    for name, value in FACTS.items():
        result = await note(tools, {name: value}, state)
        asked.append(result["next_question"])
    assert asked == [
        "Roughly what time was it?",
        "Where did it happen?",
        "In your own words, what happened?",
        "Was anyone hurt?",
        "Was it reported to the police? If so, do you have a reference?",
        "Was another vehicle or driver involved?",
        "Can your car still be driven?",
        "",
    ]
    assert result["complete"] is True
    assert state.plain("report.vehicle") == "AB12CDE"
    assert state.plain("read_back") == (
        "Shall I file your claim for the blue Toyota Corolla? On Monday the 28th of September "
        "in the evening, at the A40 near Oxford: a van reversed into my front bumper. Nobody "
        "was hurt, no police report, another driver was involved, and the car can be driven. "
        "Is that all right?"
    )


async def test_the_safety_line_goes_before_the_next_question_once() -> None:
    state, tools = await found(Kestrel())
    hurt = await note(tools, {**FACTS, "injuries": True, "vehicle_drivable": None}, state)
    assert hurt["next_question"].startswith("I'm sorry to hear that. If anyone needs medical help")
    again = await note(tools, {"injuries": True}, state)
    assert not again["next_question"].startswith("I'm sorry")


async def test_a_future_day_is_refused_with_words_for_the_caller() -> None:
    state, tools = await found(Kestrel())
    with pytest.raises(ToolError, match="That day hasn't happened yet"):
        await tools.call("note_facts", {"incident_day": "2026-10-05"}, state)
    assert state.plain("report.incident_day") is None


async def test_two_cars_are_asked_about_and_a_name_picks_one() -> None:
    state, tools = await found(Kestrel(), "KP731508", "1992-01-23")
    asked = await note(tools, FACTS, state)
    assert asked["next_question"] == (
        "Which car was it: the white Kia Niro or the black Ford Fiesta?"
    )
    await note(tools, {"vehicle": "the Kia"}, state)
    assert state.plain("report.vehicle") == "LM70XYZ"


def test_a_car_name_that_fits_none_or_two_is_no_car() -> None:
    cars = [
        Vehicle(plate="LM70XYZ", words="the white Kia Niro"),
        Vehicle(plate="GH19RST", words="the black Ford Fiesta"),
    ]
    assert car_for("Priya's car", cars) is None
    assert car_for("the black one", cars) is not None
    assert car_for("gh19 rst", cars) is not None


async def test_a_claim_is_filed_once_per_call() -> None:
    desk = ClaimsDesk()
    state, tools = await found(Kestrel(desk))
    with pytest.raises(ToolError, match="not complete yet"):
        await tools.call("file_claim", {}, state)
    await note(tools, FACTS, state)
    filed = await tools.call("file_claim", {}, state)
    again = await tools.call("file_claim", {}, state)
    assert isinstance(filed, dict) and isinstance(again, dict)
    assert filed["claim_number"] == again["claim_number"] == "KC100231"
    assert len(desk.filed) == 1


def test_words() -> None:
    assert spoken_day(date(2026, 9, 28)) == "Monday the 28th of September"
    assert spoken_when("night") == "at night"
    assert spoken_day(date(2026, 9, 28) + timedelta(days=4)) == "Friday the 2nd of October"
