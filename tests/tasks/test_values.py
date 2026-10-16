"""Finish values: their schema, the one-batch save, hand-backs, success rules and repairs.

The cases include Unmute's required fields, blank values and appends (unmute@4a738ad
internal/generate/call_state.go StepResult, _appended).
"""

import json
from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel, Field

from tests.tasks.job import job, plan, state
from unmute_core.state import CallState, Phone, StateRefused, StateSchema
from unmute_core.tasks import SaveOutcome, TaskSpec, TaskValueRefused
from unmute_core.tasks._values import (
    derived_arguments,
    finish_parameters,
    hand_back_keeps_values,
    merge_retained,
    result_model,
    save_task_values,
    success_matches,
    values_from_result,
)
from unmute_core.tools import check_parameters


def _save(
    task_plan: Any,
    call_state: CallState,
    step: str,
    hand_back_saves: bool = False,
    /,
    **values: Any,
) -> SaveOutcome:
    return save_task_values(
        task_plan.tasks[step],
        values,
        state=call_state,
        model=task_plan.result_model(step),
        hand_back_saves=hand_back_saves,
    )


def _settled(call_state: CallState) -> dict[str, Any]:
    """The values that are set and settled, by name."""
    return {
        name: call_state.plain(name)
        for name in ("order_id", "status", "note")
        if call_state.is_confirmed(name)
    }


def test_finish_schema_is_typed_per_value_and_ends_on_the_hand_back() -> None:
    task_plan = plan()
    schema = finish_parameters(task_plan.result_model("find_order"), record=True)
    assert list(schema["properties"]) == ["order_id", "status", "unserved_request"]
    assert schema["properties"]["status"] == {"enum": ["open", "closed"], "type": "string"}
    assert schema["properties"]["unserved_request"] == {
        "type": "string",
        "description": (
            "Leave empty unless the caller asked for something this step cannot serve. Then put "
            "that request here in one short plain sentence, in the caller's own terms, so the "
            "agent that owns this step can take it. Pass the values the caller already confirmed "
            "in this step in the same call: they are saved before the hand-back."
        ),
    }
    assert schema["properties"]["unserved_request"]["description"].endswith("the hand-back.")
    status = finish_parameters(task_plan.result_model("find_order"), record=False)
    assert status["properties"]["unserved_request"]["description"].endswith("can take it.")
    assert schema["required"] == [] and schema["additionalProperties"] is False


class Visit(BaseModel):
    """One visit."""

    day: date
    phone: Phone


class Booking(BaseModel):
    """A state with a nested model and a phone."""

    visit: Visit | None = Field(None, description="The visit booked.")
    caller: Phone | None = None


def test_finish_schema_inlines_refs_and_words_formats_for_the_provider() -> None:
    task = TaskSpec(
        name="book",
        instructions="Book.",
        when="w",
        assign=[
            {"variable": "visit", "path": ["visit"]},
            {"variable": "caller", "path": ["caller"]},
        ],
    )
    schema = finish_parameters(result_model(task, Booking), record=False)
    check_parameters(schema)
    visit = schema["properties"]["visit"]
    assert "$ref" not in str(schema) and "$defs" not in schema and "null" not in str(schema)
    assert visit["description"] == "The visit booked. One visit."
    day = visit["properties"]["day"]
    assert day == {"type": "string", "description": "Expected a date written YYYY-MM-DD."}
    caller = schema["properties"]["caller"]
    assert (
        caller["description"]
        == "Expected a phone number with its country code: a plus, the country code, then the "
        "number."
    )


def test_values_from_a_tool_result_follow_their_paths() -> None:
    task = plan().tasks["find_order"]
    values = values_from_result(task, {"order": {"id": "A-12"}, "status": "open", "found": True})
    assert values == {"order_id": "A-12", "status": "open"}


def test_saving_is_all_or_nothing_and_typed() -> None:
    task_plan = plan()
    call_state = state(task_plan)
    with pytest.raises(TaskValueRefused, match="status: Input should be 'open' or 'closed'"):
        _save(task_plan, call_state, "find_order", order_id="A-12", status="lost")
    assert _settled(call_state) == {}
    outcome = _save(task_plan, call_state, "find_order", order_id="A-12", status="open")
    assert outcome == SaveOutcome("completed", saved=("order_id", "status"))
    assert _settled(call_state) == {"order_id": "A-12", "status": "open"}


def test_an_unserved_finish_saves_nothing() -> None:
    task_plan = plan()
    call_state = state(task_plan)
    outcome = _save(
        task_plan, call_state, "find_order", order_id="A-12", unserved_request=" I want a refund "
    )
    assert outcome == SaveOutcome("unserved", unserved_request="I want a refund")
    assert _settled(call_state) == {}


def test_a_hand_back_saves_the_values_given_with_it() -> None:
    task_plan = plan()
    call_state = state(task_plan)
    call_state.save({"status": "open"}, step="find_order")
    outcome = _save(
        task_plan,
        call_state,
        "find_order",
        True,
        order_id="A-12",
        unserved_request="I want a refund",
    )
    assert outcome == SaveOutcome(
        "unserved", saved=("order_id",), unserved_request="I want a refund"
    )
    # Only what was given: the value left out keeps what it held.
    assert _settled(call_state) == {"order_id": "A-12", "status": "open"}


def test_a_hand_back_with_no_values_saves_nothing() -> None:
    task_plan = plan()
    call_state = state(task_plan)
    outcome = _save(
        task_plan, call_state, "find_order", True, order_id=None, unserved_request="A refund"
    )
    assert outcome == SaveOutcome("unserved", unserved_request="A refund")
    assert _settled(call_state) == {}


def test_a_hand_back_saves_the_values_that_fit_and_leaves_the_rest() -> None:
    task_plan = plan()
    call_state = state(task_plan)
    call_state.save({"status": "open"}, step="find_order")
    outcome = _save(
        task_plan,
        call_state,
        "find_order",
        True,
        order_id="A-12",
        status="lost",
        unserved_request="A refund",
    )
    assert outcome == SaveOutcome(
        "unserved", saved=("order_id",), unserved_request="A refund", refused=("status",)
    )
    assert _settled(call_state) == {"order_id": "A-12", "status": "open"}


def test_a_hand_back_whose_write_fails_saves_nothing_and_still_hands_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_plan = plan()
    call_state = state(task_plan)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise StateRefused("order_id: no")

    monkeypatch.setattr(CallState, "save", refuse)
    outcome = _save(
        task_plan,
        call_state,
        "find_order",
        True,
        order_id="A-12",
        status="lost",
        unserved_request="A refund",
    )
    assert outcome == SaveOutcome(
        "unserved", unserved_request="A refund", refused=("status", "order_id")
    )


def test_a_save_that_is_not_a_hand_back_is_still_refused_whole() -> None:
    task_plan = plan()
    call_state = state(task_plan)
    with pytest.raises(TaskValueRefused, match="status: Input should be"):
        _save(task_plan, call_state, "find_order", True, order_id="A-12", status="lost")
    assert _settled(call_state) == {}


def test_hand_back_keeps_values_once_the_caller_spoke_and_no_action_is_unrecorded() -> None:
    assert hand_back_keeps_values(caller_spoke=True, action_unrecorded=False)
    assert not hand_back_keeps_values(caller_spoke=False, action_unrecorded=False)
    assert not hand_back_keeps_values(caller_spoke=True, action_unrecorded=True)


def test_a_value_is_confirmed_only_by_its_confirming_tasks_save() -> None:
    task_plan = plan(confirm={"order_id": "find_order"})
    call_state = state(task_plan)
    _save(task_plan, call_state, "take_note", note="hi")
    call_state.save({"order_id": "A-12"})
    assert not call_state.is_confirmed("order_id")
    _save(task_plan, call_state, "find_order", order_id="A-12", status="open")
    assert call_state.is_confirmed("order_id") and call_state.is_confirmed("note")


def test_a_required_value_must_come_with_a_served_finish() -> None:
    payload = job()
    payload["tasks"][1]["assign"][0]["optional"] = False
    task_plan = plan(payload)
    call_state = state(task_plan)
    with pytest.raises(TaskValueRefused, match="note: Field required"):
        _save(task_plan, call_state, "take_note")
    assert (
        _save(task_plan, call_state, "take_note", unserved_request="A refund").status == "unserved"
    )
    # An optional value left out is saved as empty.
    optional = plan()
    optional_state = state(optional)
    optional_state.save({"note": "old"})
    assert _save(optional, optional_state, "take_note").saved == ()
    assert optional_state.plain("note") is None


def test_an_empty_string_reads_as_no_value_unless_the_type_takes_one() -> None:
    task_plan = plan()
    call_state = state(task_plan)
    outcome = _save(task_plan, call_state, "find_order", order_id="A-12", status="")
    assert outcome.saved == ("order_id",) and call_state.plain("status") is None
    assert _save(task_plan, call_state, "take_note", note="").saved == ()
    assert call_state.plain("note") == ""


def test_an_append_adds_one_entry_and_never_the_same_object_twice() -> None:
    payload = job()
    payload["tasks"][1]["assign"] = [{"variable": "tags", "path": ["tag"], "append": True}]
    task_plan = plan(payload, confirm=None)
    call_state = state(task_plan)
    assert _save(task_plan, call_state, "take_note", tag="vip").saved == ("tags",)
    assert _save(task_plan, call_state, "take_note", tag="vip").saved == ("tags",)
    assert call_state.plain("tags") == ["vip", "vip"]
    # Nothing new onto a list already settled is not a write.
    assert _save(task_plan, call_state, "take_note").saved == ()
    assert call_state.plain("tags") == ["vip", "vip"]
    schema = finish_parameters(task_plan.result_model("take_note"), record=True)
    assert schema["properties"]["tag"] == {"type": "string"}


def test_success_rules_read_words_as_the_author_wrote_them() -> None:
    rule = plan().tasks["find_order"].finish[0]
    assert success_matches(rule, {"found": True})
    assert not success_matches(rule, {"found": False})
    assert not success_matches(rule, {"found": "yes"})
    assert not success_matches(rule, "not a dict")


def test_retained_tool_values_win_a_repair() -> None:
    task_plan = plan()
    merged = merge_retained(
        task_plan.tasks["find_order"],
        {"order_id": "made-up", "status": "open"},
        {"order": {"id": "A-12"}, "status": "bogus"},
        task_plan.result_model("find_order"),
    )
    assert merged == {"order_id": "A-12", "status": "open"}


def test_a_nested_value_is_read_from_its_parent_not_asked_for() -> None:
    """A nested value is read from the value that holds it, so the step does not ask for it."""
    schema = StateSchema(
        [
            {"name": "contact", "type": "NameEmail | None"},
            {"name": "caller_email", "type": "EmailStr | None"},
        ]
    )
    task = TaskSpec(
        name="take_details",
        when="The caller gives their details.",
        instructions="Take the details.",
        assign=[
            {"variable": "contact", "path": ["contact"]},
            {"variable": "caller_email", "path": ["contact", "email"]},
        ],
    )
    model = result_model(task, schema.model)
    parameters = finish_parameters(model, record=True, derived=derived_arguments(task))
    assert "caller_email" not in parameters["properties"]
    assert "contact" in parameters["properties"]
    call_state = CallState(schema.model)
    contact = {"name": "Robin Vega", "email": "robin.vega@example.com"}
    save_task_values(task, {"contact": contact}, state=call_state, model=model)
    assert call_state.plain("caller_email") == "robin.vega@example.com"
    moved = {"name": "Robin Vega", "email": "robin@example.com"}
    sent = {"contact": json.dumps(moved), "caller_email": "other@example.com"}
    save_task_values(task, sent, state=call_state, model=model)
    assert call_state.plain("caller_email") == "robin@example.com"
