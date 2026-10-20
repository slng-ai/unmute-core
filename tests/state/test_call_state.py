"""CallState behaves as Unmute's emitted CallState does, on both paths.

Cases follow unmute@d0c67f1 internal/generate/typed_state_smoke_test.go. Each test runs on
the Unmute author's State and on the same state built with create_model (see conftest.py).
"""

import asyncio
import logging
from typing import Any

import pytest
from pydantic import BaseModel, Field, create_model
from pydantic_core import to_json

from unmute_core.settings import SettingsError
from unmute_core.state import (
    CallInfo,
    CallState,
    HeldSave,
    StateRefused,
    StateSchema,
    call_info,
)
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.tools import ToolRegistry, values_block

APPOINTMENT = {
    "scheduled_date": "2026-03-19",
    "scheduled_time": "09:30",
    "appointment_type": "haircut",
}


def test_empty_values_read_as_words_and_lists_as_empty_json(state: CallState) -> None:
    assert state.render("caller_phone") == "none recorded yet."
    assert state.render("appointments") == "[]"
    assert state.render("caller_reason") == "[]"


def test_false_and_zero_are_values_not_their_absence(state: CallState) -> None:
    state.save({"count": 0, "accepted": False})
    assert (state.render("count"), state.render("accepted")) == ("0", "false")


def test_an_unconfirmed_value_is_seen_only_by_its_confirming_step(state: CallState) -> None:
    state.save({"caller_phone": "+34600111222"})
    assert state.render("caller_phone") == "none recorded yet."
    assert state.render("caller_phone", site="task:confirm_number") == "+34600111222"
    assert not state.is_confirmed("caller_phone")

    state.save({"caller_phone": "+34 600 111 222"}, step="confirm_number")
    assert state.render("caller_phone") == "+34600111222"
    assert state.is_confirmed("caller_phone")


def test_another_step_saving_a_confirmed_value_withdraws_its_confirmation(
    state: CallState,
) -> None:
    state.save({"caller_phone": "+34600111222"}, step="confirm_number")
    state.save({"caller_phone": "+34600111222"}, step="book")
    assert not state.is_confirmed("caller_phone")


def test_an_empty_value_is_never_confirmed(state: CallState) -> None:
    state.save({"caller_phone": None}, step="confirm_number")
    assert not state.is_confirmed("caller_phone")


def test_withdraw_takes_back_what_a_step_confirmed(state: CallState) -> None:
    state.save({"caller_phone": "+34600111222", "count": 2}, step="confirm_number")
    state.withdraw("book")
    assert state.is_confirmed("caller_phone")
    state.withdraw("confirm_number")
    assert not state.is_confirmed("caller_phone")
    assert state.is_confirmed("count")


def test_a_batch_is_all_or_nothing(state: CallState) -> None:
    before = state.snapshot()
    with pytest.raises(StateRefused, match=r"^count: "):
        state.save({"appointments": [APPOINTMENT], "count": "not-a-number"})
    assert state.snapshot() == before
    assert state.values.appointments == []


def test_values_are_kept_typed_and_read_as_json(state: CallState) -> None:
    state.save({"appointments": [APPOINTMENT], "last_appointment": APPOINTMENT})
    assert type(state.values.appointments[0]).__name__ == "Appointment"
    assert str(state.values.appointments[0].scheduled_date) == "2026-03-19"
    assert state.plain("last_appointment.scheduled_time") == "09:30:00"
    assert state.render("last_appointment") == (
        '{"scheduled_date":"2026-03-19","scheduled_time":"09:30:00","appointment_type":"haircut"}'
    )


def test_a_path_past_an_absent_value_reads_as_empty(state: CallState) -> None:
    assert state.plain("last_appointment.scheduled_date") is None
    assert state.render("last_appointment.scheduled_date") == "none recorded yet."
    with pytest.raises(KeyError, match="no state value named 'nope'"):
        state.plain("nope")


def test_a_value_outside_its_type_is_refused_by_name(state: CallState) -> None:
    with pytest.raises(StateRefused, match=r"^appointments\.0\.scheduled_date: .*valid date"):
        state.save({"appointments": [{**APPOINTMENT, "scheduled_date": "tomorrow"}]})
    with pytest.raises(StateRefused, match=r"^caller_reason\.0: Input should be 'create_booking'"):
        state.save({"caller_reason": ["other"]})
    with pytest.raises(StateRefused, match=r"^caller_phone: .*phone number"):
        state.save({"caller_phone": "600 111 222"})
    with pytest.raises(StateRefused, match=r"^nope: not a value of this state"):
        state.save({"nope": 1})


def test_an_email_is_normalized_where_it_enters(state: CallState) -> None:
    state.save({"booked_for": {"name": "Fred Bloggs", "email": "Fred.Bloggs@EXAMPLE.com"}})
    assert state.values.booked_for.email == "Fred.Bloggs@example.com"
    with pytest.raises(StateRefused, match=r"^booked_for\.email: .*email address"):
        state.save({"booked_for": {"name": "Fred", "email": "fred dot bloggs at example dot com"}})
    assert state.values.booked_for.email == "Fred.Bloggs@example.com"


@pytest.mark.parametrize(
    "said",
    [
        "fred dot bloggs at example dot com",
        "fred.bloggs at example.com",
        "fred.bloggs@example",
        "fred bloggs@example.com",
        "Fred Bloggs fred.bloggs@example.com",
        "fred.bloggs@example..com",
    ],
)
def test_each_misheard_email_is_refused_with_its_defect_named(state: CallState, said: str) -> None:
    with pytest.raises(StateRefused) as refused:
        state.save({"reminder_email": said})
    field, reason = refused.value.message.split(": ", 1)
    assert field == "reminder_email"
    assert "email address" in reason and len(reason) > 10


def test_a_long_value_reaches_the_prompt_whole() -> None:
    state = CallState(create_model("Notes", note=(str | None, None)))
    state.save({"note": "x" * 5000})
    assert state.render("note") == "x" * 5000


def test_a_value_past_the_stored_limit_is_refused(state: CallState) -> None:
    with pytest.raises(StateRefused, match=r"^appointments: too long to keep: \d+ characters"):
        state.save({"appointments": [APPOINTMENT] * 1000})


@pytest.mark.parametrize(
    ("text", "saved"),
    [("x" * 64_000, True), ("x" * 64_001, False), ('"' * 40_000, True)],
    ids=["at-the-limit", "one-past", "quotes-counted-once"],
)
def test_the_stored_limit_counts_the_value_and_nothing_around_it(text: str, saved: bool) -> None:
    state = CallState(create_model("Notes", note=(str | None, None)))
    if saved:
        state.save({"note": text})
        assert state.plain("note") == text
    else:
        with pytest.raises(StateRefused, match=r"^note: too long to keep: 64001 characters"):
            state.save({"note": text})


@pytest.mark.parametrize(
    "values",
    [
        {"count": "not-a-number"},
        {"appointments": [{**APPOINTMENT, "scheduled_date": "tomorrow"}]},
        {"nope": 1},
        {"call": {}},
        {"appointments": [APPOINTMENT] * 1000},
    ],
)
def test_check_refuses_what_a_save_refuses_with_its_words(state: CallState, values: dict) -> None:
    with pytest.raises(StateRefused) as saved:
        state.save(values)
    with pytest.raises(StateRefused) as checked:
        state.check(values)
    assert checked.value.message == saved.value.message


def test_check_returns_the_values_as_json_and_saves_nothing(state: CallState) -> None:
    events = []
    state.listen(events.append)
    before = state.snapshot()
    assert state.check({"count": "3", "last_appointment": APPOINTMENT}) == {
        "count": 3,
        "last_appointment": {
            "scheduled_date": "2026-03-19",
            "scheduled_time": "09:30:00",
            "appointment_type": "haircut",
        },
    }
    assert state.check({}) == {}
    assert state.snapshot() == before
    assert events == []


def test_snapshot_gives_every_value_as_the_prompt_reads_it(state: CallState) -> None:
    state.save({"count": 3, "caller_phone": "+34600111222"})
    assert state.snapshot() == {
        "count": "3",
        "accepted": "none recorded yet.",
        "caller_reason": "[]",
        "appointments": "[]",
        "caller_phone": "none recorded yet.",
        "last_appointment": "none recorded yet.",
        "reminder_email": "none recorded yet.",
        "booked_for": "none recorded yet.",
    }
    assert state.snapshot(site="task:confirm_number")["caller_phone"] == "+34600111222"


def test_a_model_field_without_a_default_is_refused() -> None:
    class Strict(BaseModel):
        name: str

    with pytest.raises(TypeError, match=r"Strict\.name has no default"):
        CallState(Strict)


class Note(BaseModel):
    note: str | None = None


class OwnCall(BaseModel):
    call: str | None = None


class AliasedCall(BaseModel):
    phone: str | None = Field(None, alias="call")


@pytest.mark.parametrize(
    "build",
    [
        lambda: CallState(OwnCall),
        lambda: CallState(AliasedCall),
        lambda: CallState(Note, confirm={"call": "verify"}),
        lambda: CallState(Note, call_start={"call": "optional"}),
        lambda: StateSchema([{"name": "call"}]),
        lambda: TaskPlan(TasksSpec(), state=OwnCall, tools=ToolRegistry([])),
    ],
)
def test_a_state_or_a_rule_that_names_call_is_refused_when_built(build: Any) -> None:
    with pytest.raises(ValueError, match="names 'call', which unmute-core reserves"):
        build()


def test_the_call_is_read_only_and_never_listed_with_the_values() -> None:
    state = CallState(Note, call=call_info("inbound", remote_number="+34600111222"))
    session = state.call.session_id

    with pytest.raises(StateRefused, match="call: the library sets it"):
        state.save({"call": CallInfo()})
    state.save({"note": "late"})

    assert state.call.session_id == session
    assert state.render("call.calling_number") == "+34600111222"
    assert state.render("call.end_reason") == ""
    assert state.names == ("note",)
    assert state.dump() == {"note": "late"}
    assert "call" not in state.snapshot()
    assert values_block(state).splitlines()[1:] == ["- note: late"]


@pytest.mark.parametrize(
    ("given", "env", "expected"),
    [
        (None, "salon", "salon"),
        ("desk", "salon", "desk"),
        ("  desk  ", None, "desk"),
    ],
)
def test_the_call_takes_its_agent_id_from_the_code_else_the_environment(
    monkeypatch: pytest.MonkeyPatch, given: str | None, env: str | None, expected: str
) -> None:
    if env is None:
        monkeypatch.delenv("UNMUTE_AGENT_ID", raising=False)
    else:
        monkeypatch.setenv("UNMUTE_AGENT_ID", env)
    state = CallState(Note, call=call_info("inbound", agent_id=given))
    assert state.call.agent_id == expected


def test_a_save_keeps_the_calls_ids_without_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNMUTE_AGENT_ID", raising=False)
    state = CallState(Note, call=call_info("inbound", agent_id="desk", session_id="call-1"))
    state.save({"note": "late"})
    assert (state.call.agent_id, state.call.session_id) == ("desk", "call-1")


def test_a_call_without_an_agent_id_names_both_fixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNMUTE_AGENT_ID", raising=False)
    with pytest.raises(SettingsError, match=r"pass agent_id= to CallInfo.*set UNMUTE_AGENT_ID"):
        CallState(Note)


@pytest.mark.parametrize("field", ["agent_id", "session_id"])
@pytest.mark.parametrize("value", ["sal on", "a,b", "a" * 129, "café"])
def test_an_id_slng_would_refuse_fails_when_the_call_is_built(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        CallInfo.model_validate({field: value})


@pytest.mark.parametrize("model", [int, "State", None])
def test_a_state_that_is_not_a_pydantic_model_class_is_a_type_error(model: Any) -> None:
    with pytest.raises(TypeError, match="CallState needs a Pydantic model class"):
        CallState(model)


def test_confirm_must_name_a_value_of_the_model(model: type[BaseModel]) -> None:
    with pytest.raises(ValueError, match="confirm names 'phone'"):
        CallState(model, confirm={"phone": "verify"})


def test_names_pydantic_keeps_for_itself_still_work_by_their_own_name() -> None:
    model = create_model(
        "Reserved",
        x=(int | None, Field(None, alias="_x")),
        dump=(str | None, Field(None, alias="model_dump")),
    )
    state = CallState(model)
    state.save({"_x": "5", "model_dump": "kept"})
    assert state.snapshot() == {"_x": "5", "model_dump": "kept"}
    with pytest.raises(StateRefused, match=r"^_x: Input should be a valid integer"):
        state.save({"_x": "five"})


async def test_a_held_save_the_caller_said_nothing_after_is_made_as_it_was(
    state: CallState,
) -> None:
    saved: list[object] = []
    state.hold("book", HeldSave(save=saved.append, caller_spoke_since=lambda: False))
    assert await state.settle_held() == ["book"]
    assert saved == [None]
    assert await state.settle_held() == []


async def test_a_held_save_the_caller_spoke_after_follows_the_step_decision(
    state: CallState,
) -> None:
    saved: list[object] = []

    async def keep() -> dict[str, str]:
        return {"note": "corrected"}

    async def drop() -> None:
        return None

    state.hold("kept", HeldSave(saved.append, lambda: True, keep))
    state.hold("dropped", HeldSave(saved.append, lambda: True, drop))
    state.hold("undecided", HeldSave(saved.append, lambda: True))
    assert await state.settle_held() == ["kept"]
    assert saved == [{"note": "corrected"}]


async def test_a_released_held_save_is_never_made(state: CallState) -> None:
    saved: list[object] = []
    state.hold("book", HeldSave(saved.append, lambda: False))
    state.release("book")
    assert await state.settle_held() == []
    assert saved == []


async def test_a_held_save_that_fails_or_is_refused_is_skipped_and_the_rest_are_made(
    state: CallState, caplog: pytest.LogCaptureFixture
) -> None:
    saved: list[object] = []

    async def slow() -> dict[str, str]:
        await asyncio.sleep(1)
        return {}

    async def broken() -> dict[str, str]:
        raise RuntimeError("model down")

    state.hold("slow", HeldSave(saved.append, lambda: True, slow))
    state.hold("broken", HeldSave(saved.append, lambda: True, broken))
    state.hold("refused", HeldSave(lambda _: False, lambda: False))
    state.hold("made", HeldSave(saved.append, lambda: False))
    with caplog.at_level(logging.ERROR):
        assert await state.settle_held(timeout_s=0.01) == ["made"]
    assert saved == [None]
    assert "held save of broken could not be made" in caplog.text


def test_names_come_as_an_ordered_tuple_and_a_set(state) -> None:
    assert set(state.names) == state.name_set
    assert state.names is state.names
    assert isinstance(state.name_set, frozenset)


def test_a_whole_float_renders_as_an_integer_and_a_fraction_stays() -> None:
    class Prices(BaseModel):
        total: float | None = None
        parts: list[float] = Field(default_factory=list)

    state = CallState(Prices)
    state.save({"total": 3.0, "parts": [1.0, 2.5]})
    assert state.render("total") == "3"
    assert state.render("parts") == "[1,2.5]"
    assert state.snapshot()["total"] == "3"
    state.save({"total": 2.5})
    assert state.render("total") == "2.5"


def test_a_float_too_large_to_hold_its_int_exactly_keeps_its_float_text() -> None:
    class Prices(BaseModel):
        total: float | None = None

    state = CallState(Prices)
    state.save({"total": 1e300})
    assert state.render("total") == to_json(1e300).decode()
    state.save({"total": 2.0**53 - 1})
    assert state.render("total") == "9007199254740991"
