"""``CallState.listen``: one event per save that changed something, with who changed it."""

import logging

import pytest
from pydantic import BaseModel

from unmute_core.state import CallState, StateEvent, StateOrigin

STEP = StateOrigin("step", "confirm_number")


class Booking(BaseModel):
    """A state with a prefetched value that is looked up from another."""

    phone: str | None = None
    customer: str | None = None
    count: int | None = None


@pytest.fixture
def events(state: CallState) -> list[StateEvent]:
    """Every event the state's listener gets."""
    told: list[StateEvent] = []
    state.listen(told.append)
    return told


def test_a_save_that_changed_something_tells_what_and_who(
    state: CallState, events: list[StateEvent]
) -> None:
    state.save({"count": 2}, step="book")
    assert len(events) == 1
    assert events[0].origin == StateOrigin("step", "book")
    [change] = events[0].changes
    assert (change.field, change.value, change.confirmed) == ("count", 2, None)
    assert change.by == StateOrigin("step", "book")


def test_a_save_that_changed_nothing_tells_nothing(
    state: CallState, events: list[StateEvent]
) -> None:
    state.save({"count": 2})
    state.save({"count": 2})
    state.save({"count": None}, step="x")
    assert len(events) == 2


def test_a_confirmed_value_says_whether_the_caller_agreed(
    state: CallState, events: list[StateEvent]
) -> None:
    state.save({"caller_phone": "+34600111222"})
    state.save({"caller_phone": "+34600111222"}, step="confirm_number")
    assert [e.changes[0].confirmed for e in events] == [False, True]
    assert events[1].changes[0].value == "+34600111222"


def test_withdrawing_a_confirmation_tells_the_value_is_unconfirmed(
    state: CallState, events: list[StateEvent]
) -> None:
    state.save({"caller_phone": "+34600111222"}, step="confirm_number")
    state.withdraw("confirm_number")
    assert events[-1].origin == StateOrigin("withdraw", "confirm_number")
    assert [(c.field, c.confirmed) for c in events[-1].changes] == [("caller_phone", False)]


def test_withdrawing_what_holds_nothing_tells_nothing(
    state: CallState, events: list[StateEvent]
) -> None:
    state.withdraw("confirm_number")
    assert events == []


def test_a_value_looked_up_from_a_changed_one_goes_back_to_its_default() -> None:
    state = CallState(Booking, dependencies={"customer": ("phone",)})
    state.save({"phone": "1"})
    state.save({"customer": "Ana"}, inputs=("phone",), origin=StateOrigin("prefetch", "lookup"))
    told: list[StateEvent] = []
    state.listen(told.append)
    state.save({"phone": "2"}, step="ask")
    [event] = told
    by = {change.field: (change.value, change.by) for change in event.changes}
    assert by == {
        "phone": ("2", StateOrigin("step", "ask")),
        "customer": (None, StateOrigin("reset", "ask")),
    }


def test_the_origin_defaults_to_the_step_a_prefetch_or_the_call_start(state: CallState) -> None:
    told: list[StateEvent] = []
    state.listen(told.append)
    state.save({"count": 1})
    state.save({"count": 2}, inputs=())
    state.save({"count": 3}, step="book")
    assert [e.origin for e in told] == [
        StateOrigin("call_start"),
        StateOrigin("prefetch"),
        StateOrigin("step", "book"),
    ]


def test_replay_tells_the_values_already_set_as_one_start_event(state: CallState) -> None:
    state.save({"count": 1}, origin=StateOrigin("prefetch", "today"))
    state.save({"caller_phone": "+34600111222"}, origin=StateOrigin("prefetch", "caller"))
    told: list[StateEvent] = []
    state.listen(told.append, replay=True)
    [event] = told
    assert event.origin == StateOrigin("start")
    assert [(c.field, c.by, c.confirmed) for c in event.changes] == [
        ("count", StateOrigin("prefetch", "today"), None),
        ("caller_phone", StateOrigin("prefetch", "caller"), False),
    ]


def test_replay_of_an_empty_state_tells_nothing(state: CallState) -> None:
    told: list[StateEvent] = []
    state.listen(told.append, replay=True)
    assert told == []


def test_a_listener_that_raises_is_logged_removed_and_never_fails_the_save(
    state: CallState, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[StateEvent] = []

    def broken(event: StateEvent) -> None:
        raise RuntimeError("boom")

    state.listen(broken)
    state.listen(calls.append)
    with caplog.at_level(logging.WARNING):
        state.save({"count": 1})
        state.save({"count": 2})
    assert state.plain("count") == 2
    assert len(calls) == 2
    assert "state listener failed" in caplog.text
    assert caplog.text.count("state listener failed") == 1


def test_stopping_a_listener_stops_its_events(state: CallState) -> None:
    told: list[StateEvent] = []
    stop = state.listen(told.append)
    state.save({"count": 1})
    stop()
    stop()
    state.save({"count": 2})
    assert len(told) == 1
