"""Values a prefetch looked up from other values: confirmation follows, a change invalidates.

Ported from unmute@d0c67f1 internal/generate/prefetch_smoke_test.go (provenanceScript).
"""

from copy import deepcopy
from datetime import date

import pytest
from pydantic import BaseModel, create_model

from unmute_core.state import CallState, StateRefused
from unmute_core.state._call_state import EMPTY_TEXT


class ProfileState(BaseModel):
    """A caller's number, the profile looked up from it, and a record looked up from that."""

    caller_phone: str | None = None
    caller_name: str = ""
    customer_id: str | None = None
    booking_date: date | None = None
    verified: bool | None = None


BUILT_PROFILE_STATE = create_model(
    "ProfileState",
    caller_phone=(str | None, None),
    caller_name=(str, ""),
    customer_id=(str | None, None),
    booking_date=(date | None, None),
    verified=(bool | None, None),
)
CONFIRM = {
    "caller_phone": "verify_caller",
    "caller_name": "verify_caller",
    "customer_id": "verify_caller",
}
DEPENDENCIES = {"caller_name": ["caller_phone"], "customer_id": ["caller_name"]}


@pytest.fixture(params=[ProfileState, BUILT_PROFILE_STATE], ids=["author", "built"])
def state(request: pytest.FixtureRequest) -> CallState:
    """A call whose prefetch already ran: number, profile, record and a clock reading."""
    state = CallState(request.param, confirm=CONFIRM, dependencies=DEPENDENCIES)
    seed(state)
    return state


def seed(state: CallState) -> None:
    """Save what the prefetch entries would, each with the inputs it read."""
    state.save({"caller_phone": "+34600111222"}, inputs=())
    state.save({"caller_name": "OLD_PROFILE"}, inputs=("caller_phone",))
    state.save({"customer_id": "OLD_RELATED"}, inputs=("caller_name",))
    state.save({"booking_date": "2026-09-06"}, inputs=())


def everything(state: CallState) -> tuple:
    """Everything a save may change."""
    values = state.values.model_dump()
    return deepcopy((values, {name: state.is_unconfirmed(name) for name in values}))


def test_a_looked_up_value_is_hidden_until_its_source_is_confirmed(state: CallState) -> None:
    assert state.render("caller_name") == EMPTY_TEXT
    assert state.render("caller_name", "task:verify_caller") == "OLD_PROFILE"
    assert state.is_unconfirmed("caller_name")


def test_agreeing_to_the_same_value_keeps_and_confirms_what_was_looked_up(
    state: CallState,
) -> None:
    state.save({"caller_phone": "+34600111222"}, step="verify_caller")

    assert state.render("caller_name") == "OLD_PROFILE"
    assert state.render("customer_id") == "OLD_RELATED"
    assert state.is_confirmed("customer_id")


def test_a_changed_input_resets_everything_looked_up_from_it(state: CallState) -> None:
    state.save({"caller_phone": "+34600999888"}, step="verify_caller")

    assert state.values.caller_name == ""
    assert state.values.customer_id is None
    assert state.plain("booking_date") == "2026-09-06"


def test_a_refused_batch_changes_nothing(state: CallState) -> None:
    before = everything(state)

    with pytest.raises(StateRefused):
        state.save({"caller_phone": "+34600999888", "verified": "invalid"}, step="verify_caller")

    assert everything(state) == before


def test_a_replacement_in_the_same_batch_survives_the_reset(state: CallState) -> None:
    state.save({"caller_phone": "+34600999888", "caller_name": "NEW_PROFILE"}, step="verify_caller")

    assert state.values.caller_name == "NEW_PROFILE"
    assert state.values.customer_id is None
    assert state.is_confirmed("caller_name")


def test_another_writer_is_not_agreement(state: CallState) -> None:
    state.save({"caller_phone": "+34600111222"}, step="verify_caller")
    state.save({"caller_phone": "+34600111222"}, step="another_task")

    assert state.render("caller_phone") == EMPTY_TEXT
    assert state.render("caller_name") == EMPTY_TEXT


def test_a_withdrawn_source_takes_its_looked_up_values_with_it() -> None:
    state = CallState(
        ProfileState,
        confirm={"caller_phone": "verify_caller"},
        dependencies={"caller_name": ["caller_phone"]},
    )
    seed(state)
    state.save({"caller_phone": "+34600111222"}, step="verify_caller")
    assert state.is_confirmed("caller_name")

    state.withdraw("verify_caller")

    assert state.is_unconfirmed("caller_name")


def test_a_fresh_call_shares_nothing(state: CallState) -> None:
    fresh = CallState(ProfileState, confirm=CONFIRM, dependencies=DEPENDENCIES)

    fresh.save({"caller_phone": "+34600111222"}, step="verify_caller")

    assert fresh.render("caller_name") == EMPTY_TEXT
    assert state.is_unconfirmed("caller_name")
    assert not state.is_confirmed("caller_phone")


def test_dependencies_must_name_values_of_the_model() -> None:
    with pytest.raises(ValueError, match="'caller_email'"):
        CallState(ProfileState, dependencies={"caller_name": ["caller_email"]})


def test_a_looked_up_value_no_step_confirms_is_hidden_everywhere_until_its_source_is() -> None:
    state = CallState(
        ProfileState,
        confirm={"caller_phone": "verify_caller"},
        dependencies={"caller_name": ["caller_phone"]},
    )
    seed(state)

    assert state.render("caller_name") == EMPTY_TEXT
    assert state.render("caller_name", "task:verify_caller") == EMPTY_TEXT
    assert state.snapshot()["caller_name"] == EMPTY_TEXT
    state.save({"caller_phone": "+34600111222"}, step="verify_caller")
    assert state.snapshot()["caller_name"] == "OLD_PROFILE"
