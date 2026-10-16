"""The same state two ways: the author's class, and one built at runtime from type text."""

from datetime import date, time
from typing import Any, Literal

import pytest
from pydantic import BaseModel, EmailStr, Field, create_model

from tests.state.unmute_state import State
from unmute_core.state import CallState, Phone

_CONTACT: Any = create_model("Contact", name=str, email=EmailStr)
_APPOINTMENT: Any = create_model(
    "Appointment",
    scheduled_date=date,
    scheduled_time=time,
    appointment_type=Literal["haircut", "dry_cut"],
)
BUILT_STATE = create_model(
    "State",
    count=(int | None, None),
    accepted=(bool | None, None),
    caller_reason=(list[Literal["create_booking", "cancel_booking"]], Field(default_factory=list)),
    appointments=(list[_APPOINTMENT], Field(default_factory=list)),
    caller_phone=(Phone | None, None),
    last_appointment=(_APPOINTMENT | None, None),
    reminder_email=(EmailStr | None, None),
    booked_for=(_CONTACT | None, None),
)
CONFIRM = {"caller_phone": "confirm_number"}


@pytest.fixture(params=[State, BUILT_STATE], ids=["author", "built"])
def model(request: pytest.FixtureRequest) -> type[BaseModel]:
    """Each test that takes it runs on both paths."""
    return request.param


@pytest.fixture
def state(model: type[BaseModel]) -> CallState:
    """A fresh call state on each path."""
    return CallState(model, confirm=CONFIRM)
