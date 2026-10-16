"""An Unmute author's state.py, as the typed_state example writes it.

Copied from unmute@d0c67f1 internal/testdata/typed_state/state.py. One change: Phone comes
from unmute_core.state instead of a local PhoneNumberValidator, as Unmute will emit it.
"""

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

from unmute_core.state import Phone


class Contact(BaseModel):
    """A person's name and email address, held as two separate parts."""

    name: str
    email: EmailStr


class Appointment(BaseModel):
    """One thing being booked, moved or cancelled."""

    scheduled_date: date
    scheduled_time: time = Field(description="The time the caller agreed to.")
    appointment_type: Literal["haircut", "dry_cut"] = Field(
        description="The service the caller asked for, in the salon's own words."
    )


class State(BaseModel):
    """What one call knows, shared by every agent and task in it."""

    count: int | None = None
    accepted: bool | None = None
    caller_reason: list[Literal["create_booking", "cancel_booking"]] = Field(
        [],
        description="Why the caller rang. More than one, because one call can do more than one thing.",
    )
    appointments: list[Appointment] = Field(
        [],
        description="What this call has booked so far, in the order the caller gave them.",
    )
    caller_phone: Phone | None = Field(
        None,
        description="The number the caller rang from, once they have agreed it is theirs.",
    )
    last_appointment: Appointment | None = Field(
        None,
        description="The appointment most recently booked on this call, so a prompt can name one part of it.",
    )
    reminder_email: EmailStr | None = Field(
        None,
        description="Where the caller wants the reminder sent, saved only once they have spelled it out.",
    )
    booked_for: Contact | None = Field(
        None,
        description="Who the appointment is under, with their name and their email address held apart.",
    )
