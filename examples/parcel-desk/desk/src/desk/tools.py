"""The agent's tools: plain Python functions over an in-memory depot.

A real deployment calls the carrier's systems here. The library runs them the same way as
any other tool: their arguments are checked against the schema their signature gives, and
what they return against the Pydantic model they declare. A failure the model should read
is a ``ToolError``.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from unmute_core.tools import ToolError

TrackingId = Annotated[str, Field(description="The parcel's tracking id, like PD-1042.")]


class Parcel(BaseModel):
    """One parcel, as the depot knows it: the spec's ``Parcel`` shape."""

    tracking_id: str
    status: Literal["in_transit", "out_for_delivery", "delivered", "held"]
    delivery_day: date | None
    town: str


class Found(BaseModel):
    """What ``find_parcel`` returns: the parcel, or a hint for the caller."""

    found: bool
    parcel: Parcel | None = None
    hint: str = ""


class Rescheduled(BaseModel):
    """What ``reschedule_delivery`` returns."""

    status: Literal["rescheduled"]
    parcel: Parcel


class NoteSaved(BaseModel):
    """What ``save_callback_note`` returns."""

    saved: bool
    note: str


class Ended(BaseModel):
    """What ``end_call`` returns."""

    ended: bool


@dataclass
class Depot:
    """The parcels the depot holds, and the call-back notes left on this process."""

    parcels: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "PD-1042": {
                "tracking_id": "PD-1042",
                "status": "in_transit",
                "delivery_day": (date.today() + timedelta(days=2)).isoformat(),
                "town": "Leeds",
            },
            "PD-2077": {
                "tracking_id": "PD-2077",
                "status": "held",
                "delivery_day": None,
                "town": "York",
            },
        }
    )
    notes: list[dict[str, str]] = field(default_factory=list)

    def find_parcel(self, tracking_id: TrackingId) -> Found:
        """Find a parcel by its tracking id: where it is and the day it is due."""
        parcel = self.parcels.get(tracking_id.strip().upper())
        if parcel is None:
            return Found(
                found=False, hint="No parcel has that id. Ask the caller to read it again."
            )
        return Found(found=True, parcel=Parcel.model_validate(parcel))

    def reschedule_delivery(self, tracking_id: TrackingId, day: date) -> Rescheduled:
        """Move a parcel's delivery to another day, from tomorrow on."""
        parcel = self.parcels.get(tracking_id)
        if parcel is None:
            raise ToolError("no parcel has that id")
        if day <= date.today():
            raise ToolError("a delivery can be moved to tomorrow at the earliest")
        parcel["delivery_day"] = day.isoformat()
        return Rescheduled(status="rescheduled", parcel=Parcel.model_validate(parcel))

    def save_callback_note(self, note: str, call: str) -> NoteSaved:
        """Leave a note for a person to call the caller back about."""
        self.notes.append({"note": note, "call": call})
        return NoteSaved(saved=True, note=note)


def end_call() -> Ended:
    """End the call once the caller says they are done. Say your goodbye in the same reply."""
    return Ended(ended=True)
