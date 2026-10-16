"""What one Nosh call knows: one Pydantic model, typed field by field.

The model saves none of these directly. Each tool's checked result saves them, through the
registry's ``assign`` rows, and replaces what rested on them: a new area takes back the
places on offer and the details said.
"""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

CustomerName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]


class Spot(BaseModel):
    """The location the caller's words resolved to."""

    formatted_address: str
    latitude: float
    longitude: float
    at_address: bool = Field(description="An exact address or a point they are at.")


class Offered(BaseModel):
    """One place on offer, by the id the caller picks it with."""

    id: str
    name: str
    place_id: str


class Offer(BaseModel):
    """The places on offer for the spot and the craving."""

    places: list[Offered]


class Details(BaseModel):
    """What code said about the place the caller picked."""

    place: str
    name: str
    address: str
    rating: float | None = None
    price: str = ""
    today: str = ""
    phone: str | None = None
    website: str | None = None


class State(BaseModel):
    """One call. Only the caller's name is known when it starts.

    Attributes:
        customer_name: The name of the person who asked for the call. The dispatch's
            ``call_start`` must carry it: a call without one never starts.
        spot: The location read back to the caller.
        craving: What they are in the mood for.
        offer: The places on offer, ``p1`` to ``p6``.
        details: The details of the place they picked.
    """

    customer_name: CustomerName | None = None
    spot: Spot | None = None
    craving: str | None = None
    offer: Offer | None = None
    details: Details | None = None
