"""Nosh's three tools: find the area, find places there, and say one place's details.

Each tool returns a typed result that carries code's next line, and the values it saves.
The registry in ``nosh.agent`` saves them (``Assign``) and speaks the line (``ToolReply``),
so the model is asked for no reply. Two rules hold in code, not in the prompt:

- a search runs only on the spot read back to the caller: ``find_places`` locks it;
- a pick counts only if it was offered: ``place_details`` locks the offer and refuses an id
  it does not hold.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, TypeAdapter
from unmute_core.tools import ToolError

from nosh import speech
from nosh.places import Located, NotFoundError, Place, PlaceInfo, Places, places
from nosh.state import Details, Offer, Offered, Spot

TIMEZONE = "Europe/Madrid"
OFFER_AT_MOST = 6
"""How many places go on offer. The caller hears the first three."""
SEARCH = TypeAdapter(list[Place])


class FoundArea(BaseModel):
    """What ``find_area`` returns. A new area takes back the offer and the details.

    ``ask_craving`` is code's next step: with no craving and no exact address, the read-back
    asks what they are in the mood for, so the caller's yes never has to pick one.
    """

    found: bool
    spot: Spot | None = None
    formatted_address: str = ""
    ask_craving: bool = False
    craving: str | None = None
    offer: Offer | None = None
    details: Details | None = None


class FoundPlaces(BaseModel):
    """What ``find_places`` returns: the offer, or why there is none."""

    found: bool
    ask_craving: bool = False
    listed: str = ""
    craving: str | None = None
    offer: Offer | None = None
    details: Details | None = None


class FoundDetails(BaseModel):
    """What ``place_details`` returns: what code says about the place, or that it has none."""

    found: bool
    name: str
    spoken: str = ""
    details: Details | None = None


def madrid_today() -> date:
    """Return today on the Madrid clock, which "today's hours" are read for."""
    return datetime.now(ZoneInfo(TIMEZONE)).date()


@dataclass
class Finder:
    """The tools, on one place backend.

    Attributes:
        places: Where places are looked up.
        today: The day whose opening hours are said.
    """

    places: Places = field(default_factory=places)
    today: Callable[[], date] = madrid_today

    def find_area(
        self,
        area: Annotated[
            str,
            Field(
                min_length=1,
                description=(
                    "Where they want to eat or drink, as they said it: an address, a cross "
                    "street, a landmark, a neighbourhood or a town, with the city if they named one."
                ),
            ),
        ],
        at_address: Annotated[
            bool,
            Field(description="True only for an exact street address or a point they are at."),
        ],
        craving: Annotated[
            str | None,
            Field(
                description=(
                    'What they said they are in the mood for along with the area, like "tapas". '
                    "Leave it out if they have not said."
                )
            ),
        ] = None,
    ) -> FoundArea:
        """Find where the caller wants to look. The application reads the place back to them."""
        try:
            located = Located.model_validate(self.places.post("resolve-location", {"query": area}))
        except NotFoundError:
            return FoundArea(found=False)
        wanted = (craving or "").strip() or None
        spot = Spot(**located.model_dump(), at_address=at_address, craving=wanted)
        return FoundArea(
            found=True,
            spot=spot,
            formatted_address=located.formatted_address,
            ask_craving=wanted is None and not at_address,
            craving=wanted,
        )

    def find_places(
        self,
        spot: Spot,
        craving: Annotated[
            str | None,
            Field(
                description=(
                    'What they are in the mood for: a cuisine, "bar", "restaurant", "cocktail '
                    'bar", a venue by name, or "no preference". Leave it out to search for what '
                    "they said with the area."
                )
            ),
        ] = None,
    ) -> FoundPlaces:
        """Search places at the location the caller agreed to. The application says them."""
        wanted = (craving or "").strip() or spot.craving or ""
        anything = wanted.lower() in ("", "no preference")
        if not wanted and not spot.at_address:
            return FoundPlaces(found=False, ask_craving=True)
        point = {"latitude": spot.latitude, "longitude": spot.longitude}
        if spot.at_address and anything:
            raw = self.places.post("search-nearby", point)
        else:
            area = speech.address(spot.formatted_address) or spot.formatted_address
            query = f"{'restaurants and bars' if anything else wanted} in {area}"
            raw = self.places.post("search-text", {"query": query, **point})
        found = SEARCH.validate_python(raw)[:OFFER_AT_MOST]
        if not found:
            return FoundPlaces(found=False, craving=wanted or None)
        offer = Offer(
            places=[
                Offered(id=f"p{index}", name=place.name, place_id=place.place_id)
                for index, place in enumerate(found, 1)
            ]
        )
        return FoundPlaces(
            found=True, listed=speech.listed(found), craving=wanted or None, offer=offer
        )

    def place_details(
        self,
        offer: Offer,
        place: Annotated[
            str, Field(description="The id of the place they picked from the offer, like p1.")
        ],
    ) -> FoundDetails:
        """Say the details of the place the caller picked: where it is, price, today's hours."""
        picked = next((p for p in offer.places if p.id == place.strip().lower()), None)
        if picked is None:
            # The model may only pick what the caller was offered.
            ids = ", ".join(f"{p.id} ({p.name})" for p in offer.places)
            raise ToolError(f"{place} was never offered. The offer is: {ids}.")
        try:
            info = PlaceInfo.model_validate(
                self.places.post("place-details", {"place_id": picked.place_id})
            )
        except NotFoundError:
            return FoundDetails(found=False, name=picked.name)
        spoken, price, hours = speech.details(info, self.today())
        details = Details(
            place=picked.id,
            name=info.name,
            address=speech.address(info.address),
            rating=info.rating,
            price=price,
            today=hours,
            phone=info.phone,
            website=info.website,
        )
        return FoundDetails(found=True, name=info.name, spoken=spoken, details=details)
