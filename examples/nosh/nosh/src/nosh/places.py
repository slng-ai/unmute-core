"""Where Nosh finds places: the real place service, or a demo backend with no key.

Both answer the same four routes, as JSON:

| Route | Body | Answer |
|---|---|---|
| ``resolve-location`` | ``{query}`` | ``{latitude, longitude, formatted_address}``, or not found |
| ``search-text`` | ``{query, latitude, longitude}`` | a list of places |
| ``search-nearby`` | ``{latitude, longitude}`` | a list of places |
| ``place-details`` | ``{place_id}`` | one place's details, or not found |

``PlaceService`` calls the hosted service (``TOOLS_URL``, ``TOOLS_API_KEY``) with the
standard library. ``DemoPlaces`` answers from a few Barcelona places, so the example runs
with no key. ``places()`` picks one from the environment.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pydantic import BaseModel

DEFAULT_URL = "https://tools-agent-infra.onrender.com"


class NotFoundError(Exception):
    """The service has nothing for that location or place."""


class Places(Protocol):
    """Anything that answers the four place routes."""

    def post(self, route: str, body: dict[str, Any]) -> Any:
        """Return one route's JSON answer, or raise ``NotFoundError``."""
        ...


class Located(BaseModel):
    """What ``resolve-location`` answers."""

    latitude: float
    longitude: float
    formatted_address: str


class Place(BaseModel):
    """One search result: a search knows a place's name, kind, rating and reviews only."""

    place_id: str
    name: str
    address: str
    rating: float | None = None
    user_rating_count: int | None = None
    primary_type_display_name: str | None = None


class PlaceInfo(BaseModel):
    """What ``place-details`` answers for one place."""

    place_id: str
    name: str
    address: str
    rating: float | None = None
    price_level: str | None = None
    opening_hours: list[str] | None = None
    phone: str | None = None
    website: str | None = None
    primary_type_display_name: str | None = None


@dataclass
class PlaceService:
    """The hosted place tools: one bearer-token ``POST`` per route, no retry."""

    url: str = DEFAULT_URL
    key: str = ""
    timeout: float = 20.0

    def post(self, route: str, body: dict[str, Any]) -> Any:
        """Return one route's JSON answer. A 404 is ``NotFoundError``; any other failure raises."""
        request = Request(
            f"{self.url.rstrip('/')}/tools/{route}",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 404:
                raise NotFoundError(f"{route}: nothing found") from error
            raise


def _place(gid: str, name: str, rating: float, count: int, kind: str) -> dict[str, Any]:
    """Return one demo search result."""
    return {
        "place_id": gid,
        "name": name,
        "address": f"{name} street 1, 08002 Barcelona, Spain",
        "rating": rating,
        "user_rating_count": count,
        "primary_type_display_name": kind,
    }


WEEK = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
AREAS = (
    (("gothic", "gotic", "gòtic"), (41.3833, 2.1777, "Barri Gòtic, Barcelona, Spain")),
    (("gràcia", "gracia"), (41.4036, 2.1561, "Gràcia, Barcelona, Spain")),
    (("born",), (41.3851, 2.1823, "El Born, Barcelona, Spain")),
    (
        ("sant joan 55", "sant joan, 55"),
        (41.3946, 2.1745, "Passeig de Sant Joan, 55, 08009 Barcelona, Spain"),
    ),
)
"""The demo's areas, by words a caller may use for them. Anything else is not found."""
TAPAS = [
    _place("g-canete", "Bar Cañete", 4.5, 3200, "Tapas Restaurant"),
    _place("g-gone", "El Xampanyet", 4.4, 800, "Bar"),
    _place("g-palma", "Bodega La Palma", 4.0, 1500, "Spanish Restaurant"),
    _place("g-paixano", "Can Paixano", 4.2, 9000, "Bar"),
]
COCKTAILS = [
    _place("g-paradiso", "Paradiso", 4.6, 12000, "Cocktail Bar"),
    _place("g-boadas", "Boadas", 4.3, 2400, "Cocktail Bar"),
]
NEARBY = [_place("g-snack", "Bar Snack 55", 4.0, 600, "Bar")]
DETAILS = {
    "g-canete": {
        "place_id": "g-canete",
        "name": "Bar Cañete",
        "address": "Carrer de la Unió, 17, Ciutat Vella, 08001 Barcelona, Spain",
        "rating": 4.5,
        "price_level": "PRICE_LEVEL_MODERATE",
        "opening_hours": [f"{day}: 1:00 PM \u2013 12:00 AM" for day in WEEK],
        "phone": "+34 932 70 34 58",
        "website": "https://barcanete.com",
        "primary_type_display_name": "Tapas Restaurant",
    },
    "g-palma": {
        "place_id": "g-palma",
        "name": "Bodega La Palma",
        "address": "Carrer de la Palma de Sant Just, 7, 08002 Barcelona, Spain",
        "rating": 4.0,
    },
    "g-paradiso": {
        "place_id": "g-paradiso",
        "name": "Paradiso",
        "address": "Carrer de Rera Palau, 4, 08003 Barcelona, Spain",
        "rating": 4.6,
        "price_level": "PRICE_LEVEL_EXPENSIVE",
        "opening_hours": [f"{day}: 7:00 PM \u2013 2:30 AM" for day in WEEK],
        "phone": "+34 933 60 72 22",
        "primary_type_display_name": "Cocktail Bar",
    },
    "g-snack": {
        "place_id": "g-snack",
        "name": "Bar Snack 55",
        "address": "Passeig de Sant Joan, 55, 08009 Barcelona, Spain",
        "rating": 4.0,
        "price_level": "PRICE_LEVEL_INEXPENSIVE",
        "opening_hours": [f"{day}: 8:00 AM \u2013 11:00 PM" for day in WEEK],
    },
}
"""Invented details for real-sounding places. A place missing here has no details."""


@dataclass
class DemoPlaces:
    """A few Barcelona places, answered the way the place service answers. Keeps every request."""

    sent: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def post(self, route: str, body: dict[str, Any]) -> Any:
        """Return one route's answer from the demo data, or raise ``NotFoundError``."""
        self.sent.append((route, body))
        if route == "resolve-location":
            query = " ".join(body["query"].lower().split())
            found = next((at for names, at in AREAS if any(n in query for n in names)), None)
            if found is None:
                raise NotFoundError(route)
            latitude, longitude, address = found
            return {"latitude": latitude, "longitude": longitude, "formatted_address": address}
        if route == "search-text":
            query = body["query"].lower()
            if "sushi" in query:
                return []
            return COCKTAILS if "cocktail" in query else TAPAS
        if route == "search-nearby":
            return NEARBY
        if route == "place-details":
            if (answer := DETAILS.get(body["place_id"])) is None:
                raise NotFoundError(route)
            return answer
        raise ValueError(f"Nosh never calls {route}")

    def routes(self, name: str) -> list[dict[str, Any]]:
        """Return the bodies sent to one route, in order."""
        return [body for route, body in self.sent if route == name]


def places() -> Places:
    """Return the place service when ``TOOLS_API_KEY`` is set, else the demo places."""
    if key := os.getenv("TOOLS_API_KEY"):
        return PlaceService(url=os.getenv("TOOLS_URL") or DEFAULT_URL, key=key)
    return DemoPlaces()
