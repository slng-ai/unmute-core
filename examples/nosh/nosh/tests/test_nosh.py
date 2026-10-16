"""Nosh's tools and speech, offline: no model, no network, the demo places.

The whole calls run live, on both frameworks, from ``../scenarios`` (see the README).
"""

from datetime import date

import pytest
from unmute_core.call import CallStartError
from unmute_core.tools import ToolError

from nosh.agent import GREETING, NOSH, Nosh
from nosh.finder import Finder
from nosh.places import DemoPlaces, PlaceInfo
from nosh.speech import address, details, today

THURSDAY = date(2026, 10, 1)
CANETE = (
    "Bar Cañete is on Carrer de la Unió, 17, Ciutat Vella, Barcelona. It's a 4.5 out of "
    "five, and mid-range. Today they're open 1:00 PM to 12:00 AM. I've got their number and "
    "a website too, if you'd like them. Want to hear about another one, or is this the one?"
)


def finder() -> tuple[Finder, DemoPlaces]:
    demo = DemoPlaces()
    return Finder(places=demo, today=lambda: THURSDAY), demo


async def test_the_greeting_names_the_caller_and_asks_where(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNMUTE_CALL_START", raising=False)
    state = await Nosh().new_call(call_start={"customer_name": "  Maya "})
    assert GREETING.line(state) == (
        "Hey Maya, it's Nosh! You wanted a hand finding somewhere good. Whereabouts are you "
        "looking? An address or an area is fine."
    )


@pytest.mark.parametrize(
    ("dispatched", "refusal"),
    [
        (None, "Missing call_start fields: customer_name"),
        ({"customer_name": "   "}, "call_start: customer_name"),
        ({"customer_name": "x" * 41}, "call_start: customer_name"),
    ],
)
async def test_a_call_starts_only_with_the_callers_name(
    monkeypatch: pytest.MonkeyPatch, dispatched: dict[str, str] | None, refusal: str
) -> None:
    monkeypatch.delenv("UNMUTE_CALL_START", raising=False)
    with pytest.raises(CallStartError, match=refusal):
        await Nosh().new_call(call_start=dispatched)


def test_a_whole_search_is_said_by_code() -> None:
    tools, demo = finder()
    found = tools.find_area("Gothic Quarter, Barcelona", at_address=False)
    assert found.formatted_address == "Barri Gòtic, Barcelona, Spain"
    assert found.spot is not None
    places = tools.find_places(found.spot, "tapas")
    assert places.listed == (
        "Bar Cañete, a tapas restaurant with a 4.5 out of five from lots of reviews; "
        "El Xampanyet, a bar with a 4.4 out of five from 800 reviews; and Bodega La Palma, a "
        "spanish restaurant with a 4 out of five from lots of reviews"
    )
    assert places.offer is not None
    picked = tools.place_details(places.offer, "p1")
    assert picked.spoken == CANETE
    assert picked.details is not None and picked.details.price == "mid-range"
    assert [route for route, _ in demo.sent] == ["resolve-location", "search-text", "place-details"]


def test_an_area_with_no_craving_asks_first_and_an_address_searches_nearby() -> None:
    tools, demo = finder()
    area = tools.find_area("Gràcia, Barcelona", at_address=False)
    assert area.ask_craving and area.spot is not None
    assert tools.find_places(area.spot).ask_craving
    assert not demo.routes("search-text")
    street = tools.find_area("Passeig de Sant Joan 55, Barcelona", at_address=True)
    assert not street.ask_craving and street.spot is not None
    nearby = tools.find_places(street.spot)
    assert nearby.listed == "Bar Snack 55, a bar with a 4 out of five from 600 reviews"
    assert len(demo.routes("search-nearby")) == 1


def test_the_craving_said_with_the_area_is_searched_on_the_callers_yes() -> None:
    tools, demo = finder()
    area = tools.find_area("Gothic Quarter, Barcelona", at_address=False, craving=" tapas ")
    assert not area.ask_craving and area.craving == "tapas" and area.spot is not None
    places = tools.find_places(area.spot)
    assert places.craving == "tapas" and places.listed.startswith("Bar Cañete")
    assert demo.routes("search-text")


def test_a_new_area_takes_back_the_offer_and_the_details() -> None:
    tools, _ = finder()
    found = tools.find_area("Born, Barcelona", at_address=False)
    assert found.offer is None and found.details is None


def test_an_invented_place_is_refused_before_any_lookup() -> None:
    tools, demo = finder()
    spot = tools.find_area("Gothic Quarter, Barcelona", at_address=False).spot
    assert spot is not None
    offer = tools.find_places(spot, "tapas").offer
    assert offer is not None
    with pytest.raises(ToolError, match="never offered"):
        tools.place_details(offer, "p9")
    assert not demo.routes("place-details")


def test_a_place_with_no_details_goes_back_to_the_model() -> None:
    tools, _ = finder()
    spot = tools.find_area("Gothic Quarter, Barcelona", at_address=False).spot
    assert spot is not None
    offer = tools.find_places(spot, "tapas").offer
    assert offer is not None
    lost = tools.place_details(offer, "p2")
    assert not lost.found and lost.name == "El Xampanyet"


def test_nowhere_is_not_found_and_sushi_finds_nothing() -> None:
    tools, _ = finder()
    assert not tools.find_area("Nowheresville", at_address=False).found
    spot = tools.find_area("Gothic Quarter, Barcelona", at_address=False).spot
    assert spot is not None
    nothing = tools.find_places(spot, "sushi")
    assert not nothing.found and not nothing.ask_craving


def test_every_reply_and_save_row_fits_its_tool() -> None:
    assert Nosh().registry is not None
    assert NOSH.registry.ends_call("end_call", {"ended": True})


def test_speech() -> None:
    assert address("Carrer de la Unió, 17, Ciutat Vella, 08001 Barcelona, Spain") == (
        "Carrer de la Unió, 17, Ciutat Vella, Barcelona"
    )
    assert today(["Thursday: Closed"], THURSDAY) == "They're closed today."
    assert today(["Thursday: Open 24 hours"], THURSDAY) == "They're open all day today."
    info = PlaceInfo(place_id="x", name="X", address="A, B", rating=4.0)
    assert details(info, THURSDAY)[0] == (
        "X is on A. It's a 4 out of five. Want to hear about another one, or is this the one?"
    )
