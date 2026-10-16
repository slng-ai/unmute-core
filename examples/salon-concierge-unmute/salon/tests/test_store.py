"""The diary's rules. Ported from ``_demo()`` in Unmute's examples/salon-concierge/tools/salon.py."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from datetime import time as TimeOfDay
from zoneinfo import ZoneInfo

import pytest
from unmute_core.state import call_info
from unmute_core.tools import ToolError

from salon.agent import AGENT_ID, new_call, tools
from salon.store import (
    SERVICES,
    TIMES,
    TIMEZONE,
    Salon,
    normalize_phone,
    offer,
    salon_today,
    slot_parts,
)

CALLER = "+34671520201"
RIVAL = "+34671520202"
NINE = TimeOfDay(9, 0)


@pytest.fixture
def salon() -> Salon:
    return Salon()


def _day(offset: int) -> date:
    return salon_today() + timedelta(days=offset)


def test_a_number_that_cannot_be_right_never_reaches_the_diary(salon: Salon) -> None:
    lookup = tools(salon)
    with pytest.raises(ToolError, match="too long for its country"):
        asyncio.run(lookup.call("find_or_create_customer", {"phone": "+34 671 520 2020"}))
    with pytest.raises(ToolError, match="too short for its country"):
        asyncio.run(lookup.call("find_or_create_customer", {"phone": "+34 671"}))
    assert salon.customers == set()
    found = asyncio.run(lookup.call("find_or_create_customer", {"phone": "+34 671 520 201"}))
    assert found == {
        "status": "created",
        "customer_phone": CALLER,
        "summary": "A new customer was verified and created.",
    }


def test_phone_numbers_normalize_to_one_key() -> None:
    assert normalize_phone("+34 671-520-201") == "34671520201"
    assert normalize_phone(CALLER) == "34671520201"


def test_the_offer_is_said_only_for_the_common_case() -> None:
    friday = "2026-09-18"
    slots = [{"service": "haircut", "time": time} for time in ("09:00", "15:00")]
    one = [{"service": "hair-color", "date": friday, "time": "11:30"}]
    assert offer("book", friday, slots, []) == (
        "For a haircut on Friday, I have 9:00 AM or 3:00 PM. Which works best for you?"
    )
    assert offer("move", friday, slots[:1], one) == (
        "I can move it to Friday at 9:00 AM. Which works best for you?"
    )
    assert offer("book", friday, [], [], "hair-color") == (
        "I'm sorry, there's nothing free for a hair color on Friday. Would another day work?"
    )
    assert offer("book", salon_today().isoformat(), [], [], "haircut").endswith(
        "for a haircut today. Would another day work?"
    )
    other = [*slots, {"service": "blowout", "time": "09:00"}]
    for intent, rows, bookings in [
        ("cancel", [], one),
        ("cancel", slots, []),
        ("book", [], []),
        ("book", other, []),
        ("book", slots, one),
        ("move", slots, one * 2),
    ]:
        assert offer(intent, friday, rows, bookings) == ""


def test_a_customer_is_created_once(salon: Salon) -> None:
    created = salon.find_or_create_customer(CALLER)
    repeated = salon.find_or_create_customer(CALLER)
    assert (created["status"], repeated["status"]) == ("created", "existing")
    assert repeated["customer_phone"] == created["customer_phone"] == CALLER


def test_concurrent_lookups_create_one_customer(salon: Salon) -> None:
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: salon.find_or_create_customer(RIVAL), range(16)))
    assert [r["status"] for r in results].count("created") == 1


def test_find_slots_reads_the_diary(salon: Salon) -> None:
    salon.find_or_create_customer(CALLER)
    assert salon.find_slots(CALLER) == {
        "bookings": [],
        "slots": [],
        "status": "need_date",
        "offer": "Sure, which day would you like to come in?",
        "offered": True,
    }
    asked = salon.find_slots(CALLER, service="hair-color")
    assert asked["offer"] == "Sure, which day would you like your hair color?"
    assert salon.find_slots(CALLER, _day(-1), "any")["status"] == "invalid_date"
    every = salon.find_slots(CALLER, _day(1), "any")
    assert len(every["slots"]) == len(SERVICES) * len(TIMES)
    haircuts = salon.find_slots(CALLER, _day(1), "haircut")
    assert {row["service"] for row in haircuts["slots"]} == {"haircut"}


def test_find_slots_takes_only_a_real_day(salon: Salon) -> None:
    state = asyncio.run(new_call(call_info("inbound", remote_number=CALLER, agent_id=AGENT_ID)))
    state.save({"customer_phone": CALLER}, step="verify_customer")
    with pytest.raises(ToolError, match="date"):
        tools(salon).check("find_slots", {"date": "next week"}, state)
    assert tools(salon).check("find_slots", {}, state) == {}


def test_a_past_slot_today_is_never_offered(salon: Salon) -> None:
    now = datetime.now(ZoneInfo(TIMEZONE)).strftime("%H:%M")
    rows = salon.find_slots(CALLER, _day(0), "haircut")["slots"]
    assert all(row["time"] > now for row in rows)
    assert slot_parts(f"{_day(0).isoformat()}|haircut|00:00") is None


def test_booking_moving_and_cancelling(salon: Salon) -> None:
    salon.find_or_create_customer(CALLER)
    assert salon.book_appointment(RIVAL, "haircut", _day(1), NINE, False)["status"] == "not_found"
    booked = salon.book_appointment(CALLER, "haircut", _day(1), NINE, False)
    assert booked["status"] == "booked"
    assert booked["appointment"]["action"] == "book"
    assert date.fromisoformat(booked["appointment"]["date"]) == _day(1)
    assert salon.find_slots(CALLER)["status"] == "ok"

    salon.find_or_create_customer(RIVAL)
    assert salon.book_appointment(RIVAL, "haircut", _day(1), NINE, False)["status"] == "slot_taken"

    refused = salon.book_appointment(CALLER, "haircut", _day(2), NINE, False)
    assert refused["status"] == "has_booking"
    assert [row["booking_id"] for row in refused["existing"]] == [booked["booking_id"]]
    extra = salon.book_appointment(CALLER, "blowout", _day(2), NINE, True)
    assert extra["status"] == "booked"
    wrong = salon.cancel_appointment(CALLER, extra["booking_id"], "haircut", _day(2), NINE)
    assert wrong["status"] == "not_found"
    gone = salon.cancel_appointment(CALLER, extra["booking_id"], "blowout", _day(2), NINE)
    assert gone["status"] == "cancelled"

    moved = salon.move_appointment(CALLER, booked["booking_id"], _day(2), NINE)
    assert moved["status"] == "moved"
    assert salon.find_slots(CALLER)["bookings"][0]["date"] == _day(2).isoformat()

    cancel = (CALLER, booked["booking_id"], "haircut", _day(2), NINE)
    assert salon.cancel_appointment(*cancel)["status"] == "cancelled"
    again = salon.cancel_appointment(*cancel)
    assert again["status"] == "not_found" and "already cancelled" in again["summary"]


def test_a_complaint_needs_a_known_customer_and_a_summary(salon: Salon) -> None:
    salon.find_or_create_customer(CALLER)
    assert salon.record_complaint(CALLER, "   ", "")["status"] == "invalid"
    assert salon.record_complaint(RIVAL, "Uneven cut.", "")["status"] == "customer_not_found"
    complaint = salon.record_complaint(CALLER, "My cut  was   uneven.", "A redo")
    assert complaint["complaint"] == {
        "complaint_id": complaint["complaint_id"],
        "summary": "My cut was uneven.",
        "requested_resolution": "A redo",
    }
