from datetime import date, datetime
from enum import StrEnum
from typing import Literal
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel

from unmute_core.state import BirthDate, Spelled
from unmute_core.tools import local_tool, spoken


@pytest.mark.parametrize(
    ("language", "day", "hour"),
    [
        ("en", "Friday, October 16, 2026", "9:00 AM"),
        ("es", "viernes, 16 de octubre de 2026", "9:00"),
        ("fr", "vendredi 16 octobre 2026", "09:00"),
        ("de", "Freitag, 16. Oktober 2026", "09:00"),
        ("es-MX", "viernes, 16 de octubre de 2026", "9:00 a.m."),
        ("xx-YY", "Friday, October 16, 2026", "9:00 AM"),
    ],
)
def test_dates_and_times_are_said_in_the_call_language(language: str, day: str, hour: str) -> None:
    assert spoken("2026-10-16", "date", language) == day
    assert spoken("09:00:00", "time", language) == hour


def test_a_phone_is_said_in_its_country_groups() -> None:
    assert spoken("+34671520202", "phone", "en") == "+34 671 52 02 02"


@pytest.mark.parametrize(
    ("code", "said"),
    [
        ("KP482913", "K P, 4 8 2, 9 1 3"),
        ("kc-1002", "K C, 1 0, 0 2"),
        ("WA-1001", "W A, 1 0, 0 1"),
        ("AB12CDE", "A B, 1 2, C D E"),
        ("1234567", "1 2 3, 4 5, 6 7"),
    ],
)
def test_a_spelled_code_is_said_in_short_groups(code: str, said: str) -> None:
    assert spoken(code, "spelled", "en") == said


@pytest.mark.parametrize(
    ("language", "said"),
    [("en", "June 5, 1979"), ("en-GB", "5 June 1979"), ("es", "5 de junio de 1979")],
)
def test_a_birth_date_is_said_with_its_year_and_no_weekday(language: str, said: str) -> None:
    assert spoken("1979-06-05", "birth-date", language, today=TODAY) == said


@pytest.mark.parametrize(("value", "form"), [("haircut", None), ("soon", "date"), (3, "time")])
def test_anything_else_is_said_as_given(value: object, form: str | None) -> None:
    assert spoken(value, form, "en") == value


TODAY = date(2026, 10, 8)


@pytest.mark.parametrize(
    ("language", "said"),
    [
        (
            "en",
            [
                "today, October 8",
                "tomorrow, October 9",
                "the day after tomorrow, October 10",
                "Friday, October 16",
            ],
        ),
        (
            "es",
            [
                "hoy, 8 de octubre",
                "mañana, 9 de octubre",
                "pasado mañana, 10 de octubre",
                "viernes, 16 de octubre",
            ],
        ),
        (
            "de",
            [
                "heute, 8. Oktober",
                "morgen, 9. Oktober",
                "übermorgen, 10. Oktober",
                "Freitag, 16. Oktober",
            ],
        ),
    ],
)
def test_a_day_is_named_from_today_then_dated(language: str, said: list[str]) -> None:
    days = ["2026-10-08", "2026-10-09", "2026-10-10", "2026-10-16"]
    assert [spoken(day, "date", language, TODAY) for day in days] == said


def test_the_year_is_said_only_for_another_year() -> None:
    assert spoken("2027-01-08", "date", "en", TODAY) == "Friday, January 8, 2027"
    assert spoken("2027-01-01", "date", "en", date(2026, 12, 31)) == "tomorrow, January 1"


def test_a_language_without_day_words_says_the_weekday() -> None:
    assert spoken("2026-10-08", "date", "sv", TODAY) == "torsdag, 8 oktober"


def test_an_enum_value_is_said_as_words() -> None:
    assert spoken("hair-color", "enum", "en") == "hair color"


@pytest.mark.parametrize(
    ("value", "language", "zone", "said"),
    [
        ("2026-10-09T09:30:00", "en", None, "tomorrow, October 9, 9:30 AM"),
        (
            "2026-10-09T07:30:00+00:00",
            "en",
            ZoneInfo("Europe/Madrid"),
            "tomorrow, October 9, 9:30 AM",
        ),
    ],
)
def test_a_date_time_says_the_day_then_the_time(
    value: str, language: str, zone: ZoneInfo | None, said: str
) -> None:
    assert spoken(value, "date-time", language, TODAY, zone) == said


class Service(StrEnum):
    HAIRCUT = "haircut"


class Appointment(BaseModel):
    day: date
    at: datetime | None = None
    service: Service


class Booking(BaseModel):
    status: Literal["booked", "full"]
    appointment: Appointment | None = None
    ref: Spelled
    born: BirthDate | None = None


BOOK = local_tool(lambda: {}, output=Booking, name="book")


@pytest.mark.parametrize(
    ("path", "form"),
    [
        (("appointment", "at"), "date-time"),
        (("appointment", "service"), "enum"),
        (("ref",), "spelled"),
        (("born",), "birth-date"),
        (("nowhere",), None),
    ],
)
def test_a_value_is_spoken_by_its_place_in_a_schema(
    path: tuple[str, ...], form: str | None
) -> None:
    assert BOOK.result_form(path) == form
