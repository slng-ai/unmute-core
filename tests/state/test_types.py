"""The value types: phone repair, dates, clock times, identifiers and names with addresses.

In the phone cases libphonenumber stays the judge, so a number that cannot exist is still
refused.
"""

import datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from unmute_core.state import Date, Id, NameEmail, Phone, Time, region_of, use_phone_region

PHONE = TypeAdapter(Phone | None)
FULLWIDTH = 0xFF10


def _fullwidth(text: str) -> str:
    """``text`` with its ASCII digits written as fullwidth digits."""
    return "".join(chr(FULLWIDTH + int(c)) if c.isdigit() else c for c in text)


@pytest.mark.parametrize(
    ("said", "stored"),
    [
        ("+34 612 345 678", "+34612345678"),
        ("(+34) 612.345.678", "+34612345678"),
        ("0034 612 345 678", "+34612345678"),
        ("+1 (415) 555-2671", "+14155552671"),
    ],
)
def test_a_phone_number_drops_its_separators_and_a_leading_00(said, stored):
    assert PHONE.validate_python(said) == stored


@pytest.mark.parametrize("sent", [34600111222, 34600111222.0, "34600111222"])
def test_a_number_that_lost_its_plus_is_repaired_when_only_one_number_fits(sent):
    assert PHONE.validate_python(sent) == "+34600111222"


@pytest.mark.parametrize("sent", [12345678, 12345678.0])
def test_a_json_number_no_valid_number_fits_is_told_to_quote_it(sent):
    with pytest.raises(ValidationError, match="send it as text in quotes"):
        PHONE.validate_python(sent)


def test_true_is_not_a_phone_number():
    with pytest.raises(ValidationError):
        PHONE.validate_python(True)


def test_a_national_number_is_read_in_the_calls_own_country():
    with pytest.raises(ValidationError, match="It has no plus"):
        PHONE.validate_python("600 111 222")
    with use_phone_region(lambda: "ES"):
        assert PHONE.validate_python("600 111 222") == "+34600111222"
        assert PHONE.validate_python(600111222) == "+34600111222"
    with use_phone_region(lambda: "US"):
        assert PHONE.validate_python(4155552671) == "+14155552671"
    with pytest.raises(ValidationError, match="It has no plus"):
        PHONE.validate_python("600 111 222")


def test_two_numbers_that_fit_are_refused_not_guessed():
    with use_phone_region(lambda: "ES"):
        with pytest.raises(ValidationError, match="It has no plus"):
            PHONE.validate_python("988389990")
        with pytest.raises(ValidationError, match="send it as text in quotes"):
            PHONE.validate_python(988389990)


def test_a_number_that_cannot_exist_is_still_refused():
    with pytest.raises(ValidationError, match="too short"):
        PHONE.validate_python("+34 61")


@pytest.mark.parametrize(
    "said",
    [
        "+34 612 345 678 ext 5",
        "+34 612 345 678 ext. 5",
        "+34 612 345 678 (extension 5)",
        "+44 20 7946 0958 x 123",
        "+34612345678;ext=5",
        "+34612345678#5",
        "+34612345678,5",
        "612 345 678 ext 5",
        "612 345 678 x5",
        "612-345-678 extension 12",
        "612345678#5",
    ],
)
def test_a_number_with_an_extension_is_refused_not_cut(said):
    with pytest.raises(ValidationError, match="It has an extension"):
        PHONE.validate_python(said)


@pytest.mark.parametrize("said", ["+1-800-FLOWERS", "+34 612 ABC 678", "612 ABC 678"])
def test_a_number_with_letters_is_refused_not_spelled_out(said):
    with pytest.raises(ValidationError, match="It has letters"):
        PHONE.validate_python(said)


@pytest.mark.parametrize(
    "said",
    [
        "Phone: +34 612 345 678",
        "mobile +34612345678",
        "tel:+34612345678",
        "(mobile) +34 612 345 678",
        "#+34612345678",
        "my number is 612345678",
    ],
)
def test_words_or_symbols_before_a_number_are_refused(said):
    with use_phone_region(lambda: "ES"), pytest.raises(ValidationError, match="before the number"):
        PHONE.validate_python(said)


def test_a_plus_or_digits_of_another_script_are_still_read():
    assert PHONE.validate_python("\uff0b34 612 345 678") == "+34612345678"
    assert PHONE.validate_python("+" + _fullwidth("34612345678")) == "+34612345678"


def test_the_country_of_a_full_number():
    assert region_of("+34600111222") == "ES"
    assert region_of("+14155552671") == "US"
    assert region_of("") is None
    assert region_of("not a number") is None


def test_a_date_is_a_real_day_written_year_month_day():
    dates = TypeAdapter(Date)
    assert dates.validate_python("2026-03-19") == datetime.date(2026, 3, 19)
    assert dates.validate_python(datetime.date(2026, 3, 19)) == datetime.date(2026, 3, 19)
    for bad in ("19/03/2026", "tomorrow", _fullwidth("2026-03-19"), 1773878400):
        with pytest.raises(ValidationError, match="expected a day written year-month-day"):
            dates.validate_python(bad)
    with pytest.raises(ValidationError):
        dates.validate_python("2026-02-30")
    assert dates.json_schema() == {"type": "string", "format": "date"}


def test_a_time_is_hours_and_minutes_on_the_24_hour_clock():
    times = TypeAdapter(Time)
    assert times.validate_python(" 09:30 ") == "09:30"
    for bad in ("9:30", "24:00", "09:30:00", _fullwidth("09:30")):
        with pytest.raises(ValidationError, match="expected a time of day"):
            times.validate_python(bad)
    assert times.json_schema() == {"type": "string", "format": "time"}


def test_an_id_takes_a_json_number_as_its_digits():
    ids = TypeAdapter(Id)
    assert ids.validate_python(1234) == "1234"
    assert ids.validate_python("A-12.b:3") == "A-12.b:3"
    with pytest.raises(ValidationError):
        ids.validate_python(True)
    with pytest.raises(ValidationError):
        ids.validate_python("-starts-with-a-dash")


def test_a_name_email_reads_one_string_into_its_two_parts():
    assert NameEmail.model_validate("Fred Bloggs <fred@Example.com>") == NameEmail(
        name="Fred Bloggs", email="fred@example.com"
    )
    assert NameEmail.model_validate("ana@example.com").name == "ana"
    with pytest.raises(ValidationError, match="expected an email address"):
        NameEmail.model_validate("not an address")
    schema = NameEmail.model_json_schema()
    assert schema["description"].startswith("A person and their email address.")
    assert set(schema["properties"]) == {"name", "email"}
