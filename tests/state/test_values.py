"""Value checks of the library types, and of the Pydantic types a State uses.

A type word in the type grammar means a Pydantic type, so the expectations follow that type.
"""

from datetime import date, time
from typing import Any, Literal

import pytest
from pydantic import EmailStr, NameEmail, TypeAdapter, ValidationError, create_model

from tests.state.conftest import BUILT_STATE
from unmute_core.state import CallState, Id, Phone, StateRefused, StateSchema
from unmute_core.state._types import Text


def _state(**fields: Any) -> CallState:
    """Return a call state whose values have the given types, each starting as None."""
    definitions: dict[str, Any] = {name: (kind | None, None) for name, kind in fields.items()}
    return CallState(create_model("S", **definitions))


def _check(annotation: Any, value: object) -> Any:
    """Validate one value against one type, as a save does."""
    return TypeAdapter(annotation).validate_python(value)


def _digits(text: str, zero: int) -> str:
    """Return ``text`` with its ASCII digits written in another script's digits."""
    return "".join(chr(zero + int(c)) if c.isdigit() else c for c in text)


FULLWIDTH, ARABIC_INDIC = 0xFF10, 0x0660


@pytest.mark.parametrize(
    ("said", "stored"),
    [
        ("+34 612 345 678", "+34612345678"),
        ("+34-612-345-678", "+34612345678"),
        ("(+34) 612.345.678", "+34612345678"),
        ("0034 612 345 678", "+34612345678"),
        ("+1 (415) 555-0100", "+14155550100"),
        ("+44 7911 123456", "+447911123456"),
        ("+86 138 0013 8000", "+8613800138000"),
        ("+81 3-1234-5678", "+81312345678"),
        (_digits("+34 612 345 678", FULLWIDTH), "+34612345678"),
        (_digits("+34 612 345 678", ARABIC_INDIC), "+34612345678"),
    ],
)
def test_a_phone_number_is_tidied_and_saved_as_e164(said: str, stored: str) -> None:
    assert _check(Phone, said) == stored


def test_a_phone_number_without_a_plus_is_told_what_it_lacks() -> None:
    with pytest.raises(ValidationError, match="It has no plus"):
        _check(Phone, "612 345 678")
    with pytest.raises(ValidationError, match="expected a phone number") as refused:
        _check(Phone, "+34 61")
    assert "no plus" not in str(refused.value)


@pytest.mark.parametrize("said", ["+1000000", "+34 612 345", "", "anonymous", "+999 123456789"])
def test_a_phone_number_that_cannot_exist_is_refused(said: str) -> None:
    with pytest.raises(ValidationError, match="expected a phone number with its country code"):
        _check(Phone, said)


@pytest.mark.parametrize(
    ("said", "reason"),
    [
        ("+34 671 520 2020", "too long for its country, so a digit was misheard or said twice"),
        ("+34 671", "too short for its country: ask the caller for the rest"),
        ("+999 123456789", "No country has that code"),
    ],
)
def test_a_wrong_phone_number_says_what_to_ask_next(said: str, reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        _check(Phone, said)


def test_phone_tells_tool_schemas_it_is_a_phone_number() -> None:
    assert TypeAdapter(Phone).json_schema() == {"type": "string", "format": "phone"}


@pytest.mark.parametrize(
    ("written", "value"),
    [
        (date, _digits("2026-03-19", FULLWIDTH)),
        (time, "0" + _digits("9", FULLWIDTH) + ":3" + _digits("0", FULLWIDTH)),
        (int, _digits("12", FULLWIDTH)),
    ],
)
def test_dates_times_and_numbers_take_ascii_digits_only(written: type, value: str) -> None:
    with pytest.raises(StateRefused, match=r"^v: "):
        _state(v=written).save({"v": value})


@pytest.mark.parametrize("value", ["-abc", "a" * 65, "a b"])
def test_an_id_checks_the_whole_text(value: str) -> None:
    with pytest.raises(ValidationError, match="expected an identifier: letters, digits"):
        _check(Id, value)


def test_an_id_takes_letters_digits_and_four_marks() -> None:
    assert _check(Id, "A-1042:x_y.z") == "A-1042:x_y.z"
    assert TypeAdapter(Id).json_schema() == {"type": "string", "format": "id"}


def test_a_date_inside_a_shape_checks_the_whole_text() -> None:
    state = CallState(BUILT_STATE)
    appointment = {"scheduled_time": "09:30", "appointment_type": "haircut"}
    with pytest.raises(StateRefused, match=r"^last_appointment\.scheduled_date: "):
        state.save({"last_appointment": {**appointment, "scheduled_date": "2026-03-19\n"}})
    with pytest.raises(StateRefused, match=r"day value is outside expected range"):
        state.save({"last_appointment": {**appointment, "scheduled_date": "2026-02-30"}})


def test_oversized_numbers_are_refused_as_values_not_raised() -> None:
    state = _state(n=int, f=float)
    with pytest.raises(StateRefused, match=r"^n: "):
        state.save({"n": "1" * 5000})
    with pytest.raises(StateRefused, match=r"^f: "):
        state.save({"f": 10**400})
    state.save({"n": "1" * 40})
    assert state.render("n") == "1" * 40


def test_email_text_is_normalized() -> None:
    state = _state(email=EmailStr)
    state.save({"email": " Fred@Example.COM "})
    assert state.plain("email") == "Fred@example.com"
    with pytest.raises(StateRefused, match=r"^email: .*email address"):
        state.save({"email": "fred@"})


def test_name_email_is_one_text_as_in_pydantic() -> None:
    state = _state(who=NameEmail)
    state.save({"who": "Fred Bloggs <Fred@Example.com>"})
    assert state.plain("who") == "Fred Bloggs <Fred@example.com>"
    assert state.values.who.name == "Fred Bloggs"
    with pytest.raises(StateRefused, match=r"^who: .*email address"):
        state.save({"who": "not an address"})


def test_a_shape_is_checked_field_by_field() -> None:
    customer = create_model("Customer", contact=EmailStr, status=Literal["new", "known"])
    state = _state(customer=customer)
    state.save({"customer": {"contact": "fred@Example.com", "status": "new"}})
    assert state.plain("customer") == {"contact": "fred@example.com", "status": "new"}
    assert state.render("customer.status") == "new"
    with pytest.raises(StateRefused, match=r"^customer\.status: Input should be 'new' or"):
        state.save({"customer": {"contact": "fred@example.com", "status": "vip"}})
    with pytest.raises(StateRefused, match=r"^customer\.contact: .*email address"):
        state.save({"customer": {"contact": "not an address", "status": "new"}})


def test_a_list_checks_every_entry() -> None:
    state = CallState(create_model("S", days=(list[date], [])))
    with pytest.raises(StateRefused, match=r"^days\.1: "):
        state.save({"days": ["2026-03-19", "tomorrow"]})
    state.save({"days": ["2026-03-19"]})
    assert state.render("days") == '["2026-03-19"]'


def test_a_shape_field_may_start_with_model_when_pydantic_does_not_use_it() -> None:
    state = _state(car=create_model("Car", model_year=int))
    state.save({"car": {"model_year": 2020}})
    assert state.plain("car") == {"model_year": 2020}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_float_value_refuses_nan_and_infinity(value: float) -> None:
    with pytest.raises(StateRefused, match=r"^v: "):
        CallState(StateSchema([{"name": "v", "type": "float | None"}]).model).save({"v": value})


@pytest.mark.parametrize("annotation", [Text, Id])
def test_a_non_finite_number_is_not_stored_as_text(annotation: Any) -> None:
    with pytest.raises(ValidationError, match="expected a finite number"):
        _check(annotation, float("nan"))


def test_text_ids_and_choices_are_trimmed() -> None:
    assert _check(Text, "  hello \n") == "hello"
    assert _check(Id, "abc\n") == "abc"
    schema = StateSchema(
        [
            {"name": "note", "type": "str"},
            {"name": "tier", "type": 'Literal["gold", "silver"]'},
            {"name": "flag", "type": "bool"},
        ]
    )
    state = CallState(schema.model)
    state.save({"note": " padded ", "tier": " gold\n", "flag": "true"})
    assert state.plain("note") == "padded"
    assert state.plain("tier") == "gold"
    assert state.plain("flag") is True


@pytest.mark.parametrize("sent", ["small", " small", "small \n"])
def test_a_padded_choice_is_saved_as_declared(sent: str) -> None:
    schema = StateSchema([{"name": "size", "type": 'Literal[" small", "large "]'}])
    state = CallState(schema.model)
    state.save({"size": sent})
    assert state.plain("size") == " small"
