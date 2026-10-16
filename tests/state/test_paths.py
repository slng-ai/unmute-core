"""Paths into the State, written ``{{a.b}}`` or ``{{a__b}}``, and what they name."""

import pytest

from unmute_core.state import StateSchema
from unmute_core.state._paths import (
    ValuePath,
    ValuePathError,
    is_single_value,
    parse_path,
    path_annotation,
)

SCHEMA = StateSchema(
    [
        {"name": "contact", "type": "Contact | None"},
        {"name": "bookings", "type": "list[Booking]"},
        {"name": "status", "type": 'Literal["open", "closed"] | None'},
        {"name": "a__b", "type": "str"},
    ],
    [
        {
            "name": "Contact",
            "fields": [{"name": "name", "type": "str"}, {"name": "who", "type": "NameEmail"}],
        },
        {"name": "Booking", "fields": [{"name": "day", "type": "Date"}]},
    ],
)


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("contact", ValuePath("contact")),
        ("contact.name", ValuePath("contact", ("name",))),
        ("contact__name", ValuePath("contact", ("name",))),
        ("contact.who.email", ValuePath("contact", ("who", "email"))),
        ("a__b", ValuePath("a__b")),
    ],
)
def test_both_forms_read_the_same_path(name, path):
    assert parse_path(name, SCHEMA.model.model_fields) == path


def test_a_path_has_a_dotted_text_and_a_router_key():
    path = ValuePath("contact", ("who", "email"))
    assert (path.text, path.key) == ("contact.who.email", "contact__who__email")


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("contact.who__email", "mixes dots and double underscores"),
        ("a." + ".".join("f" * 9), "at most 8"),
        ("contact__", "not a value name"),
    ],
)
def test_bad_paths_are_refused(name, message):
    with pytest.raises(ValuePathError, match=message):
        parse_path(name)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("ghost", "names no value of the State: 'ghost'"),
        ("bookings.day", "bookings is a list, and a path cannot name a field inside one"),
        ("status.word", "status is Literal"),
        ("contact.phone", "shape 'Contact' declares no field 'phone'. It declares name, who"),
    ],
)
def test_a_path_that_names_nothing_says_what_is_there(name, message):
    with pytest.raises(ValuePathError, match=message):
        path_annotation(SCHEMA.model, parse_path(name))


def test_a_path_names_a_value_a_shape_or_a_list():
    assert is_single_value(path_annotation(SCHEMA.model, parse_path("contact.who.email")))
    assert not is_single_value(path_annotation(SCHEMA.model, parse_path("contact.who")))
    assert not is_single_value(path_annotation(SCHEMA.model, parse_path("bookings")))


def test_the_path_functions_are_the_value_path_methods() -> None:
    path = ValuePath.parse("contact__who", SCHEMA.model.model_fields)
    assert path == parse_path("contact__who", SCHEMA.model.model_fields)
    assert path.annotation(SCHEMA.model) == path_annotation(SCHEMA.model, path)
