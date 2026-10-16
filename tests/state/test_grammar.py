"""The type grammar takes the type texts it documents, and no others."""

import pytest

from unmute_core.state import TypeTextError, parse_type
from unmute_core.state._grammar import reserved_field_name, reserved_shape_name


@pytest.mark.parametrize(
    ("written", "canonical"),
    [
        ("EmailStr", "EmailStr"),
        ("string", "str"),
        ("integer | None", "int | None"),
        ("NameEmail | None", "NameEmail | None"),
        ("list[Appointment]", "list[Appointment]"),
        ("list[Literal['x', \"y\"]]", 'list[Literal["x", "y"]]'),
        ("None | Phone", "Phone | None"),
    ],
)
def test_the_grammar_writes_each_type_one_way(written, canonical):
    assert parse_type(written).canonical == canonical


@pytest.mark.parametrize(
    ("written", "message"),
    [
        ("list[str] | None", "a list that may be absent"),
        ("list[list[str]]", "a list of lists"),
        ("str | int", "a union of str and int"),
        ("None", '"None" and nothing else'),
        ("str | None | None", '"None" written twice'),
        ("dict", "declare the fields as a shape"),
        ("Optional[str]", 'write "T | None"'),
        ("Appointment", "no shape named 'Appointment' is declared"),
        ('Literal["a", "a"]', "twice"),
        ("Literal[1, 2]", "not a quoted entry"),
        ("Literal[]", "Literal with no entries"),
        ("Phone[x]", "takes no brackets"),
        ("list[str", "the type ends too early"),
        ("str]", "unexpected ']' after the type"),
        ('"open"', "where a type belongs"),
    ],
)
def test_refusals_say_what_to_write(written, message):
    with pytest.raises(TypeTextError, match=message):
        parse_type(written, shapes=set())


def test_a_literal_takes_many_long_choices_but_no_empty_one():
    many = ", ".join(f'"c{index}"' for index in range(65))
    assert len(parse_type(f"Literal[{many}]").choices) == 65
    assert parse_type(f'Literal["{"x" * 101}"]').choices == ("x" * 101,)
    with pytest.raises(TypeTextError, match="must not be empty"):
        parse_type('Literal[""]')


def test_a_type_nested_past_the_stack_is_refused_not_a_crash():
    deep = "list[" * 1000 + "str" + "]" * 1000
    with pytest.raises(TypeTextError, match="nests more than 16 brackets deep"):
        parse_type(deep[:8000])
    assert parse_type('list[Literal["a"]]').canonical == 'list[Literal["a"]]'


def test_any_other_name_is_a_shape_until_the_shapes_are_known():
    later = parse_type("Contact | None")
    assert (later.base, later.shape, later.optional) == ("shape", "Contact", True)
    assert parse_type("list[Contact]").shape_names() == {"Contact"}


def test_names_pydantic_and_the_grammar_keep_are_reserved():
    assert reserved_field_name("model_dump")
    assert not reserved_field_name("model_year")
    assert "primitive" in reserved_shape_name("str")
    assert "validated shape" in reserved_shape_name("Phone")
    assert "built-in shapes" in reserved_shape_name("NameEmail")
    assert "grammar" in reserved_shape_name("Literal")
    assert not reserved_shape_name("Contact")
