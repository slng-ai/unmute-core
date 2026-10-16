import dataclasses
from typing import Annotated, Any, cast

import pytest
from pydantic import AfterValidator, Field, TypeAdapter, ValidationError, validate_call
from pydantic.dataclasses import dataclass
from pydantic_core import PydanticCustomError

from unmute_core._pydantic import VALUE, named_errors, original_errors, own_error, retitled


class SpecError(ValueError):
    pass


def _named(name: str) -> str:
    if not name:
        raise SpecError("a spec needs a name: give one")
    return name


@original_errors
@dataclass(frozen=True, config=VALUE)
class Spec:
    name: Annotated[str, AfterValidator(_named)]
    size: Annotated[int, Field(ge=1)] = 1

    def __post_init__(self) -> None:
        if self.name == "reserved":
            raise SpecError("'reserved' is taken: pick another name")


def test_a_check_of_its_own_raises_its_own_error() -> None:
    with pytest.raises(SpecError, match=r"^a spec needs a name: give one$"):
        Spec("")


def test_a_post_init_check_raises_its_own_error() -> None:
    with pytest.raises(SpecError, match=r"^'reserved' is taken"):
        Spec("reserved")


def test_a_built_in_constraint_raises_a_validation_error_naming_the_field() -> None:
    with pytest.raises(ValidationError, match=r"for Spec\nsize\n.*greater than or equal to 1"):
        Spec("a", 0)


@pytest.mark.parametrize(
    ("arguments", "text"),
    [
        ({}, "missing the required argument 'name'"),
        ({"name": "a", "colour": "red"}, "unexpected keyword argument 'colour'"),
    ],
)
def test_an_argument_mistake_is_a_type_error(arguments: dict[str, object], text: str) -> None:
    with pytest.raises(TypeError, match=text):
        cast(Any, Spec)(**arguments)


def test_positional_arguments_and_replace_still_work() -> None:
    spec = Spec("a", 2)
    assert dataclasses.replace(spec, size=3) == Spec(name="a", size=3)
    with pytest.raises(SpecError):
        dataclasses.replace(spec, name="")


def _lines(error: BaseException) -> list[str]:
    return str(error).splitlines()


def test_retitled_prefixes_a_value_checked_on_its_own() -> None:
    with pytest.raises(ValidationError) as caught:
        TypeAdapter(Annotated[int, Field(ge=1)]).validate_python(0)
    error = retitled(caught.value, "tool", prefix=("timezone",))
    assert _lines(error)[:2] == ["1 validation error for tool", "timezone"]


def test_retitled_keeps_the_error_a_library_validator_raised() -> None:
    with pytest.raises(ValidationError) as caught:
        TypeAdapter(Annotated[str, AfterValidator(_named)]).validate_python("")
    error = retitled(caught.value, "Spec", prefix=("name",))
    assert own_error(error).__class__ is SpecError


def _custom(value: int) -> int:
    raise PydanticCustomError("my_error", "no {why}", {"why": "x"})


def test_retitled_keeps_a_custom_error_type() -> None:
    with pytest.raises(ValidationError) as caught:
        TypeAdapter(Annotated[int, AfterValidator(_custom)]).validate_python(1)
    error = retitled(caught.value, "Spec", prefix=("name",))
    assert error.errors()[0]["type"] == "my_error"
    assert "no x" in str(error)


def test_named_errors_titles_a_validate_call_function_and_names_a_positional_parameter() -> None:
    @named_errors("shout")
    @validate_call
    def shout(text: str, times: Annotated[int, Field(ge=1)] = 1) -> str:
        return text * times

    assert shout("a", 2) == "aa"
    with pytest.raises(ValidationError) as caught:
        shout("a", 0)
    assert _lines(caught.value)[:2] == ["1 validation error for shout", "times"]
