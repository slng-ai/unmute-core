"""What every Pydantic value class in the library shares.

A value class is a ``pydantic.dataclasses.dataclass`` with ``config=VALUE``, wrapped in
``original_errors``:

```python
@original_errors
@dataclass(frozen=True, config=VALUE)
class Transfer:
    name: Annotated[str, StringConstraints(min_length=1)]
```

Callers still build it by position or by keyword, and ``dataclasses.replace`` still works.
``original_errors`` keeps the errors callers saw before the class became Pydantic.
"""

import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any, LiteralString, cast

from pydantic import ConfigDict, ValidationError
from pydantic_core import ErrorDetails, InitErrorDetails, PydanticCustomError

VALUE = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
"""The config of every value class. An unknown argument fails. A framework type in a field is
checked with ``isinstance``. A Protocol field needs ``SkipValidation[...]``."""

_ARGUMENT_ERRORS = frozenset(
    {
        "unexpected_keyword_argument",
        "unexpected_positional_argument",
        "missing",
        "missing_argument",
        "multiple_argument_values",
    }
)


def _argument_text(detail: ErrorDetails) -> str:
    """Word one argument mistake as the stdlib dataclass ``__init__`` does."""
    where = detail["loc"][0] if detail["loc"] else "?"
    if detail["type"] in {"missing", "missing_argument"}:
        return f"missing the required argument {where!r}"
    if detail["type"] == "unexpected_positional_argument":
        return "takes fewer positional arguments than were given"
    if detail["type"] == "multiple_argument_values":
        return f"got multiple values for the argument {where!r}"
    return f"got an unexpected keyword argument {where!r}"


def _original(cls: type, error: ValidationError) -> Exception:
    """Return the error the caller saw before the class was Pydantic.

    An argument mistake becomes a ``TypeError``. An error one of the library's own checks
    raised comes back as itself, with its type and text. Anything else, like a broken
    ``Field(ge=1)``, stays the ``ValidationError``.
    """
    if wrong := [d for d in error.errors() if d["type"] in _ARGUMENT_ERRORS]:
        return TypeError(f"{cls.__name__}() {_argument_text(wrong[0])}")
    return own_error(error)


def own_error(error: ValidationError) -> Exception:
    """Return the first error one of the library's own validators raised, else ``error``.

    Use it where a ``TypeAdapter`` or a model validates outside a value class, so the
    caller still gets the library's error and its text.
    """
    return next(
        (
            own
            for detail in error.errors()
            if isinstance(own := detail.get("ctx", {}).get("error"), Exception)
        ),
        error,
    )


def _line(detail: ErrorDetails, title: str, location: tuple[int | str, ...]) -> InitErrorDetails:
    """Return one error line at a new location, keeping its type, input and context.

    A type Pydantic does not know, like a custom one, goes through ``PydanticCustomError``.
    """
    line = InitErrorDetails(type=detail["type"], loc=location, input=detail["input"])
    if "ctx" in detail:
        line["ctx"] = detail["ctx"]
    try:
        ValidationError.from_exception_data(title, [line])
    except (KeyError, TypeError, ValueError):
        line["type"] = PydanticCustomError(
            cast("LiteralString", detail["type"]), cast("LiteralString", detail["msg"])
        )
    return line


def retitled(
    error: ValidationError,
    title: str,
    parameters: Sequence[str] = (),
    prefix: tuple[int | str, ...] = (),
) -> ValidationError:
    """Return ``error`` under the public name of what failed, with names for its locations.

    Pydantic's own sentence stays. The first line names ``title``, and each location names
    the field.

    Args:
        error: The error Pydantic raised.
        title: The public class or function, like ``CallLimit``.
        parameters: The positional parameters of the call, so a location that is an index
            becomes the parameter's name.
        prefix: Put in front of each location, for a value checked on its own that has none.
    """
    lines = [
        _line(
            detail,
            title,
            prefix
            + tuple(
                parameters[part] if isinstance(part, int) and part < len(parameters) else part
                for part in detail["loc"][:1]
            )
            + detail["loc"][1:],
        )
        for detail in error.errors()
    ]
    return ValidationError.from_exception_data(title, lines)


def named_errors[**P, R](title: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Make a ``validate_call`` function name its public self in a constraint error.

    Put it above ``@validate_call``. The error stays a ``ValidationError``, with Pydantic's
    sentence. Its first line reads ``1 validation error for <title>`` and each location is
    the parameter's name, not its position.

    Args:
        title: The public name, like ``CallLimit`` or ``turn_window``.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        """Wrap one validated function."""
        parameters = tuple(inspect.signature(function).parameters)

        @functools.wraps(function)
        def naming_errors(*args: P.args, **kwargs: P.kwargs) -> R:
            """Call the function and rename the error it raises."""
            try:
                return function(*args, **kwargs)
            except ValidationError as error:
                raise retitled(error, title, parameters) from error

        return naming_errors

    return decorate


def original_errors[C: type](cls: C) -> C:
    """Make a Pydantic dataclass raise the errors it raised as a stdlib dataclass.

    A ``ParameterSchemaError`` from a field validator reaches the caller as itself, not
    wrapped in a ``ValidationError``. A missing or unknown argument is a ``TypeError``. A
    constraint error names the field, also for an argument given by position.

    Args:
        cls: A class built by ``pydantic.dataclasses.dataclass``.

    Returns:
        The same class, with its ``__init__`` wrapped.
    """
    init = cls.__init__
    positional = tuple(
        name
        for name, parameter in inspect.signature(cls).parameters.items()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    )

    @functools.wraps(init)
    def keeping_errors(self: object, *args: Any, **kwargs: Any) -> None:
        """Run the Pydantic ``__init__`` and raise the error the caller expects."""
        try:
            init(self, *args, **kwargs)
        except ValidationError as error:
            original = _original(cls, error)
            if original is error:
                raise retitled(error, cls.__name__, positional) from None
            raise original from error

    cls.__init__ = keeping_errors  # ty: ignore[invalid-assignment]
    return cls


__all__ = ["VALUE", "named_errors", "original_errors", "own_error", "retitled"]
