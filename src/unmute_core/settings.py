"""Typed settings read from the environment, and errors that name what to set.

Each part of the library that needs a key or a URL has one settings class, like
``SlngSettings`` or ``LangfuseSettings``. Values passed to the constructor win over the
environment. Nothing reads the environment at import. ``check_settings`` loads several
classes at startup and names every missing variable in one error. ``env_flag`` is the one
way to read an on/off variable. ``env_value`` reads a variable whose name comes from config.
Nothing else in the library reads the environment. docs/core/reference/settings.mdx lists
every variable.
"""

import functools
from itertools import chain
from typing import Any, ClassVar

from pydantic import AliasChoices, Field, TypeAdapter, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic_core import ErrorDetails
from pydantic_settings import BaseSettings, SettingsConfigDict

_FLAG = TypeAdapter(bool)
_UNSET_ERRORS = frozenset({"missing", "too_short", "string_too_short"})
FLAG_WORDS = "1, true, yes, y, on, t or 0, false, no, n, off, f"
UNSET_ANY = "unset_any"
"""The error type a validator raises when none of several variables is set. Its ``names``
context names them, like ``SLNG_ROUTER_URL or SLNG_WORLD_PART``."""


class SettingsError(ValueError):
    """A setting is missing or holds a bad value. The message names each variable to fix."""


class EnvSettings(BaseSettings):
    """The base of every settings class in the library.

    Constructor values win over the environment. An empty variable counts as unset. A
    ``.env`` file is read only when you ask: ``SlngSettings(_env_file=".env")``.

    Attributes:
        config_error: The error a subclass raises when a value is missing or bad.
    """

    model_config = SettingsConfigDict(
        frozen=True,
        extra="ignore",
        env_file=None,
        env_ignore_empty=True,
        validate_by_name=True,
    )
    config_error: ClassVar[type[SettingsError]] = SettingsError

    def __init__(self, **values: Any) -> None:
        """Read the settings, or raise ``config_error`` naming every variable to fix."""
        try:
            super().__init__(**values)
        except ValidationError as error:
            raise self.config_error(_describe(type(self), error)) from error

    @classmethod
    def env_names(cls, field: str) -> tuple[str, ...]:
        """Return the environment variables that set ``field``, the preferred one first."""
        info: FieldInfo = cls.model_fields[field]
        alias = info.validation_alias
        if isinstance(alias, AliasChoices):
            return tuple(str(choice) for choice in alias.choices)
        if isinstance(alias, str):
            return (alias,)
        return (f"{cls.model_config.get('env_prefix', '')}{field}".upper(),)


def _field_of(settings: type[EnvSettings], location: str) -> str | None:
    """Return the field a validation error points at, by field name or env name."""
    for name in settings.model_fields:
        if location == name or location.upper() in settings.env_names(name):
            return name
    return None


def _line(settings: type[EnvSettings], detail: ErrorDetails) -> str:
    """Word one validation error as a line that names what to set."""
    location = str(detail["loc"][0]) if detail["loc"] else ""
    field = _field_of(settings, location)
    names = " or ".join(settings.env_names(field)) if field else location or settings.__name__
    if detail["type"] == UNSET_ANY:
        return f"Set {detail['ctx']['names']} in the environment."
    if detail["type"] in _UNSET_ERRORS:
        return f"Set {names} in the environment."
    return f"{names}: {detail['msg']}."


def _describe(settings: type[EnvSettings], error: ValidationError) -> str:
    """Turn a validation error into one line per variable, naming what to set."""
    return " ".join(_line(settings, detail) for detail in error.errors())


def check_settings[S: EnvSettings](*classes: type[S]) -> tuple[S, ...]:
    """Load every settings class, or raise one error that names every variable to fix.

    Call it once at startup, so a missing key fails the deploy and not the first call.

    Args:
        *classes: The settings classes the app needs, like ``SlngSettings``.

    Returns:
        One loaded settings object per class, in order.

    Raises:
        SettingsError: If any class is missing a value or holds a bad one.
    """
    loaded: list[S] = []
    problems: list[str] = []
    for settings in classes:
        try:
            loaded.append(settings())
        except SettingsError as error:
            problems.append(str(error))
    if problems:
        raise SettingsError(" ".join(problems))
    return tuple(loaded)


class _ExactNames(EnvSettings):
    """Reads a variable by its exact name: ``orders_token`` does not answer ``ORDERS_TOKEN``."""

    model_config = SettingsConfigDict(case_sensitive=True)


@functools.cache
def _variable(name: str, kind: type) -> type[EnvSettings]:
    """Return a settings class that reads only ``name``, built once per name and type."""
    return create_model(
        f"Variable_{name}",
        __base__=_ExactNames,
        value=(kind | None, Field(None, validation_alias=name)),
    )


def env_value[T](name: str, kind: type[T]) -> T | None:
    """Read one environment variable whose name comes from config.

    A webhook's ``token_env`` is one. Use a settings class when the name is fixed.

    Args:
        name: The variable, like ``ORDERS_TOKEN``.
        kind: The type to read it as, like ``str``.

    Returns:
        The value, or None when the variable is unset or empty.

    Raises:
        SettingsError: If the value is not a ``kind``.
    """
    return _variable(name, kind)().value  # ty: ignore[unresolved-attribute]


def env_flag(name: str, *, default: bool = False) -> bool:
    """Read an on/off environment variable.

    Args:
        name: The variable, like ``UNMUTE_LOCAL_RUN``.
        default: The value when the variable is unset or empty.

    Returns:
        True for 1, true, yes, y, on or t. False for 0, false, no, n, off or f. Any case.

    Raises:
        SettingsError: If the value is not one of those words.
    """
    value = (env_value(name, str) or "").strip()
    if not value:
        return default
    try:
        return _FLAG.validate_python(value)
    except ValidationError:
        raise SettingsError(f"{name} must be one of {FLAG_WORDS}, not {value!r}.") from None


def env_example(*classes: type[EnvSettings]) -> str:
    """Return ``.env`` example text for the given settings classes.

    Each variable gets its field description as a comment. A required variable is left
    empty, and one with a default shows it commented out.
    """
    lines = chain.from_iterable(
        _example_lines(settings, field, info)
        for settings in classes
        for field, info in settings.model_fields.items()
    )
    return "\n".join(lines) + "\n"


def _example_lines(settings: type[EnvSettings], field: str, info: FieldInfo) -> list[str]:
    """Return the comment and the variable line for one field."""
    name = settings.env_names(field)[0]
    if info.is_required():
        return [f"# {info.description}", f"{name}="]
    default = "" if info.default is None else info.default
    return [f"# {info.description}", f"# {name}={default}"]


__all__ = [
    "FLAG_WORDS",
    "UNSET_ANY",
    "EnvSettings",
    "SettingsError",
    "check_settings",
    "env_example",
    "env_flag",
    "env_value",
]
