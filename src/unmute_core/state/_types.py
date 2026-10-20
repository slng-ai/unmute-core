"""Value types the library ships for a State: ``Phone``, ``Date``, ``Time``, ``Id``, ``NameEmail``.

Each one declares a JSON Schema ``format``, so a tool schema or another reader can name it.
Their wording follows the type grammar in ``unmute_core.state``, which Unmute uses too.
"""

import contextlib
import contextvars
import datetime
import functools
import json
import math
import re
from collections.abc import Callable, Generator
from types import MappingProxyType
from typing import Annotated, Any

import phonenumbers
from email_validator import EmailNotValidError, validate_email
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    EmailStr,
    Field,
    WithJsonSchema,
    model_validator,
)
from pydantic_core import PydanticCustomError

_PHONE_SEPARATORS = re.compile(r"[\s().\-/]")
_DIGITS_ONLY = re.compile(r"\d{6,15}", re.ASCII)
_LETTER_AFTER_DIGIT = re.compile(r"[+\d].*[^\W\d_]")
_NUMBER_START = re.compile(r"[+\uff0b\d]")
"""What a tidied number starts with: a plus, ASCII or fullwidth, or a digit of any script."""
_EXTENSION_MARK = re.compile(r"(?:;ext=|extension|ext|x|#)\d+$", re.ASCII | re.IGNORECASE)
"""An extension at the end of a tidied number that libphonenumber could not read."""
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
_CLOCK_TIME = re.compile(r"([01]\d|2[0-3]):[0-5]\d", re.ASCII)
_PHONE_PHRASE = "a phone number with its country code: a plus, the country code, then the number"
"""No example number: a model with no digits from the caller copies one."""
DATE_PHRASE = "a day written year-month-day, like 2026-03-19"
ID_PHRASE = "an identifier: letters, digits, and then any of dot, dash, underscore or colon"
TIME_PHRASE = "a time of day on the 24-hour clock, like 09:30 or 17:45"
_NO_PLUS = (
    "It has no plus: put one before the country code, or ask the caller for the country "
    "code if they haven't said it"
)
_AS_NUMBER = (
    "It came as a number, which cannot hold the plus: send it as text in quotes, like "
    '"+34600111222"'
)
_TOO_LONG = (
    "It is too long for its country, so a digit was misheard or said twice. Say the digits "
    "back to the caller and ask which part is wrong; nothing is missing"
)
_TOO_SHORT = "It is too short for its country: ask the caller for the rest"
_NO_COUNTRY = "No country has that code: ask for the whole number again, with the country code"
_NO_SUCH_NUMBER = "No such number exists: ask for the whole number again"
_EXTENSION = "It has an extension, which the number cannot hold: send the number without it"
_BEFORE = (
    "It has words or symbols before the number: send only the number, from its plus or its "
    "first digit"
)
_LETTERS = "It has letters: ask the caller for the number in digits"
_PARSE_REASONS = MappingProxyType(
    {
        phonenumbers.NumberParseException.INVALID_COUNTRY_CODE: _NO_COUNTRY,
        phonenumbers.NumberParseException.TOO_LONG: _TOO_LONG,
    }
)
_LENGTH_REASONS = MappingProxyType(
    {
        phonenumbers.ValidationResult.TOO_LONG: _TOO_LONG,
        phonenumbers.ValidationResult.TOO_SHORT: _TOO_SHORT,
        phonenumbers.ValidationResult.INVALID_LENGTH: _NO_SUCH_NUMBER,
        phonenumbers.ValidationResult.INVALID_COUNTRY_CODE: _NO_COUNTRY,
    }
)

_PHONE_REGION: contextvars.ContextVar[Callable[[], str | None] | None] = contextvars.ContextVar(
    "unmute_core_phone_region", default=None
)


@contextlib.contextmanager
def use_phone_region(region: Callable[[], str | None] | None) -> Generator[None]:
    """Read a phone number sent without its plus as one of the call's own country.

    The call's numbers are known only once the caller is on the line, so ``region`` is a
    function, asked each time a number needs it. Adapters enter this once per call, before
    they start the session, so every task the call runs inherits it.

    Args:
        region: Returns the call's country as ISO 3166 alpha-2, like ``"ES"``, or None.

    Yields:
        Nothing. The region holds until the block ends.
    """
    token = _PHONE_REGION.set(region)
    try:
        yield
    finally:
        _PHONE_REGION.reset(token)


def region_of(number: str | None) -> str | None:
    """Return the country (ISO 3166 alpha-2) a full number belongs to, or None.

    Args:
        number: A phone number with its plus, like ``+34612345678``.
    """
    if not number:
        return None
    try:
        region = phonenumbers.region_code_for_number(phonenumbers.parse(number))
    except phonenumbers.NumberParseException:
        return None
    return region if region and region != "ZZ" else None


@functools.lru_cache(maxsize=1024)
def _valid_e164(text: str, region: str | None) -> str | None:
    """Return ``text`` as E.164 when libphonenumber finds it a valid number, else None.

    Kept per text and region: the same number is read again on every turn it is repaired.
    """
    try:
        number = phonenumbers.parse(text, region)
    except phonenumbers.NumberParseException:
        return None
    if not phonenumbers.is_valid_number(number):
        return None
    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)


def repaired_phone(digits: str) -> str | None:
    """Return the one valid number that digits sent without a plus can be, or None.

    Two readings are tried: the digits as an international number, and as a national number
    of the call's country (see ``use_phone_region``). A number is kept only when exactly one
    valid reading exists, so the repair is never a guess.

    Args:
        digits: Only digits, like ``34600111222`` or ``600111222``.
    """
    source = _PHONE_REGION.get()
    region = source() if source is not None else None
    readings = {_valid_e164("+" + digits, None)}
    if region is not None:
        readings.add(_valid_e164(digits, region))
    readings.discard(None)
    return readings.pop() if len(readings) == 1 else None


def _refused(reason: str) -> PydanticCustomError:
    """Return the phone error for a number that does not fit, with what to ask next."""
    return PydanticCustomError(
        "phone", "expected {phrase}. {reason}", {"phrase": _PHONE_PHRASE, "reason": reason}
    )


def _phone_number(value: object) -> object:
    """Read a phone sent as a JSON number, which cannot carry its plus.

    The digits are kept when they can be only one valid number. Otherwise the model is told
    to send the number as text.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    if float(value).is_integer() and (repaired := repaired_phone(str(int(value)))):
        return repaired
    raise _refused(_AS_NUMBER)


def _unread_reason(compact: str, error: phonenumbers.NumberParseException) -> str:
    """Return why a tidied number libphonenumber could not read is refused.

    An extension or letters come first: without its plus, a national number with an
    extension cannot be read at all, and "no country has that code" would send the model
    after a country code the caller never needed to give.
    """
    if _EXTENSION_MARK.search(compact):
        return _EXTENSION
    if _LETTER_AFTER_DIGIT.search(compact):
        return _LETTERS
    short = compact[1:].isdigit()
    return _PARSE_REASONS.get(error.error_type, _TOO_SHORT if short else _NO_SUCH_NUMBER)


def _e164(value: str) -> str:
    """Tidy a phone number, check it with libphonenumber, and write it as E.164.

    Spaces, dashes, dots, slashes and brackets are dropped, and a leading 00 becomes +.
    Digits with no plus are repaired when they can be only one valid number. The model is
    refused only for what it cannot tidy away, with libphonenumber's own reason, so it asks
    the caller the right next question: a missing country code, a number too long (a digit
    misheard or said twice), too short (the caller is not done), or one that cannot exist.
    A number with an extension, with letters among its digits, or with words or symbols
    before it (``Phone: +34...``, ``tel:+34...``) is refused too: the stored number must be
    exactly what the caller said, and text around it can change what it means. A digit or
    plus of another script, like a fullwidth one, is the same digit, and is read.
    """
    compact = _PHONE_SEPARATORS.sub("", value)
    if compact.startswith("00"):
        compact = "+" + compact[2:]
    if compact and not _NUMBER_START.match(compact):
        raise _refused(_BEFORE)
    if _DIGITS_ONLY.fullmatch(compact):
        if repaired := repaired_phone(compact):
            return repaired
        raise _refused(_NO_PLUS)
    try:
        number = phonenumbers.parse(compact, None)
    except phonenumbers.NumberParseException as error:
        raise _refused(_unread_reason(compact, error)) from None
    if number.extension:
        raise _refused(_EXTENSION)
    if _LETTER_AFTER_DIGIT.search(compact):
        raise _refused(_LETTERS)
    if not phonenumbers.is_valid_number(number):
        possible = phonenumbers.is_possible_number_with_reason(number)
        raise _refused(_LENGTH_REASONS.get(possible, _NO_SUCH_NUMBER))
    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)


Phone = Annotated[
    str,
    BeforeValidator(_phone_number),
    AfterValidator(_e164),
    WithJsonSchema({"type": "string", "format": "phone"}),
]
"""A phone number, saved as E.164 like ``+34612345678``. JSON Schema ``format: phone``."""


def json_text(value: object) -> object:
    """Read an object or a list a model sent as JSON text. Anything else is as it is."""
    if not isinstance(value, str) or not value.lstrip().startswith(("{", "[")):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _number_as_text(value: object) -> object:
    """Store a JSON number sent to a text value as its text, and trim text. True stays refused."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int | float) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise PydanticCustomError(
                "finite", "expected a finite number, not {value}", {"value": str(value)}
            )
        return str(value)
    return value


Text = Annotated[str, BeforeValidator(_number_as_text)]
"""Text, which also stores a JSON number sent to it as its digits. The grammar's ``str``."""


def _identifier(value: str) -> str:
    """Accept an identifier whole; refuse anything else in words the model can act on."""
    if _ID.fullmatch(value):
        return value
    raise PydanticCustomError("id", "expected {phrase}", {"phrase": ID_PHRASE})


Id = Annotated[
    str,
    BeforeValidator(_number_as_text),
    AfterValidator(_identifier),
    WithJsonSchema({"type": "string", "format": "id"}),
]
"""An identifier: a letter or digit, then up to 63 of those or ``. _ : -``. ``format: id``.

A JSON number is read as its digits.
"""


def _iso_date(value: object) -> object:
    """Accept a date, or text written exactly ``YYYY-MM-DD``; refuse anything else."""
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str) and _ISO_DATE.fullmatch(value.strip()):
        return value.strip()
    raise PydanticCustomError("date", "expected {phrase}", {"phrase": DATE_PHRASE})


Date = Annotated[datetime.date, BeforeValidator(_iso_date)]
"""A calendar day, read from ``YYYY-MM-DD`` text only. JSON Schema ``format: date``."""


BirthDate = Annotated[
    datetime.date,
    BeforeValidator(_iso_date),
    WithJsonSchema({"type": "string", "format": "birth-date"}),
]
"""A date of birth, read from ``YYYY-MM-DD`` text only. JSON Schema ``format: birth-date``.

Code says it with its year and no weekday: "June 5, 1979", "5 de junio de 1979".
"""


def _code(value: object) -> object:
    """Trim a code, and store a JSON number sent to it as its digits."""
    value = _number_as_text(value)
    if isinstance(value, str) and not value:
        raise PydanticCustomError("spelled", "expected a code, like KP482913")
    return value


Spelled = Annotated[
    str, BeforeValidator(_code), WithJsonSchema({"type": "string", "format": "spelled"})
]
"""A code said a character at a time, like a policy or booking number. ``format: spelled``.

Code says "KP482913" as "K P, 4 8 2, 9 1 3", so a voice reads each character on its own.
It is saved as given, trimmed. Check its pattern with your own validator.
"""


def _clock_time(value: object) -> object:
    """Accept ``HH:MM`` on the 24-hour clock, ASCII digits only."""
    if isinstance(value, str) and _CLOCK_TIME.fullmatch(value.strip()):
        return value.strip()
    raise PydanticCustomError("time", "expected {phrase}", {"phrase": TIME_PHRASE})


Time = Annotated[
    str, BeforeValidator(_clock_time), WithJsonSchema({"type": "string", "format": "time"})
]
"""A time of day, saved as ``HH:MM`` text like ``09:30``. JSON Schema ``format: time``.

Text, not ``datetime.time``, so it reads back exactly as the format words say.
"""


class NameEmail(BaseModel):
    """A person and their email address, held as two parts.

    It reads the address on its own, or a name in front of it like ``Jo Bloggs
    <jo@example.com>``. Paths read its parts: ``{{contact.email}}``. Built in to the type
    grammar under this name.

    Attributes:
        name: The name as the person gave it. With only an address, the part before the @.
        email: The address on its own.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "description": (
                "A person and their email address. Accepts either the address on its own or "
                "a display name in front of it, and holds the two parts separately."
            )
        },
    )

    name: str = Field(
        description=(
            "The person's name as they gave it. When only an address is known this is the "
            "part of it before the at sign."
        )
    )
    email: EmailStr = Field(description="The email address on its own, with no name around it.")

    @model_validator(mode="before")
    @classmethod
    def _one_string(cls, value: Any) -> Any:
        """Read ``Name <address>``, a bare address, or the two parts as JSON text."""
        if not isinstance(value, str):
            return value
        if value.lstrip().startswith("{"):
            return json_text(value)
        try:
            info = validate_email(value, check_deliverability=False, allow_display_name=True)
        except EmailNotValidError as error:
            raise PydanticCustomError(
                "name_email",
                "expected an email address, on its own or with a name in front of it, like "
                "Fred Bloggs <fred@example.com>: {reason}",
                {"reason": str(error)},
            ) from None
        return {"name": info.display_name or info.local_part, "email": info.normalized}
