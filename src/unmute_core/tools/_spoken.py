"""How a value is said out loud by code, in the call's language.

A value's JSON Schema ``format`` picks the form. CLDR data (through Babel) gives each
language its own words and order. With the call's date known, "2026-10-16" is said
"Friday, October 16" in English and "viernes, 16 de octubre" in Spanish, and today's date
"today, October 8". A ``birth-date`` is said with its year and no weekday. A ``spelled``
code is said in short groups, a character at a time, so "KP482913" is "K P, 4 8 2, 9 1 3".
An ``enum`` value is said as words, so "hair-color" is "hair color". Any other value is
said as given.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, tzinfo
from functools import cache
from types import MappingProxyType
from typing import Any

import phonenumbers
from babel import Locale, UnknownLocaleError
from babel.dates import format_date, format_skeleton, format_time, get_datetime_format

logger = logging.getLogger(__name__)

_NARROW_SPACES = str.maketrans({chr(0x202F): " ", chr(0x00A0): " "})

_NEAR_DAYS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "en": ("today", "tomorrow", "the day after tomorrow"),
        "es": ("hoy", "mañana", "pasado mañana"),
        "fr": ("aujourd'hui", "demain", "après-demain"),
        "de": ("heute", "morgen", "übermorgen"),
        "it": ("oggi", "domani", "dopodomani"),
        "pt": ("hoje", "amanhã", "depois de amanhã"),
        "nl": ("vandaag", "morgen", "overmorgen"),
        "ca": ("avui", "demà", "demà passat"),
    }
)
"""Words for today and the next two days. CLDR has them, but Babel does not ship them.

A language missing here says the weekday instead, so every language still works.
"""


@cache
def locale_of(language: str) -> Locale:
    """Return the CLDR locale for a BCP-47 tag, falling back to its base language, then English.

    Args:
        language: A tag like ``es-MX`` or ``en``.
    """
    for tag in (language, language.replace("_", "-").split("-")[0]):
        try:
            return Locale.parse(tag, sep="-" if "-" in tag else "_")
        except (UnknownLocaleError, ValueError):
            continue
    logger.warning("no CLDR data for language %r; spoken lines say dates in English", language)
    return Locale.parse("en")


def _day(day: date, today: date | None, locale: Locale) -> str:
    """Say a day as a person does: "today, October 8", "Friday, October 16".

    The day is named first, then its date. The year is said only for another year, or
    when today is unknown.
    """
    if today is None:
        return format_date(day, "full", locale=locale)
    near = _NEAR_DAYS.get(locale.language, ())
    offset = (day - today).days
    if 0 <= offset < len(near):
        named = near[offset]
    elif day.year != today.year:
        return format_date(day, "full", locale=locale)
    else:
        named = format_date(day, "EEEE", locale=locale)
    return f"{named}, {format_skeleton('MMMMd', day, locale=locale)}"


_RUNS = re.compile(r"[^\W\d_]+|\d+")
"""A code's runs of letters and of digits. Anything else, like a dash, is not said."""


def _digit_groups(digits: str) -> list[str]:
    """Split digits into groups of three, ending on two twos rather than a lone digit."""
    count = len(digits)
    sizes = [3] * (count // 3)
    if count % 3 == 2 or (count % 3 == 1 and not sizes):
        sizes.append(count % 3)
    elif count % 3 == 1:
        sizes[-1:] = [2, 2]
    groups, start = [], 0
    for size in sizes:
        groups.append(digits[start : start + size])
        start += size
    return groups


def _spelled(value: str) -> str:
    """Say a code a character at a time, in groups: "KP482913" is "K P, 4 8 2, 9 1 3".

    A text-to-speech voice reads each spaced character on its own, in the call's language.
    """
    groups = [
        group
        for run in _RUNS.findall(value)
        for group in (_digit_groups(run) if run.isdigit() else [run.upper()])
    ]
    if not groups:
        raise ValueError("nothing to spell")
    return ", ".join(" ".join(group) for group in groups)


def _moment(value: str, today: date | None, zone: tzinfo | None, locale: Locale) -> str:
    """Say a date and time: the day as ``_day`` says it, then the time, in CLDR's order.

    A value with an offset is moved to the call's zone first.
    """
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is not None and zone is not None:
        moment = moment.astimezone(zone)
    said_time = format_time(moment.time(), "short", locale=locale)
    pattern = str(get_datetime_format("long", locale=locale)).replace("'", "")
    return pattern.replace("{0}", said_time).replace("{1}", _day(moment.date(), today, locale))


def spoken(
    value: object,
    form: str | None,
    language: str,
    today: date | None = None,
    zone: tzinfo | None = None,
) -> object:
    """Return a value as code says it.

    Args:
        value: The validated value, as JSON data.
        form: Its JSON Schema ``format``: ``date``, ``date-time``, ``time``, ``phone``,
            ``birth-date`` and ``spelled`` are said in their spoken form, and ``enum`` as
            words. Anything else, or a value that does not parse, is returned as it is.
        language: The call's language, as a BCP-47 tag.
        today: The call's date, where the caller is. Without it a date is said in full,
            with its year.
        zone: The call's time zone. A ``date-time`` with an offset is said in it.
    """
    if not isinstance(value, str):
        return value
    locale = locale_of(language)
    try:
        if form == "date":
            text = _day(date.fromisoformat(value), today, locale)
        elif form == "date-time":
            text = _moment(value, today, zone, locale)
        elif form == "birth-date":
            text = format_date(date.fromisoformat(value), "long", locale=locale)
        elif form == "spelled":
            text = _spelled(value)
        elif form == "time":
            text = format_time(time.fromisoformat(value), "short", locale=locale)
        elif form == "phone":
            number = phonenumbers.parse(value, None)
            text = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        elif form == "enum":
            text = value.replace("-", " ").replace("_", " ")
        else:
            return value
    except (ValueError, phonenumbers.NumberParseException):
        return value
    return text.translate(_NARROW_SPACES)


def _resolved(node: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    """Follow a ``$ref`` into the root's ``$defs``, and an optional value's ``anyOf``."""
    while True:
        if isinstance(ref := node.get("$ref"), str) and ref.startswith("#/$defs/"):
            node = root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})
            continue
        options = [option for option in node.get("anyOf", ()) if option.get("type") != "null"]
        if len(options) == 1 and "format" not in node and "enum" not in node:
            node = options[0]
            continue
        return node


def form_at(schema: Mapping[str, Any], path: Sequence[str]) -> str | None:
    """Return how a value inside a JSON Schema is spoken: its ``format``, or ``enum``.

    The path walks ``properties``, through ``$ref`` (a nested model or an ``Enum`` class)
    and an optional value's ``anyOf``.

    Args:
        schema: A tool's input or output JSON Schema, with its ``$defs``.
        path: Field names, like ``("appointment", "date")``.

    Returns:
        The ``format``, ``enum`` for a set of words, or None when the value has neither or
        the path is not in the schema.
    """
    node = _resolved(schema, schema)
    for part in path:
        node = _resolved(node.get("properties", {}).get(part, {}), schema)
    if isinstance(found := node.get("format"), str):
        return found
    return "enum" if "enum" in node or "const" in node else None
