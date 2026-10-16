"""How Kestrel says days and times in the claim's read-back: plain functions, no state.

The library says a field by its JSON Schema format: a ``Spelled`` code a character at a
time, a ``BirthDate`` with its year. The read-back is one line built from the whole
report, so its days and times are worded here.
"""

from datetime import date, time


def ordinal(n: int) -> str:
    """Say 1 as "1st", 22 as "22nd" and 13 as "13th"."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def spoken_day(value: date) -> str:
    """Say a day as "Monday the 28th of September"."""
    return f"{value:%A} the {ordinal(value.day)} of {value:%B}"


def spoken_when(value: time | str) -> str:
    """Say a time as "at 6:30 PM", or a part of the day as "in the evening" or "at night"."""
    if isinstance(value, str):
        return "at night" if value == "night" else f"in the {value}"
    hour = value.hour % 12 or 12
    minutes = f":{value.minute:02d}" if value.minute else ""
    return f"at {hour}{minutes} {'AM' if value.hour < 12 else 'PM'}"
