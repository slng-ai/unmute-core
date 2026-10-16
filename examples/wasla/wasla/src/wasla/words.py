"""How Wasla says money, days and times out loud: plain functions, no state.

The library says a ``date`` or ``time`` field by its JSON Schema format. An amount has no
such format, and a TTS voice reads "434.50" its own way, so code says it in words here:
"four hundred thirty-four dirhams and fifty fils".
"""

from datetime import date, time
from decimal import Decimal

ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number(n: int) -> str:
    """Say 0 to 999,999 in words: 434 is "four hundred thirty-four"."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10 - 2] + (f"-{ONES[n % 10]}" if n % 10 else "")
    if n < 1000:
        return f"{ONES[n // 100]} hundred" + (f" {number(n % 100)}" if n % 100 else "")
    return f"{number(n // 1000)} thousand" + (f" {number(n % 1000)}" if n % 1000 else "")


def amount(value: Decimal) -> str:
    """Say an amount in UAE dirhams and fils, in words, never in digits."""
    dirhams, fils = divmod(int(value.quantize(Decimal("0.01")) * 100), 100)
    said = f"{number(dirhams)} dirham{'' if dirhams == 1 else 's'}"
    return f"{said} and {number(fils)} fils" if fils else said


def ordinal(n: int) -> str:
    """Say 1 as "1st", 22 as "22nd" and 13 as "13th"."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def day(value: date) -> str:
    """Say a day as "Friday the 9th of October"."""
    return f"{value:%A} the {ordinal(value.day)} of {value:%B}"


def clock(value: time) -> str:
    """Say a time as "5 PM" or "5:30 PM"."""
    hour = value.hour % 12 or 12
    minutes = f":{value.minute:02d}" if value.minute else ""
    return f"{hour}{minutes} {'AM' if value.hour < 12 else 'PM'}"
