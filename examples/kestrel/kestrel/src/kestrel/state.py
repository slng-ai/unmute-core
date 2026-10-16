"""What a Kestrel claim call knows: one Pydantic model, typed field by field.

The model never writes these values directly. Each one is saved from a tool's checked
result (``Assign``), so a value that does not fit its type never reaches the call:

- ``policy`` comes only from ``lookup_policy``, after the caller said yes to the number and
  date of birth read back to them.
- ``report`` holds the facts of the incident. ``note_facts`` merges each answer into it,
  and its result saves it whole as soon as it returns.
- ``read_back`` is the claim summary in code's words. The confirm line of ``file_claim``
  says it, so the caller hears exactly what will be filed.
"""

from datetime import date, time
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field
from unmute_core.state import BirthDate, Spelled


def policy_number(value: str) -> str:
    """Return a policy number in its one spelling: KP and six digits, upper case.

    Spaces, dashes and dots are dropped, as speech to text writes "KP-482913".

    Raises:
        ValueError: When it is not KP and six digits.
    """
    value = "".join(character for character in value if character.isalnum()).upper()
    if len(value) != 8 or not value.startswith("KP") or not value[2:].isdigit():
        raise ValueError(
            f"{value} is not a policy number: it is KP and six digits, like KP482913. "
            "Ask for it again."
        )
    return value


PolicyNumber = Annotated[
    Spelled,
    AfterValidator(policy_number),
    Field(description="The policy number: KP and six digits, like KP482913. Never guess a digit."),
]
DayPart = Literal["morning", "afternoon", "evening", "night"]


class Vehicle(BaseModel):
    """One car on a policy."""

    plate: str
    words: str = Field(description='How the car is said, like "the blue Toyota Corolla".')


class Policy(BaseModel):
    """The policy the caller's number and date of birth found."""

    policy_number: str
    first_name: str
    vehicles: list[Vehicle]


class Report(BaseModel):
    """The facts of the incident, each empty until the caller gives it."""

    incident_day: date | None = None
    incident_time: time | DayPart | None = None
    place: str | None = None
    what_happened: str | None = None
    injuries: bool | None = None
    police_report: str | None = None
    other_party: bool | None = None
    vehicle_drivable: bool | None = None
    vehicle: str | None = Field(default=None, description="The plate of the car it happened to.")


class State(BaseModel):
    """One claim call. Every value starts empty.

    Attributes:
        today: Today's date on the claims desk's clock, read as the call starts.
        weekday: Today's weekday, so the model can work out "on Monday".
        policy_number: The number the caller gave, once a lookup ran on it.
        birth_date: The holder's date of birth the caller gave.
        policy: The policy found. Nothing about a policy is known before it.
        report: The incident's facts so far.
        read_back: The whole claim in code's words, read back before it is filed.
        claim_number: The claim number, once filed.
    """

    today: date | None = None
    weekday: str | None = None
    policy_number: PolicyNumber | None = None
    birth_date: BirthDate | None = Field(
        default=None, description="The policy holder's date of birth, as YYYY-MM-DD."
    )
    policy: Policy | None = None
    report: Report = Field(default_factory=Report)
    read_back: str | None = None
    claim_number: Spelled | None = None
