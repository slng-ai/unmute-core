"""Kestrel's claims desk and the tools the agent calls on it.

``ClaimsDesk`` stands in for the insurer's backend: two invented policies, and the claims
filed against them, in memory. Its three tools are plain Python methods with typed
arguments and typed results:

- ``lookup_policy`` finds a policy by its number and the holder's date of birth.
- ``note_facts`` merges what the caller just said into the report, and returns the next
  question in code's words, or the whole claim read back once every fact is in. Its
  result saves the report and the read-back into the call's state at once.
- ``file_claim`` files the report once per call.

Nothing the caller hears from these tools is worded by the model: each result carries the
line code says, or a typed ``status`` its replies turn into a line. The library checks each
argument against its type before a method runs, and each result against its model when it
returns. A rule about the claim, like "not in the future", is the tool's: an argument type
only says what the caller meant.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time
from threading import Lock
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from unmute_core.state import BirthDate, Spelled
from unmute_core.tools import ToolError

from kestrel.state import DayPart, Policy, PolicyNumber, Report, Vehicle
from kestrel.words import spoken_day, spoken_when

TIMEZONE = "Europe/London"
ZONE = ZoneInfo(TIMEZONE)
QUESTIONS = (
    ("incident_day", "I can take your claim now. On which day did it happen?"),
    ("incident_time", "Roughly what time was it?"),
    ("place", "Where did it happen?"),
    ("what_happened", "In your own words, what happened?"),
    ("injuries", "Was anyone hurt?"),
    ("police_report", "Was it reported to the police? If so, do you have a reference?"),
    ("other_party", "Was another vehicle or driver involved?"),
    ("vehicle_drivable", "Can your car still be driven?"),
)
"""The incident's facts in the order code asks for them, each with its question."""
SAFETY = (
    "I'm sorry to hear that. If anyone needs medical help, please call the emergency "
    "services first."
)
SEED: tuple[dict[str, Any], ...] = (
    {
        "policy_number": "KP482913",
        "first_name": "Daniel",
        "birth_date": date(1979, 6, 5),
        "vehicles": [{"plate": "AB12CDE", "words": "the blue Toyota Corolla"}],
    },
    {
        "policy_number": "KP731508",
        "first_name": "Priya",
        "birth_date": date(1992, 1, 23),
        "vehicles": [
            {"plate": "LM70XYZ", "words": "the white Kia Niro"},
            {"plate": "GH19RST", "words": "the black Ford Fiesta"},
        ],
    },
)
"""Invented people and cars. Any resemblance to a real one is a coincidence."""


def today() -> date:
    """Return today's date on the claims desk's clock. Tests replace it."""
    return datetime.now(ZONE).date()


IncidentDay = Annotated[
    date | None,
    Field(
        description=(
            "The day it happened, as YYYY-MM-DD. Work it out from today's date. Pass it even "
            "when it is in the future: the application checks it."
        )
    ),
]


class PolicyFound(BaseModel):
    """What ``lookup_policy`` returns."""

    found: bool
    policy: Policy | None = None
    first_name: str = ""
    hint: str = ""


class Noted(BaseModel):
    """What ``note_facts`` returns: the report so far, and what code says next.

    Attributes:
        report: Every fact so far, the new ones merged in.
        status: ``noted``, or ``future_day`` when the day given has not happened yet. That
            day is left out of the report, and the tool's reply asks for it again.
        complete: True once every fact and the car are known.
        next_question: The question for the first missing fact, while one is missing.
        read_back: The whole claim in code's words, once it is complete.
    """

    report: Report
    status: Literal["noted", "future_day"] = "noted"
    complete: bool
    next_question: str = ""
    read_back: str | None = None


class Filed(BaseModel):
    """What ``file_claim`` returns. The claim number is said a character at a time."""

    filed: bool
    claim_number: Spelled


def car_for(said: str, vehicles: list[Vehicle]) -> Vehicle | None:
    """Return the one car a caller named, by its plate or by a word of its make or model.

    A name that fits no car, or more than one, is no car: code then asks which car it was.
    """
    key = "".join(said.split()).upper()
    words = set(said.lower().split()) - {"the", "my", "car", "a"}
    named = [
        vehicle
        for vehicle in vehicles
        if key == vehicle.plate or words & (set(vehicle.words.lower().split()) - {"the"})
    ]
    return named[0] if len(named) == 1 else None


def read_back(report: Report, vehicles: list[Vehicle]) -> str:
    """Return the whole claim in code's words, ending in the question the caller answers."""
    assert report.incident_day is not None and report.incident_time is not None
    car = next((v.words for v in vehicles if v.plate == report.vehicle), "your car")
    place = (report.place or "").strip().rstrip(".")
    happened = (report.what_happened or "").strip().rstrip(".")
    police = (report.police_report or "").strip()
    reported = "no police report" if police.lower() in ("", "none") else f"police report {police}"
    return " ".join(
        [
            f"Shall I file your claim for {car}?",
            f"On {spoken_day(report.incident_day)} {spoken_when(report.incident_time)},",
            f"at {place}: {happened}.",
            "Someone was hurt," if report.injuries else "Nobody was hurt,",
            f"{reported},",
            "another driver was involved,"
            if report.other_party
            else "no other driver was involved,",
            f"and the car {'can' if report.vehicle_drivable else 'cannot'} be driven.",
            "Is that all right?",
        ]
    )


@dataclass
class ClaimsDesk:
    """The policies, and every claim filed, by call. Thread-safe: tools may run in threads."""

    policies: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {p["policy_number"]: p for p in SEED}
    )
    filed: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def lookup_policy(self, policy_number: PolicyNumber, birth_date: BirthDate) -> PolicyFound:
        """Find the caller's policy by its number and the policy holder's date of birth.

        Call it as soon as you have both. It first reads both back to the caller, so never
        read them back yourself. Call it again with the same values after their yes.
        """
        found = self.policies.get(policy_number)
        if found is None or found["birth_date"] != birth_date:
            return PolicyFound(
                found=False,
                hint=(
                    "No policy matches that number and date of birth. Reveal nothing about "
                    "any policy."
                ),
            )
        policy = Policy.model_validate(found)
        return PolicyFound(found=True, policy=policy, first_name=policy.first_name)

    def note_facts(
        self,
        report: Report,
        policy: Policy,
        incident_day: IncidentDay = None,
        incident_time: Annotated[
            time | DayPart | None,
            Field(description="HH:MM on a 24-hour clock, or morning, afternoon, evening, night."),
        ] = None,
        place: Annotated[
            str | None, Field(description="Where it happened, in the caller's own words.")
        ] = None,
        what_happened: Annotated[
            str | None,
            Field(description="What happened, in the caller's own words: the shortest excerpt."),
        ] = None,
        injuries: Annotated[bool | None, Field(description="Whether anyone was hurt.")] = None,
        police_report: Annotated[
            str | None,
            Field(description='The police reference they gave, or "none" for no report.'),
        ] = None,
        other_party: Annotated[
            bool | None, Field(description="Whether another vehicle or driver was involved.")
        ] = None,
        vehicle_drivable: Annotated[
            bool | None, Field(description="Whether their car can still be driven.")
        ] = None,
        vehicle: Annotated[
            str | None, Field(description="The car it happened to: its plate or its name.")
        ] = None,
    ) -> Noted:
        """Note every fact of the incident the caller just gave, all in one call.

        Pass only what they said. The application asks the next question itself. When it
        answers complete, call file_claim at once.
        """
        known = report
        future = incident_day is not None and incident_day > today()
        given: dict[str, object] = {
            "incident_day": None if future else incident_day,
            "incident_time": incident_time,
            "place": place,
            "what_happened": what_happened,
            "injuries": injuries,
            "police_report": police_report,
            "other_party": other_party,
            "vehicle_drivable": vehicle_drivable,
        }
        if vehicle is not None and (car := car_for(vehicle, policy.vehicles)) is not None:
            given["vehicle"] = car.plate
        newly_hurt = injuries is True and known.injuries is not True
        report = known.model_copy(
            update={name: value for name, value in given.items() if value is not None}
        )
        if report.vehicle is None and len(policy.vehicles) == 1:
            report.vehicle = policy.vehicles[0].plate
        noted = next_step(report, policy.vehicles, newly_hurt)
        if noted.complete:
            noted.read_back = read_back(report, policy.vehicles)
        if future:
            noted.status = "future_day"
        return noted

    def file_claim(self, report: Report, policy_number: str, call: str) -> Filed:
        """File the claim exactly as it was read back. Call it once the report is complete.

        It first reads the whole claim back to the caller, so never read it back yourself.
        Call it again after their yes.
        """
        if not next_step(report, [], False).complete:
            raise ToolError("the report is not complete yet: note the missing facts first")
        with self.lock:
            if call not in self.filed:
                number = f"KC{100231 + len(self.filed)}"
                self.filed[call] = {
                    "claim_number": number,
                    "policy_number": policy_number,
                    **report.model_dump(mode="json"),
                }
            number = self.filed[call]["claim_number"]
        return Filed(filed=True, claim_number=number)


def next_step(report: Report, vehicles: list[Vehicle], newly_hurt: bool) -> Noted:
    """Return what code says next: the first missing fact's question, or that it is complete.

    The safety line goes before the next question once, when the caller first says
    someone was hurt.
    """
    question = next((ask for name, ask in QUESTIONS if getattr(report, name) is None), None)
    if question is None and report.vehicle is None:
        question = f"Which car was it: {' or '.join(v.words for v in vehicles)}?"
    if question is not None:
        said = f"{SAFETY} {question}" if newly_hurt else question
        return Noted(report=report, complete=False, next_question=said)
    return Noted(report=report, complete=True)


COVERAGE = """\
- A claims handler calls within one working day of a claim being filed, and decides what
  the policy covers. Never promise cover yourself.
- Comprehensive policies cover damage to the caller's own car, whoever was at fault, less
  the excess. Third-party policies cover only damage to others.
- The excess is on the policy schedule. The handler confirms it.
- A courtesy car is offered while the car is repaired at a Kestrel approved garage, if
  the policy includes one.
- A car that cannot be driven can be recovered by Kestrel's recovery partner: the handler
  arranges it.
- Photos of the damage and the scene help: the handler asks for them.
"""


class Guide(BaseModel):
    """What ``coverage_guide`` returns."""

    guide: str


def coverage_guide() -> Guide:
    """Read the coverage guide: cover, the excess, a courtesy car, recovery, what happens next.

    Call it only when the caller asks about one of these, and answer from it.
    """
    return Guide(guide=COVERAGE)
