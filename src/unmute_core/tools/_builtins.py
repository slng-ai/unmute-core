"""Tools the library ships ready to use: the current date and time, and the caller's number.

Unmute reads the same values through prefetch instead (unmute@2c78a5a internal/ir/prefetch.go):
a ``clock`` entry and the caller's number, here ``state.call.calling_number``. These tools let
the model ask for them during the call.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, BaseModel, ConfigDict, TypeAdapter, ValidationError

from unmute_core._pydantic import retitled
from unmute_core.state import CallState
from unmute_core.tools._spec import ToolSpec


def _no_arguments() -> dict[str, Any]:
    """Return the parameters of a tool that takes none: a new dict for each tool."""
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _zone(name: str) -> str:
    """Refuse a name that is not an IANA time zone."""
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"{name!r} is not an IANA time zone, like 'Europe/Madrid'") from None
    return name


IanaZone = Annotated[str, AfterValidator(_zone)]
"""The name of an IANA time zone, like ``Europe/Madrid``."""
_IANA_ZONE = TypeAdapter(IanaZone)


class _Result(BaseModel):
    """What a built-in tool answers: it dumps to the dict the model reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class _Now(_Result):
    """The date and time in one zone."""

    timezone: str
    local_datetime: str
    local_date: str
    local_time: str
    day_of_week: str
    utc_offset: str
    is_dst: bool


class _PhoneNumber(_Result):
    """The caller's number and whether the call came in or went out, or why there is none."""

    available: bool
    phone_number: str | None = None
    call_type: Literal["inbound", "outbound"] | None = None
    reason: str | None = None


class _Builtin:
    """Runs a built-in tool: a function of the call state, with no arguments."""

    def __init__(self, answer: Callable[[CallState | None], dict[str, Any]]) -> None:
        """Keep the function that answers a call."""
        self.answer = answer

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        """Return the answer for this call."""
        return self.answer(state)


def current_datetime_tool(
    timezone: str = "UTC",
    *,
    name: str = "current_datetime",
    description: str = (
        "Get the current date, time and day of the week. Call it whenever an answer depends "
        "on today or now."
    ),
) -> ToolSpec:
    """Build a tool that tells the model the date and time in one time zone.

    The result holds ``timezone``, ``local_datetime``, ``local_date``, ``local_time``,
    ``day_of_week``, ``utc_offset`` (like ``+02:00``) and ``is_dst``.

    Args:
        timezone: An IANA zone, like ``Europe/Madrid``. Never the container's clock.
        name: The tool name.
        description: What the model reads.

    Raises:
        ValueError: If the zone is not an IANA zone.
    """
    try:
        zone = ZoneInfo(_IANA_ZONE.validate_python(timezone))
    except ValidationError as error:
        raise retitled(error, "current_datetime_tool", prefix=("timezone",)) from None

    def now(_state: CallState | None) -> dict[str, Any]:
        """Read the clock in the zone."""
        moment = datetime.now(zone)
        offset = moment.strftime("%z")
        return _Now(
            timezone=timezone,
            local_datetime=moment.isoformat(timespec="seconds"),
            local_date=moment.date().isoformat(),
            local_time=moment.strftime("%H:%M:%S"),
            day_of_week=moment.strftime("%A"),
            utc_offset=f"{offset[:3]}:{offset[3:]}",
            is_dst=bool(moment.dst()),
        ).model_dump()

    return ToolSpec(name, description, _no_arguments(), _Now, _Builtin(now))


def user_phone_number_tool(
    *,
    name: str = "user_phone_number",
    description: str = "Get the caller's phone number. It is known only on a phone call.",
) -> ToolSpec:
    """Build a tool that tells the model the other party's number, from ``state.call``.

    The result is ``{"available": true, "phone_number": "+34...", "call_type": "inbound",
    "reason": null}``, with ``call_type`` null when the call has no ``direction``, or
    ``{"available": false, "phone_number": null, "call_type": null, "reason": ...}`` when the
    call has no number, like a web call.

    Args:
        name: The tool name.
        description: What the model reads.
    """

    def number(state: CallState | None) -> dict[str, Any]:
        """Read ``call.remote_number`` and ``call.direction``."""
        call = state.call if state is not None else None
        if call is None or not call.remote_number:
            return _PhoneNumber(
                available=False, reason="this call has no phone number, like a web call"
            ).model_dump()
        return _PhoneNumber(
            available=True, phone_number=call.remote_number, call_type=call.direction
        ).model_dump()

    return ToolSpec(name, description, _no_arguments(), _PhoneNumber, _Builtin(number))
