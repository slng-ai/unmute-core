"""What the call itself carries, kept in every state as the reserved value ``call``.

The adapter reads the line: SIP, a carrier websocket, or a web session. The library adds
``call`` to every State model, so prompts, tools and tasks read ``{{call.calling_number}}``
and Python reads ``state.call.calling_number``. Nobody but the library writes it. An author
who needs to confirm a number with the caller copies it into a value of their own.
"""

import logging
import uuid
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    create_model,
    model_validator,
)

from unmute_core.settings import EnvSettings, SettingsError
from unmute_core.state._types import Phone, region_of

logger = logging.getLogger(__name__)

CALL = "call"
"""The State value the library reserves for what the call carries."""
CallChannel = Literal["phone", "web"]
Direction = Literal["inbound", "outbound"]
DIRECTIONS: tuple[Direction, ...] = ("inbound", "outbound")
ID_MAX = 256
TRACKING_ID_MAX = 128
"""The longest agent or session id. SLNG speech refuses a longer one."""
ID_FORBIDDEN = frozenset(", \t\n\r{}|")
AGENT_ID_ENV = "UNMUTE_AGENT_ID"

_KEYPAD = "22233344455566677778889999"
_WITHHELD = tuple(
    "".join(_KEYPAD[ord(letter) - ord("a")] for letter in word)
    for word in ("anonymous", "restricted", "unknown", "unavailable", "private")
)
_PHONE: TypeAdapter[str] = TypeAdapter(Phone)


class ReservedNameError(ValueError):
    """A State declares ``call``, which the library fills from the call itself."""


def reserved_name_error(where: str) -> ReservedNameError:
    """Return the error for a State, a rule or a task that names ``call``.

    Args:
        where: What named it, like ``State`` or ``confirm``.
    """
    return ReservedNameError(
        f"{where} names 'call', which unmute-core reserves for what the call carries "
        "(state.call.calling_number, {{call.session_id}}). Rename your value, or copy a call "
        "value into one of your own with a CallValue prefetch entry"
    )


def _line_number(value: object, info: ValidationInfo) -> str | None:
    """Return a number the line sent as E.164, or None for a placeholder or a non-number.

    A number is a plus and 2 to 15 ASCII digits. A value whose digits start a keypad spelling
    of a withheld-caller word is a placeholder. Only that direction: "unknown" spells 8656696,
    and +86 5669 6xxx is an ordinary Chinese mobile. The number itself is never logged.
    """
    text = ("" if value is None else str(value)).strip()
    digits = text.removeprefix("+")
    if not text:
        return None
    if not (
        text.startswith("+") and digits.isascii() and digits.isdigit() and 2 <= len(digits) <= 15
    ):
        logger.info("call: %s was not a phone number, treating it as absent", info.field_name)
        return None
    if any(spelled.startswith(digits) for spelled in _WITHHELD):
        logger.info("call: %s is a withheld caller ID, treating it as absent", info.field_name)
        return None
    try:
        return _PHONE.validate_python(text)
    except ValidationError:
        logger.info("call: %s is no valid number, treating it as absent", info.field_name)
        return None


LineNumber = Annotated[Phone | None, BeforeValidator(_line_number)]
"""A number from the line. One that is withheld or no number reads as None, never an error."""
CallId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=ID_MAX)]


def no_forbidden(value: str, info: ValidationInfo) -> str:
    """Refuse an id that holds a comma, whitespace, ``{``, ``}`` or ``|``, or non-ASCII text."""
    if bad := sorted(ID_FORBIDDEN & set(value)):
        raise ValueError(
            f"{info.field_name} must not contain a comma, whitespace, '{{', '}}' or '|', "
            f"found {bad[0]!r}"
        )
    if not value.isascii() or not value.isprintable():
        raise ValueError(
            f"{info.field_name} must be printable ASCII, found a character outside it: "
            "use ASCII letters, digits and punctuation"
        )
    return value


TrackingId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=TRACKING_ID_MAX),
    AfterValidator(no_forbidden),
]
"""An agent or session id every SLNG service reads: stripped, 1 to 128 printable ASCII
characters, no comma, whitespace, braces or bar."""


class AgentSettings(EnvSettings):
    """The agent id a call takes when your code passes none.

    Attributes:
        agent_id: One id per agent, from ``UNMUTE_AGENT_ID``.
    """

    agent_id: TrackingId = Field(validation_alias=AGENT_ID_ENV, description="One id per agent.")


def _agent_id_from_env() -> str:
    """Return ``UNMUTE_AGENT_ID``, or raise an error that names both ways to set it."""
    try:
        return AgentSettings().agent_id
    except SettingsError as error:
        raise SettingsError(
            f"Every call needs an agent id: pass agent_id= to CallInfo, call_info, "
            f"sip_call_info or telephony_call_info, or set {AGENT_ID_ENV}. {error}"
        ) from error


def _new_session_id() -> str:
    """Return a fresh id for one call."""
    return uuid.uuid4().hex


class CallInfo(BaseModel):
    """What one call carries: who is on it, how it came in, and its ids.

    The adapter builds it from the line: ``sip_call_info`` on LiveKit, ``telephony_call_info``
    on Pipecat, or ``call_info`` by hand. A web call carries no numbers and no direction,
    unless a dispatch or ``UNMUTE_CALL_START`` simulates a phone call.

    Attributes:
        agent_id: One id per agent, the same on LiveKit and Pipecat. Your code or the
            dispatch sets it, else ``UNMUTE_AGENT_ID``. A call without one fails to build.
        session_id: One id per call. Made once when the call's state is built, and the same
            on every turn, task and handoff. The dispatch may set it. Every SLNG model,
            speech-to-text and text-to-speech request sends both ids.
        channel: ``phone`` for SIP or a carrier websocket, ``web`` otherwise.
        direction: Who placed this call: ``inbound`` or ``outbound``. None on a web call.
        calling_number: The party that placed the call. Inbound: the caller. Outbound: the
            agent's own number.
        called_number: The party that was dialled. Inbound: the agent's own number.
            Outbound: the customer.
        remote_number: The other party, either way: filled from ``direction``.
        carrier_call_id: The carrier's own id of the call, like the SIP Call-ID or Twilio's
            CallSid. It is not ``session_id``.
        first_user_message: What the caller said first. The library saves it as their
            first message reaches the model, before the ``first_user_message`` tools run.
        end_reason: Why the call ended. The library saves it as the call ends.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: TrackingId = Field(
        default_factory=_agent_id_from_env, description="One id per agent."
    )
    session_id: TrackingId = Field(
        default_factory=_new_session_id, description="One id per call, the same on every turn."
    )
    channel: CallChannel = Field("web", description="phone for SIP or a carrier, else web.")
    direction: Direction | None = Field(None, description="Who placed the call.")
    calling_number: LineNumber = Field(None, description="The party that placed the call.")
    called_number: LineNumber = Field(None, description="The party that was dialled.")
    remote_number: LineNumber = Field(None, description="The other party on the call.")
    carrier_call_id: CallId | None = Field(None, description="The carrier's id of the call.")
    first_user_message: str | None = Field(None, description="What the caller said first.")
    end_reason: str | None = Field(None, description="Why the call ended.")

    @model_validator(mode="before")
    @classmethod
    def _remote_from_direction(cls, data: Any) -> Any:
        """Fill ``remote_number`` with the party on the other side of ``direction``."""
        if not isinstance(data, Mapping):
            return data
        direction = data.get("direction")
        remote = {"inbound": "calling_number", "outbound": "called_number"}.get(str(direction))
        return {**data, "remote_number": data.get(remote) if remote else None}


_MODELS: dict[type[BaseModel], type[BaseModel]] = {}
"""Each State model to its call model. Built once per class, never per call."""
_BUILT: set[type[BaseModel]] = set()


def call_model[M: BaseModel](model: type[M]) -> type[M]:
    """Return the State model with the reserved ``call`` value added, built once per class.

    Args:
        model: The author's State model, or one this function already returned.

    Raises:
        ReservedNameError: If the model declares ``call`` itself.
    """
    if model in _BUILT:
        return model
    if (built := _MODELS.get(model)) is not None:
        return built  # ty: ignore[invalid-return-type]
    if any(name == CALL or info.alias == CALL for name, info in model.model_fields.items()):
        raise reserved_name_error(model.__name__)
    built = create_model(
        model.__name__,
        __base__=model,
        __module__=model.__module__,
        call=(
            CallInfo,
            Field(
                default_factory=CallInfo,
                alias=CALL,
                description="What the call carries. Set by the library.",
            ),
        ),
    )
    # ponytail: kept for the process, as a class is; a State built per call would grow it
    _MODELS[model] = built
    _BUILT.add(built)
    return built


def call_info(
    direction: Direction,
    *,
    remote_number: object = None,
    local_number: object = None,
    channel: CallChannel = "phone",
    carrier_call_id: object = None,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> CallInfo:
    """Return what a phone call carries, its two numbers named by who called whom.

    On an inbound call the caller is ``calling_number`` and the agent's own number is
    ``called_number``. On an outbound call it is the other way round.

    Args:
        direction: ``inbound`` or ``outbound``.
        remote_number: The other party's number: the caller, or the one dialled.
        local_number: The agent's own number, its trunk.
        channel: ``phone`` for a phone line, ``web`` to simulate one in a web session.
        carrier_call_id: The carrier's id of the call.
        agent_id: The agent's id. ``UNMUTE_AGENT_ID`` when None.
        session_id: The call's id. A fresh one when None.

    Raises:
        ValidationError: If the direction is neither, or an id is empty, too long or holds
            a comma, whitespace, a brace or a bar.
        SettingsError: If there is no agent id and ``UNMUTE_AGENT_ID`` is unset.

    Ported from unmute@868ae23 templates/livekit_v1/utils_telephony.py.tmpl:112-147.
    """
    outbound = direction == "outbound"
    given = {
        "direction": direction,
        "channel": channel,
        "calling_number": local_number if outbound else remote_number,
        "called_number": remote_number if outbound else local_number,
        "carrier_call_id": carrier_call_id or None,
        "agent_id": agent_id,
        "session_id": session_id,
    }
    return CallInfo.model_validate({name: value for name, value in given.items() if value})


def call_phone_region(call: CallInfo) -> str | None:
    """Return the country of the call's own numbers, for reading a national phone number.

    The caller's side first, then the agent's. Give it to ``use_phone_region``.

    Args:
        call: What the call carries, like ``state.call``.

    Returns:
        ISO 3166 alpha-2, like ``"ES"``, or None for a call without numbers, like a web call.
    """
    caller = call.calling_number if call.direction == "inbound" else call.called_number
    numbers = (caller, call.calling_number, call.called_number)
    return next((region for number in numbers if (region := region_of(number))), None)
