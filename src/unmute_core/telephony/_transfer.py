"""A transfer to a person, as a tool the model calls, on any phone line.

Ported from unmute@2c78a5a templates/livekit_v1/agents.py.tmpl:355-545 and
templates/pipecat_v1/agents.py.tmpl:386-533. The model reads the same texts on both
frameworks. The tool says its own line before the transfer. Left to the prompt, a model
can say "putting you through" and never call the tool.
"""

import logging
import time
import weakref
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal, Protocol, Self

from pydantic import AfterValidator, BaseModel, Field, model_validator
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.call import AGENT_ENDED, TRANSFERRED
from unmute_core.state import CallState
from unmute_core.telephony._numbers import check_destination, sip_user
from unmute_core.tools import ToolError, ToolSpec

logger = logging.getLogger(__name__)

TransferMode = Literal["cold", "warm"]
"""``cold`` hands the caller's leg over. ``warm`` calls the person first and briefs them."""
OnUnavailable = Literal["return_to_caller", "hangup"]
"""What a transfer nobody takes does: the agent keeps helping, or the call ends."""
OnVoicemail = Literal["hangup", "leave_message"]
"""What an outbound call that reaches voicemail does."""

VOICEMAIL_MESSAGE = "You reached voicemail. Leave a brief message and say you will call back."
"""What the model hears when it should leave a message."""
NO_PHONE_CALLER = (
    "There is no phone caller to transfer, because this session did not arrive by phone. "
    "Tell the caller you cannot put them through and keep helping them."
)
"""What the model reads when the call has no phone leg, like a web call."""
_UNAVAILABLE = MappingProxyType(
    {
        "cold": "The transfer failed ({reason}). Tell the caller and keep helping them.",
        "warm": "Nobody took the transfer ({reason}). Tell the caller and keep helping them.",
    }
)
_ANNOUNCE = MappingProxyType(
    {
        "cold": "Putting you through now, one moment.",
        "warm": "One moment while I bring a colleague on the line.",
    }
)
CONNECTED_LINE = "You are on the line with my colleague now. I will leave you to it."
"""Said once a warm transfer has merged the calls."""
IN_PROGRESS = "A transfer is already under way. Do not start another."
"""What the model reads when it starts a second transfer on the same call."""
UNAVAILABLE_LINE = "I'm sorry, nobody can take your call right now. Goodbye."
"""Said before hanging up when nobody takes a transfer that ends the call."""
_NO_ARGUMENTS: Mapping[str, Any] = MappingProxyType(
    {"type": "object", "properties": {}, "additionalProperties": False}
)


class NoPhoneCaller(Exception):  # noqa: N818 - it names the situation, like ToolRefused
    """The call has no phone leg to transfer, like a web call."""


class TransferError(Exception):
    """Nobody took the transfer, or the carrier refused it. The caller is still on the call.

    Attributes:
        reason: Why, in words the model can pass on, like ``busy``.
    """

    def __init__(self, reason: str) -> None:
        """Keep the reason."""
        super().__init__(reason)
        self.reason = reason


@original_errors
@dataclass(frozen=True, config=VALUE)
class Transfer:
    """A transfer to a person: where to, and what happens when nobody takes it.

    Attributes:
        name: The tool's name, like ``send_to_billing``.
        destination: An E.164 number, or a ``tel:``, ``sip:`` or ``sips:`` URI.
        description: When the model should call it.
        mode: ``cold`` or ``warm``.
        briefing: Extra words for the person a warm transfer calls, after the chat.
        ring_timeout_s: How long the destination may ring. None uses the carrier's default.
        on_unavailable: ``return_to_caller`` lets the agent keep helping; ``hangup`` says
            ``unavailable_line`` and ends the call.
        announce: Said before the transfer starts. The default is the mode's line.
        connected_line: Said once a warm transfer has merged the calls.
        unavailable_line: Said before hanging up, with ``on_unavailable="hangup"``. Write
            it in the call's language.
    """

    name: str
    destination: Annotated[str, AfterValidator(check_destination)]
    description: str = "Transfer the caller to a human agent."
    mode: TransferMode = "cold"
    briefing: str = ""
    ring_timeout_s: Annotated[float, Field(gt=0)] | None = None
    on_unavailable: OnUnavailable = "return_to_caller"
    announce: str | None = None
    connected_line: str = CONNECTED_LINE
    unavailable_line: str = UNAVAILABLE_LINE

    @model_validator(mode="after")
    def _fits_the_mode(self) -> Self:
        """Check the fields that depend on the mode.

        Raises:
            ValueError: If ``briefing`` comes with a cold transfer, or a warm transfer
                names no one to call.
        """
        if self.briefing and self.mode != "warm":
            raise ValueError(f"transfer {self.name}: briefing is legal with a warm transfer only")
        if self.mode == "warm" and not sip_user(self.destination):
            raise ValueError(f"transfer {self.name}: {self.destination!r} names no one to call")
        return self

    @property
    def spoken(self) -> str:
        """The line said before the transfer starts."""
        return self.announce if self.announce is not None else _ANNOUNCE[self.mode]


class PhoneLine(Protocol):
    """The phone leg of one call: what a transfer acts on. Build one per call.

    The adapters ship one per route: ``unmute_core.livekit.SipLine``, and
    ``unmute_core.pipecat.TwilioLine``, ``TelnyxLine`` and ``DailySipLine``. A line of
    your own must be hashable and weak-referenceable, as a plain class is, so the library
    can tell which lines have a transfer under way without keeping them alive.

    Attributes:
        warm: True when the line can make a warm transfer.
    """

    warm: ClassVar[bool]

    async def transfer(self, transfer: Transfer) -> None:
        """Hand the caller over. Return once the transfer is done, or accepted.

        Raises:
            NoPhoneCaller: If the call has no phone leg.
            TransferError: If nobody took it. The caller is still on the call.
        """
        ...

    async def say(self, line: str) -> None:
        """Say a line to the caller, before the call ends."""
        ...


_BUSY: weakref.WeakSet[PhoneLine] = weakref.WeakSet()
"""The lines with a transfer under way, or done. A line serves one call."""


class _TransferCall:
    """Runs a transfer on the call's line, and tells the model how it went.

    One transfer at a time per line: a model may call the tool twice in one reply, or a
    second transfer tool while the first runs. Each would call the destination again.
    """

    def __init__(self, transfer: Transfer, line: PhoneLine) -> None:
        """Keep the transfer and the call's line."""
        self._transfer = transfer
        self._line = line

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        """Transfer the caller.

        Returns:
            ``{"transferred": true}``, or, with ``on_unavailable="hangup"``, a result that
            ends the call once the line has been said.

        Raises:
            ToolError: When the call has no phone leg, or nobody took the transfer and the
                agent keeps helping.
        """
        transfer = self._transfer
        if self._line in _BUSY:
            raise ToolError(IN_PROGRESS)
        _BUSY.add(self._line)
        logger.info("%s transfer %s starting", transfer.mode, transfer.name)
        began = time.monotonic()
        try:
            await self._line.transfer(transfer)
        except NoPhoneCaller:
            _BUSY.discard(self._line)
            logger.info("%s transfer %s skipped: no phone caller", transfer.mode, transfer.name)
            raise ToolError(NO_PHONE_CALLER) from None
        except TransferError as failed:
            logger.info(
                "%s transfer %s unavailable after %.0f s: %s",
                transfer.mode,
                transfer.name,
                time.monotonic() - began,
                failed.reason,
            )
            if transfer.on_unavailable == "return_to_caller":
                _BUSY.discard(self._line)
                raise ToolError(_UNAVAILABLE[transfer.mode].format(reason=failed.reason)) from None
            await self._line.say(transfer.unavailable_line)
            return {"transferred": False, "ended": True, "reason": failed.reason}
        except BaseException:
            _BUSY.discard(self._line)
            raise
        logger.info(
            "%s transfer %s done after %.0f s",
            transfer.mode,
            transfer.name,
            time.monotonic() - began,
        )
        return {"transferred": True}


class _Transferred(BaseModel):
    """What a transfer returns: whether the caller was handed over, or why the call ended."""

    transferred: bool
    ended: bool = False
    reason: str | None = None


def _end_reason(result: object) -> str:
    """Return ``transferred`` for a transfer that went through, else the agent's own end."""
    handed_over = isinstance(result, Mapping) and result.get("transferred") is True
    return TRANSFERRED if handed_over else AGENT_ENDED


def transfer_tool(transfer: Transfer, line: PhoneLine) -> ToolSpec:
    """Return a tool that transfers the caller on this call's line. Build it per call.

    The tool takes no arguments: the model never sees the destination. It says the
    announce line first, then transfers, and ends the call once the caller is handed
    over. The call then ends as ``transferred``. The caller's speech never cancels a transfer under way. A second transfer on
    the same call is refused: the model reads that one is already under way.

    Args:
        transfer: Where to, and what happens when nobody takes it.
        line: This call's phone line.

    Raises:
        TypeError: If the line is not hashable or not weak-referenceable.
        ValueError: If the transfer is warm and the line cannot make one.
    """
    try:
        weakref.ref(line)
        hash(line)
    except TypeError:
        raise TypeError(
            f"{type(line).__name__} must be hashable and weak-referenceable to be a PhoneLine: "
            "use a plain class, or add __weakref__ to its __slots__ and keep __hash__"
        ) from None
    if transfer.mode == "warm" and not getattr(line, "warm", False):
        raise ValueError(
            f"transfer {transfer.name} is warm, and {type(line).__name__} has no warm "
            "transfer: run it on LiveKit SIP (SipLine), or make it cold"
        )
    return ToolSpec(
        transfer.name,
        transfer.description,
        dict(_NO_ARGUMENTS),
        _Transferred,
        _TransferCall(transfer, line),
        announce=transfer.spoken,
        announce_wait=True,
        effect="ends_conversation",
        interruption="continue",
        ends_as=_end_reason,
    )
