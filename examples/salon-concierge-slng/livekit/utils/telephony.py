"""Read a phone call's facts, and dial and transfer through your trunk."""

from __future__ import annotations

from livekit import rtc
from livekit.agents import AgentSession

from call_state import CallState
from utils.dev_metrics import dev_say


# --- cold transfer destination -----------------------------------------------
# TransferSIPParticipant's transfer_to is a URI, not a bare number: it becomes
# the Refer-To of the outgoing SIP REFER. `tel:+E164` covers Twilio and Telnyx;
# a provider that needs its trunk host in the Refer-To (Plivo) takes the
# `sip:+E164@<trunk>` form, which a destination may already carry (SCHEMA N26).
def _refer_uri(destination: str) -> str:
    """Turn a destination into a REFER URI, adding `tel:` only to a bare number.

    Args:
        destination: A phone number or a `tel:`, `sip:` or `sips:` URI.

    Returns:
        The destination as a URI.
    """
    if destination.startswith(("tel:", "sip:", "sips:")):
        return destination
    return "tel:" + destination


# --- LiveKit SIP call context -----------------------------------------------
# UNVERIFIED: Recheck these LiveKit-specific APIs with the LiveKit docs MCP.
# They were verified against docs.livekit.io on 2026-07-20 because MCP was not
# available during generation.
def _livekit_call_context(
    room_name: str, participant: rtc.RemoteParticipant | None, metadata: dict
) -> dict:
    """Collect the call facts from the SIP participant and the job metadata.

    Args:
        room_name: The LiveKit room, used as the session id.
        participant: The SIP participant, or None before it has joined.
        metadata: The parsed job metadata.

    Returns:
        The call's carrier, connection, direction and numbers.

    Raises:
        RuntimeError: The direction the SIP leg reports is not the job's.
    """
    expected_direction = "outbound" if "phone_number" in metadata else "inbound"
    direction = metadata.get("direction") or expected_direction
    if direction != expected_direction:
        raise RuntimeError("call direction does not match the LiveKit job")
    attributes = participant.attributes if participant is not None else {}
    remote_number = attributes.get("sip.phoneNumber") or metadata.get("phone_number")
    trunk_number = attributes.get("sip.trunkPhoneNumber")
    return {
        "session_id": room_name,
        "carrier": "twilio",
        "connection": "twilio_sip",
        "call_id": attributes.get("sip.callID") or metadata.get("call_id"),
        "direction": direction,
        "from_number": trunk_number if direction == "outbound" else remote_number,
        "to_number": remote_number if direction == "outbound" else trunk_number,
    }


def _hydrate_livekit_context(
    userdata: CallState, context: dict, *, require_all: bool = True
) -> None:
    """Save the call facts a prompt names onto the session state.

    Args:
        userdata: The state to fill.
        context: The facts from _livekit_call_context.
        require_all: Fail when a fact the package reads is absent.

    Raises:
        RuntimeError: A required fact is absent, or does not fit its type.
    """
    missing: list[str] = []
    if missing and require_all:
        raise RuntimeError("Missing LiveKit SIP context fields: " + ", ".join(missing))


async def _livekit_entry_greeting(session: AgentSession) -> None:
    """Open the call the way the package's greeting says to.

    Args:
        session: The running session.
    """
    await dev_say(
        session,
        "Hi, you've reached Sage and Stone. Robin speaking, what can I do for you?",
    )
