"""Read a phone call's facts and control the call at your carrier."""

from __future__ import annotations

import os
from typing import Any

import aiohttp
from loguru import logger
from pipecat.runner.types import RunnerArguments

from settings import CALL_REQUIRED_ENV, PHONE_SAMPLE_RATE_HZ

# --- the carrier stream ------------------------------------------------------
# Your carrier streams the call's audio straight to Pipecat Cloud, which starts
# this agent. Nothing of yours is in the path, so there is nothing here that
# answers a webhook or holds a call open. The runner has already parsed the
# carrier's handshake by the time run_bot starts; everything below reads that
# parsed result and never re-parses it.


def _phone_session(runner_args: RunnerArguments) -> dict[str, Any] | None:
    """Read the parsed carrier handshake, or None when this is not a phone call.

    The runner sets transport_type and call_data while building the transport, so
    this reads facts rather than deriving them. A browser or console session has
    neither and must behave exactly as it does on a package with no telephony.

    Args:
        runner_args: The arguments the runner started this session with.

    Returns:
        The call's handshake data, or None for a browser or console session.

    Raises:
        RuntimeError: When a phone call is missing an environment variable it needs.
    """
    if getattr(runner_args, "transport_type", None) != "twilio":
        return None
    missing = [name for name in CALL_REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Missing environment for a phone call: {', '.join(missing)}"
        )
    call = dict(getattr(runner_args, "call_data", None) or {})
    direction = (call.get("body") or {}).get("direction", "inbound")
    logger.info("phone call {} ({})", call.get("call_id"), direction)
    return call


def _pipeline_audio_rates(phone_call: dict[str, Any] | None) -> dict[str, Any]:
    """Choose the pipeline's sample rates: the carrier's own on a phone call only.

    Twilio Media Streams is 8 kHz mono in both directions. Pipecat's defaults are
    16 kHz in and 24 kHz out, so without this every inbound frame is upsampled and
    every outbound frame is downsampled, for the whole call, in exchange for no
    information at all: there is nothing above 4 kHz in the audio to recover. The
    transport's own guide asks for these two values for exactly that reason.

    A browser or console session on the same package keeps the defaults, where
    8 kHz would throw away quality those channels really have.

    Args:
        phone_call: The parsed carrier handshake, or None off a phone call.

    Returns:
        The two pipeline parameters to set, or an empty dict to keep the defaults.
    """
    if not phone_call:
        return {}
    return {
        "audio_in_sample_rate": PHONE_SAMPLE_RATE_HZ,
        "audio_out_sample_rate": PHONE_SAMPLE_RATE_HZ,
    }


def _transfer_twiml(destination: str, ring_timeout: int) -> str:
    """Build the whole transfer as one document your carrier will run in order.

    Announce, dial, then end the original call when the destination leg ends.
    Twilio moves to the next verb after a completed, declined, or unanswered
    dial, so the same final hangup covers every outcome without starting a new
    agent that has none of this call's context.

    Args:
        destination: The markup for whom to dial.
        ring_timeout: How many seconds the destination may ring.

    Returns:
        The TwiML document.
    """
    return (
        "<Response>"
        "<Say>Connecting you to a colleague now.</Say>"
        f'<Dial answerOnBridge="true" timeout="{ring_timeout}">{destination}</Dial>'
        "<Hangup/>"
        "</Response>"
    )


async def _update_carrier_call(call_id: str, twiml: str) -> None:
    """Replace the live call's document at your carrier, by its own call id.

    This is the one request in the whole project made in the carrier's own words,
    and it is call control rather than provisioning: the call already exists. One
    HTTPS request, so no carrier SDK.

    Args:
        call_id: The carrier's id for the live call.
        twiml: The document the call should run from now on.

    Raises:
        RuntimeError: When the carrier answers with a non-2xx status.
    """
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    base = "https://api.twilio.com"
    url = f"{base}/2010-04-01/Accounts/{account_sid}/Calls/{call_id}.json"
    auth = aiohttp.BasicAuth(account_sid, os.environ["TWILIO_AUTH_TOKEN"])
    async with (
        aiohttp.ClientSession() as session,
        session.post(url, auth=auth, data={"Twiml": twiml}) as response,
    ):
        if not 200 <= response.status < 300:
            raise RuntimeError(
                f"carrier call update failed with {response.status}: {await response.text()}"
            )
