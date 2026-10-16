"""Live telephony checks against real accounts. Skipped unless you set their variables.

Each one places a real call to a phone you name and costs a few cents. Answer it, then
hear it move to the transfer number. Run them one at a time::

    UNMUTE_LIVE_DIAL_NUMBER=+34... UNMUTE_LIVE_TRANSFER_NUMBER=+34... \\
    LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... \\
    UNMUTE_LIVE_SIP_ADDRESS=... UNMUTE_LIVE_SIP_USERNAME=... UNMUTE_LIVE_SIP_PASSWORD=... \\
    UNMUTE_LIVE_FROM_NUMBER=+34... \\
    uv run pytest -m live tests/live -k livekit -s

- ``livekit``: ``SipLine.dial`` through your trunk, then a SIP REFER to the transfer number.
- ``twilio``: a Twilio call that waits, then ``TwilioLine`` moves it to the transfer number.
  Needs ``UNMUTE_LIVE_TWILIO_ACCOUNT_SID`` and ``UNMUTE_LIVE_TWILIO_AUTH_TOKEN``.
- ``telnyx``: a Telnyx call, then ``TelnyxLine``'s transfer command. Needs
  ``UNMUTE_LIVE_TELNYX_API_KEY`` and ``UNMUTE_LIVE_TELNYX_CONNECTION_ID``.
"""

import os
import uuid
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from unmute_core.telephony import Transfer

pytestmark = pytest.mark.live

COMMON = ("UNMUTE_LIVE_DIAL_NUMBER", "UNMUTE_LIVE_TRANSFER_NUMBER", "UNMUTE_LIVE_FROM_NUMBER")
WAIT_TWIML = (
    "<Response><Say>Unmute core live test. You will be transferred.</Say>"
    '<Pause length="40"/></Response>'
)


def need(*names: str) -> dict[str, str]:
    """Return the variables, or skip the test when one is unset."""
    missing = [name for name in (*COMMON, *names) if not os.environ.get(name)]
    if missing:
        pytest.skip(f"set {', '.join(missing)} to run this live test")
    return {name: os.environ[name] for name in (*COMMON, *names)}


def transfer(env: dict[str, str]) -> Transfer:
    """The transfer every live test makes."""
    return Transfer("live_transfer", env["UNMUTE_LIVE_TRANSFER_NUMBER"], ring_timeout_s=30)


class Worker:
    """A worker that only takes the lines a failed transfer would say."""

    async def queue_frame(self, frame: Any) -> None:
        return None


def worker() -> Any:
    """A ``Worker``, typed as the pipeline worker a line takes."""
    return Worker()


async def test_livekit_dials_then_refers(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("livekit.agents")
    from livekit import api, rtc

    from unmute_core.livekit import SipLine, _phone

    env = need(
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "UNMUTE_LIVE_SIP_ADDRESS",
        "UNMUTE_LIVE_SIP_USERNAME",
        "UNMUTE_LIVE_SIP_PASSWORD",
    )
    client = api.LiveKitAPI()
    room = f"unmute-live-{uuid.uuid4().hex[:8]}"
    participants: dict[str, Any] = {}

    async def wait_for_participant(*, identity: str) -> Any:
        participants[identity] = SimpleNamespace(
            identity=identity, kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        )
        return participants[identity]

    job = SimpleNamespace(
        room=SimpleNamespace(name=room, remote_participants=participants),
        api=client,
        wait_for_participant=wait_for_participant,
        shutdown=lambda reason="": None,
    )
    monkeypatch.setattr(_phone, "get_job_context", lambda: job)
    trunk = api.SIPOutboundConfig(
        hostname=env["UNMUTE_LIVE_SIP_ADDRESS"],
        auth_username=env["UNMUTE_LIVE_SIP_USERNAME"],
        auth_password=env["UNMUTE_LIVE_SIP_PASSWORD"],
    )
    session: Any = SimpleNamespace(room_io=None)
    line = SipLine(session, trunk=trunk, caller_id=env["UNMUTE_LIVE_FROM_NUMBER"])
    try:
        await client.room.create_room(api.CreateRoomRequest(name=room))
        dialled = await line.dial(
            env["UNMUTE_LIVE_DIAL_NUMBER"], ring_timeout_s=30, detect_voicemail=False
        )
        assert dialled.participant.identity == "phone_user"
        await line.transfer(transfer(env))
    finally:
        await client.room.delete_room(api.DeleteRoomRequest(room=room))
        await client.aclose()


async def test_twilio_moves_a_live_call() -> None:
    pytest.importorskip("pipecat")
    from unmute_core.pipecat import TwilioLine

    env = need("UNMUTE_LIVE_TWILIO_ACCOUNT_SID", "UNMUTE_LIVE_TWILIO_AUTH_TOKEN")
    sid = env["UNMUTE_LIVE_TWILIO_ACCOUNT_SID"]
    auth = (sid, env["UNMUTE_LIVE_TWILIO_AUTH_TOKEN"])
    calls = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls"
    async with httpx.AsyncClient(auth=auth, timeout=20) as client:
        placed = await client.post(
            f"{calls}.json",
            data={
                "To": env["UNMUTE_LIVE_DIAL_NUMBER"],
                "From": env["UNMUTE_LIVE_FROM_NUMBER"],
                "Twiml": WAIT_TWIML,
            },
        )
        placed.raise_for_status()
        call_sid = placed.json()["sid"]
        try:
            line = TwilioLine(
                call_sid, account_sid=sid, auth_token=auth[1], worker=worker(), serializer=None
            )
            input("Answer the call, then press Enter to transfer it... ")
            await line.transfer(transfer(env))
            input("Check the call moved to the transfer number, then press Enter... ")
        finally:
            await client.post(f"{calls}/{call_sid}.json", data={"Status": "completed"})


async def test_telnyx_moves_a_live_call() -> None:
    pytest.importorskip("pipecat")
    from unmute_core.pipecat import TelnyxLine

    env = need("UNMUTE_LIVE_TELNYX_API_KEY", "UNMUTE_LIVE_TELNYX_CONNECTION_ID")
    headers = {"Authorization": f"Bearer {env['UNMUTE_LIVE_TELNYX_API_KEY']}"}
    async with httpx.AsyncClient(headers=headers, timeout=20) as client:
        placed = await client.post(
            "https://api.telnyx.com/v2/calls",
            json={
                "connection_id": env["UNMUTE_LIVE_TELNYX_CONNECTION_ID"],
                "to": env["UNMUTE_LIVE_DIAL_NUMBER"],
                "from": env["UNMUTE_LIVE_FROM_NUMBER"],
            },
        )
        placed.raise_for_status()
        call_id = placed.json()["data"]["call_control_id"]
        try:
            line = TelnyxLine(
                call_id, api_key=env["UNMUTE_LIVE_TELNYX_API_KEY"], worker=worker(), serializer=None
            )
            input("Answer the call, then press Enter to transfer it... ")
            await line.transfer(transfer(env))
            input("Check the call moved to the transfer number, then press Enter... ")
        finally:
            await client.post(f"https://api.telnyx.com/v2/calls/{call_id}/actions/hangup")
