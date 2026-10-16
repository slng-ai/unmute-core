"""``SipLine`` through LiveKit's real API client, against a local server that answers as LiveKit.

The fakes in ``test_phone.py`` stand in for ``api.sip``. Here the request goes through
``livekit.api.LiveKitAPI`` itself: the Twirp path, the signed token with its SIP grant,
the protobuf body, and a SIP failure as LiveKit sends it.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from aiohttp import web
from livekit import api, rtc

from unmute_core.livekit import DialError, SipLine, _phone
from unmute_core.telephony import Transfer, TransferError

SECRET = "s" * 32


@dataclass
class LiveKitServer:
    """A local LiveKit SIP service: it keeps each request and answers with ``error``."""

    url: str = ""
    requests: list[tuple[str, dict[str, Any], bytes]] = field(default_factory=list)
    error: dict[str, Any] | None = None

    async def handle(self, request: web.Request) -> web.Response:
        """Answer one Twirp call."""
        body = await request.read()
        self.requests.append((request.path, dict(request.headers), body))
        if self.error is not None:
            return web.json_response(self.error, status=400)
        method = request.path.rsplit("/", 1)[1]
        answer = {
            "TransferSIPParticipant": api.TransferSIPParticipantResponse(),
            "CreateSIPParticipant": api.SIPParticipantInfo(participant_identity="phone_user"),
        }[method]
        return web.Response(body=answer.SerializeToString(), content_type="application/protobuf")


@pytest.fixture
async def livekit(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[LiveKitServer]:
    """The server, and a job whose API is a real ``LiveKitAPI`` pointed at it."""
    server = LiveKitServer()
    app = web.Application()
    app.router.add_post("/twirp/{method:.*}", server.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    client = api.LiveKitAPI(f"http://127.0.0.1:{port}", "key", SECRET, failover=False)
    job = SimpleNamespace(
        room=SimpleNamespace(
            name="call-1",
            remote_participants={
                "sip": SimpleNamespace(
                    identity="sip_caller", kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                )
            },
        ),
        api=client,
        shutdowns=[],
    )

    async def wait_for_participant(*, identity: str) -> Any:
        return SimpleNamespace(identity=identity)

    job.wait_for_participant = wait_for_participant
    job.shutdown = job.shutdowns.append
    monkeypatch.setattr(_phone, "get_job_context", lambda: job)
    yield server
    await client.aclose()
    await runner.cleanup()


def session() -> Any:
    """The parts of a session ``SipLine`` reads."""
    return SimpleNamespace(
        room_io=SimpleNamespace(set_participant=lambda identity: None, linked_participant=None)
    )


def grants(headers: dict[str, Any]) -> dict[str, Any]:
    """Return the claims of the request's signed token."""
    token = headers["Authorization"].removeprefix("Bearer ")
    return jwt.decode(token, SECRET, algorithms=["HS256"], options={"verify_aud": False})


async def test_a_refer_reaches_livekit_as_a_signed_twirp_call(livekit: LiveKitServer) -> None:
    await SipLine(session()).transfer(Transfer("t", "+34910000000", ring_timeout_s=20))
    ((path, headers, body),) = livekit.requests
    assert path == "/twirp/livekit.SIP/TransferSIPParticipant"
    request = api.TransferSIPParticipantRequest.FromString(body)
    assert (request.room_name, request.participant_identity) == ("call-1", "sip_caller")
    assert request.transfer_to == "tel:+34910000000"
    assert request.ringing_timeout.seconds == 20
    claims = grants(headers)
    assert claims["sip"]["call"] is True
    assert claims["video"]["roomAdmin"] is True


async def test_a_busy_refer_as_livekit_reports_it_is_a_failed_transfer(
    livekit: LiveKitServer,
) -> None:
    livekit.error = {
        "code": "unavailable",
        "msg": "transfer failed",
        "meta": {"sip_status_code": "486", "sip_status": "Busy Here"},
    }
    with pytest.raises(TransferError) as raised:
        await SipLine(session()).transfer(Transfer("t", "+34910000000"))
    assert raised.value.reason == "busy"


async def test_a_dial_reaches_livekit_with_the_inline_trunk(livekit: LiveKitServer) -> None:
    trunk = api.SIPOutboundConfig(hostname="trunk.example", auth_username="u", auth_password="p")
    line = SipLine(session(), trunk=trunk, caller_id="+34900000000")
    dialled = await line.dial("+34910000000", detect_voicemail=False)
    assert dialled.participant.identity == "phone_user"
    ((path, _headers, body),) = livekit.requests
    assert path == "/twirp/livekit.SIP/CreateSIPParticipant"
    request = api.CreateSIPParticipantRequest.FromString(body)
    assert (request.sip_call_to, request.sip_number) == ("+34910000000", "+34900000000")
    assert request.trunk.hostname == "trunk.example"
    assert request.wait_until_answered


async def test_a_dial_nobody_answers_as_livekit_reports_it(livekit: LiveKitServer) -> None:
    livekit.error = {
        "code": "unavailable",
        "msg": "no answer",
        "meta": {"sip_status_code": "480", "sip_status": "Temporarily Unavailable"},
    }
    with pytest.raises(DialError) as raised:
        await SipLine(session()).dial("+34910000000", detect_voicemail=False)
    assert (raised.value.reason, raised.value.sip_status) == ("no answer", 480)
