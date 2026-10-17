"""``SipLine`` on LiveKit: REFER, warm transfer and outbound dial, against fakes."""

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from livekit import api, rtc
from livekit.agents import llm
from livekit.agents.beta.workflows import WarmTransferTask

from tests.livekit.fake import until
from tests.outputs import AnyResult
from unmute_core.livekit import BRIEFING_PERSONA, DialError, SipLine, _phone
from unmute_core.telephony import NoPhoneCaller, Transfer, TransferError

SIP = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
STANDARD = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
TRUNK = api.SIPOutboundConfig(hostname="trunk.example", auth_username="u", auth_password="p")


class FakeSpeech:
    """A line that has already played."""

    async def wait_for_playout(self) -> None:
        return None


class FakeSession:
    """The parts of an ``AgentSession`` a line uses."""

    def __init__(self) -> None:
        self.said: list[str] = []
        self.replies: list[str] = []
        self.listening: list[str] = []
        self.room_io = SimpleNamespace(
            set_participant=self.listening.append, linked_participant=None
        )
        self.llm = None
        self.current_agent = SimpleNamespace(chat_ctx=llm.ChatContext(), llm="agent-llm")

    def say(self, text: str, **_: Any) -> FakeSpeech:
        self.said.append(text)
        return FakeSpeech()

    def generate_reply(self, *, instructions: str) -> FakeSpeech:
        self.replies.append(instructions)
        return FakeSpeech()


class FakeSip:
    """``api.sip`` that records requests and answers as told."""

    def __init__(self, error: Exception | None = None, response: Any = None) -> None:
        self.error = error
        self.response = response or api.TransferSIPParticipantResponse()
        self.requests: list[Any] = []

    async def transfer_sip_participant(self, request: Any) -> Any:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response

    async def create_sip_participant(self, request: Any) -> Any:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return api.SIPParticipantInfo()


class FakeJob:
    """The job context: its room, its API, and how it shuts down."""

    def __init__(self, participants: dict[str, Any], sip: FakeSip) -> None:
        self.room = SimpleNamespace(name="call-1", remote_participants=participants)
        self.api = SimpleNamespace(sip=sip)
        self.shutdowns: list[str] = []

    async def wait_for_participant(self, *, identity: str) -> Any:
        return SimpleNamespace(identity=identity, kind=SIP)

    def shutdown(self, reason: str = "") -> None:
        self.shutdowns.append(reason)


def caller(identity: str = "sip_caller", kind: Any = SIP) -> Any:
    """A remote participant."""
    return SimpleNamespace(identity=identity, kind=kind)


@pytest.fixture
def job(monkeypatch: pytest.MonkeyPatch) -> FakeJob:
    """A job with a SIP caller and a web participant, as the line's job context."""
    fake = FakeJob({"web": caller("web", STANDARD), "sip": caller()}, FakeSip())
    monkeypatch.setattr(_phone, "get_job_context", lambda: fake)
    return fake


def sip_error(code: int) -> api.SipCallError:
    """The error LiveKit raises for a SIP status."""
    return api.SipCallError(
        "unavailable",
        "call failed",
        status=400,
        metadata={"sip_status_code": str(code), "sip_status": "Busy Here"},
    )


async def test_a_cold_transfer_refers_the_sip_caller(job: FakeJob) -> None:
    line = SipLine(FakeSession())  # ty: ignore[invalid-argument-type]
    await line.transfer(Transfer("t", "+34910000000", ring_timeout_s=20))
    (request,) = job.api.sip.requests
    assert request.room_name == "call-1"
    assert request.participant_identity == "sip_caller"
    assert request.transfer_to == "tel:+34910000000"
    assert request.play_dialtone
    assert request.ringing_timeout.seconds == 20


async def test_a_cold_transfer_with_no_sip_caller_says_so(job: FakeJob) -> None:
    job.room.remote_participants = {"web": caller("web", STANDARD)}
    with pytest.raises(NoPhoneCaller):
        await SipLine(FakeSession()).transfer(Transfer("t", "+34910000000"))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("sip", "reason"),
    [
        (FakeSip(error=sip_error(486)), "busy"),
        (FakeSip(error=TimeoutError()), "TimeoutError"),
        (
            FakeSip(
                response=api.TransferSIPParticipantResponse(
                    status=api.SIPTransferStatus.STS_TRANSFER_FAILED,
                    sip_status=api.SIPStatus(
                        code=api.SIPStatusCode.SIP_STATUS_TEMPORARILY_UNAVAILABLE
                    ),
                )
            ),
            "no answer",
        ),
    ],
)
async def test_a_refused_refer_is_a_failed_transfer(
    job: FakeJob, sip: FakeSip, reason: str
) -> None:
    job.api.sip = sip
    with pytest.raises(TransferError) as raised:
        await SipLine(FakeSession()).transfer(Transfer("t", "+34910000000"))  # ty: ignore[invalid-argument-type]
    assert raised.value.reason == reason


class FakeWarm:
    """``WarmTransferTask``: records how it was built, then merges or fails."""

    built: ClassVar[list[dict[str, Any]]] = []
    error: ClassVar[Exception | None] = None

    def __init__(self, sip_call_to: str, **options: Any) -> None:
        inspect.signature(WarmTransferTask).bind(sip_call_to, **options)
        if FakeWarm.error is not None and isinstance(FakeWarm.error, ValueError):
            raise FakeWarm.error
        FakeWarm.built.append({"sip_call_to": sip_call_to, **options})

    def __await__(self) -> Any:
        async def run() -> Any:
            if FakeWarm.error is not None:
                raise FakeWarm.error
            return SimpleNamespace(human_agent_identity="human-agent-sip")

        return run().__await__()


@pytest.fixture
def warm(monkeypatch: pytest.MonkeyPatch) -> type[FakeWarm]:
    """``WarmTransferTask`` replaced by ``FakeWarm``."""
    FakeWarm.built, FakeWarm.error = [], None
    monkeypatch.setattr(_phone, "WarmTransferTask", FakeWarm)
    return FakeWarm


async def test_a_warm_transfer_briefs_then_says_the_merge(warm: type[FakeWarm]) -> None:
    session = FakeSession()
    line = SipLine(session, trunk=TRUNK, caller_id="+34900000000")  # ty: ignore[invalid-argument-type]
    transfer = Transfer(
        "escalate",
        "sip:+34910000000@trunk.example",
        mode="warm",
        briefing="VIP.",
        ring_timeout_s=15,
    )
    await line.transfer(transfer)
    (built,) = warm.built
    assert built["sip_call_to"] == "+34910000000"
    assert built["sip_connection"] is TRUNK
    assert built["sip_number"] == "+34900000000"
    assert built["ringing_timeout"] == 15
    assert built["chat_ctx"] is session.current_agent.chat_ctx
    assert built["instructions"].persona == BRIEFING_PERSONA
    assert built["instructions"].extra == "VIP."
    assert session.said == [transfer.connected_line]


async def test_a_warm_transfer_nobody_takes_is_a_failed_transfer(warm: type[FakeWarm]) -> None:
    warm.error = llm.ToolError("voicemail detected")
    session = FakeSession()
    line = SipLine(session, trunk="ST_trunk")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TransferError) as raised:
        await line.transfer(Transfer("escalate", "+34910000000", mode="warm"))
    assert raised.value.reason == "voicemail detected"
    assert warm.built[0]["sip_trunk_id"] == "ST_trunk"
    assert session.said == []


class FakeDetector:
    """``AMD``: holds the agent's speech, then gives a verdict."""

    verdict = "human"
    used = False
    llm: Any = None

    def __init__(self, session: Any, *, llm: Any, participant_identity: str) -> None:
        self.identity = participant_identity
        FakeDetector.llm = llm

    async def __aenter__(self) -> "FakeDetector":
        FakeDetector.used = True
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self) -> Any:
        return SimpleNamespace(category=FakeDetector.verdict)


@pytest.fixture
def amd(monkeypatch: pytest.MonkeyPatch) -> type[FakeDetector]:
    """``AMD`` replaced by ``FakeDetector``."""
    FakeDetector.verdict, FakeDetector.used = "human", False
    monkeypatch.setattr(_phone, "AMD", FakeDetector)
    return FakeDetector


async def test_a_dial_that_a_person_answers(job: FakeJob, amd: type[FakeDetector]) -> None:
    session = FakeSession()
    line = SipLine(session, trunk=TRUNK, caller_id="+34900000000")  # ty: ignore[invalid-argument-type]
    dialled = await line.dial("+34910000000", ring_timeout_s=40)
    assert (dialled.verdict, dialled.ended) == ("human", False)
    assert amd.llm == "agent-llm"
    assert dialled.participant.identity == "phone_user"
    assert session.listening == ["phone_user"]
    (request,) = job.api.sip.requests
    assert request.sip_call_to == "+34910000000"
    assert request.wait_until_answered
    assert request.trunk.hostname == "trunk.example"
    assert request.sip_number == "+34900000000"
    assert request.ringing_timeout.seconds == 40
    assert job.shutdowns == []


@pytest.mark.parametrize(
    ("verdict", "on_voicemail", "replies", "shutdown"),
    [
        ("machine-vm", "hangup", [], "voicemail detected"),
        ("machine-vm", "leave_message", [_phone.VOICEMAIL_MESSAGE], "voicemail: left a message"),
        ("machine-unavailable", "leave_message", [], "mailbox unavailable"),
    ],
)
async def test_voicemail_hangs_up_or_leaves_a_message(
    job: FakeJob,
    amd: type[FakeDetector],
    verdict: str,
    on_voicemail: Any,
    replies: list[str],
    shutdown: str,
) -> None:
    amd.verdict = verdict
    session = FakeSession()
    dialled = await SipLine(session, trunk="ST_trunk").dial(  # ty: ignore[invalid-argument-type]
        "+34910000000", on_voicemail=on_voicemail
    )
    assert dialled.ended
    assert session.replies == replies
    assert job.shutdowns == [shutdown]
    assert job.api.sip.requests[0].sip_trunk_id == "ST_trunk"


async def test_a_dial_without_detection_skips_it(job: FakeJob, amd: type[FakeDetector]) -> None:
    dialled = await SipLine(FakeSession()).dial("+34910000000", detect_voicemail=False)  # ty: ignore[invalid-argument-type]
    assert dialled.verdict is None
    assert not amd.used


async def test_dial_options_reach_the_request_only_when_given(job: FakeJob) -> None:
    await SipLine(FakeSession()).dial("+34910000000", detect_voicemail=False)  # ty: ignore[invalid-argument-type]
    await SipLine(FakeSession()).dial(  # ty: ignore[invalid-argument-type]
        "+34910000000",
        detect_voicemail=False,
        participant_name="Ana",
        display_name="Salon Sol",
        metadata="m",
        attributes={"k": "v"},
        krisp=True,
    )
    plain, full = job.api.sip.requests
    assert (plain.participant_name, plain.display_name, plain.participant_metadata) == ("", "", "")
    assert not plain.participant_attributes
    assert not plain.krisp_enabled
    assert (full.participant_name, full.display_name, full.participant_metadata) == (
        "Ana",
        "Salon Sol",
        "m",
    )
    assert dict(full.participant_attributes) == {"k": "v"}
    assert full.krisp_enabled


class FakeRoom:
    """A room that keeps its event handlers, and can emit a subscribed audio track."""

    def __init__(self, name: str = "call-1") -> None:
        self.name = name
        self.handlers: dict[str, list[Any]] = {}

    def on(self, event: str, handler: Any) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Any) -> None:
        self.handlers[event].remove(handler)

    def emit_audio(self, identity: str) -> None:
        track = SimpleNamespace(kind=rtc.TrackKind.KIND_AUDIO)
        for handler in list(self.handlers.get("track_subscribed", [])):
            handler(track, None, SimpleNamespace(identity=identity))


def waiting_job(job: FakeJob, publications: list[Any], monkeypatch: pytest.MonkeyPatch) -> FakeRoom:
    """Give the job a room with events, and a callee holding the given publications."""
    room = FakeRoom()
    job.room = room  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(_phone, "AUDIO_SETTLE_S", 0)

    async def join(*, identity: str) -> Any:
        return SimpleNamespace(
            identity=identity,
            kind=SIP,
            track_publications={str(n): pub for n, pub in enumerate(publications)},
        )

    job.wait_for_participant = join  # ty: ignore[invalid-assignment]
    return room


async def test_a_dial_waits_for_the_callees_audio_track(
    job: FakeJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = waiting_job(job, [], monkeypatch)
    dial = asyncio.create_task(
        SipLine(FakeSession()).dial(  # ty: ignore[invalid-argument-type]
            "+34910000000", detect_voicemail=False, audio_wait_s=5
        )
    )
    await until(lambda: room.handlers.get("track_subscribed"))
    assert not dial.done()
    room.emit_audio("phone_user")
    assert (await dial).participant.identity == "phone_user"
    assert room.handlers["track_subscribed"] == []


async def test_a_dial_does_not_wait_for_a_track_already_subscribed(
    job: FakeJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = SimpleNamespace(kind=rtc.TrackKind.KIND_AUDIO, subscribed=True)
    waiting_job(job, [ready], monkeypatch)
    dialled = await asyncio.wait_for(
        SipLine(FakeSession()).dial(  # ty: ignore[invalid-argument-type]
            "+34910000000", detect_voicemail=False, audio_wait_s=5
        ),
        1,
    )
    assert dialled.verdict is None


async def test_a_dial_goes_on_when_no_audio_arrives(
    job: FakeJob, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    waiting_job(job, [], monkeypatch)
    dialled = await SipLine(FakeSession()).dial(  # ty: ignore[invalid-argument-type]
        "+34910000000", detect_voicemail=False, audio_wait_s=0.05
    )
    assert dialled.participant.identity == "phone_user"
    assert "no audio from phone_user" in caplog.text


async def test_detection_ignores_the_audio_wait(
    job: FakeJob, amd: type[FakeDetector], monkeypatch: pytest.MonkeyPatch
) -> None:
    waiting_job(job, [], monkeypatch)
    dialled = await asyncio.wait_for(
        SipLine(FakeSession()).dial("+34910000000", audio_wait_s=60),  # ty: ignore[invalid-argument-type]
        1,
    )
    assert dialled.verdict == "human"


async def test_a_busy_number_is_a_dial_error(job: FakeJob, amd: type[FakeDetector]) -> None:
    job.api.sip = FakeSip(error=sip_error(486))
    with pytest.raises(DialError) as raised:
        await SipLine(FakeSession()).dial("+34910000000")  # ty: ignore[invalid-argument-type]
    assert (raised.value.reason, raised.value.sip_status) == ("busy", 486)


async def test_a_dial_needs_an_e164_number(job: FakeJob) -> None:
    with pytest.raises(ValueError, match=r"E\.164"):
        await SipLine(FakeSession()).dial("910000000")  # ty: ignore[invalid-argument-type]


async def test_the_detector_takes_the_model_it_is_given(
    job: FakeJob, amd: type[FakeDetector]
) -> None:
    session = FakeSession()
    session.llm = "session-llm"
    await SipLine(session).dial("+34910000000")  # ty: ignore[invalid-argument-type]
    assert amd.llm == "session-llm"
    await SipLine(session).dial("+34910000000", detector_llm="openai/gpt-4.1-mini")  # ty: ignore[invalid-argument-type]
    assert amd.llm == "openai/gpt-4.1-mini"


async def test_the_sessions_own_sip_participant_is_referred_first(job: FakeJob) -> None:
    session = FakeSession()
    session.room_io.linked_participant = caller("phone_user")
    await SipLine(session).transfer(Transfer("t", "+34910000000"))  # ty: ignore[invalid-argument-type]
    assert job.api.sip.requests[0].participant_identity == "phone_user"


async def test_a_warm_transfer_with_no_trunk_is_a_failed_transfer(warm: type[FakeWarm]) -> None:
    warm.error = ValueError("`sip_trunk_id` must be provided")
    with pytest.raises(TransferError, match="no trunk"):
        await SipLine(FakeSession()).transfer(Transfer("w", "+34910000000", mode="warm"))  # ty: ignore[invalid-argument-type]


async def test_a_continue_tool_turns_interruptions_off_on_livekit() -> None:
    from unmute_core.livekit import function_tools
    from unmute_core.telephony import transfer_tool
    from unmute_core.tools import ToolRegistry, local_tool

    class Line:
        warm: ClassVar[bool] = False

        async def transfer(self, transfer: Transfer) -> None:
            return None

        async def say(self, line: str) -> None:
            return None

    def lookup(order_id: str) -> dict[str, str]:
        """Look up an order."""
        return {"order_id": order_id}

    registry = ToolRegistry(
        [transfer_tool(Transfer("t", "+34910000000"), Line()), local_tool(lookup, output=AnyResult)]
    )
    disallowed: list[str] = []
    session = FakeSession()
    session.shutdown = lambda: None  # ty: ignore[unresolved-attribute]
    session.once = lambda name, handler: None  # ty: ignore[unresolved-attribute]
    for tool in function_tools(registry):
        name = tool.info.name
        context = SimpleNamespace(
            session=session,
            function_call=SimpleNamespace(name=name),
            disallow_interruptions=lambda name=name: disallowed.append(name),
        )
        arguments = {"order_id": "A-1"} if name == "lookup" else {}
        await tool(arguments, context)
    assert disallowed == ["t"]
