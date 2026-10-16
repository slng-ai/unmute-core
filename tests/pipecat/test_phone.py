"""Pipecat phone lines: Twilio and Telnyx over REST, Daily over SIP, against fakes."""

import asyncio
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest
from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer

from unmute_core.call import CALLER_HUNG_UP, CallHooks
from unmute_core.pipecat import (
    DailySipLine,
    TelnyxLine,
    TwilioLine,
    _phone,
    dial_out,
    disarm_hang_up,
    forward_dial_in,
    transfer_twiml,
)
from unmute_core.telephony import NoPhoneCaller, Transfer, TransferError, transfer_tool

BILLING = Transfer("send_to_billing", "+34910000000", ring_timeout_s=20)


def new_worker() -> Any:
    """A ``FakeWorker``, typed as the worker a line takes."""
    return FakeWorker()


class FakeWorker:
    """The parts of a ``PipelineWorker`` a line uses."""

    def __init__(self) -> None:
        self.frames: list[Any] = []
        self.cancelled: list[str | None] = []

    async def queue_frame(self, frame: Any) -> None:
        self.frames.append(frame)

    async def cancel(self, *, reason: str | None = None) -> None:
        self.cancelled.append(reason)


class Carrier:
    """A carrier's REST API: every request, answered with ``status``.

    ``armed`` keeps whether a watched serializer could still hang up as each request
    arrived.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status = 200
        self.watched: Any = None
        self.armed: list[bool] = []

    def answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.watched is not None:
            self.armed.append(not self.watched._hangup_attempted)
        return httpx.Response(self.status, json={})


@pytest.fixture
def carrier(monkeypatch: pytest.MonkeyPatch) -> Carrier:
    """The carrier the lines reach, in place of the network."""
    fake = Carrier()
    real = httpx.AsyncClient

    def client(**options: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(fake.answer), **options)

    monkeypatch.setattr(_phone.httpx, "AsyncClient", client)
    return fake


def test_the_twiml_dials_then_hangs_up() -> None:
    assert transfer_twiml("+34910000000", 20) == (
        '<Response><Dial answerOnBridge="true" timeout="20">+34910000000</Dial><Hangup/></Response>'
    )
    assert 'timeout="1"' in transfer_twiml("+34910000000", 0.2)
    assert "<Sip>sip:desk@pbx.example?a=1&amp;b=2</Sip>" in transfer_twiml(
        "sip:desk@pbx.example?a=1&b=2", 25
    )
    assert ">+34910000000</Dial>" in transfer_twiml("tel:+34910000000", 25)


async def test_twilio_replaces_the_calls_twiml(carrier: Carrier) -> None:
    serializer = TwilioFrameSerializer("MZ1", "CA1", "AC1", "secret")
    line = TwilioLine(
        "CA1", account_sid="AC1", auth_token="secret", worker=new_worker(), serializer=serializer
    )
    carrier.watched = serializer
    await line.transfer(BILLING)
    assert carrier.armed == [False]
    (request,) = carrier.requests
    assert str(request.url) == "https://api.twilio.com/2010-04-01/Accounts/AC1/Calls/CA1.json"
    assert request.headers["authorization"].startswith("Basic ")
    form = dict(httpx.QueryParams(request.content.decode()))
    assert form["Twiml"] == transfer_twiml("+34910000000", 20)
    assert serializer._hangup_attempted


async def test_telnyx_sends_its_transfer_command(carrier: Carrier) -> None:
    line = TelnyxLine("v3:abc", api_key="KEY", worker=new_worker(), serializer=None)
    await line.transfer(Transfer("t", "tel:+34910000000"))
    (request,) = carrier.requests
    assert str(request.url) == "https://api.telnyx.com/v2/calls/v3:abc/actions/transfer"
    assert request.headers["authorization"] == "Bearer KEY"
    assert request.read() == b'{"to":"+34910000000","timeout_secs":25}'


async def test_a_refused_request_keeps_the_caller_and_the_hang_up(
    carrier: Carrier,
) -> None:
    carrier.status = 404
    serializer = TwilioFrameSerializer("MZ1", "CA1", "AC1", "secret")
    line = TwilioLine(
        "CA1", account_sid="AC1", auth_token="x", worker=new_worker(), serializer=serializer
    )
    with pytest.raises(TransferError, match="refused it with 404"):
        await line.transfer(BILLING)
    assert not serializer._hangup_attempted


async def test_a_web_call_has_no_phone_caller() -> None:
    with pytest.raises(NoPhoneCaller):
        await TwilioLine(
            None, account_sid="AC1", auth_token="x", worker=new_worker(), serializer=None
        ).transfer(BILLING)


async def test_the_line_queues_what_it_says() -> None:
    worker = new_worker()
    await TelnyxLine("v3:abc", api_key="KEY", worker=worker, serializer=None).say("Adiós.")
    assert [frame.text for frame in worker.frames if isinstance(frame, TTSSpeakFrame)] == ["Adiós."]


def test_pipecat_lines_make_no_warm_transfer() -> None:
    warm = Transfer("escalate", "+34910000000", mode="warm")
    with pytest.raises(ValueError, match="TwilioLine has no warm transfer"):
        transfer_tool(
            warm,
            TwilioLine(
                "CA1", account_sid="AC1", auth_token="x", worker=new_worker(), serializer=None
            ),
        )


@pytest.mark.parametrize(
    "serializer",
    [
        lambda: TwilioFrameSerializer("MZ1", "CA1", "AC1", "secret"),
        lambda: TelnyxFrameSerializer("s1", "PCMU", "PCMU", "v3:abc", "KEY"),
        lambda: PlivoFrameSerializer("s1", "call-1", "MA1", "secret"),
    ],
)
async def test_a_disarmed_serializer_does_not_hang_up_on_the_end(
    serializer: Callable[[], Any],
) -> None:
    built = serializer()
    hung_up: list[bool] = []

    async def hang_up() -> None:
        hung_up.append(True)

    built._hang_up_call = hang_up
    disarm_hang_up(built)
    await built.serialize(EndFrame())
    assert hung_up == []


class FakeDaily:
    """A ``DailyTransport``: its handlers, its SIP calls, and what they answer."""

    def __init__(self, transfer_error: str | Exception | None = None) -> None:
        self.handlers: dict[str, list[Callable[..., Any]]] = {}
        self.transfers: list[dict[str, Any]] = []
        self.dialouts: list[dict[str, Any]] = []
        self.transfer_error = transfer_error

    def add_event_handler(self, event: str, handler: Callable[..., Any]) -> None:
        self.handlers.setdefault(event, []).append(handler)

    async def fire(self, event: str, *args: Any) -> None:
        for handler in self.handlers.get(event, []):
            await handler(self, *args)

    async def sip_call_transfer(self, settings: dict[str, Any]) -> str | None:
        self.transfers.append(settings)
        if isinstance(self.transfer_error, Exception):
            raise self.transfer_error
        return self.transfer_error

    async def start_dialout(self, settings: dict[str, Any]) -> tuple[str, str | None]:
        self.dialouts.append(settings)
        return "leg-1", None


async def transfer_then(daily: FakeDaily, event: str, data: Any, transfer: Transfer) -> None:
    """Start a Daily transfer, then fire ``event`` while it waits."""
    line = DailySipLine(cast(Any, daily), new_worker(), sip_host="trunk.example")
    await daily.fire("on_dialin_connected", {"sessionId": "caller-leg"})
    running = asyncio.ensure_future(line.transfer(transfer))
    await asyncio.sleep(0)
    await daily.fire(event, data)
    await running


async def test_daily_transfers_through_the_trunk_and_waits_for_the_answer() -> None:
    daily = FakeDaily()
    await transfer_then(daily, "on_dialout_answered", {}, BILLING)
    assert daily.transfers == [{"toEndPoint": "sip:+34910000000@trunk.example"}]


@pytest.mark.parametrize(
    ("event", "data", "reason"),
    [
        ("on_dialout_error", {"errorMsg": "486 Busy Here"}, "486 Busy Here"),
        ("on_dialout_stopped", {}, "the call to the destination ended"),
    ],
)
async def test_a_daily_leg_that_fails_returns_the_caller(
    event: str, data: Any, reason: str
) -> None:
    with pytest.raises(TransferError) as raised:
        await transfer_then(FakeDaily(), event, data, BILLING)
    assert raised.value.reason == reason


async def test_a_daily_transfer_nobody_answers_times_out() -> None:
    line = DailySipLine(cast(Any, FakeDaily()), new_worker(), sip_host="trunk.example")
    await line._connected(None, {})
    with pytest.raises(TransferError, match="no answer"):
        await line.transfer(Transfer("t", "+34910000000", ring_timeout_s=0.01))


async def test_daily_refuses_or_has_no_phone_leg() -> None:
    line = DailySipLine(cast(Any, FakeDaily("no session")), new_worker(), sip_host="trunk.example")
    with pytest.raises(NoPhoneCaller):
        await line.transfer(BILLING)
    await line._connected(None, {})
    with pytest.raises(TransferError, match="Daily refused it: no session"):
        await line.transfer(BILLING)


async def test_dial_in_is_forwarded_once(carrier: Carrier) -> None:
    daily = FakeDaily()
    forward_dial_in(cast(Any, daily), "CA9", account_sid="AC1", auth_token="secret")
    await daily.fire("on_dialin_ready", "sip:room@daily.example")
    await daily.fire("on_dialin_ready", "sip:room@daily.example")
    (request,) = carrier.requests
    assert str(request.url).endswith("/Accounts/AC1/Calls/CA9.json")
    form = dict(httpx.QueryParams(request.content.decode()))
    assert form["Twiml"] == "<Response><Dial><Sip>sip:room@daily.example</Sip></Dial></Response>"


async def test_dial_out_starts_on_join_and_ends_with_its_own_leg() -> None:
    daily, worker = FakeDaily(), new_worker()
    dial_out(cast(Any, daily), worker, "sip:+34910000000@trunk.example", display_name="Salon")
    await daily.fire("on_joined", {})
    assert daily.dialouts == [
        {"sipUri": "sip:+34910000000@trunk.example", "provider": "daily", "displayName": "Salon"}
    ]
    await daily.fire("on_dialout_stopped", {"sessionId": "transfer-leg"})
    assert worker.cancelled == []
    await daily.fire("on_dialout_stopped", {"sessionId": "leg-1"})
    assert worker.cancelled == ["dial-out ended"]


async def test_a_dial_out_leg_that_stopped_is_the_callee_hanging_up() -> None:
    daily, worker, call = FakeDaily(), new_worker(), CallHooks()
    dial_out(cast(Any, daily), worker, "sip:+34910000000@trunk.example", call=call)
    await daily.fire("on_joined", {})
    await daily.fire("on_dialout_stopped", {"sessionId": "transfer-leg"})
    assert worker.cancelled == []
    await daily.fire("on_dialout_stopped", {"sessionId": "leg-1"})
    assert worker.cancelled == ["dial-out ended"]
    assert (await call.end("ended")).reason == CALLER_HUNG_UP


async def test_a_dial_out_leg_that_failed_is_no_hang_up() -> None:
    daily, worker, call = FakeDaily(), new_worker(), CallHooks()
    dial_out(cast(Any, daily), worker, "sip:+34910000000@trunk.example", call=call)
    await daily.fire("on_joined", {})
    await daily.fire("on_dialout_error", {"sessionId": "leg-1"})
    assert worker.cancelled == ["dial-out ended"]
    assert (await call.end("ended")).reason == "ended"


async def test_the_callers_own_leg_ending_settles_no_transfer() -> None:
    daily = FakeDaily()
    line = DailySipLine(cast(Any, daily), new_worker(), sip_host="trunk.example")
    await daily.fire("on_dialout_connected", {"sessionId": "caller-leg"})
    running = asyncio.ensure_future(line.transfer(BILLING))
    await asyncio.sleep(0)
    await daily.fire("on_dialout_stopped", {"sessionId": "caller-leg"})
    await asyncio.sleep(0)
    assert not running.done()
    await daily.fire("on_dialout_answered", {"sessionId": "transfer-leg"})
    await running


async def test_daily_that_raises_is_a_failed_transfer_and_one_runs_at_a_time() -> None:
    line = DailySipLine(
        cast(Any, FakeDaily(RuntimeError("no session"))), new_worker(), sip_host="h"
    )
    await line._connected(None, {})
    with pytest.raises(TransferError, match="Daily refused it: no session"):
        await line.transfer(BILLING)
    waiting = DailySipLine(cast(Any, FakeDaily()), new_worker(), sip_host="h")
    await waiting._connected(None, {})
    first = asyncio.ensure_future(waiting.transfer(BILLING))
    await asyncio.sleep(0)
    with pytest.raises(TransferError, match="another transfer is under way"):
        await waiting.transfer(BILLING)
    first.cancel()
