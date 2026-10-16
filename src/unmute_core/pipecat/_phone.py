"""The phone leg of a Pipecat call: transfers on a carrier websocket or a Daily SIP room.

Ported from unmute@2c78a5a templates/pipecat_v1/utils_telephony.py.tmpl:66-209,
agents.py.tmpl:386-533 and call.py.tmpl:552-622.

A carrier websocket call is transferred through the carrier's REST API: Twilio replaces the
call's TwiML, Telnyx has a transfer command. Plivo's transfer fetches XML from a URL you
host, and Exotel has none for a stream, so neither has a line here. Twilio, Telnyx and
Plivo serializers hang up the call on the pipeline's first ``EndFrame`` or ``CancelFrame``;
a line turns that off once a transfer is accepted, or the end would drop the caller just
handed over.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, ClassVar
from xml.sax.saxutils import escape

import httpx
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.worker import PipelineWorker

if TYPE_CHECKING:
    from pipecat.transports.daily.transport import DailyTransport

from unmute_core.call import CallHooks
from unmute_core.telephony import NoPhoneCaller, Transfer, TransferError, sip_uri

logger = logging.getLogger(__name__)

TWILIO_API = "https://api.twilio.com/2010-04-01"
TELNYX_API = "https://api.telnyx.com/v2"
RING_TIMEOUT_S = 25.0
"""How long a transfer rings when the ``Transfer`` sets no timeout. Unmute's default."""
_HTTP_TIMEOUT = httpx.Timeout(10.0)


def disarm_hang_up(serializer: object, *, armed: bool = False) -> bool:
    """Stop a carrier serializer from hanging up the call when the pipeline ends.

    Twilio's, Telnyx's and Plivo's serializers hang up through the carrier's REST API on
    the first ``EndFrame`` or ``CancelFrame``. After a transfer that call is the caller's.

    Args:
        serializer: The call's serializer.
        armed: Turn the hang-up back on instead, after a transfer that failed.

    Returns:
        True when the hang-up was on before.
    """
    if not hasattr(serializer, "_hangup_attempted"):
        logger.warning("%s has no hang-up to turn off", type(serializer).__name__)
        return False
    was_armed = not serializer._hangup_attempted
    serializer._hangup_attempted = not armed  # ty: ignore[invalid-assignment]
    return was_armed


def transfer_twiml(destination: str, ring_timeout_s: float) -> str:
    """Return the TwiML that dials the destination, then ends the caller's call.

    Twilio moves to the next verb after a dial that completes, is declined or rings out,
    so one ``<Hangup/>`` covers each.
    """
    if destination.startswith(("sip:", "sips:")):
        target = f"<Sip>{escape(destination)}</Sip>"
    else:
        target = escape(destination.removeprefix("tel:"))
    return (
        "<Response>"
        f'<Dial answerOnBridge="true" timeout="{max(1, round(ring_timeout_s))}">{target}</Dial>'
        "<Hangup/>"
        "</Response>"
    )


async def _post(url: str, **request: Any) -> None:
    """Send one carrier request.

    Raises:
        TransferError: If it could not be sent, or the carrier answered with an error.
    """
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.post(url, **request)
    except httpx.HTTPError as error:
        raise TransferError(f"the carrier could not be reached: {type(error).__name__}") from error
    if not response.is_success:
        logger.warning("carrier request failed with %s: %s", response.status_code, response.text)
        raise TransferError(f"the carrier refused it with {response.status_code}")


class _CarrierLine:
    """A carrier websocket call, transferred through the carrier's REST API."""

    warm: ClassVar[bool] = False

    def __init__(self, call_id: str | None, worker: PipelineWorker, serializer: object) -> None:
        """Keep the call, the worker that speaks, and the serializer to disarm."""
        self._call_id = call_id
        self._worker = worker
        self._serializer = serializer

    async def transfer(self, transfer: Transfer) -> None:
        """Ask the carrier to transfer the call, with the pipeline's end kept off it.

        The serializer's hang-up is off before the request goes out, since the carrier may
        close the stream as soon as it accepts. A refused request turns it back on.

        Raises:
            NoPhoneCaller: If the call has no carrier id, like a web call.
            TransferError: If the carrier refused the request.
        """
        if not self._call_id:
            raise NoPhoneCaller
        serializer = self._serializer
        armed = serializer is not None and disarm_hang_up(serializer)
        try:
            await self._send(self._call_id, transfer)
        except BaseException:
            if armed:
                disarm_hang_up(serializer, armed=True)
            raise

    async def say(self, line: str) -> None:
        """Queue a line. The end of the call waits for it to play."""
        await self._worker.queue_frame(TTSSpeakFrame(line))

    async def _send(self, call_id: str, transfer: Transfer) -> None:
        """Send the carrier's transfer request."""
        raise NotImplementedError


class TwilioLine(_CarrierLine):
    """A Twilio Media Streams call. Build one per call.

    The transfer replaces the call's TwiML with a ``<Dial>``. The stream closes, Twilio
    rings the destination, and the call ends when that leg ends. The agent does not learn
    whether the destination answered.
    """

    def __init__(
        self,
        call_sid: str | None,
        *,
        account_sid: str,
        auth_token: str,
        worker: PipelineWorker,
        serializer: object | None,
        api_url: str = TWILIO_API,
    ) -> None:
        """Keep the call and the account.

        Args:
            call_sid: The call's ``CallSid``, ``state.call.carrier_call_id``. None on a web
                call, where a transfer tells the model there is no phone caller.
            account_sid: The Twilio account.
            auth_token: Its auth token.
            worker: The call's pipeline worker, which says the lines.
            serializer: The call's ``TwilioFrameSerializer``, whose own hang-up is turned
                off for a transfer. Pass None only for a serializer built with
                ``auto_hang_up=False``: otherwise the pipeline's end drops the caller.
            api_url: Twilio's REST API.
        """
        super().__init__(call_sid, worker, serializer)
        self._auth = (account_sid, auth_token)
        self._api_url = api_url.rstrip("/")

    async def _send(self, call_id: str, transfer: Transfer) -> None:
        """Replace the call's TwiML with the dial."""
        account_sid = self._auth[0]
        twiml = transfer_twiml(transfer.destination, transfer.ring_timeout_s or RING_TIMEOUT_S)
        await _post(
            f"{self._api_url}/Accounts/{account_sid}/Calls/{call_id}.json",
            auth=self._auth,
            data={"Twiml": twiml},
        )


class TelnyxLine(_CarrierLine):
    """A Telnyx media streaming call. Build one per call.

    Telnyx keeps the caller's leg when the destination does not answer, but says so later
    by webhook. So the transfer counts as done once Telnyx accepts it.
    """

    def __init__(
        self,
        call_control_id: str | None,
        *,
        api_key: str,
        worker: PipelineWorker,
        serializer: object | None,
        api_url: str = TELNYX_API,
    ) -> None:
        """Keep the call and the key.

        Args:
            call_control_id: The call's ``call_control_id``, ``state.call.carrier_call_id``.
                None on a web call.
            api_key: The Telnyx API key.
            worker: The call's pipeline worker, which says the lines.
            serializer: The call's ``TelnyxFrameSerializer``, whose own hang-up is turned
                off for a transfer. Pass None only for a serializer built with
                ``auto_hang_up=False``: otherwise the pipeline's end drops the caller.
            api_url: Telnyx's REST API.
        """
        super().__init__(call_control_id, worker, serializer)
        self._api_key = api_key
        self._api_url = api_url.rstrip("/")

    async def _send(self, call_id: str, transfer: Transfer) -> None:
        """Send the transfer command."""
        await _post(
            f"{self._api_url}/calls/{call_id}/actions/transfer",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "to": transfer.destination.removeprefix("tel:"),
                "timeout_secs": max(5, round(transfer.ring_timeout_s or RING_TIMEOUT_S)),
            },
        )


class DailySipLine:
    """A phone call in a Daily room, over SIP. Build one per call, before it joins.

    The transfer asks Daily to call the destination through your trunk and waits until it
    answers. The agent then leaves, and Daily keeps bridging the two legs, so both are
    billed until the call ends. A transfer nobody answers leaves the caller with the agent.
    """

    warm: ClassVar[bool] = False

    def __init__(
        self, transport: "DailyTransport", worker: PipelineWorker, *, sip_host: str
    ) -> None:
        """Follow the transport's SIP events.

        Args:
            transport: The call's ``DailyTransport``.
            worker: The call's pipeline worker, which says the lines.
            sip_host: Your trunk's termination host. A number is dialled as
                ``sip:<number>@<sip_host>``.
        """
        self._transport = transport
        self._worker = worker
        self._sip_host = sip_host
        self._legs: set[str] = set()
        self._pending: asyncio.Future[str | None] | None = None
        for event in ("on_dialin_connected", "on_dialout_connected"):
            transport.add_event_handler(event, self._connected)
        transport.add_event_handler("on_dialout_answered", self._answered)
        for event in ("on_dialout_error", "on_dialout_stopped"):
            transport.add_event_handler(event, self._failed)

    async def transfer(self, transfer: Transfer) -> None:
        """Call the destination and wait until it answers.

        Raises:
            NoPhoneCaller: If no phone leg ever joined the room.
            TransferError: If Daily refused it, the leg failed, or nobody answered.
        """
        if not self._legs:
            raise NoPhoneCaller
        if self._pending is not None:
            raise TransferError("another transfer is under way")
        timeout = transfer.ring_timeout_s or RING_TIMEOUT_S
        self._pending = asyncio.get_running_loop().create_future()
        try:
            try:
                error = await self._transport.sip_call_transfer(
                    {"toEndPoint": sip_uri(transfer.destination, self._sip_host)}
                )
            except Exception as raised:
                raise TransferError(f"Daily refused it: {raised}") from raised
            if error:
                raise TransferError(f"Daily refused it: {error}")
            async with asyncio.timeout(timeout):
                failure = await self._pending
        except TimeoutError:
            raise TransferError("no answer") from None
        finally:
            self._pending = None
        if failure is not None:
            raise TransferError(failure)

    async def say(self, line: str) -> None:
        """Queue a line. The end of the call waits for it to play."""
        await self._worker.queue_frame(TTSSpeakFrame(line))

    async def _connected(self, _transport: object, data: object) -> None:
        """Note the caller's leg: a phone call is in the room."""
        self._legs.add(_leg(data) or "caller")

    def _waiting(self, data: object) -> "asyncio.Future[str | None] | None":
        """Return the transfer that waits, unless the event is about the caller's leg."""
        pending = self._pending
        if pending is None or pending.done() or _leg(data) in self._legs:
            return None
        return pending

    async def _answered(self, _transport: object, data: object) -> None:
        """Settle a transfer that is waiting: the destination answered."""
        if (pending := self._waiting(data)) is not None:
            pending.set_result(None)

    async def _failed(self, _transport: object, data: object) -> None:
        """Settle a transfer that is waiting: its leg failed or stopped."""
        if (pending := self._waiting(data)) is not None:
            detail = data.get("errorMsg") if isinstance(data, dict) else None
            pending.set_result(str(detail or "the call to the destination ended"))


def _leg(data: object) -> str | None:
    """Return the SIP session a Daily event is about, when it names one."""
    leg = data.get("sessionId") if isinstance(data, dict) else None
    return str(leg) if leg else None


def forward_dial_in(
    transport: "DailyTransport", call_sid: str, *, account_sid: str, auth_token: str
) -> None:
    """Send a Twilio call into the Daily room once the room can take it. Once per call.

    Unmute's Daily dial-in: the caller waits on Twilio, and the room's SIP address arrives
    in ``on_dialin_ready``. A failure is logged, and the caller keeps waiting.

    Args:
        transport: The call's ``DailyTransport``.
        call_sid: The Twilio call to send.
        account_sid: The Twilio account.
        auth_token: Its auth token.
    """
    sent = False

    async def ready(_transport: object, sip_endpoint: str) -> None:
        """Point the Twilio call at the room, the first time only."""
        nonlocal sent
        if sent:
            return
        sent = True
        twiml = f"<Response><Dial><Sip>{escape(sip_endpoint)}</Sip></Dial></Response>"
        try:
            await _post(
                f"{TWILIO_API}/Accounts/{account_sid}/Calls/{call_sid}.json",
                auth=(account_sid, auth_token),
                data={"Twiml": twiml},
            )
        except TransferError as error:
            logger.error("the call could not be sent to the room: %s", error.reason)

    transport.add_event_handler("on_dialin_ready", ready)


def dial_out(
    transport: "DailyTransport",
    worker: PipelineWorker,
    sip_address: str,
    *,
    display_name: str | None = None,
    call: CallHooks | None = None,
) -> None:
    """Call a SIP address from the Daily room once it joins, and end the call with that leg.

    Unmute's Daily dial-out. When that leg fails or stops, the pipeline is cancelled. Another
    leg of the room, like a transfer's, ends nothing.

    Args:
        transport: The call's ``DailyTransport``.
        worker: The call's pipeline worker.
        sip_address: Who to call, like ``sip:+34910000000@trunk.example``.
        display_name: The name the callee sees.
        call: The call's hooks. A leg that stopped is the callee hanging up, so the call
            ends with ``caller_hung_up``. A leg that failed is no hang-up.
    """
    settings: dict[str, Any] = {"sipUri": sip_address, "provider": "daily"}
    if display_name:
        settings["displayName"] = display_name
    dialled: list[str] = []

    async def joined(_transport: object, _data: object) -> None:
        """Start the call."""
        session_id, error = await transport.start_dialout(settings)
        if error:
            logger.error("the outbound call could not start: %s", error)
            await worker.cancel(reason="dial-out failed")
        elif session_id:
            dialled.append(session_id)

    async def ended(_transport: object, data: object) -> None:
        """End the call with its phone leg, not with another leg of the room."""
        leg = _leg(data)
        if leg and dialled and leg not in dialled:
            return
        logger.info("the outbound leg ended: %s", data)
        await worker.cancel(reason="dial-out ended")

    async def stopped(_transport: object, data: object) -> None:
        """Note the callee's hang-up, then end the call with the leg."""
        leg = _leg(data)
        if call is not None and not (leg and dialled and leg not in dialled):
            call.caller_left()
        await ended(_transport, data)

    transport.add_event_handler("on_joined", joined)
    transport.add_event_handler("on_dialout_error", ended)
    transport.add_event_handler("on_dialout_stopped", stopped)
