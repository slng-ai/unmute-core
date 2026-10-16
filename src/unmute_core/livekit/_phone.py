"""The LiveKit SIP leg of a call: cold and warm transfers, and outbound dialling.

Ported from unmute@2c78a5a templates/livekit_v1/agents.py.tmpl:355-545 (transfers),
call.py.tmpl:269-386 (dial and voicemail) and prompts.py.tmpl:46-68 (the briefing).
"""

import asyncio
import contextlib
import logging
from collections.abc import Mapping
from typing import ClassVar

import aiohttp
from livekit import api, rtc
from livekit.agents import AMD, NOT_GIVEN, AgentSession, get_job_context, llm
from livekit.agents.beta.workflows import WarmTransferTask
from livekit.agents.beta.workflows.utils import WorkflowInstructions
from pydantic import SkipValidation
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.call import AGENT_ENDED, CallHooks
from unmute_core.livekit._step import say
from unmute_core.telephony import (
    VOICEMAIL_MESSAGE,
    E164Number,
    NoPhoneCaller,
    OnVoicemail,
    Transfer,
    TransferError,
    refer_uri,
    sip_reason,
    sip_user,
)

logger = logging.getLogger(__name__)

BRIEFING_PERSONA = """\
# Identity

You are the assistant that took a caller's call. You are now on the phone to a colleague
you called for help with it. The caller is not on this line: they are on hold and cannot
hear any of this.

# Goal

Hand the caller over. Your first words are the handover, not a greeting. Do not say hello,
do not offer help, and never wait to be asked what the call is about. Say who is on hold,
what they want, and what was already tried or offered, then ask one question your colleague
can answer. A few sentences, no more.

Tell your colleague you will put the caller through as soon as they say they are ready.

If the conversation history below is empty, or has no words from the caller in it, say that
someone is on hold asking for a person, that you do not have their details yet, and ask
whether your colleague can take the call. Say exactly that. Never greet instead.

Your colleague may not be able to take it. If they say they cannot, if they go quiet, or if
the conversation moves on without an answer, decline the transfer and give their reason. Do
not leave the question open: the caller is holding for the whole of this conversation."""
"""Who the agent is when it briefs the person a warm transfer calls."""

AUDIO_SETTLE_S = 0.3
"""Seconds a dial waits after the callee's audio is subscribed, so the first word is not clipped."""

_CARRIER_ERRORS = (api.ServerError, aiohttp.ClientError, TimeoutError)


class DialError(Exception):
    """The outbound call was not answered, or the carrier refused it.

    Attributes:
        reason: ``busy``, ``no answer``, ``sip 503`` or ``failed``.
        sip_status: The SIP status, like 486, or None.
    """

    def __init__(self, reason: str, sip_status: int | None = None) -> None:
        """Keep the reason and the status."""
        super().__init__(reason)
        self.reason = reason
        self.sip_status = sip_status


@original_errors
@dataclass(frozen=True, config=VALUE)
class Dialled:
    """An outbound call that was answered.

    Attributes:
        participant: The person or machine that answered.
        verdict: The voicemail detector's category, like ``human`` or ``machine-vm``, or
            None without detection.
        ended: True when the call ended at once: voicemail, or no mailbox. Do not greet.
    """

    participant: SkipValidation[rtc.RemoteParticipant]
    verdict: str | None = None
    ended: bool = False


@original_errors
@dataclass(frozen=True, config=VALUE)
class _DialOptions:
    """What one dial asks of LiveKit: the number, which must be E.164, and who the callee is."""

    number: E164Number
    identity: str
    ring_timeout_s: float | None = None
    participant_name: str = ""
    display_name: str | None = None
    metadata: str = ""
    attributes: SkipValidation[Mapping[str, str] | None] = None
    krisp: bool = False

    def request(
        self, room_name: str, trunk: str | api.SIPOutboundConfig | None, caller_id: str | None
    ) -> api.CreateSIPParticipantRequest:
        """Return the request that makes LiveKit call the number and wait until it is answered."""
        request = api.CreateSIPParticipantRequest(
            room_name=room_name,
            sip_call_to=self.number,
            participant_identity=self.identity,
            wait_until_answered=True,
            sip_number=caller_id or "",
        )
        if isinstance(trunk, api.SIPOutboundConfig):
            request.trunk.CopyFrom(trunk)
        elif trunk:
            request.sip_trunk_id = trunk
        if self.ring_timeout_s is not None:
            request.ringing_timeout.FromNanoseconds(int(self.ring_timeout_s * 1e9))
        if self.participant_name:
            request.participant_name = self.participant_name
        if self.display_name is not None:
            request.display_name = self.display_name
        if self.metadata:
            request.participant_metadata = self.metadata
        if self.attributes:
            request.participant_attributes.update(self.attributes)
        if self.krisp:
            request.krisp_enabled = True
        return request


class SipLine:
    """The SIP leg of one LiveKit call. Build one per call, once the session has started.

    Pass it to ``transfer_tool`` for each transfer, and call ``dial`` for an outbound call.
    A warm transfer and a dial go out through ``trunk``.
    """

    warm: ClassVar[bool] = True

    def __init__(
        self,
        session: AgentSession,
        *,
        trunk: str | api.SIPOutboundConfig | None = None,
        caller_id: str | None = None,
        call: CallHooks | None = None,
    ) -> None:
        """Keep the session and the trunk the line dials through.

        Args:
            session: The call's session.
            trunk: A stored outbound trunk id, or the trunk's settings sent with each dial.
                None leaves it to ``LIVEKIT_SIP_OUTBOUND_TRUNK`` for a warm transfer.
            caller_id: The number the person called sees, like ``+34910000000``.
            call: The call's hooks. A dial that ends on a voicemail then ends the call with
                the reason ``agent_ended``.
        """
        self._session = session
        self._call = call
        self._trunk = trunk
        self._caller_id = caller_id

    async def transfer(self, transfer: Transfer) -> None:
        """Transfer the caller: a SIP REFER when cold, a call and a briefing when warm.

        Raises:
            NoPhoneCaller: If no SIP caller is in the room, for a cold transfer.
            TransferError: If the carrier refused it, or nobody took it.
        """
        if transfer.mode == "warm":
            await self._warm(transfer)
        else:
            await self._cold(transfer)

    async def say(self, line: str) -> None:
        """Say a line, and wait until it has played."""
        if (speech := say(self._session, line, allow_interruptions=False)) is not None:
            await speech.wait_for_playout()

    async def _cold(self, transfer: Transfer) -> None:
        """Refer the SIP caller to the destination. The caller leaves the room.

        The caller is the session's own participant when it is a SIP one, else the first
        SIP participant. A REFER LiveKit reports as ongoing counts as done.
        """
        job = get_job_context()
        room_io = self._session.room_io
        linked = room_io.linked_participant if room_io is not None else None
        candidates = [linked, *job.room.remote_participants.values()]
        caller = next(
            (
                participant
                for participant in candidates
                if participant is not None
                and participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
            ),
            None,
        )
        if caller is None:
            raise NoPhoneCaller
        request = api.TransferSIPParticipantRequest(
            room_name=job.room.name,
            participant_identity=caller.identity,
            transfer_to=refer_uri(transfer.destination),
            play_dialtone=True,
        )
        if transfer.ring_timeout_s is not None:
            request.ringing_timeout.FromNanoseconds(int(transfer.ring_timeout_s * 1e9))
        try:
            response = await job.api.sip.transfer_sip_participant(request)
        except api.SipCallError as error:
            raise TransferError(sip_reason(error.sip_status_code)) from error
        except _CARRIER_ERRORS as error:
            raise TransferError(str(error) or type(error).__name__) from error
        if response.status == api.SIPTransferStatus.STS_TRANSFER_FAILED:
            raise TransferError(sip_reason(response.sip_status.code or None))

    async def _warm(self, transfer: Transfer) -> None:
        """Call the person, brief them from the chat, then merge the calls."""
        trunk_id = self._trunk if isinstance(self._trunk, str) else NOT_GIVEN
        connection = self._trunk if isinstance(self._trunk, api.SIPOutboundConfig) else NOT_GIVEN
        try:
            task = WarmTransferTask(
                sip_user(transfer.destination),
                sip_trunk_id=trunk_id,
                sip_connection=connection,
                sip_number=self._caller_id or NOT_GIVEN,
                chat_ctx=self._session.current_agent.chat_ctx,
                instructions=WorkflowInstructions(
                    persona=BRIEFING_PERSONA, extra=transfer.briefing
                ),
                ringing_timeout=transfer.ring_timeout_s
                if transfer.ring_timeout_s is not None
                else NOT_GIVEN,
            )
        except ValueError as error:
            logger.error("the warm transfer cannot dial: %s", error)
            raise TransferError("there is no trunk to call the person through") from error
        try:
            result = await task
        except llm.ToolError as error:
            raise TransferError(error.message) from error
        logger.info("warm transfer merged with %s", result.human_agent_identity)
        await self.say(transfer.connected_line)

    async def dial(
        self,
        number: str,
        *,
        identity: str = "phone_user",
        ring_timeout_s: float | None = None,
        on_voicemail: OnVoicemail = "hangup",
        detect_voicemail: bool = True,
        detector_llm: llm.LLM | str | None = None,
        participant_name: str = "",
        display_name: str | None = None,
        metadata: str = "",
        attributes: Mapping[str, str] | None = None,
        krisp: bool = False,
        audio_wait_s: float | None = None,
    ) -> Dialled:
        """Call a number from the started session, and wait until someone answers.

        The session listens to ``identity``. With ``detect_voicemail``, LiveKit's ``AMD``
        holds the agent's speech until it knows who answered. Voicemail then hangs up or
        leaves a message, and a full or missing mailbox hangs up.

        Args:
            number: The number to call, in E.164.
            identity: The participant identity the callee gets.
            ring_timeout_s: How long it may ring. None is LiveKit's 30 s.
            on_voicemail: ``hangup``, or ``leave_message``, which the model words.
            detect_voicemail: Run LiveKit's answering machine detection.
            detector_llm: The model that judges who answered. None uses the session's
                LLM, else the agent's: ``UnmuteAgent`` holds its model on the agent.
            participant_name: The callee's display name in the room. Empty keeps LiveKit's.
            display_name: The caller name the callee's phone shows. None keeps the trunk's.
            metadata: Metadata for the callee's participant.
            attributes: Attributes for the callee's participant.
            krisp: Turn on Krisp noise cancellation for the callee.
            audio_wait_s: Without detection, wait at most this long for the callee's audio
                track after they join, then ``AUDIO_SETTLE_S`` more. A timeout is logged and
                the dial goes on. None does not wait. Detection ignores it, because it
                already listens for the callee.

        Returns:
            Who answered. Greet them unless ``ended`` is True.

        Raises:
            ValueError: If the number is not E.164.
            DialError: If nobody answered, or the carrier refused the call.
        """
        options = _DialOptions(
            number,
            identity,
            ring_timeout_s,
            participant_name,
            display_name,
            metadata,
            attributes,
            krisp,
        )
        job = get_job_context()
        session = self._session
        if session.room_io is not None:
            session.room_io.set_participant(identity)
        request = options.request(job.room.name, self._trunk, self._caller_id)
        async with contextlib.AsyncExitStack() as stack:
            detector = (
                await stack.enter_async_context(
                    AMD(
                        session,
                        llm=detector_llm or session.llm or session.current_agent.llm,
                        participant_identity=identity,
                    )
                )
                if detect_voicemail
                else None
            )
            try:
                await job.api.sip.create_sip_participant(request)
            except api.SipCallError as error:
                raise DialError(sip_reason(error.sip_status_code), error.sip_status_code) from error
            except _CARRIER_ERRORS as error:
                raise DialError(str(error) or type(error).__name__) from error
            participant = await job.wait_for_participant(identity=identity)
            verdict = (await detector.execute()).category if detector is not None else None
        if detector is None and audio_wait_s is not None:
            await _wait_for_audio(job.room, participant, audio_wait_s)
        if verdict == "machine-vm":
            if on_voicemail == "leave_message":
                await session.generate_reply(instructions=VOICEMAIL_MESSAGE).wait_for_playout()
                self._end_requested()
                job.shutdown("voicemail: left a message")
            else:
                self._end_requested()
                job.shutdown("voicemail detected")
            return Dialled(participant, verdict, ended=True)
        if verdict == "machine-unavailable":
            self._end_requested()
            job.shutdown("mailbox unavailable")
            return Dialled(participant, verdict, ended=True)
        return Dialled(participant, verdict)

    def _end_requested(self) -> None:
        """Tell the call's hooks that the agent ends the call, before the job shuts down."""
        if self._call is not None:
            self._call.end_requested(AGENT_ENDED)


def _audio_subscribed(participant: rtc.RemoteParticipant) -> bool:
    """Return whether the participant has an audio track the session is subscribed to."""
    return any(
        publication.kind == rtc.TrackKind.KIND_AUDIO and publication.subscribed
        for publication in participant.track_publications.values()
    )


async def _wait_for_audio(
    room: rtc.Room, participant: rtc.RemoteParticipant, timeout_s: float
) -> None:
    """Wait for the callee's audio track, then let it settle. A timeout is logged, not raised."""
    arrived = asyncio.Event()

    def on_track(
        track: rtc.Track, publication: rtc.RemoteTrackPublication, who: rtc.RemoteParticipant
    ) -> None:
        """Note when the callee's audio track is subscribed."""
        if who.identity == participant.identity and track.kind == rtc.TrackKind.KIND_AUDIO:
            arrived.set()

    room.on("track_subscribed", on_track)
    try:
        if not _audio_subscribed(participant):
            await asyncio.wait_for(arrived.wait(), timeout_s)
    except TimeoutError:
        logger.warning(
            "no audio from %s within %.1f s. The dial goes on", participant.identity, timeout_s
        )
    finally:
        room.off("track_subscribed", on_track)
    await asyncio.sleep(AUDIO_SETTLE_S)
