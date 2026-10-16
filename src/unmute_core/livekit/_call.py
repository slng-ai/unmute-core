"""Opening a LiveKit call: what it carries, its greeting, its limit, and its ``CallHooks``.

Ported from unmute@868ae23 templates/livekit_v1/utils_telephony.py.tmpl:112-147 and
agents.py.tmpl:301-324.
"""

import asyncio
import contextlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from livekit import rtc
from livekit.agents import NOT_GIVEN, Agent, AgentSession, utils
from livekit.agents.voice import AgentStateChangedEvent, CloseEvent, ErrorEvent, SpeechHandle

from unmute_core.call import MAX_DURATION, CallHooks, transcript_messages
from unmute_core.greeting import GREETING_HOLD_SECONDS, Greeting
from unmute_core.idle import CallLimit
from unmute_core.livekit._close import CloseGuard
from unmute_core.livekit._model import ModelCallLLM
from unmute_core.livekit._realtime import realtime_model, speak
from unmute_core.livekit._trace import end_reason
from unmute_core.state import CallInfo, CallState, Direction, call_info

SIP_RULE_ID = "sip.ruleID"
"""The attribute LiveKit sets on a SIP participant that came in through a dispatch rule."""


def _sip_direction(
    participant: rtc.RemoteParticipant | None, metadata: Mapping[str, object]
) -> Direction:
    """Read the direction: the metadata's own word, the SIP participant, then the metadata.

    LiveKit gives an inbound SIP participant the dispatch rule it came through, so one with
    a rule is inbound whatever the metadata holds. Otherwise a job that names
    ``phone_number`` dials out.
    """
    if (said := metadata.get("direction")) in ("inbound", "outbound"):
        return cast("Direction", said)
    if participant is not None and participant.attributes.get(SIP_RULE_ID):
        return "inbound"
    return "outbound" if "phone_number" in metadata else "inbound"


def sip_call_info(
    participant: rtc.RemoteParticipant | None,
    metadata: Mapping[str, object],
    *,
    trunk_number: str | None = None,
    direction: Direction | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> CallInfo:
    """Return what a SIP call carries: its direction, its two numbers and its call id.

    The direction is ``direction`` when given. Else the metadata's ``direction``. Else a SIP
    participant that came in through a dispatch rule is inbound. Else a job that names
    ``phone_number`` dials out, and any other is inbound. The SIP participant's attributes
    win over the metadata for the numbers and the call id.

    Args:
        participant: The SIP participant, or None before it has joined.
        metadata: The job's metadata, parsed.
        trunk_number: The agent's own number, when the participant does not report it.
        direction: ``inbound`` or ``outbound``, when your dispatch knows it.
        agent_id: The agent's id, the same on every framework. ``UNMUTE_AGENT_ID`` when None.
        session_id: The call's id, when your dispatch gives one. A fresh one otherwise.
    """
    attributes = participant.attributes if participant is not None else {}
    return call_info(
        direction or _sip_direction(participant, metadata),
        remote_number=attributes.get("sip.phoneNumber") or metadata.get("phone_number"),
        local_number=attributes.get("sip.trunkPhoneNumber") or trunk_number,
        carrier_call_id=attributes.get("sip.callID") or metadata.get("call_id"),
        agent_id=agent_id,
        session_id=session_id,
    )


def greet(
    session: AgentSession, greeting: Greeting, state: CallState | None
) -> SpeechHandle | None:
    """Open the call: say the greeting's line, have the model word one, or wait.

    Call it once the session has started and prefetch has run, from the entry agent's
    ``on_enter`` or after ``session.start``.

    Args:
        session: The call's session.
        greeting: How the call opens.
        state: The call state the line's values come from.

    Returns:
        The greeting's speech, or None when the caller speaks first.
    """
    if greeting.speaks_first == "user":
        return None
    interruptible = NOT_GIVEN if greeting.interruptible else False
    if (line := greeting.line(state)) is not None:
        return speak(session, line, allow_interruptions=interruptible)
    return session.generate_reply(
        instructions=greeting.instruction, allow_interruptions=interruptible
    )


LINE_WAIT_SECONDS = 10.0
"""The longest the failure line may play before the call hangs up."""
_COMPONENTS = MappingProxyType(
    {
        "stt_error": "stt",
        "llm_error": "llm",
        "tts_error": "tts",
        "realtime_model_error": "llm",
    }
)


def attach_call(session: AgentSession, call: CallHooks) -> None:
    """Let ``call`` speak and hang up on the session, fail it and end it. Once per call.

    A part that fails for good, as LiveKit reports in an ``error`` event, fails the call.
    The session's close ends it, and the job's shutdown waits for that end.

    Args:
        session: The call's session.
        call: The call's hooks.
    """
    if call.bound:
        return

    async def say(line: str) -> None:
        """Say a line nobody can cut off, and wait until it played."""
        handle = speak(session, line, allow_interruptions=False)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(handle.wait_for_playout(), LINE_WAIT_SECONDS)

    call.bind(say=say, hang_up=session.shutdown, transcript=lambda: transcript(session))

    def failed(event: ErrorEvent) -> None:
        """Fail the call on the first error LiveKit cannot recover from."""
        error = event.error
        component = _COMPONENTS.get(str(error.type))
        if component is not None and not error.recoverable:
            call.fail_soon(component, error.error or RuntimeError(str(error)))

    ending: list[asyncio.Future[object]] = []
    guard = CloseGuard(lambda: ending)

    def closed(event: CloseEvent) -> None:
        """Start the call's end."""
        ending.append(asyncio.ensure_future(call.end(end_reason(event.reason.value))))
        guard.closed()

    def thinking(event: AgentStateChangedEvent) -> None:
        """At the first reply, warn when the agent's model request skips the call's hooks."""
        if event.new_state != "thinking":
            return
        session.off("agent_state_changed", thinking)
        agent = session.current_agent
        model = agent.llm if utils.is_given(agent.llm) else session.llm
        if (
            type(agent).llm_node is Agent.llm_node
            and not isinstance(model, ModelCallLLM)
            and realtime_model(session) is None
        ):
            call.warn_skipped_model_hooks(
                "use filled_llm or traced_llm in the agent's llm_node, or ModelCallLLM, or "
                "UnmuteAgent"
            )

    session.on("error", failed)
    session.once("close", closed)
    if call.model_hooks:
        session.on("agent_state_changed", thinking)
    guard.hold_job()


def transcript(session: AgentSession) -> list[dict[str, str]]:
    """Return what the caller and the agent said, as ``{"role", "content"}`` messages."""
    messages, _ = session.history.to_provider_format("openai")
    return transcript_messages(messages)


SESSION_NOT_RUNNING = "AgentSession isn't running"
"""What ``say`` and ``generate_reply`` raise on a closed session (voice/agent_session.py:1553)."""


async def open_call(
    session: AgentSession,
    greeting: Greeting,
    call: CallHooks,
    *,
    max_hold_s: float = GREETING_HOLD_SECONDS,
) -> SpeechHandle | None:
    """Pass the checks before the greeting, then greet, holding the caller's audio until then.

    Call it once the session has started. The caller's audio comes back when the greeting
    starts, or when it ends for a greeting nobody may cut off. A failed check plays the
    failure line and hangs up, and the caller's audio stays off.

    Args:
        session: The call's session.
        greeting: How the call opens.
        call: The call's hooks, with the checks to pass.
        max_hold_s: The longest the caller's audio stays off once the greeting is queued,
            in case its speech never starts. Pipecat's ``GreetingHold`` takes the same.

    Returns:
        The greeting's speech, or None when the caller speaks first, a check failed, or the
        session closed while the checks ran.
    """
    attach_call(session, call)
    session.input.set_audio_enabled(False)
    if not await call.open():
        return None
    if (
        greeting.speaks_first == "agent"
        and greeting.line(call.state) is None
        and realtime_model(session) is not None
    ):
        await _await_prefetch(session, call)
    try:
        handle = greet(session, greeting, call.state)
    except BaseException as error:
        if isinstance(error, RuntimeError) and SESSION_NOT_RUNNING in str(error):
            return None
        session.input.set_audio_enabled(True)
        raise
    if handle is None:
        session.input.set_audio_enabled(True)
        return None

    loop = asyncio.get_running_loop()
    hold = loop.call_later(max_hold_s, lambda: release())

    released: list[bool] = []

    def release(*_: object) -> None:
        """Give the caller's audio back, once."""
        if not released:
            released.append(True)
            hold.cancel()
            session.off("agent_state_changed", speaking)
            session.input.set_audio_enabled(True)

    def speaking(event: AgentStateChangedEvent) -> None:
        """Give the audio back as the agent starts to speak, so the caller can cut in."""
        if event.new_state == "speaking":
            release()

    if greeting.interruptible:
        session.on("agent_state_changed", speaking)
    handle.add_done_callback(release)
    return handle


async def _await_prefetch(session: AgentSession, call: CallHooks) -> None:
    """Wait for the lookups and the instructions a realtime model's worded greeting reads.

    A realtime model has no request the call could hold: the greeting is a reply it
    words from its instructions, so the lookups end and the instructions are sent first.
    """
    await call.before_request(None)
    from unmute_core.livekit._agent import UnmuteAgent  # _agent imports this module

    if isinstance(agent := session.current_agent, UnmuteAgent):
        await agent.instructions_settled()


def max_duration(
    session: AgentSession, seconds: float, *, call: CallHooks | None = None
) -> CallLimit:
    """End the call ``seconds`` after now, with no closing line. Call it once it has opened.

    The count stops when the session closes first.

    Args:
        session: The call's session.
        seconds: The longest the call may run.
        call: The call's hooks. The limit first tells them the call ends for
            ``max_duration``, so the call's end reports that reason.

    Returns:
        The running limit.
    """

    def end() -> None:
        """Note the call's end, then close the session."""
        if call is not None:
            call.end_requested(MAX_DURATION)
        session.shutdown()

    limit = CallLimit(seconds, end)
    session.on("close", lambda _event: limit.cancel())
    limit.start()
    return limit
