"""A LiveKit session on a realtime model: who speaks a fixed line, and which model it is.

LiveKit's ``session.say`` needs a TTS, or a model that can say a line itself
(voice/agent_activity.py:1720-1729). No realtime model of LiveKit 1.8.7 can. So when the
model speaks itself, a fixed line becomes a reply it is told to speak word for word, as
Unmute's realtime greeting does.
"""

import contextlib

from livekit.agents import NOT_GIVEN, AgentSession, NotGivenOr, llm, tts
from livekit.agents.utils import is_given
from livekit.agents.voice import SpeechHandle

from unmute_core.realtime import say_line


def realtime_model(session: AgentSession) -> llm.RealtimeModel | None:
    """Return the realtime model the session's current agent talks to, or None."""
    model: object = session.llm
    with contextlib.suppress(RuntimeError):
        if is_given(agent_llm := session.current_agent.llm):
            model = agent_llm
    return model if isinstance(model, llm.RealtimeModel) else None


def _tts(session: AgentSession) -> tts.TTS | None:
    """Return the TTS of the session's current agent, or the session's own."""
    with contextlib.suppress(RuntimeError):
        if is_given(agent_tts := session.current_agent.tts):
            return agent_tts
    return session.tts


def speaks_itself(session: AgentSession) -> bool:
    """Return True when a realtime model speaks and no TTS can say a fixed line."""
    model = realtime_model(session)
    if model is None or _tts(session) is not None:
        return False
    return model.capabilities.audio_output and not model.capabilities.supports_say


def fixed_instructions(session: AgentSession) -> bool:
    """Return True when the session's realtime model fixes its instructions at the start."""
    model = realtime_model(session)
    return model is not None and not model.capabilities.mutable_instructions


def speak(
    session: AgentSession,
    line: str,
    *,
    allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    add_to_chat_ctx: bool = True,
) -> SpeechHandle:
    """Say a fixed line: with ``session.say``, or by the realtime model that speaks itself.

    The model's reply joins its conversation, whatever ``add_to_chat_ctx`` says: the
    model holds it on its own server.
    """
    if speaks_itself(session):
        return session.generate_reply(
            instructions=say_line(line), allow_interruptions=allow_interruptions
        )
    return session.say(
        line, allow_interruptions=allow_interruptions, add_to_chat_ctx=add_to_chat_ctx
    )


def session_closing(session: AgentSession) -> bool:
    """Return True while the session closes: LiveKit's private ``_closing`` flag.

    LiveKit has no public way to ask. ``tests/livekit/test_sdk_pins.py`` pins the flag.
    """
    return session._closing
