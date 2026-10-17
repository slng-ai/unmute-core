"""Drop a reply the agent has not started saying when the caller speaks again.

A caller pauses ("Hi." … "I'd like to book a haircut"), and LiveKit ends their turn at
the pause. When they go on before the reply has played, LiveKit 1.8.8 keeps the reply: the
commit never checks that the caller is speaking (voice/agent_activity.py:2658-2743),
resumed speech only pauses a reply (:2404-2435), and the pause lifts after a short silence
inside their sentence (:2492-2496). The reply then plays over them.

Pipecat interrupts a reply in flight as soon as the caller starts speaking, so this brings
LiveKit to the same behaviour. The caller's words are already in the history when the agent
is "thinking" (:3676-3696), so the next reply answers all of them.

``interruption`` sets what happens to speech over a line that already plays.
"""

import logging

from livekit.agents import (
    AgentSession,
    AgentStateChangedEvent,
    InterruptionOptions,
    ToolExecutionUpdatedEvent,
    UserStateChangedEvent,
)

from unmute_core.settings import env_value
from unmute_core.turn import Backchannels, InterruptionRule

logger = logging.getLogger(__name__)

INFERENCE_KEYS = ("LIVEKIT_INFERENCE_API_KEY", "LIVEKIT_API_KEY")
"""The variables LiveKit's adaptive interruption detector reads its key from."""
_LONGER_THAN_ANY_LINE = 3600.0
"""A backchannel boundary end, in seconds, that reaches back to the start of every line."""


def interruption(
    backchannels: Backchannels,
    *,
    enabled: bool = True,
    min_words: int = 0,
    min_speech_s: float | None = None,
) -> InterruptionOptions:
    """Return the session's interruption options for one backchannel rule.

    LiveKit's adaptive detector judges speech over the agent: an interruption stops the
    agent, a backchannel does not. LiveKit then drops the backchannel's words unless they
    ended in the last second of the line (voice/audio_recognition.py:689-717).

    - ``keep``: the agent finishes its line, then answers the words. A "yes" over a
      question is the answer to it.
    - ``drop``: the agent finishes its line and the caller's words are dropped. Right
      when the agent reads long text and the caller's "mm-hm" asks for nothing. It is not
      LiveKit's default in production: LiveKit turns the detector on by itself only in dev
      mode and on LiveKit Cloud, and otherwise uses plain voice detection, which is ``stop``.
    - ``stop``: no detector. Any speech of half a second stops the agent, as on Pipecat.

    The detector runs on LiveKit's inference service and reads ``LIVEKIT_API_KEY`` or
    ``LIVEKIT_INFERENCE_API_KEY``. Without either, LiveKit falls back to ``stop`` and this
    function logs a warning.

    Unmute's ``conversation.interruption`` maps on as ``enabled`` and ``minimum_words``
    (unmute@868ae23 internal/generate/livekit_v1_build.go:886-893).

    Args:
        backchannels: The rule.
        enabled: False lets the agent finish every line, whatever the caller says.
        min_words: Words the caller must say before speech counts as an interruption.
            0 keeps LiveKit's default.
        min_speech_s: Seconds of speech over the agent before it stops, for ``stop``.
            None keeps LiveKit's 0.5 s. A cough or a "mm" shorter than this does not
            cut the agent off.

    Raises:
        ValueError: If ``min_words`` or ``min_speech_s`` is negative, or ``min_speech_s``
            comes with ``keep`` or ``drop``.
    """
    InterruptionRule(backchannels, min_words, min_speech_s)
    if not enabled:
        return {"enabled": False}
    if backchannels != "stop" and not any(env_value(name, str) for name in INFERENCE_KEYS):
        logger.warning(
            "interruption(%r) runs on LiveKit's adaptive detector, which needs "
            "LIVEKIT_API_KEY or LIVEKIT_INFERENCE_API_KEY. Without one, LiveKit stops the "
            "agent on any speech, as 'stop' does",
            backchannels,
        )
    options: InterruptionOptions
    if backchannels == "stop":
        options = {"enabled": True, "mode": "vad"}
    elif backchannels == "drop":
        options = {"enabled": True, "mode": "adaptive"}
    else:
        options = {
            "enabled": True,
            "mode": "adaptive",
            "backchannel_boundary": (1.0, _LONGER_THAN_ANY_LINE),
        }
    if min_words:
        options["min_words"] = min_words
    if min_speech_s is not None:
        options["min_duration"] = min_speech_s
    return options


def yield_to_caller(session: AgentSession) -> None:
    """Drop the reply being prepared whenever the caller is speaking while the agent thinks.

    Audio that already plays is left to LiveKit's interruption settings. A reply that runs
    a tool is left alone too, because cutting it off would leave the tool's result for a
    later reply. Only that reply:
    a task's delegate tool runs for the whole task, and the step's own replies inside it
    still yield. Call it once per session, after building it.

    Args:
        session: The call's session.
    """
    running: dict[str, str] = {}

    def drop(why: str) -> None:
        """Interrupt the current reply, unless it plays, runs a tool, or must not stop."""
        speech = session.current_speech
        if speech is None or speech.done() or speech.interrupted:
            return
        if speech.id in running.values():
            return
        if not speech.allow_interruptions:
            return
        logger.info("dropped a reply the caller spoke over: %s", why)
        speech.interrupt()

    def caller(event: UserStateChangedEvent) -> None:
        """The caller went on speaking after their turn was committed."""
        if event.new_state == "speaking" and session.agent_state == "thinking":
            drop("the caller resumed while the agent was thinking")

    def agent(event: AgentStateChangedEvent) -> None:
        """The turn was committed while the caller was already speaking again."""
        if event.new_state == "thinking" and session.user_state == "speaking":
            drop("the turn was committed while the caller was speaking")

    def tools(event: ToolExecutionUpdatedEvent) -> None:
        """Keep which reply runs each tool call, so that reply is never cut."""
        update = event.update
        speech = session.current_speech
        if update.type == "tool_call_started" and speech is not None:
            running[update.function_call.call_id] = speech.id
        elif update.type == "tool_call_ended":
            running.pop(update.call_id, None)

    session.on("user_state_changed", caller)
    session.on("agent_state_changed", agent)
    session.on("tool_execution_updated", tools)
