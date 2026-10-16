"""Run ``IdleNudges`` on a LiveKit session, also while a task runs.

LiveKit's ``user_away_timeout`` never starts while a tool runs (voice/agent_session.py:2065,
"a tool in flight will speak when it lands"). A task or a flow runs inside its delegate
tool, so a caller who goes silent during one is never marked away. This reads only the
agent's state and the caller's words, so a running delegate does not stop it.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from livekit.agents import AgentSession, AgentStateChangedEvent, UserInputTranscribedEvent

from unmute_core.call import IDLE, CallHooks
from unmute_core.idle import IdleNudgeController, IdleNudges
from unmute_core.livekit._realtime import speak


def idle_nudges(
    session: AgentSession,
    config: IdleNudges | None,
    *,
    hangup: Callable[[], Awaitable[object]] | None = None,
    on_event: Callable[[str, dict[str, Any] | None], None] | None = None,
    call: CallHooks | None = None,
) -> IdleNudgeController:
    """Say each nudge after its quiet, then the final line, then end the call.

    Leave the session's ``user_away_timeout`` unset (None). The default hang-up closes
    the session, which drops a phone caller only when the session was started with
    ``RoomOptions(delete_room_on_close=True)``.

    Args:
        session: The call's session, before it starts.
        config: The lines and waits. None turns it off.
        hangup: How to end the call, in place of ``session.shutdown()``.
        on_event: Called with each ``idle_nudges.*`` event, for the trace.
        call: The call's hooks. The hang-up first tells them the call ends for ``idle``,
            so the call's end reports that reason and a later failure plays no line.

    Returns:
        The controller, to ``suspend`` and ``resume`` it around something that must not
        draw a nudge.
    """

    async def say(text: str) -> bool:
        """Say a line and return True when the caller cut in."""
        return (await speak(session, text)).interrupted

    async def generate(instructions: str) -> bool:
        """Have the model word a nudge and return True when the caller cut in."""
        return (await session.generate_reply(instructions=instructions)).interrupted

    async def shutdown() -> None:
        """Note the call's end, then close the session once the final line has played."""
        if call is not None:
            call.end_requested(IDLE)
        if hangup is not None:
            await hangup()
        else:
            session.shutdown()

    controller = IdleNudgeController(
        config, speak=say, hangup=shutdown, on_event=on_event, generate=generate
    )

    def on_agent_state(event: AgentStateChangedEvent) -> None:
        """Pass the agent's state on."""
        controller.on_agent_state_changed(event.old_state, event.new_state)

    def on_transcript(event: UserInputTranscribedEvent) -> None:
        """Pass the caller's words on, interim ones too."""
        controller.on_user_transcript(event.transcript)

    session.on("agent_state_changed", on_agent_state)
    session.on("user_input_transcribed", on_transcript)
    session.on("close", lambda _event: controller.close())
    return controller
