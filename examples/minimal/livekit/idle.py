"""Speak to a caller who went quiet, then hang up.

``IdleNudges`` holds up to three lines and the quiet before each. ``idle_nudges`` runs them
on the session: no model request, only code saying the lines. Leave the session's
``user_away_timeout`` unset, because LiveKit's own timer waits for a running tool.

``IdleNudges(first_nudge_delay_seconds=None, second_nudge_delay_seconds=None,
hangup_delay_seconds=30)`` ends a quiet call with no nudge. ``hangup_delay_seconds=None``
only nudges and never hangs up.

Run it against your microphone::

    uv run python idle.py console
"""

import os

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.voice import room_io
from livekit.plugins import openai, silero, slng
from unmute_core.idle import IdleNudges
from unmute_core.livekit import idle_nudges

IDLE = IdleNudges(
    first_nudge_delay_seconds=10,
    first_nudge_text="Are you still there?",
    second_nudge_delay_seconds=10,
    second_nudge_text="I'm here whenever you're ready.",
    hangup_delay_seconds=10,
    final_hangup_text="I'll let you go for now. Goodbye.",
)

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one call that nudges a quiet caller and then ends."""
    slng_host = f"{os.getenv('SLNG_WORLD_PART', 'eu-north')}.api.slng.ai"
    stt = slng.STT(slng_base_url=slng_host, model="soniox/speech-ai:rt-v5")
    session = AgentSession(
        stt=stt,
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=slng.TTS(
            slng_base_url=slng_host,
            model="slng/fish/tts:s2.1-pro",
            voice="8d8aec63a56c460cb7def795bdf754fa",
        ),
        vad=silero.VAD.load(),
    )
    stt.attach_to_session(session)
    idle_nudges(session, IDLE)
    agent = Agent(instructions="You answer calls for a shop. Keep replies short.")
    await session.start(
        agent=agent, room=ctx.room, room_options=room_io.RoomOptions(delete_room_on_close=True)
    )


if __name__ == "__main__":
    cli.run_app(server)
