"""A handoff between two agents, with a line said first.

Both agents are ``UnmuteAgent``s that share one call state. The sales agent's tool returns
``hand_off(...)``: the line is said, the next agent hears the last few messages, and it
speaks first.

Run it against your microphone::

    uv run python handoff.py console
"""

import os

from livekit.agents import AgentServer, AgentSession, JobContext, cli, function_tool
from livekit.plugins import openai, silero, slng
from unmute_core.agent import AgentSpec
from unmute_core.handoff import HandoffContext
from unmute_core.livekit import UnmuteAgent


class Sales(UnmuteAgent):
    """Answers sales questions and passes complaints to support."""

    def __init__(self) -> None:
        """Build the sales agent."""
        super().__init__(AgentSpec("You are sales at a shop. Keep replies short.", name="sales"))

    @function_tool
    async def to_support(self) -> UnmuteAgent | str:
        """The caller has a complaint or a problem with an order."""
        return await self.hand_off(
            Support(),
            announce="Putting you through to support.",
            context=HandoffContext(history="last_n", max_messages=6),
        )


class Support(UnmuteAgent):
    """Takes complaints and problems with orders."""

    def __init__(self) -> None:
        """Build the support agent."""
        super().__init__(
            AgentSpec("You are support at a shop. Keep replies short.", name="support")
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one call that starts with the sales agent."""
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
    await session.start(agent=Sales(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
