"""Traces for a plain LiveKit agent, sent to Langfuse.

``trace_session`` turns the session's events into turns, transcripts and tool calls.
``traced_llm`` in the agent's ``llm_node`` adds an ``llm`` span for each request, and
``traced_tts`` in the ``tts_node`` adds the speech spans. The model is LiveKit's own OpenAI
plugin, which keeps its own client and retries. It needs ``OPENAI_API_KEY``, ``SLNG_API_KEY``
and the Langfuse keys.

Run it against your microphone::

    uv run python telemetry.py console
"""

import os
from collections.abc import AsyncIterable, AsyncIterator

from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, ModelSettings, cli, llm
from livekit.plugins import openai, silero, slng
from unmute_core.livekit import SessionTrace, trace_session, traced_llm, traced_tts
from unmute_core.telemetry import Langfuse, Telemetry


class TracedAgent(Agent):
    """An agent whose model requests and speech are recorded as spans."""

    def __init__(self, trace: SessionTrace, **kwargs: object) -> None:
        """Keep the session's trace and build the agent."""
        super().__init__(**kwargs)  # ty: ignore[invalid-argument-type]
        self.trace = trace

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterator[llm.ChatChunk | str]:
        """Run the session's LLM plugin and record each request as an ``llm`` span."""
        async for chunk in traced_llm(self, self.trace, chat_ctx, tools, model_settings):
            yield chunk

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        """Run LiveKit's own speech and record it."""
        async for frame in traced_tts(self, self.trace, text, model_settings):
            yield frame


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one call and send its trace to Langfuse."""
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
    telemetry = Telemetry(Langfuse.from_env(), agent="minimal", call_id=ctx.room.name)
    trace = trace_session(session, telemetry)
    agent = TracedAgent(
        trace,
        instructions="You answer calls for a shop. Keep replies short: they are spoken aloud.",
    )
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
