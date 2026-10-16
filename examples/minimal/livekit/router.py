"""The SLNG Context Router as the agent's LLM.

``RouterModelCall`` sends each request to the router with the agent and call ids. Each
request carries your own router config, which sends the model to OpenAI on your key. The
router fills the prompt from the call state's values. ``ModelCallLLM`` makes it the LLM of a
plain LiveKit agent, so the session has no LLM plugin. Needs ``OPENAI_API_KEY``,
``SLNG_API_KEY`` and ``SLNG_WORLD_PART`` (or ``SLNG_ROUTER_URL``). ``SLNG_MODEL`` is optional.

Run it against your microphone::

    uv run python router.py console
"""

import os

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import silero, slng
from pydantic import BaseModel, Field, SecretStr
from unmute_core.livekit import ModelCallLLM
from unmute_core.slng import (
    OpenAIEndpoint,
    RouterConfig,
    RouterModelCall,
    SlngSettings,
)
from unmute_core.state import CallState


class RouterSettings(SlngSettings):
    """The router settings, with your own OpenAI key. ``SLNG_*`` variables override them.

    Attributes:
        model: The model your router config sends to OpenAI.
        openai_api_key: Your OpenAI key, from ``OPENAI_API_KEY``. Every request carries a
            router config that sends ``model`` to OpenAI with this key.
    """

    model: str = "gpt-5.6-luna"
    openai_api_key: SecretStr = Field(min_length=1, validation_alias="OPENAI_API_KEY")

    def config(self) -> RouterConfig:
        """Return your router config: one model, on your OpenAI key."""
        endpoint = OpenAIEndpoint(
            url="https://api.openai.com/v1", api_key=self.openai_api_key.get_secret_value()
        )
        return RouterConfig.single(self.model, endpoint)


class State(BaseModel):
    """What the call remembers."""

    caller_name: str | None = None


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one call whose model requests go through the router."""
    state = CallState(State)
    state.save({"caller_name": "Sam"})
    settings = RouterSettings()
    router = RouterModelCall(
        state.call,
        model=settings.model,
        config=settings.config(),
    )
    slng_host = f"{os.getenv('SLNG_WORLD_PART', 'eu-north')}.api.slng.ai"
    stt = slng.STT(slng_base_url=slng_host, model="soniox/speech-ai:rt-v5")
    session = AgentSession(
        stt=stt,
        tts=slng.TTS(
            slng_base_url=slng_host,
            model="slng/fish/tts:s2.1-pro",
            voice="8d8aec63a56c460cb7def795bdf754fa",
        ),
        vad=silero.VAD.load(),
    )
    stt.attach_to_session(session)
    agent = Agent(
        instructions="You talk to {{caller_name}}. Keep replies short: they are spoken aloud.",
        llm=ModelCallLLM(router, state=state),
    )
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
