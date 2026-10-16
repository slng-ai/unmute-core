"""How the call opens: a fixed line at once, a tool that runs as the call starts, and its end.

``Greeting(text=...)`` says a line. ``Greeting()`` has the model word one, and
``Greeting(speaks_first="user")`` waits for the caller. ``open_call`` holds the caller's
audio, passes the checks before the greeting, then opens the call with it.

A prefetch ``Lookup`` runs ``store_hours`` as the call starts, beside the greeting
(``background=True``), and saves the hours into State. The greeting does not wait for it,
but the first model request does, and the prompt reads ``{{closes_at}}``. The ``report`` hook is
told how the call ended. A part that fails for good says a short line and hangs up.

Run it against your microphone::

    uv run python greeting.py console
"""

import logging
import os

from livekit.agents import AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai, silero, slng
from pydantic import BaseModel
from unmute_core.agent import AgentSpec
from unmute_core.call import CallEnd, CallHooks
from unmute_core.greeting import Greeting
from unmute_core.hooks import Hooks
from unmute_core.livekit import UnmuteAgent, open_call
from unmute_core.prefetch import Lookup, Prefetch
from unmute_core.state import CallState
from unmute_core.tools import local_tool

logger = logging.getLogger(__name__)

GREETING = Greeting(text="Hi, thanks for calling the shop. How can I help?")


PROMPT = "You answer calls for a shop that closes at {{closes_at}}. Keep replies short."


class State(BaseModel):
    """What the call knows."""

    closes_at: str | None = None


class Hours(BaseModel):
    """Today's opening hours."""

    open: str
    close: str


async def store_hours() -> Hours:
    """Read today's opening hours."""
    return Hours(open="9:00", close="18:00")


PREFETCH = Prefetch(
    [Lookup("hours", local_tool(store_hours), {"closes_at": "close"}, background=True)],
    state=State,
)


HOOKS = Hooks()


@HOOKS.on("call.ended")
def report(event: str, end: CallEnd) -> None:
    """Log how the call ended."""
    logger.info("call ended: %s after %.0f s", end.reason, end.duration_s)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one call that opens with the greeting."""
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
    state = CallState(State)
    call = CallHooks(state, hooks=HOOKS, prefetch=PREFETCH)
    agent = UnmuteAgent(AgentSpec(PROMPT), state=state, call=call)
    await session.start(agent=agent, room=ctx.room)
    await open_call(session, GREETING, call)


if __name__ == "__main__":
    cli.run_app(server)
