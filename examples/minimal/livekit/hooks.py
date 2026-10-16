"""Hooks: your own code before, around and after prompts, model requests and tools.

One ``Hooks`` object holds them, and ``CallHooks`` runs them for each call:

- ``PromptSuffix`` adds the same line after every prompt, the task steps' too.
- ``before_model`` sends ``temperature=0`` with every request.
- ``before_tool`` refuses a refund. The tool never runs, and the model reads why.
- ``on("tool.*")`` logs each tool call, and ``on("call.ended")`` how the call ended.

Run it against your microphone::

    uv run python hooks.py console
"""

import logging
import os
from dataclasses import replace
from typing import Annotated

from livekit.agents import AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai, silero, slng
from pydantic import BaseModel, Field
from unmute_core.agent import AgentSpec
from unmute_core.call import CallEnd, CallHooks
from unmute_core.hooks import Hooks, PromptSuffix, Refuse, ToolRun
from unmute_core.livekit import UnmuteAgent
from unmute_core.model import ModelRequest
from unmute_core.tools import ToolRegistry, local_tool

logger = logging.getLogger(__name__)

PROMPT = "You answer calls for a shop. You can look up orders and refund them."


class OrderStatus(BaseModel):
    """Where an order is."""

    order_id: str
    status: str


def order_status(
    order_id: Annotated[str, Field(description="The order id, like A-1042.")],
) -> OrderStatus:
    """Look up an order by its id and return where it is."""
    return OrderStatus(order_id=order_id, status="out for delivery")


class Refunded(BaseModel):
    """An order that was refunded."""

    order_id: str
    refunded: bool


def refund(order_id: Annotated[str, Field(description="The order id to refund.")]) -> Refunded:
    """Refund an order."""
    return Refunded(order_id=order_id, refunded=True)


HOOKS = Hooks()
HOOKS.add(PromptSuffix("Keep every reply to one or two short sentences."))


@HOOKS.before_model
def steady(request: ModelRequest) -> ModelRequest:
    """Send every request with temperature 0."""
    return replace(request, params=request.params | {"temperature": 0})


@HOOKS.before_tool
def no_refunds(run: ToolRun) -> Refuse | None:
    """Refuse refunds on the phone."""
    if run.name == "refund":
        return Refuse("Refunds are done by email. Offer to send the form.")
    return None


@HOOKS.on("tool.*")
def log_tool(event: str, data: dict) -> None:
    """Log each tool call's stage."""
    logger.info("%s %s", event, data["name"])


@HOOKS.on("call.ended")
def report(event: str, end: CallEnd) -> None:
    """Log how the call ended."""
    logger.info("call ended: %s after %.0f s", end.reason, end.duration_s)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one call with the hooks."""
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
    call = CallHooks(hooks=HOOKS)
    tools = ToolRegistry([local_tool(order_status), local_tool(refund)])
    agent = UnmuteAgent(AgentSpec(PROMPT, tools=tools), call=call)
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
