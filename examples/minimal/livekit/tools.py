"""Tools on a plain LiveKit agent: a local tool, the built-in clock and a save tool.

``ToolRegistry`` holds the tools. ``function_tools`` hands them to ``Agent(tools=...)``.
The order tool says its announce line first, with the brand the call started with, and
runs once the line has played. It returns a typed ``OrderStatus``. The registry's reply
line then answers in place of the model, and says the arrival day in the call's language:
"tomorrow, October 16". ``save_call_start`` saves the dispatch's values, here a
fixed brand, into State. ``current_datetime_tool`` answers in one time zone. ``values_tool``
lets the model save State values, and a listener keeps the prompt showing them.

Run it against your microphone::

    uv run python tools.py console
"""

import asyncio
import os
from datetime import date, datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai, silero, slng
from pydantic import BaseModel, Field
from unmute_core.call import save_call_start
from unmute_core.livekit import function_tools
from unmute_core.model import render_prompt
from unmute_core.state import CallState, StateEvent
from unmute_core.tools import (
    ToolRegistry,
    ToolReply,
    current_datetime_tool,
    local_tool,
    values_block,
    values_tool,
)

PROMPT = "You answer calls for {{brand}}. Keep replies short: they are spoken aloud."


class State(BaseModel):
    """What the call remembers."""

    brand: str | None = None
    caller_name: str | None = None
    delivery_note: str | None = None


TIMEZONE = "Europe/Madrid"
LANGUAGE = "en"
"""The call's language. A reply line says its dates in it: try ``es`` for Spanish."""


class OrderStatus(BaseModel):
    """Where an order is, and the day it arrives."""

    order_id: str
    status: str
    arrives: date


def order_status(
    order_id: Annotated[str, Field(description="The order id, like A-1042.")],
) -> OrderStatus:
    """Look up an order by its id and return where it is and the day it arrives."""
    tomorrow = datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=1)
    return OrderStatus(order_id=order_id, status="out for delivery", arrives=tomorrow)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one call with an agent that can look up an order, tell the time and save values."""
    state = CallState(State, call_start={"brand": "optional"})
    save_call_start(state, {"brand": "Corner Shop"})
    registry = ToolRegistry(
        [
            local_tool(
                order_status,
                announce="One moment, I am checking that order at {{brand}}.",
                announce_wait=True,
            ),
            current_datetime_tool(TIMEZONE),
            values_tool(state),
        ],
        replies=[
            ToolReply(
                tool="order_status",
                say=["Your order {{result.order_id}} arrives {{result.arrives}}."],
            )
        ],
    )
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
    agent = Agent(
        instructions=render_prompt(PROMPT, state),
        tools=function_tools(registry, state, language=LANGUAGE, timezone=TIMEZONE),
    )
    updates: set[asyncio.Task[None]] = set()

    def refresh(event: StateEvent) -> None:
        """Show the saved values in the prompt."""
        task = asyncio.create_task(
            agent.update_instructions(render_prompt(PROMPT, state) + "\n\n" + values_block(state))
        )
        updates.add(task)
        task.add_done_callback(updates.discard)

    state.listen(refresh)
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
