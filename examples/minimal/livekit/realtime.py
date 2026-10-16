"""An agent on OpenAI's realtime model: it hears, thinks and speaks over one connection.

The session has no STT, LLM or TTS, only the realtime model. ``UnmuteAgent`` still opens
the call with a fixed line and gives the model its prompt filled with the saved values. The
model has no TTS beside it, so it says the line itself, word for word as far as it can. It
words its own line as it calls a tool, so no announce line is set. Tasks, and the ``before_model``, ``wrap_model`` and
``after_model`` hooks, are refused: no model request passes through the library.

Run it against your microphone::

    uv run python realtime.py console
"""

from typing import Annotated

from livekit.agents import AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai
from pydantic import BaseModel, Field
from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks
from unmute_core.greeting import Greeting
from unmute_core.livekit import UnmuteAgent, open_call
from unmute_core.state import CallState
from unmute_core.tools import ToolRegistry, local_tool

PROMPT = "You answer calls for a corner shop. Keep replies short: they are spoken aloud."
GREETING = Greeting(text="Hi, thanks for calling the corner shop. How can I help?")


class State(BaseModel):
    """What the call remembers."""

    order_id: str | None = None


class OrderStatus(BaseModel):
    """Where an order is."""

    order_id: str
    status: str


def order_status(
    order_id: Annotated[str, Field(description="The order id, like A-1042.")],
) -> OrderStatus:
    """Look up an order by its id and return where it is."""
    return OrderStatus(order_id=order_id, status="out for delivery")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one call on the realtime model."""
    session = AgentSession(llm=openai.realtime.RealtimeModel(voice="marin"))
    state = CallState(State)
    call = CallHooks(state)
    tools = ToolRegistry([local_tool(order_status)])
    agent = UnmuteAgent(AgentSpec(PROMPT, tools=tools, save_values=True), state=state, call=call)
    await session.start(agent=agent, room=ctx.room)
    await open_call(session, GREETING, call)


if __name__ == "__main__":
    cli.run_app(server)
