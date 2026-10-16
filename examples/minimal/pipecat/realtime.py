"""An agent on OpenAI's realtime service: it hears, thinks and speaks over one connection.

The pipeline has no STT or TTS, only the realtime service. ``UnmuteFlows`` gives it the
prompt filled with the saved values, and its tools. ``open_call`` takes the service too:
it opens the call with a fixed line the model says itself, word for word as far as it
can. It words its own line as it calls a tool, so no announce line is set. Tasks, and the ``before_model``,
``wrap_model`` and ``after_model`` hooks, are refused: no model request passes through the
library.

Run it, then open http://localhost:7860 and talk::

    uv run python realtime.py -t webrtc
"""

import os
from collections.abc import Callable
from typing import Annotated, Any

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner
from pydantic import BaseModel, Field
from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks
from unmute_core.greeting import Greeting
from unmute_core.pipecat import UnmuteFlows, open_call
from unmute_core.state import CallState
from unmute_core.tools import ToolRegistry, local_tool

TRANSPORTS: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)
}
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


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call on the realtime service."""
    transport = await create_transport(runner_args, TRANSPORTS)
    llm = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            session_properties=events.SessionProperties(
                audio=events.AudioConfiguration(output=events.AudioOutput(voice="marin"))
            ),
        ),
    )
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)
    state = CallState(State)
    call = CallHooks(state)
    tools = ToolRegistry([local_tool(order_status)])
    flows = UnmuteFlows(
        AgentSpec(PROMPT, tools=tools, save_values=True),
        llm=llm,
        context=context,
        aggregators=aggregators,
        state=state,
        call=call,
    )
    flows.install()
    pipeline = Pipeline(
        [transport.input(), aggregators.user(), llm, transport.output(), aggregators.assistant()]
    )
    worker = PipelineWorker(pipeline)

    @transport.event_handler("on_client_connected")
    async def greet(transport: BaseTransport, client: Any) -> None:
        """Open the call with the greeting, said by the model."""
        await open_call(worker, GREETING, call, context=context, llm=llm, transport=transport)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()
    await call.end("ended")


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
