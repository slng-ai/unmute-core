"""Tools on a plain Pipecat bot: a local tool, the built-in clock and a save tool.

``ToolRegistry`` holds the tools. ``tools_schema`` gives them, with their handlers, to the
``LLMContext``. The order tool says its announce line first, with the brand the call
started with, and runs once the line has played. It returns a typed ``OrderStatus``. The
registry's reply line then answers in place of the model, and says the arrival day in the
call's language: "tomorrow, October 16". ``save_call_start`` saves the dispatch's
values, here a fixed brand, into State. ``current_datetime_tool`` answers in one time
zone. ``values_tool`` lets the model save State values.

Run it, then open http://localhost:7860 and talk::

    uv run python tools.py -t webrtc
"""

import os
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner
from pipecat_slng import SlngSTTService, SlngTTSService
from pydantic import BaseModel, Field
from unmute_core.call import save_call_start
from unmute_core.model import render_prompt
from unmute_core.pipecat import tools_schema
from unmute_core.state import CallState
from unmute_core.tools import (
    ToolRegistry,
    ToolReply,
    current_datetime_tool,
    local_tool,
    values_tool,
)

TRANSPORTS: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)
}


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


PROMPT = "You answer calls for {{brand}}. Keep replies short."


class State(BaseModel):
    """What the call remembers."""

    brand: str | None = None
    caller_name: str | None = None
    delivery_note: str | None = None


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call with a bot that can look up an order, tell the time and save values."""
    transport = await create_transport(runner_args, TRANSPORTS)
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
    llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(
            model="gpt-4.1-mini",
            system_instruction=render_prompt(PROMPT, state),
        )
    )
    context = LLMContext(tools=tools_schema(registry, state, language=LANGUAGE, timezone=TIMEZONE))
    aggregators = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    pipeline = Pipeline(
        [
            transport.input(),
            SlngSTTService(
                api_key=os.environ["SLNG_API_KEY"],
                world_part=os.getenv("SLNG_WORLD_PART", "eu-north"),
                model="soniox/speech-ai:rt-v5",
                language=Language.EN,
            ),
            aggregators.user(),
            llm,
            SlngTTSService(
                api_key=os.environ["SLNG_API_KEY"],
                world_part=os.getenv("SLNG_WORLD_PART", "eu-north"),
                model="slng/fish/tts:s2.1-pro",
                voice="8d8aec63a56c460cb7def795bdf754fa",
                language=Language.EN,
            ),
            transport.output(),
            aggregators.assistant(),
        ]
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(PipelineWorker(pipeline))
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
