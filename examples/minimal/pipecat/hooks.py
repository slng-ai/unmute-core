"""Hooks: your own code before, around and after prompts, model requests and tools.

One ``Hooks`` object holds them, and ``CallHooks`` runs them for each call. On a plain
pipeline, ``attach_call(llm=...)`` runs the prompt and model hooks, and
``tools_schema(hooks=call)`` the tool hooks:

- ``PromptSuffix`` adds the same line after the system prompt.
- ``before_model`` sends ``temperature=0`` with every request.
- ``before_tool`` refuses a refund. The tool never runs, and the model reads why.
- ``on("tool.*")`` logs each tool call, and ``on("call.ended")`` how the call ended.

Run it, then open http://localhost:7860 and talk::

    uv run python hooks.py -t webrtc
"""

import logging
import os
from collections.abc import Callable
from dataclasses import replace
from typing import Annotated, Any

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
from unmute_core.call import CallEnd, CallHooks
from unmute_core.hooks import Hooks, PromptSuffix, Refuse, ToolRun
from unmute_core.model import ModelRequest
from unmute_core.pipecat import attach_call, tools_schema
from unmute_core.tools import ToolRegistry, local_tool

logger = logging.getLogger(__name__)

TRANSPORTS: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)
}

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


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call with the hooks."""
    transport = await create_transport(runner_args, TRANSPORTS)
    call = CallHooks(hooks=HOOKS)
    registry = ToolRegistry([local_tool(order_status), local_tool(refund)])
    llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(model="gpt-4.1-mini", system_instruction=PROMPT)
    )
    context = LLMContext(tools=tools_schema(registry, hooks=call))
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
    worker = PipelineWorker(pipeline)
    attach_call(worker, call, context, llm=llm, transport=transport)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
