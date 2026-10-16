"""How the call opens: a fixed line at once, a tool that runs as the call starts, and its end.

``Greeting(text=...)`` says a line. ``Greeting()`` has the model word one, and
``Greeting(speaks_first="user")`` waits for the caller. ``open_call`` passes the checks
before the greeting, then queues it. ``GreetingHold`` holds the caller's audio until then.

A prefetch ``Lookup`` runs ``store_hours`` as the call starts, beside the greeting
(``background=True``), and saves the hours into State. The greeting does not wait for it,
but the first model request does, and the prompt reads ``{{closes_at}}``. ``UnmuteFlows``
fills the prompt before each request. The ``report`` hook is
told how the call ended. A part that fails for good says a short line and hangs up.

Run it, then open http://localhost:7860 and talk::

    uv run python greeting.py -t webrtc
"""

import logging
import os
from collections.abc import Callable
from typing import Any

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
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner
from pipecat_slng import SlngSTTService, SlngTTSService
from pydantic import BaseModel
from unmute_core.agent import AgentSpec
from unmute_core.call import CallEnd, CallHooks
from unmute_core.greeting import Greeting
from unmute_core.hooks import Hooks
from unmute_core.pipecat import GreetingHold, UnmuteFlows, open_call
from unmute_core.prefetch import Lookup, Prefetch
from unmute_core.state import CallState
from unmute_core.tools import local_tool

logger = logging.getLogger(__name__)

TRANSPORTS: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)
}
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


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call that opens with the greeting."""
    transport = await create_transport(runner_args, TRANSPORTS)
    llm = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4.1-mini"))
    context = LLMContext()
    hold = GreetingHold()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(), user_mute_strategies=[hold]
        ),
    )
    state = CallState(State)
    call = CallHooks(state, hooks=HOOKS, prefetch=PREFETCH)
    flows = UnmuteFlows(
        AgentSpec(PROMPT), llm=llm, context=context, aggregators=aggregators, call=call
    )
    flows.install()
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

    @transport.event_handler("on_client_connected")
    async def greet(transport: BaseTransport, client: Any) -> None:
        """Open the call with the greeting."""
        await open_call(worker, GREETING, call, context=context, hold=hold, transport=transport)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()
    await call.end("ended")


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
