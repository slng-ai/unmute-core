"""A handoff between two agents, with a line said first, and an end-call tool.

The call starts with the sales agent. Its handoff tool calls ``flows.hand_off`` with the
support agent: the line is said, support hears the last few messages, and it speaks first.
Neither agent has tasks: each ``AgentSpec`` holds only a prompt and a name.
``end_call_tool`` ends the call after the goodbye, with one model request.

Run it, then open http://localhost:7860 and talk::

    uv run python handoff.py -t webrtc
"""

import os
from collections.abc import Callable
from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMUpdateSettingsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import LLMSettings
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner
from pipecat_slng import SlngSTTService, SlngTTSService
from unmute_core.agent import AgentSpec
from unmute_core.handoff import HandoffContext
from unmute_core.pipecat import UnmuteFlows, end_call_tool

TRANSPORTS: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)
}
SALES = "You are sales at a shop. Keep replies short."
SUPPORT = "You are support at a shop. Keep replies short."
END_CALL = "End the call when the caller is done. Say goodbye in the same reply."


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call that starts with sales and can move to support."""
    transport = await create_transport(runner_args, TRANSPORTS)
    llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(model="gpt-4.1-mini", system_instruction="")
    )
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    ending = end_call_tool(END_CALL, "Goodbye.", lambda: flows.reply_text)

    async def to_support(params: FunctionCallParams) -> None:
        """Give the call to support, after a line, with the last six messages."""
        await flows.hand_off(
            params,
            AgentSpec(SUPPORT, name="support"),
            native_tools=[ending],
            announce="Putting you through to support.",
            context=HandoffContext(history="last_n", max_messages=6),
        )

    handoff = FunctionSchema(
        "to_support",
        "The caller has a complaint or a problem with an order.",
        {},
        [],
        handler=to_support,
    )
    flows = UnmuteFlows(
        AgentSpec(SALES, name="sales"),
        llm=llm,
        context=context,
        aggregators=aggregators,
        native_tools=[handoff, ending],
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
    async def prompt(transport: BaseTransport, client: Any) -> None:
        """Give the bot the sales prompt."""
        delta = LLMSettings(system_instruction=flows.main_instruction())
        await worker.queue_frame(LLMUpdateSettingsFrame(delta=delta))

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
