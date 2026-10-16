"""Traces for a plain Pipecat bot, sent to Langfuse.

``TelemetryObserver`` turns the pipeline's frames into turns, model requests, tool calls
and speech spans. Add it to the worker with metrics on, and leave Pipecat's own tracing off.
Needs ``OPENAI_API_KEY``, ``SLNG_API_KEY`` and the Langfuse keys.

Run it, then open http://localhost:7860 and talk::

    uv run python telemetry.py -t webrtc
"""

import os
import uuid
from collections.abc import Callable
from typing import Any

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
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
from unmute_core.pipecat import TelemetryObserver
from unmute_core.telemetry import Langfuse, Telemetry

TRANSPORTS: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)
}
MODEL = "gpt-4.1-mini"


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call and send its trace to Langfuse."""
    transport = await create_transport(runner_args, TRANSPORTS)
    telemetry = Telemetry(
        Langfuse.from_env(), agent="minimal", call_id=runner_args.session_id or uuid.uuid4().hex
    )
    llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(
            model=MODEL, system_instruction="You answer calls for a shop. Keep replies short."
        )
    )
    context = LLMContext()
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
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[TelemetryObserver(telemetry, model=MODEL, provider="openai")],
    )
    telemetry.start_call()
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        await telemetry.close(reason="ended")


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
