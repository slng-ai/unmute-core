"""Speak to a caller who went quiet, then hang up.

``IdleNudges`` holds up to three lines and the quiet before each. ``IdleNudgeObserver``
watches the pipeline's frames, says the lines as code, makes no model request, and ends the
call after the last one.

``IdleNudges(first_nudge_delay_seconds=None, second_nudge_delay_seconds=None,
hangup_delay_seconds=30)`` ends a quiet call with no nudge. ``hangup_delay_seconds=None``
only nudges and never hangs up.

Run it, then open http://localhost:7860 and talk::

    uv run python idle.py -t webrtc
"""

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
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner
from pipecat_slng import SlngSTTService, SlngTTSService
from unmute_core.idle import IdleNudges
from unmute_core.pipecat import IdleNudgeObserver

TRANSPORTS: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)
}
IDLE = IdleNudges(
    first_nudge_delay_seconds=10,
    first_nudge_text="Are you still there?",
    second_nudge_delay_seconds=10,
    second_nudge_text="I'm here whenever you're ready.",
    hangup_delay_seconds=10,
    final_hangup_text="I'll let you go for now. Goodbye.",
)


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call that nudges a quiet caller and then ends."""
    transport = await create_transport(runner_args, TRANSPORTS)
    llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(
            model="gpt-4.1-mini",
            system_instruction="You answer calls for a shop. Keep replies short.",
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
    worker = PipelineWorker(pipeline)
    worker.add_observer(IdleNudgeObserver(IDLE, worker))
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
