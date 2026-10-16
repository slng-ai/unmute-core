"""One task with one step that saves a value.

A task is steps that each save values to the call state, in order. ``take_note`` runs when
the caller wants to leave a note, and saves it as ``note``. ``UnmuteFlows`` runs the task on
Pipecat Flows, with a node for each step.

Run it, then open http://localhost:7860 and talk::

    uv run python tasks.py -t webrtc
"""

import os
from collections.abc import Callable
from typing import Any

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
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import LLMSettings
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner
from pipecat_slng import SlngSTTService, SlngTTSService
from pydantic import BaseModel
from unmute_core.agent import AgentSpec
from unmute_core.pipecat import UnmuteFlows
from unmute_core.state import CallState
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.tools import ToolRegistry

TRANSPORTS: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)
}


class State(BaseModel):
    """What the call remembers."""

    note: str | None = None


SPEC = TasksSpec.model_validate(
    {
        "mode": "record",
        "tasks": [
            {
                "name": "take_note",
                "when": "The caller wants to leave a note.",
                "instructions": "Ask what the note says, then save it.",
                "assign": [{"variable": "note", "path": ["note"]}],
            }
        ],
    }
)


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call whose bot can take a note."""
    transport = await create_transport(runner_args, TRANSPORTS)
    state = CallState(State)
    plan = TaskPlan(SPEC, state=State, tools=ToolRegistry([]))
    llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(model="gpt-4.1-mini", system_instruction="")
    )
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    spec = AgentSpec("You answer calls for a shop. Keep replies short.", tasks=plan)
    flows = UnmuteFlows(spec, llm=llm, context=context, aggregators=aggregators, state=state)
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
        """Give the bot its prompt, with the task's tool added."""
        delta = LLMSettings(system_instruction=flows.main_instruction())
        await worker.queue_frame(LLMUpdateSettingsFrame(delta=delta))

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
