"""The SLNG Context Router as the bot's LLM.

``RouterModelCall`` sends each request to the router with the agent and call ids. Each
request carries your own router config, which sends the model to OpenAI on your key. The
router fills the prompt from the call state's values. ``ModelCallLLMService`` makes it the
LLM of a plain Pipecat pipeline. Needs ``OPENAI_API_KEY``, ``SLNG_API_KEY`` and
``SLNG_WORLD_PART`` (or ``SLNG_ROUTER_URL``). ``SLNG_MODEL`` is optional.

Run it, then open http://localhost:7860 and talk::

    uv run python router.py -t webrtc
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
from pydantic import BaseModel, Field, SecretStr
from unmute_core.pipecat import ModelCallLLMService
from unmute_core.slng import (
    OpenAIEndpoint,
    RouterConfig,
    RouterModelCall,
    SlngSettings,
)
from unmute_core.state import CallState

TRANSPORTS: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)
}


class RouterSettings(SlngSettings):
    """The router settings, with your own OpenAI key. ``SLNG_*`` variables override them.

    Attributes:
        model: The model your router config sends to OpenAI.
        openai_api_key: Your OpenAI key, from ``OPENAI_API_KEY``. Every request carries a
            router config that sends ``model`` to OpenAI with this key.
    """

    model: str = "gpt-5.6-luna"
    openai_api_key: SecretStr = Field(min_length=1, validation_alias="OPENAI_API_KEY")

    def config(self) -> RouterConfig:
        """Return your router config: one model, on your OpenAI key."""
        endpoint = OpenAIEndpoint(
            url="https://api.openai.com/v1", api_key=self.openai_api_key.get_secret_value()
        )
        return RouterConfig.single(self.model, endpoint)


class State(BaseModel):
    """What the call remembers."""

    caller_name: str | None = None


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call whose model requests go through the router."""
    transport = await create_transport(runner_args, TRANSPORTS)
    state = CallState(State)
    state.save({"caller_name": "Sam"})
    settings = RouterSettings()
    router = RouterModelCall(
        state.call,
        model=settings.model,
        config=settings.config(),
    )
    llm = ModelCallLLMService(
        router,
        state=state,
        settings=OpenAILLMService.Settings(
            model=router.model,
            system_instruction="You talk to {{caller_name}}. Keep replies short.",
        ),
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
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(PipelineWorker(pipeline))
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
