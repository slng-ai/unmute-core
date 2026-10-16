"""Parcel Desk on Pipecat, defined as data on unmute-core.

The agent is the shared ``desk`` package, built from data: ``DESK.agent(state)`` is the
call's ``AgentSpec``, the same one ``livekit/agent.py`` runs. This file only builds the
Pipecat side: the pipeline, its speech, and the call's start and end. Every model request
goes through the SLNG Context Router, through a ``ModelCallLLMService`` built on the
spec's own model call.

Run it, then open http://localhost:7860 and talk::

    uv run bot.py -t webrtc
"""

import os
from collections.abc import Callable
from typing import Any

from desk.agent import build
from dotenv import load_dotenv
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMUpdateSettingsFrame
from pipecat.observers.base_observer import BaseObserver
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
from pipecat.services.settings import LLMSettings
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.workers.runner import WorkerRunner
from pipecat_slng import SlngSTTService, SlngTTSService
from unmute_core.model import model_name
from unmute_core.pipecat import (
    ModelCallLLMService,
    TelemetryObserver,
    UnmuteFlows,
    greeting_frames,
    interruption,
    max_duration,
    smart_turn_params,
    telephony_call_info,
    vad_params,
)
from unmute_core.slng import tracking_ids
from unmute_core.state import CallInfo, call_phone_region, use_phone_region
from unmute_core.telemetry import Langfuse, Telemetry

load_dotenv()

DESK = build()
WINDOW = DESK.window("pipecat")
transport_params: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


def line_call(runner_args: RunnerArguments) -> CallInfo | None:
    """Return what a phone call carries, from the carrier's handshake. None on a web call.

    A browser call has no caller ID: ``UNMUTE_CALL_START`` can simulate one.
    """
    if call_data := getattr(runner_args, "call_data", None):
        return telephony_call_info(call_data, agent_id=DESK.agent_id)
    return None


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call. The Pipecat runner calls this for each session."""
    transport = await create_transport(runner_args, transport_params)
    state = await DESK.new_call(line_call(runner_args))
    telemetry = None
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        telemetry = Telemetry(
            Langfuse.from_env(), agent=DESK.agent_id, call_id=state.call.session_id
        )

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=vad_params(WINDOW)),
            user_turn_strategies=interruption(
                "keep",
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(params=smart_turn_params(WINDOW))
                    )
                ],
            ),
        ),
    )
    agent = DESK.agent(state)
    model_call = agent.model
    llm = ModelCallLLMService(
        model_call,
        state=state,
        settings=OpenAILLMService.Settings(model=model_name(model_call), system_instruction=""),
    )
    flows = UnmuteFlows(
        agent, llm=llm, context=context, aggregators=aggregators, state=state, telemetry=telemetry
    )
    flows.install()
    slng_key = os.environ["SLNG_API_KEY"]
    pipeline = Pipeline(
        [
            transport.input(),
            SlngSTTService(
                api_key=slng_key,
                model="soniox/speech-ai:rt-v5",
                language=Language("en"),
                world_part="eu-north",
                **tracking_ids(state.call),
            ),
            aggregators.user(),
            llm,
            SlngTTSService(
                api_key=slng_key,
                voice="8d8aec63a56c460cb7def795bdf754fa",
                model="slng/fish/tts:s2.1-pro",
                language=Language("en"),
                world_part="eu-north",
                **tracking_ids(state.call),
            ),
            transport.output(),
            aggregators.assistant(),
        ]
    )
    observers: list[BaseObserver] = []
    if telemetry is not None:
        observers.append(
            TelemetryObserver(telemetry, model=model_name(model_call), provider="slng", flows=flows)
        )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=observers,
    )
    end_reason = "ended"

    @transport.event_handler("on_client_connected")
    async def greet(transport: BaseTransport, client: Any) -> None:
        """Set the agent's prompt, greet, and start the call's clock."""
        await worker.queue_frames(
            [
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(system_instruction=flows.main_instruction())
                ),
                *greeting_frames(DESK.greeting, state),
            ]
        )
        max_duration(worker, DESK.max_duration_s)

    @transport.event_handler("on_client_disconnected")
    async def hang_up(transport: BaseTransport, client: Any) -> None:
        """Make the saves steps held for the call's end, then stop the call."""
        nonlocal end_reason
        end_reason = "caller_hung_up"
        await flows.settle()
        await worker.cancel()

    if telemetry is not None:
        telemetry.start_call()
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    try:
        with use_phone_region(lambda: call_phone_region(state.call)):
            await runner.run()
    finally:
        await flows.settle()
        if telemetry is not None:
            await telemetry.close(state=state, reason=end_reason)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
