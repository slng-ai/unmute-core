"""The salon concierge on Pipecat, built by hand on unmute-core.

The agents themselves (prompts, tools, tasks, knowledge) are the shared ``salon``
package's ``AgentSpec``s, the same ones ``livekit/agent.py`` runs. This file only builds
the Pipecat side: the pipeline, the handoffs and the end of the call.

Run it, then open http://localhost:7860 and talk::

    uv run bot.py -t webrtc
"""

import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from pipecat.adapters.schemas.function_schema import FunctionSchema
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
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.settings import LLMSettings
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.workers.runner import WorkerRunner
from pipecat_slng import SlngSTTService, SlngTTSService
from salon import agent as salon
from salon.state import State
from salon.store import Salon
from unmute_core.agent import AgentSpec
from unmute_core.pipecat import (
    Backchannels,
    IdleNudgeObserver,
    TelemetryObserver,
    UnmuteFlows,
    end_call_tool,
    greeting_frames,
    interruption,
    max_duration,
    smart_turn_params,
    telephony_call_info,
    vad_params,
)
from unmute_core.slng import tracking_ids
from unmute_core.state import CallInfo, CallState, call_phone_region, use_phone_region
from unmute_core.telemetry import Langfuse, Telemetry
from unmute_core.turn import turn_window

from vertex import VertexLLM

load_dotenv()
salon.start_knowledge()

CONCIERGE, SPECIALIST = salon.agents(Salon())
MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
BACKCHANNELS: Backchannels = "keep"
WINDOW = turn_window(salon.PACE, "pipecat")
transport_params: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


def salon_flows(
    state: CallState[State],
    llm: GoogleLLMService,
    context: LLMContext,
    aggregators: LLMContextAggregatorPair,
    telemetry: Telemetry | None,
) -> UnmuteFlows:
    """Return the call's agents on Pipecat: the concierge first, the specialist a handoff away."""
    ending = end_call_tool(
        f"{salon.END_CALL}\n{salon.SAY_GOODBYE_WITH_THE_CALL}",
        salon.GOODBYE_LINE,
        lambda: flows.reply_text,
    )

    def handoff(
        name: str,
        description: str,
        to: AgentSpec,
        tools: Callable[[], list[FunctionSchema]],
        announce: str | None = None,
    ) -> FunctionSchema:
        """Return a tool that hands the call to ``to``, with the Pipecat tools ``tools()`` gives."""

        async def handler(params: FunctionCallParams) -> None:
            """Give the call to the other agent."""
            await flows.hand_off(params, to, native_tools=tools(), announce=announce)

        return FunctionSchema(name, description, {}, [], handler=handler)

    concierge_tools = [
        handoff(
            "to_complaints",
            salon.TO_COMPLAINTS,
            SPECIALIST,
            lambda: specialist_tools,
            salon.TO_COMPLAINTS_LINE,
        ),
        ending,
    ]
    specialist_tools = [
        handoff("to_concierge", salon.TO_CONCIERGE, CONCIERGE, lambda: concierge_tools),
        ending,
    ]
    flows = UnmuteFlows(
        CONCIERGE,
        llm=llm,
        context=context,
        aggregators=aggregators,
        state=state,
        telemetry=telemetry,
        native_tools=concierge_tools,
    )
    flows.install()
    return flows


def build_llm() -> VertexLLM:
    """Return Gemini on Vertex. ``UnmuteFlows`` sets its system instruction."""
    return VertexLLM(
        location="eu",
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GoogleLLMService.Settings(
            model=MODEL,
            system_instruction="",
            thinking=GoogleLLMService.ThinkingConfig(thinking_level="MINIMAL"),
        ),
    )


def line_call(runner_args: RunnerArguments) -> CallInfo | None:
    """Return what a phone call carries, from the carrier's handshake. None on a web call.

    A browser call has no caller ID: ``UNMUTE_CALL_START`` can simulate one.
    """
    call_data = getattr(runner_args, "call_data", None)
    return telephony_call_info(call_data, agent_id=salon.AGENT_ID) if call_data else None


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call. The Pipecat runner calls this for each session."""
    transport = await create_transport(runner_args, transport_params)
    state = await salon.new_call(line_call(runner_args))
    telemetry = None
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        telemetry = Telemetry(
            Langfuse.from_env(),
            agent=salon.AGENT_ID,
            call_id=state.call.session_id,
            resource={"deployment.environment.name": os.getenv("ENVIRONMENT", "development")},
        )

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=vad_params(WINDOW)),
            user_turn_strategies=interruption(
                BACKCHANNELS,
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(params=smart_turn_params(WINDOW))
                    )
                ],
            ),
        ),
    )
    llm = build_llm()
    flows = salon_flows(state, llm, context, aggregators, telemetry)
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
                warm_standby_enabled=True,
                world_part="eu-north",
                **tracking_ids(state.call),
            ),
            transport.output(),
            aggregators.assistant(),
        ]
    )
    observers: list[BaseObserver] = []
    if telemetry is not None:
        observers.append(TelemetryObserver(telemetry, model=MODEL, provider="google", flows=flows))
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=observers,
    )

    def traced(name: str, data: dict[str, Any] | None) -> None:
        """Trace each nudge line and the hang-up, not every timer change."""
        if telemetry is not None and name.endswith((".fired", ".hangup")):
            telemetry.event(name, data)

    worker.add_observer(IdleNudgeObserver(salon.IDLE, worker, on_event=traced))
    end_reason = "ended"

    @transport.event_handler("on_client_connected")
    async def greet(transport: BaseTransport, client: Any) -> None:
        """Set the concierge's prompt, greet, and start the call's clock."""
        await worker.queue_frames(
            [
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(system_instruction=flows.main_instruction())
                ),
                *greeting_frames(salon.GREETING, state),
            ]
        )
        max_duration(worker, salon.MAX_DURATION_S)

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
