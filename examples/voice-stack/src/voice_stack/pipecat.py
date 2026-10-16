"""Run a ``VoiceApp`` on Pipecat.

An example's ``pipecat/bot.py`` is a few lines over ``run_bot``. Each call runs the same
way:

1. Read what the line carries (a carrier's handshake, or nothing on a browser call).
2. ``app.new_call`` builds the call's state, with its prefetch read in, from the
   ``call_start`` object of the runner's request body. A request that lacks a required
   value, or holds a wrong one, gets no pipeline, and the log says which value.
3. The pipeline gets SLNG speech in the stack's region and Gemini on Vertex EU.
4. ``UnmuteFlows`` runs the app's ``AgentSpec`` on the pipeline's context.
5. When the caller connects, the greeting plays and the call's longest length starts.

Run an example, then open http://localhost:7860 and talk::

    uv run bot.py -t webrtc
"""

import logging
import os
from collections.abc import Callable
from typing import Any

from google import genai
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    LLMUpdateSettingsFrame,
)
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
from unmute_core.call import CallStartError
from unmute_core.pipecat import (
    IdleNudgeObserver,
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
from unmute_core.state import CallInfo, CallState, call_phone_region, use_phone_region
from unmute_core.telemetry import Langfuse, Telemetry
from unmute_core.turn import turn_window

from voice_stack.app import Handoff, VoiceApp
from voice_stack.settings import StackSettings

logger = logging.getLogger("voice_stack")

transport_params: dict[str, Callable[..., Any]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}
"""The transports ``uv run bot.py -t ...`` may pick."""


def vertex_client(api_key: str, location: str) -> genai.Client:
    """Return a GenAI client pinned to one Vertex location, with an API key.

    Pipecat's Google service reaches Vertex only with Google Cloud credentials. This client
    uses an API key instead, on the location's own host, so EU requests stay in the EU.

    Args:
        api_key: The Vertex API key.
        location: A Vertex region, or ``eu``, ``us`` or ``global`` for a multi-region host.
    """
    if location in ("us", "eu"):
        host = f"aiplatform.{location}.rep.googleapis.com"
    elif location == "global":
        host = "aiplatform.googleapis.com"
    else:
        host = f"{location}-aiplatform.googleapis.com"
    return genai.Client(
        vertexai=True,
        api_key=api_key,
        location=location,
        http_options=genai.types.HttpOptions(
            base_url=f"https://{host}/v1beta1",
            base_url_resource_scope=genai.types.ResourceScope.COLLECTION,
            api_version="v1beta1",
            headers={"X-Vertex-AI-LLM-Shared-Request-Type": "priority"},
        ),
    )


class VertexLLM(GoogleLLMService):
    """Pipecat's Google LLM service, with a Vertex API-key client."""

    def __init__(self, *, location: str, **kwargs: Any) -> None:
        """Remember the location, then build the service.

        Args:
            location: The Vertex location.
            **kwargs: The service's own arguments, like ``api_key`` and ``settings``.
        """
        self._vertex_location = location
        super().__init__(**kwargs)

    def create_client(self) -> None:
        """Build the Vertex client the service calls through."""
        self._client = vertex_client(self._api_key, self._vertex_location)


def build_llm(settings: StackSettings) -> GoogleLLMService:
    """Return Gemini on Vertex, with minimal thinking. ``UnmuteFlows`` sets its prompt."""
    return VertexLLM(
        location=settings.vertex_location,
        api_key=settings.google_api_key.get_secret_value(),
        settings=GoogleLLMService.Settings(
            model=settings.llm_model,
            system_instruction="",
            thinking=GoogleLLMService.ThinkingConfig(thinking_level="MINIMAL"),
        ),
    )


def line_call(app: VoiceApp, runner_args: RunnerArguments) -> CallInfo | None:
    """Return what a phone call carries, from the carrier's handshake. None on a web call."""
    call_data = getattr(runner_args, "call_data", None)
    return telephony_call_info(call_data, agent_id=app.agent_id) if call_data else None


def dispatched(runner_args: RunnerArguments) -> dict[str, Any] | None:
    """Return the ``call_start`` object the call was dispatched with, or None.

    A web call carries it in the runner's request body. A phone call carries it in the
    body of the carrier's handshake, which ``create_transport`` reads.
    """
    for body in (
        getattr(runner_args, "body", None),
        (getattr(runner_args, "call_data", None) or {}).get("body"),
    ):
        if isinstance(body, dict) and isinstance(body.get("call_start"), dict):
            return body["call_start"]
    return None


def build_pipeline(
    app: VoiceApp,
    state: CallState,
    transport: BaseTransport,
    settings: StackSettings,
    telemetry: Telemetry | None,
) -> tuple[PipelineWorker, UnmuteFlows]:
    """Return the call's worker and the flows that run the app's agent on it.

    Args:
        app: The agent.
        state: The call's state. SLNG speech sends its agent and session ids.
        transport: Where the caller's audio comes from and goes to.
        settings: Where speech and the model run.
        telemetry: The call's spans, or None.
    """
    window = turn_window(app.pace, "pipecat")
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=vad_params(window)),
            user_turn_strategies=interruption(
                "keep",
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(params=smart_turn_params(window))
                    )
                ],
            ),
        ),
    )
    llm = build_llm(settings)
    first, every = app.specs(state)

    def tools_of(name: str | None) -> list[FunctionSchema]:
        """Return the handoff tools the app gives one agent, as Pipecat tools."""
        return [handoff_tool(handoff) for handoff in app.handoffs.get(name or "", ())]

    def handoff_tool(handoff: Handoff) -> FunctionSchema:
        """Return one handoff as a Pipecat tool whose handler calls ``UnmuteFlows.hand_off``."""

        async def handler(params: FunctionCallParams) -> None:
            """Save what the handoff sets, then give the call to the next agent."""
            if handoff.saves:
                state.save(dict(handoff.saves))
            target = every[handoff.to]
            await flows.hand_off(
                params,
                target,
                native_tools=tools_of(target.name),
                announce=handoff.announce,
                opens_with=handoff.opens_with,
            )

        return FunctionSchema(handoff.tool, handoff.description, {}, [], handler=handler)

    flows = UnmuteFlows(
        first,
        llm=llm,
        context=context,
        aggregators=aggregators,
        state=state,
        telemetry=telemetry,
        native_tools=tools_of(first.name),
    )
    flows.install()
    key = settings.slng_api_key.get_secret_value()
    ids = tracking_ids(state.call)
    pipeline = Pipeline(
        [
            transport.input(),
            SlngSTTService(
                api_key=key,
                model=settings.stt_model,
                language=Language(app.language),
                world_part=settings.slng_world_part,
                **ids,
            ),
            aggregators.user(),
            llm,
            SlngTTSService(
                api_key=key,
                model=settings.tts_model,
                voice=app.voice,
                language=Language(app.speaks),
                warm_standby_enabled=True,
                world_part=settings.slng_world_part,
                **ids,
            ),
            transport.output(),
            aggregators.assistant(),
        ]
    )
    observers: list[BaseObserver] = []
    if telemetry is not None:
        observers.append(
            TelemetryObserver(telemetry, model=settings.llm_model, provider="google", flows=flows)
        )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=observers,
    )
    if app.idle is not None:
        worker.add_observer(IdleNudgeObserver(app.idle, worker, llm=llm))
    return worker, flows


async def run_bot(app: VoiceApp, runner_args: RunnerArguments) -> None:
    """Run one call of ``app``. An example's ``bot()`` calls it for each session.

    A browser call has no caller ID. ``UNMUTE_CALL_START`` can simulate one.
    ``LANGFUSE_PUBLIC_KEY`` turns tracing on.

    Args:
        app: The agent.
        runner_args: What the Pipecat runner passes for the session.
    """
    settings = StackSettings()
    transport = await create_transport(runner_args, transport_params)
    try:
        state = await app.new_call(line_call(app, runner_args), dispatched(runner_args))
    except CallStartError as error:
        logger.error("%s: the call does not start: %s", app.agent_id, error)
        return
    telemetry = None
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        telemetry = Telemetry(
            Langfuse.from_env(), agent=app.agent_id, call_id=state.call.session_id
        )
    worker, flows = build_pipeline(app, state, transport, settings, telemetry)
    end_reason = "ended"

    @transport.event_handler("on_client_connected")
    async def greet(transport: BaseTransport, client: Any) -> None:
        """Set the agent's prompt, greet, and start the call's clock."""
        await worker.queue_frames(
            [
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(system_instruction=flows.main_instruction())
                ),
                *greeting_frames(app.greeting, state),
            ]
        )
        max_duration(worker, app.max_duration_s)

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
