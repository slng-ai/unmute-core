"""One call: its transport, pipeline, event handlers and the end of it."""

from __future__ import annotations

import asyncio
import os
import uuid
import sys
from collections.abc import Callable
from typing import Any

from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.bus import BusBridgeProcessor
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.llm import LLMWorkerActivationArgs
from pipecat.workers.runner import WorkerRunner
from pipecat_slng import SlngSTTService

from agents import ConciergeAgent
from session import _prefetch, build_state
from settings import MAIN_NAME, require_env
from utils.dev_metrics import DevReporter, install_dev_metrics
from utils.telephony import _phone_session, _pipeline_audio_rates
from utils.tracing import enable_agent_tracing, flush_tracing, setup_tracing, start_call

# Tasks that end the call later. A task nobody references can be collected
# while it sleeps, so each is held here until it finishes.
_END_TASKS: set[asyncio.Task[None]] = set()


async def _end_after(worker: PipelineWorker, timeout_secs: float) -> None:
    """End the call once `timeout_secs` have passed.

    Args:
        worker: The pipeline worker to send the end frame to.
        timeout_secs: How long to wait first.
    """
    await asyncio.sleep(timeout_secs)
    await worker.queue_frame(EndFrame())


def _schedule_end_after(worker: PipelineWorker, timeout_secs: float) -> None:
    """Schedule `_end_after` and hold its task until it finishes.

    Args:
        worker: The pipeline worker to send the end frame to.
        timeout_secs: How long to wait first.
    """
    task = asyncio.create_task(_end_after(worker, timeout_secs))
    _END_TASKS.add(task)
    task.add_done_callback(_END_TASKS.discard)


transport_params: dict[str, Callable[[], TransportParams]] = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    # The runner assigns an inbound call's dial-in settings and Daily credentials
    # onto whatever this returns, so on the Daily route it has to be the params
    # class that declares them. The generic one rejects the assignment.
    "daily": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    # Your carrier's stream, terminated by Pipecat Cloud. The runner attaches the
    # carrier's frame serializer itself, reading the stream and call ids out of the
    # handshake, so this declares the audio directions and nothing else.
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True, audio_out_enabled=True
    ),
}


def build_stt(slng_session_id: str) -> STTService:
    """Build the transcriber.

    Args:
        slng_session_id: This call's SLNG session id, sent with the agent id.

    Returns:
        The configured STT service.
    """
    return SlngSTTService(
        api_key=os.environ["SLNG_API_KEY"],
        model="soniox/speech-ai:rt-v5",
        language=Language("en"),
        world_part="eu-north",
        external_agent_id="salon-concierge-single-prompt",
        external_session_id=slng_session_id,
    )


async def _run_bot(
    transport: BaseTransport, runner_args: RunnerArguments, dev: DevReporter
) -> None:
    """Build the pipeline for one session and run it until the call ends.

    Args:
        transport: The transport the caller is connected on.
        runner_args: The arguments the runner started this session with.
        dev: The dev-page reporter that observes the session.
    """
    require_env()
    call_context: dict[str, Any] = {}
    # One SLNG session id per call, passed as an argument from here on. SLNG
    # speech sends it, with the agent id, so usage groups by agent and by call.
    # Not named session_id, because runner_args.session_id is a different thing.
    slng_session_id = str(uuid.uuid4())
    # Decided once, right after the transport exists: is this a phone call at all?
    # Everything a call needs is checked here, before the caller hears anything.
    # None means browser or console, which behave exactly as they do on a package
    # with no telephony.
    phone_call = _phone_session(runner_args)
    call_context["_phone_call"] = phone_call
    # The call's own facts, under the flat names a `source:` reads. The call id
    # and stream id are parsed out of the carrier's handshake; the rest are the
    # <Parameter> set the TwiML markup dictated, which is why the runbook says
    # what each one is for and what happens when an existing Bin lacks it.
    if phone_call is not None:
        _markup = phone_call.get("body") or {}
        call_context["call_id"] = phone_call.get("call_id") or ""
        call_context["stream_id"] = phone_call.get("stream_id") or ""
        call_context["direction"] = _markup.get("direction") or "inbound"
        call_context["from_number"] = _markup.get("from_number") or ""
        call_context["to_number"] = _markup.get("to_number") or ""

    # utils/telemetry.py names the backend. The attributes tie every span of
    # this call together, whichever backend reads them.
    telemetry = setup_tracing()
    trace_attributes = start_call(telemetry, runner_args)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    # turn: local — end-of-turn detection runs on-device (Silero VAD). No API
    # key, no network hop; the turn binding in targets.yaml is advisory.
    context = LLMContext()
    state = build_state(call_context)
    # Facts knowable before the greeting, resolved once, here: after the state
    # exists so there is somewhere to put them, and before the agents are built
    # so every prompt they render already sees them.
    await _prefetch(state, call_context)
    agents = [
        ConciergeAgent(
            state=state,
            context=context,
            call_context=call_context,
            dev_metrics=dev,
            slng_session_id=slng_session_id,
        )
    ]
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # The FLOOR: how long silence has to last before speech counts as
            # stopped. Authored as endpointing_delay.
            #
            # Widening this window makes some turns SLOWER, not more patient.
            # pipecat-slng asks the bridge to finalise when VAD reports the caller
            # stopped, and a final that already arrived finds no request
            # outstanding, so the frame goes out unfinalized and Pipecat waits out
            # a flat 1.0s safety net. Observed transcripts arrive from 0.27s.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            # The CEILING, on the end-of-turn analyzer. pace: snappy.
            #
            # Note the two fields both spelled stop_secs. The one above is the
            # VAD's silence window; this one is how long the turn may run before
            # closing regardless. They are different fields and different numbers.
            #
            # The analyzer is Pipecat's own default stop strategy. It is
            # constructed explicitly here only so the ceiling is reachable: left
            # implicit it runs at SmartTurnParams' own 3s.
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            params=SmartTurnParams(stop_secs=1.2)
                        )
                    )
                ],
            ),
            # The stretches of the call the caller cannot talk over, lowered from
            # conversation.interruption. On a phone route the greeting is
            # protected by default: a phone leg has no echo cancellation, so
            # without this the agent hears its own opening line and cuts itself
            # off. Barge-in stays on everywhere else.
            user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
            user_idle_timeout=15,
        ),
    )
    bridge = BusBridgeProcessor(
        bus=runner.bus, worker_name=MAIN_NAME, name=f"{MAIN_NAME}::BusBridge"
    )
    dev.observe_aggregators(user_aggregator, assistant_aggregator)
    pipeline = Pipeline(
        [
            transport.input(),
            dev.observe_stt(build_stt(slng_session_id)),
            user_aggregator,
            bridge,
            transport.output(),
            assistant_aggregator,
        ]
    )
    main = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        # The worker builds its own latency observer only when tracing is enabled,
        # and does not re-expose its events, so this one is always ours.
        observers=dev.observers(),
        conversation_id=runner_args.session_id,
        enable_tracing=True,
        additional_span_attributes=trace_attributes,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            **_pipeline_audio_rates(phone_call),
        ),
    )

    enable_agent_tracing(main, agents)

    @user_aggregator.event_handler("on_user_turn_idle")
    async def on_user_turn_idle(aggregator: Any) -> None:
        """Nudge the caller when they have gone quiet."""
        await aggregator.push_frame(
            LLMMessagesAppendFrame(
                [
                    {
                        "role": "developer",
                        "content": "The caller has gone quiet. Politely check if they are still there.",
                    }
                ],
                run_llm=True,
            )
        )

    # inactivity.end_after: the nudge above is the first half, and this is the
    # second. Without it an idle call is nudged forever and never hung up, which
    # on a phone line is a billed open call. The clock starts at the first idle
    # turn and is cancelled by the caller speaking again.
    _idle_end: asyncio.Task[None] | None = None

    @user_aggregator.event_handler("on_user_turn_idle")
    async def on_user_turn_idle_end(aggregator: Any) -> None:
        """Start the clock that ends an idle call."""
        nonlocal _idle_end
        if _idle_end is None or _idle_end.done():
            _idle_end = asyncio.create_task(_end_after(main, 45 - 15))

    @user_aggregator.event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator: Any, strategy: Any) -> None:
        """Stop the hangup clock when the caller speaks again."""
        nonlocal _idle_end
        if _idle_end is not None and not _idle_end.done():
            _idle_end.cancel()
            _idle_end = None

    runner_ready = asyncio.Event()
    pipeline_started = asyncio.Event()
    worker_start_error: Exception | None = None
    entry_started = False

    async def activate_entry() -> None:
        """Activate the entry agent once, when the client and the pipeline are ready."""
        nonlocal entry_started
        if entry_started or worker_start_error is not None:
            return
        entry_started = True
        await main.activate_worker(
            "concierge",
            args=LLMWorkerActivationArgs(run_llm=False),
        )
        await next(agent for agent in agents if agent.name == "concierge").queue_frame(
            TTSSpeakFrame(
                "Hi, you've reached Sage and Stone. Robin speaking, what can I do for you?"
            )
        )

        _schedule_end_after(main, 900)

    @runner.event_handler("on_ready")
    async def on_runner_ready(runner: WorkerRunner) -> None:
        """Note that the runner can take workers."""
        runner_ready.set()

    @main.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker: PipelineWorker, frame: Any) -> None:
        """Add the agent workers once the main pipeline has started."""
        nonlocal worker_start_error
        await runner_ready.wait()
        try:
            await runner.add_workers(*agents)
        except Exception as error:  # noqa: BLE001 - logged with its traceback, then the runner is cancelled
            worker_start_error = error
            # Logged here, where it is still known. Cancelling the runner makes
            # run() raise CancelledError, and that reaches the caller before the
            # re-raise below ever runs, so this error is destroyed on its way
            # out: measured 2026-08-21, a session died half a second in and
            # reported nothing but a failed trace flush.
            logger.exception("the agent workers failed to start")
            pipeline_started.set()
            await runner.cancel(reason="agent worker startup failed")
            return
        pipeline_started.set()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport: BaseTransport, client: Any) -> None:
        """Activate the entry agent once the carrier's stream is open."""
        # Telephony transports (Daily SIP, the platform's carrier stream) have
        # no RTVI client-ready handshake: the carrier opens the media stream and
        # the bot initiates once that connection is open and main's StartFrame
        # has traversed (Pipecat Twilio dial-in flow; SPEC V2).
        #
        # This is also all an outbound call needs. Your carrier opens the stream
        # when the person answers, so the greeting below meets a human rather
        # than ringback, and there is no ringing state to handle.
        await pipeline_started.wait()
        await activate_entry()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: BaseTransport, client: Any) -> None:
        """Stop the runner when the caller leaves."""
        await runner.cancel()

    try:
        await runner.add_workers(main)
        await runner.run()
        if worker_start_error is not None:
            raise worker_start_error
    finally:
        primary_error = sys.exception()
        if primary_error is not None:
            try:
                await asyncio.to_thread(
                    flush_tracing, telemetry, runner_args.session_id or ""
                )
            except BaseException as cleanup_error:  # noqa: BLE001 - the primary error is what propagates
                logger.error(
                    "Tracing flush failed while preserving the primary error ({})",
                    type(cleanup_error).__name__,
                )
        else:
            await asyncio.to_thread(
                flush_tracing, telemetry, runner_args.session_id or ""
            )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Run one session, reporting to the dev page and naming any error that stops it.

    Args:
        transport: The transport the caller is connected on.
        runner_args: The arguments the runner started this session with.
    """
    dev = install_dev_metrics(runner_args)
    try:
        await _run_bot(transport, runner_args, dev)
    except Exception as error:
        # The dev page is where the author is looking when a call dies, so it
        # names what stopped it. The type only: an exception's message quotes
        # whatever it was working on, and that belongs in the log rather than
        # on a page that carries no prompts, arguments or results.
        dev.finish(error=True, reason=f"The agent stopped on {type(error).__name__}.")
        raise
    finally:
        dev.finish()
