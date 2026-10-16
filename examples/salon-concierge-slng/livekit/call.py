"""One call: warming the worker, building the session, and wiring the call."""

from __future__ import annotations

import asyncio
import os
import uuid

from livekit import rtc
from livekit.agents import (
    AgentSession,
    APIConnectOptions,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    UserStateChangedEvent,
    inference,
    metrics,
)
from livekit.agents.voice import MetricsCollectedEvent
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import openai, silero, slng

from agents import Concierge
from call_state import CallState
from session import (
    _dispatched_call_start,
    _hydrate_call_start,
    _livekit_job_metadata,
    _prefetch,
)
from settings import require_call_env, require_env
from utils import knowledge
from utils.dev_metrics import install_dev_metrics
from utils.router import _slng_router_client
from utils.telephony import (
    _hydrate_livekit_context,
    _livekit_call_context,
    _livekit_entry_greeting,
)
from utils.tracing import setup_tracing

# The loop keeps only a weak reference to a task, so a timer nobody holds can be
# collected before it fires. Each one is held here until it finishes.
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


def prewarm(proc: JobProcess) -> None:
    """Load what every call needs once, before the worker accepts a job.

    Args:
        proc: The worker process being warmed.
    """
    # The FLOOR: how long silence has to last before the runtime treats the
    # caller as finished. Authored as endpointing_delay.
    #
    # Always explicit. Leaving it off inherited Silero's own 0.55s, which is
    # slower than the turn detector wants and was never anybody's choice. Hard
    # floor is 0.25s; the turn detector raises at session start below it.
    #
    # Lowering this alone does not shorten a turn. The ceiling is in the session's
    # turn_handling endpointing below, and that is where a 2.5s turn came from.
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.3)
    # Read, split and embed every knowledge base here, not on the call path.
    # LiveKit documents prewarm as the place for static data a job needs, and it
    # runs once per worker process before any job is accepted, so a caller never
    # waits for indexing and a failure stops the worker from reporting ready.
    knowledge.build_indexes()


async def run_call(ctx: JobContext) -> None:
    """Run one call: build the session, start the entry agent and wire the call.

    Args:
        ctx: The job LiveKit dispatched for this call.
    """
    require_env()
    # The call state is a local here as well as the session's user data, so the
    # router's template variable snapshot can read it beside the llm kwarg.
    #
    # It also carries one SLNG Context Router session id per call, created here
    # where the call begins, so one worker process serving several jobs keeps them
    # apart. It rides the state object rather than a local of its own because
    # every agent and task now builds its own header set at request time and has
    # to be able to reach this from a method body. Not named session_id, because
    # this file already writes a "session_id" into the telephony call context and
    # that is a different thing.
    slng_state = CallState(slng_session_id=str(uuid.uuid4()))
    # This client is ours, so closing it is ours too: the plugin closes only a
    # client it built itself. Without this a worker leaks one connection pool per
    # call it serves.
    slng_client = _slng_router_client()
    slng_state._slng_client = slng_client

    async def _close_slng_client() -> None:
        await slng_client.close()

    ctx.add_shutdown_callback(_close_slng_client)
    session = AgentSession[CallState](
        userdata=slng_state,
        # One retry, not the framework's three. A failed completion is retried
        # four times by default with 0.1 + 2.0 + 2.0 seconds of sleep between the
        # attempts, on top of four round trips to the provider: a live call on
        # 2026-09-16 (trace 917975e9) spent 18.5 seconds in two of those and the
        # caller asked whether anybody was still there.
        #
        # A caller cannot wait that out, and a retry loop tuned for a batch job
        # is the wrong default for a conversation. One retry still covers a
        # transient fault and costs 0.1s, because the first retry is nearly
        # immediate; anything worse is better surfaced than slept through.
        #
        # ponytail: fixed, not authorable. No package would want a different
        # number, and the one that does can say so when it turns up.
        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(max_retry=1),
        ),
        stt=slng.STT(
            api_key=os.environ["SLNG_API_KEY"],
            slng_base_url="eu-north.api.slng.ai",
            model="soniox/speech-ai:rt-v5",
            language="en",
            external_agent_id="salon-concierge-slng-v1",
            external_session_id=slng_state.slng_session_id,
        ),
        llm=openai.LLM(client=slng_state._slng_client, model="gpt-5.6-luna"),
        tts=slng.TTS(
            api_key=os.environ["SLNG_API_KEY"],
            slng_base_url="eu-north.api.slng.ai",
            voice="62ae83ad-4f6a-430b-af41-a9bede9286ca",
            model="cartesia/sonic:3.5",
            language="en",
            external_agent_id="salon-concierge-slng-v1",
            external_session_id=slng_state.slng_session_id,
            warm_standby_enabled=True,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version="v1-mini"),
            # The CEILING, and the shortest wait the runtime will consider.
            # pace: snappy. Without this the streaming defaults apply and
            # max_delay is 2.5s, which no package could reach.
            endpointing={"mode": "dynamic", "min_delay": 0.25, "max_delay": 1.2},
            interruption={"enabled": True},
            preemptive_generation={"enabled": True},
        ),
        vad=ctx.proc.userdata["vad"],
        user_away_timeout=15,
    )

    # utils/telemetry.py names the backend. The room is the call's session.
    setup_tracing(ctx, session)

    # Inert unless the dev loop set UNMUTE_DEV_METRICS. Reads session events, so
    # it never touches the tracer provider an opt-in trace export would own.
    install_dev_metrics(session, call_id=ctx.room.name)

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    async def _end_if_still_away() -> None:
        await asyncio.sleep(30)  # inactivity.end_after
        if session.user_state == "away":
            session.shutdown()

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
        if ev.new_state != "away":
            return
        session.generate_reply(
            instructions="The caller went quiet. Briefly check whether they are still there."
        )
        _timer = asyncio.create_task(_end_if_still_away())
        _BACKGROUND_TASKS.add(_timer)
        _timer.add_done_callback(_BACKGROUND_TASKS.discard)

    metadata = _livekit_job_metadata(ctx.job.metadata)
    call_start = _dispatched_call_start(metadata)
    outbound_job = "phone_number" in metadata
    if outbound_job:
        require_call_env()
    await ctx.connect()

    if outbound_job:
        raise RuntimeError("this LiveKit SIP route does not accept outbound jobs")
    else:
        participant = await ctx.wait_for_participant()
    if (
        not outbound_job
        and participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    ):
        require_call_env()
    # Bound whether or not a SIP participant arrived, so the pre-fetch below can
    # read it either way: a web caller has no carrier facts and every entry that
    # reads one skips, which is the documented behaviour and not an error.
    call_context = None
    if (
        not outbound_job
        and participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    ):
        call_context = _livekit_call_context(ctx.room.name, participant, metadata)
        _hydrate_livekit_context(session.userdata, call_context)
    if not outbound_job:
        _hydrate_call_start(session.userdata, call_start)
    if not outbound_job:
        await _prefetch(session.userdata, call_context)
    if not outbound_job:
        await session.start(agent=Concierge(initial=True), room=ctx.room)
    await _livekit_entry_greeting(session)

    async def _max_duration() -> None:
        await asyncio.sleep(900)
        session.shutdown()  # conversation.max_duration

    _timer = asyncio.create_task(_max_duration())
    _BACKGROUND_TASKS.add(_timer)
    _timer.add_done_callback(_BACKGROUND_TASKS.discard)
