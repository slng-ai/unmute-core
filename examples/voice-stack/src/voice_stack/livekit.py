"""Run a ``VoiceApp`` on LiveKit Agents.

``server_for(app)`` returns the worker an example's ``livekit/agent.py`` exposes. Each call
then runs the same way:

1. Read what the line carries (a SIP caller's numbers, or nothing on a browser call).
2. ``app.new_call`` builds the call's state, with its prefetch read in. A dispatch that
   lacks a required ``call_start`` value, or holds a wrong one, ends the job before the
   greeting, and the log says which value.
3. The session gets SLNG speech in the stack's region and Gemini on Vertex EU.
4. ``UnmuteAgent`` runs the app's ``AgentSpec``.
5. The greeting plays, the idle nudges and the call's longest length start.

Run an example against your microphone with ``lk agent console agent.py``, or as a worker
for a LiveKit project with ``uv run agent.py dev``.
"""

import json
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from google import genai
from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    APIConnectOptions,
    FunctionTool,
    JobContext,
    JobProcess,
    RunContext,
    TurnHandlingOptions,
    function_tool,
    inference,
    llm,
)
from livekit.agents.voice import room_io
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import google, silero, slng
from unmute_core.agent import AgentSpec
from unmute_core.call import CallStartError
from unmute_core.livekit import (
    ModelCallLLM,
    UnmuteAgent,
    endpointing,
    greet,
    idle_nudges,
    interruption,
    max_duration,
    sip_call_info,
    yield_to_caller,
)
from unmute_core.slng import tracking_ids
from unmute_core.state import CallState, call_phone_region, use_phone_region
from unmute_core.telemetry import Langfuse, Telemetry
from unmute_core.turn import turn_window

from voice_stack.app import Handoff, VoiceApp
from voice_stack.settings import StackSettings

logger = logging.getLogger("voice_stack")


def vertex_client(api_key: str, location: str) -> genai.Client:
    """Return a GenAI client pinned to one Vertex location, with an API key.

    LiveKit's Google plugin reaches Vertex only with Google Cloud credentials. This client
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


class VertexLLM(google.LLM):
    """LiveKit's Google LLM plugin, with its client swapped for a Vertex API-key one."""

    def __init__(self, *, api_key: str, location: str, **kwargs: Any) -> None:
        """Build the plugin, then replace its client before any request runs.

        Args:
            api_key: The Vertex API key.
            location: The Vertex location.
            **kwargs: The plugin's own arguments, like ``model``.
        """
        super().__init__(api_key=api_key, vertexai=False, **kwargs)
        self._client.close()
        self._client = vertex_client(api_key, location)


def build_llm(settings: StackSettings, state: CallState) -> llm.LLM:
    """Return the stack's model on LiveKit: Gemini on Vertex, or a ``gpt-`` model on OpenAI."""
    if settings.on_openai:
        return ModelCallLLM(settings.openai_model(), state=state)
    return VertexLLM(
        api_key=settings.google_api_key.get_secret_value(),
        location=settings.vertex_location,
        model=settings.llm_model,
        thinking_config={"thinking_level": settings.thinking_level},
    )


def build_session(
    app: VoiceApp, state: CallState, vad: silero.VAD, settings: StackSettings
) -> AgentSession:
    """Return the call's session: SLNG speech, Gemini, and the app's turn pace.

    Interruption is set, not left to LiveKit, so a deployed worker treats the caller
    speaking over the agent the same way as a local one. ``keep`` lets a "yes" said over a
    question answer it: most of these agents' lines end in a question.

    Args:
        app: The agent.
        state: The call's state. SLNG speech sends its agent and session ids.
        vad: The worker's voice detector.
        settings: Where speech and the model run.
    """
    key = settings.slng_api_key.get_secret_value()
    ids = tracking_ids(state.call)
    window = turn_window(app.pace, "livekit")
    stt = slng.STT(
        api_key=key,
        slng_base_url=settings.slng_host,
        model=settings.stt_model,
        language=app.language,
        **ids,
    )
    session = AgentSession(
        conn_options=SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=1)),
        stt=stt,
        llm=build_llm(settings, state),
        tts=slng.TTS(
            api_key=key,
            slng_base_url=settings.slng_host,
            model=settings.tts_model,
            voice=app.voice,
            language=app.speaks,
            warm_standby_enabled=True,
            **ids,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version="v1-mini"),
            endpointing=endpointing(window),
            interruption=interruption("keep"),
        ),
        vad=vad,
        user_away_timeout=None,
    )
    stt.attach_to_session(session)
    yield_to_caller(session)
    return session


def unmute_agent(
    app: VoiceApp,
    spec: AgentSpec,
    state: CallState,
    every: Mapping[str, AgentSpec],
    telemetry: Telemetry | None = None,
) -> UnmuteAgent:
    """Return ``spec`` as a LiveKit agent, with the handoff tools the app gives it.

    Only the first agent gets the telemetry: an agent the call is handed to records into
    the first one's spans.
    """
    return UnmuteAgent(
        spec,
        state=state,
        telemetry=telemetry,
        native_tools=[
            handoff_tool(app, handoff, state, every)
            for handoff in app.handoffs.get(spec.name or "", ())
        ],
    )


def handoff_tool(
    app: VoiceApp, handoff: Handoff, state: CallState, every: Mapping[str, AgentSpec]
) -> FunctionTool:
    """Return one handoff as a LiveKit function tool that calls ``UnmuteAgent.hand_off``."""

    async def hand_off(context: RunContext) -> object:
        """Save what the handoff sets, then give the call to the next agent."""
        if handoff.saves:
            state.save(dict(handoff.saves))
        current = context.session.current_agent
        if not isinstance(current, UnmuteAgent):
            raise TypeError("a handoff runs only from an UnmuteAgent")
        target = unmute_agent(app, every[handoff.to], state, every)
        return await current.hand_off(
            target, announce=handoff.announce, opens_with=handoff.opens_with
        )

    return function_tool(hand_off, name=handoff.tool, description=handoff.description)


def server_for(app: VoiceApp) -> tuple[AgentServer, Callable[[JobContext], Awaitable[None]]]:
    """Return the worker that runs ``app``, and its entrypoint.

    ``LIVEKIT_AGENT_NAME`` names the worker for explicit dispatch, like a SIP dispatch
    rule. Empty takes every room. ``LANGFUSE_PUBLIC_KEY`` turns tracing on.

    Args:
        app: The agent every call runs.
    """
    window = turn_window(app.pace, "livekit")
    server = AgentServer()

    def prewarm(proc: JobProcess) -> None:
        """Load the voice detector once per worker process, with the app's turn pace."""
        proc.userdata["vad"] = silero.VAD.load(min_silence_duration=window.vad_silence)

    server.setup_fnc = prewarm

    @server.rtc_session(agent_name=os.getenv("LIVEKIT_AGENT_NAME", ""))
    async def entrypoint(ctx: JobContext) -> None:
        """Run one call: read what it carries, build its state, greet, and cap its length.

        A console or browser call has no caller ID. ``UNMUTE_CALL_START`` can simulate one.
        """
        settings = StackSettings()
        await ctx.connect()
        participant = await ctx.wait_for_participant()
        metadata = json.loads(ctx.job.metadata or "{}")
        call = None
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            call = sip_call_info(participant, metadata, agent_id=app.agent_id)
        try:
            state = await app.new_call(call, metadata.get("call_start"))
        except CallStartError as error:
            logger.error("%s: the call does not start: %s", app.agent_id, error)
            ctx.shutdown(reason=str(error))
            return
        telemetry = None
        if os.getenv("LANGFUSE_PUBLIC_KEY"):
            telemetry = Telemetry(
                Langfuse.from_env(), agent=app.agent_id, call_id=state.call.session_id
            )
        session = build_session(app, state, ctx.proc.userdata["vad"], settings)
        if app.idle is not None:
            idle_nudges(session, app.idle)
        with use_phone_region(lambda: call_phone_region(state.call)):
            first, every = app.specs(state)
            await session.start(
                agent=unmute_agent(app, first, state, every, telemetry),
                room=ctx.room,
                room_options=room_io.RoomOptions(delete_room_on_close=True),
            )
        greet(session, app.greeting, state)
        max_duration(session, app.max_duration_s)

    return server, entrypoint
