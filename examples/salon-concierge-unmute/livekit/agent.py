"""The salon concierge on LiveKit Agents, built by hand on unmute-core.

The agents themselves (prompts, tools, tasks, knowledge) are the shared ``salon``
package's ``AgentSpec``s, the same ones ``pipecat/bot.py`` runs. This file only builds the
LiveKit side: the session, an ``UnmuteAgent`` for each spec, and their handoffs.

Run it against your microphone, with the LiveKit CLI::

    lk agent console agent.py
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    APIConnectOptions,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    function_tool,
    inference,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.voice import room_io
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import silero, slng
from salon import agent as salon
from salon.state import State
from salon.store import Salon
from unmute_core.livekit import (
    Backchannels,
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

from vertex import VertexLLM

load_dotenv()

CONCIERGE, SPECIALIST = salon.agents(Salon())
BACKCHANNELS: Backchannels = "keep"
WINDOW = turn_window(salon.PACE, "livekit")


def end_call() -> EndCallTool:
    """Return LiveKit's end-call tool, with the salon's wording.

    The goodbye rides in the reply that calls the tool, so the call ends after one model
    request. With ``end_instructions`` set, LiveKit would ask the model a second time and
    the caller could hear two goodbyes. The session deletes the room as it closes, which
    hangs up a phone caller, so the tool does not delete it too.
    """
    return EndCallTool(
        extra_description=f"{salon.END_CALL}\n{salon.SAY_GOODBYE_WITH_THE_CALL}",
        end_instructions=None,
        delete_room=False,
    )


class Concierge(UnmuteAgent):
    """Books, moves and cancels appointments, and answers questions about the salon."""

    def __init__(
        self,
        state: CallState[State],
        telemetry: Telemetry | None = None,
    ) -> None:
        """Build the concierge for one call."""
        super().__init__(CONCIERGE, state=state, telemetry=telemetry, native_tools=[end_call()])

    @function_tool
    async def to_complaints(self) -> UnmuteAgent | str:
        """The caller has a complaint or service problem."""
        assert self.state is not None
        return await self.hand_off(
            ComplaintSpecialist(self.state), announce=salon.TO_COMPLAINTS_LINE
        )


class ComplaintSpecialist(UnmuteAgent):
    """Takes a complaint down and quotes the refund policy."""

    def __init__(self, state: CallState[State]) -> None:
        """Build the complaint specialist. It records into the concierge's spans."""
        super().__init__(SPECIALIST, state=state, native_tools=[end_call()])

    @function_tool
    async def to_concierge(self) -> UnmuteAgent | str:
        """The caller changes to a request owned by another specialist."""
        assert self.state is not None
        return await self.hand_off(Concierge(self.state))


def prewarm(proc: JobProcess) -> None:
    """Load the voice detector and both knowledge bases once per worker process."""
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=WINDOW.vad_silence)
    salon.start_knowledge()


def build_session(vad: silero.VAD, state: CallState[State]) -> AgentSession:
    """Return the session: SLNG speech, Gemini on Vertex, the local turn detector.

    The STT hears the session's user state, so it finalises the transcript as soon as
    the caller goes quiet instead of waiting for its own endpoint. ``LLM_MODEL`` picks
    another Gemini model to compare against.

    Interruption is set, not left to LiveKit: it turns adaptive detection on under
    ``console`` and ``dev`` but off under ``start`` (voice/agent_activity.py:5116-5122), so
    a deployed worker would handle the caller speaking over it differently from a local one.
    ``BACKCHANNELS`` keeps a "yes" said over the agent: most lines end in a question.
    SLNG speech sends the call's agent and session ids.
    """
    slng_key = os.environ["SLNG_API_KEY"]
    stt = slng.STT(
        api_key=slng_key,
        slng_base_url="eu-north.api.slng.ai",
        model="soniox/speech-ai:rt-v5",
        language="en",
        **tracking_ids(state.call),
    )
    session = AgentSession(
        conn_options=SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=1)),
        stt=stt,
        llm=VertexLLM(
            api_key=os.environ["GOOGLE_API_KEY"],
            location="eu",
            model=os.getenv("LLM_MODEL", "gemini-3.1-flash-lite"),
            thinking_config={"thinking_level": "MINIMAL"},
        ),
        tts=slng.TTS(
            api_key=slng_key,
            slng_base_url="eu-north.api.slng.ai",
            voice="8d8aec63a56c460cb7def795bdf754fa",
            model="slng/fish/tts:s2.1-pro",
            language="en",
            **tracking_ids(state.call),
            warm_standby_enabled=True,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version="v1-mini"),
            endpointing=endpointing(WINDOW),
            interruption=interruption(BACKCHANNELS),
            preemptive_generation={"enabled": True},
        ),
        vad=vad,
        user_away_timeout=None,
    )
    stt.attach_to_session(session)
    yield_to_caller(session)
    return session


server = AgentServer()
server.setup_fnc = prewarm


@server.rtc_session(agent_name=os.getenv("LIVEKIT_AGENT_NAME", ""))
async def entrypoint(ctx: JobContext) -> None:
    """Run one call: read what it carries, start the concierge, and greet the caller.

    A console or browser call has no caller ID: ``UNMUTE_CALL_START`` can simulate one.
    """
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    metadata = json.loads(ctx.job.metadata or "{}")
    call = None
    phone_call = participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    if phone_call and not os.getenv("IGNORE_CALLER_ID"):
        call = sip_call_info(participant, metadata, agent_id=salon.AGENT_ID)
    state = await salon.new_call(call, metadata.get("call_start"))
    telemetry = None
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        telemetry = Telemetry(
            Langfuse.from_env(),
            agent=salon.AGENT_ID,
            call_id=participant.attributes.get("coval.simulation_id") or state.call.session_id,
            resource={"deployment.environment.name": os.getenv("ENVIRONMENT", "development")},
        )

    session = build_session(ctx.proc.userdata["vad"], state)

    def traced(name: str, data: dict[str, Any] | None) -> None:
        """Trace each nudge line and the hang-up, not every timer change."""
        if telemetry is not None and name.endswith((".fired", ".hangup")):
            telemetry.event(name, data)

    idle_nudges(session, salon.IDLE, on_event=traced)

    with use_phone_region(lambda: call_phone_region(state.call)):
        await session.start(
            agent=Concierge(state, telemetry),
            room=ctx.room,
            room_options=room_io.RoomOptions(delete_room_on_close=True),
        )
    greet(session, salon.GREETING, state)
    max_duration(session, salon.MAX_DURATION_S)
