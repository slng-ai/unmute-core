"""Parcel Desk on LiveKit Agents, defined as data on unmute-core.

The agent is the shared ``desk`` package, built from data: ``DESK.agent(state)`` is the
call's ``AgentSpec``, the same one ``pipecat/bot.py`` runs. This file only builds the
LiveKit side: the session, its speech, and the call's start and end. Every model request
goes through the SLNG Context Router, so the session has no LLM plugin.

Run it against your microphone, with the LiveKit CLI::

    lk agent console agent.py
"""

import json
import os

from desk.agent import build
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    inference,
)
from livekit.agents.voice import room_io
from livekit.plugins import silero, slng
from unmute_core.livekit import (
    UnmuteAgent,
    endpointing,
    greet,
    interruption,
    max_duration,
    sip_call_info,
    yield_to_caller,
)
from unmute_core.slng import tracking_ids
from unmute_core.state import CallInfo, CallState, call_phone_region, use_phone_region
from unmute_core.telemetry import Langfuse, Telemetry

load_dotenv()

DESK = build()
WINDOW = DESK.window("livekit")


def prewarm(proc: JobProcess) -> None:
    """Load the voice detector once per worker process, with the spec's turn pace."""
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=WINDOW.vad_silence)


def build_session(vad: silero.VAD, state: CallState) -> AgentSession:
    """Return the session: SLNG speech, and the spec's turn pace. No LLM: the agent has one.

    SLNG speech sends the call's agent and session ids, the same ones the router gets.
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
        stt=stt,
        tts=slng.TTS(
            api_key=slng_key,
            slng_base_url="eu-north.api.slng.ai",
            voice="8d8aec63a56c460cb7def795bdf754fa",
            model="slng/fish/tts:s2.1-pro",
            language="en",
            **tracking_ids(state.call),
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version="v1-mini"),
            endpointing=endpointing(WINDOW),
            interruption=interruption("keep"),
        ),
        vad=vad,
    )
    stt.attach_to_session(session)
    yield_to_caller(session)
    return session


def line_call(participant: rtc.RemoteParticipant, metadata: dict) -> CallInfo | None:
    """Return what a SIP call carries. None on a console or browser call.

    A console or browser call has no caller ID: ``UNMUTE_CALL_START`` can simulate one.
    """
    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return sip_call_info(participant, metadata, agent_id=DESK.agent_id)
    return None


server = AgentServer()
server.setup_fnc = prewarm


@server.rtc_session(agent_name=os.getenv("LIVEKIT_AGENT_NAME", ""))
async def entrypoint(ctx: JobContext) -> None:
    """Run one call: read what it carries, build the agent's state, greet, and cap its length."""
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    metadata = json.loads(ctx.job.metadata or "{}")
    state = await DESK.new_call(line_call(participant, metadata), metadata.get("call_start"))
    telemetry = None
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        telemetry = Telemetry(
            Langfuse.from_env(), agent=DESK.agent_id, call_id=state.call.session_id
        )
    agent = UnmuteAgent(DESK.agent(state), state=state, telemetry=telemetry)
    session = build_session(ctx.proc.userdata["vad"], state)
    with use_phone_region(lambda: call_phone_region(state.call)):
        await session.start(
            agent=agent,
            room=ctx.room,
            room_options=room_io.RoomOptions(delete_room_on_close=True),
        )
    greet(session, DESK.greeting, state)
    max_duration(session, DESK.max_duration_s)
