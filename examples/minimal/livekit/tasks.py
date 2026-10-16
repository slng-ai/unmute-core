"""One task with one step that saves a value.

A task is steps that each save values to the call state, in order. ``take_note`` runs when
the caller wants to leave a note, and saves it as ``note``. ``UnmuteAgent`` runs the task
and the session's own LLM plugin answers.

Run it against your microphone::

    uv run python tasks.py console
"""

import os

from livekit.agents import AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai, silero, slng
from pydantic import BaseModel
from unmute_core.agent import AgentSpec
from unmute_core.livekit import UnmuteAgent
from unmute_core.state import CallState
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.tools import ToolRegistry


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

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Run one call whose agent can take a note."""
    slng_host = f"{os.getenv('SLNG_WORLD_PART', 'eu-north')}.api.slng.ai"
    stt = slng.STT(slng_base_url=slng_host, model="soniox/speech-ai:rt-v5")
    session = AgentSession(
        stt=stt,
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=slng.TTS(
            slng_base_url=slng_host,
            model="slng/fish/tts:s2.1-pro",
            voice="8d8aec63a56c460cb7def795bdf754fa",
        ),
        vad=silero.VAD.load(),
    )
    stt.attach_to_session(session)
    spec = AgentSpec(
        "You answer calls for a shop. Keep replies short.",
        tasks=TaskPlan(SPEC, state=State, tools=ToolRegistry([])),
    )
    agent = UnmuteAgent(spec, state=CallState(State))
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
