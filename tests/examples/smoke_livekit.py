"""Run one text turn through a LiveKit example's own entrypoint.

``scripts/check-examples.sh`` runs it inside the example's venv, from the project folder::

    .venv/bin/python -I ../../../tests/examples/smoke_livekit.py tools [--live]

The entrypoint runs as written, on a fake job with no room. The session drops speech
in and out, so the call is text only. Without ``--live`` a scripted model answers and
no request leaves the machine. With ``--live`` the example's own model answers.
"""

import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

sys.path[:0] = [os.getcwd(), str(Path(__file__).resolve().parents[2])]

from livekit import rtc  # noqa: E402
from livekit.agents import AgentSession  # noqa: E402
from livekit.agents.testing import fake_job_context  # noqa: E402

from tests.examples.offline import CALLER, PROMPTS, REPLY, check, go_offline  # noqa: E402
from tests.livekit.fake import ScriptedLLM  # noqa: E402
from unmute_core.livekit import ModelCallLLM  # noqa: E402

SPEECH = ("stt", "tts", "vad", "turn_handling", "turn_detection")


class TextSession(AgentSession):
    """An ``AgentSession`` with no speech and no room. A script answers unless live."""

    live: ClassVar[bool] = False
    scripted: ScriptedLLM | None = None
    started: ClassVar[list["TextSession"]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Drop the speech plugins and, unless live, swap a framework model for a script."""
        for key in SPEECH:
            kwargs.pop(key, None)
        if not self.live and not isinstance(kwargs.get("llm"), ModelCallLLM):
            # ponytail: enough replies for any one-turn example, a cycling LLM if one needs more
            kwargs["llm"] = self.scripted = ScriptedLLM(*[REPLY] * 10)
        super().__init__(**kwargs)

    async def start(self, agent: Any, **kwargs: Any) -> Any:
        """Start with no room, so no audio is read or played."""
        kwargs.pop("room", None)
        kwargs.pop("room_options", None)
        TextSession.started.append(self)
        return await super().start(agent, **kwargs)


class SmokeRoom(rtc.Room):
    """A room that is never connected, with the name a dispatched room has."""

    @property
    def name(self) -> str:
        """The room name the entrypoint reads."""
        return "smoke-room"


async def caller(*args: object, **kwargs: object) -> SimpleNamespace:
    """The participant a web call has."""
    return SimpleNamespace(
        identity="caller",
        name="caller",
        sid="PA_smoke",
        metadata="",
        attributes={},
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
    )


def shared(module: Any) -> list[Any]:
    """Return the module, and the shared stack an example may build its session in."""
    return [module, *(m for n, m in list(sys.modules.items()) if n.startswith("voice_stack"))]


async def smoke(name: str, live: bool) -> str:
    """Run the module's entrypoint and one caller turn. Return the agent's reply."""
    TextSession.live = live
    if not live:
        go_offline()
    module: Any = importlib.import_module(name)
    for loaded in shared(module):
        if getattr(loaded, "AgentSession", None) is AgentSession:
            loaded.AgentSession = TextSession
    with fake_job_context(room=SmokeRoom(), job_id=f"smoke-{name}") as ctx:
        setup = getattr(module.server, "setup_fnc", None)
        if setup is not None:
            setup(ctx.proc)
        ctx.wait_for_participant = caller  # ty: ignore[invalid-assignment]
        await module.entrypoint(ctx)
        [session] = TextSession.started
        try:
            result = await session.run(user_input=CALLER)
        finally:
            await session.aclose()
    if session.scripted is not None:
        PROMPTS.extend(request.prompt for request in session.scripted.requests)
    replies = [
        event.item.text_content or ""
        for event in result.events
        if event.type == "message" and event.item.role == "assistant"
    ]
    return check(name, replies, live=live)


def main() -> None:
    """Run the smoke for the module named on the command line."""
    name, *flags = sys.argv[1:]
    reply = asyncio.run(asyncio.wait_for(smoke(name, "--live" in flags), timeout=30))
    print(f"ok: {name} answered: {reply[:80]!r}")


if __name__ == "__main__":
    main()
