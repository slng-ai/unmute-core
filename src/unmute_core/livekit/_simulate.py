"""The LiveKit driver of a simulated call: ``UnmuteAgent`` in a text-only ``AgentSession``.

No STT, TTS or room. The session's transcription output stands in for the speaker:
LiveKit sends it both the lines code says and the model's replies when there is no audio.
Each caller line is a typed turn (``session.run``). A turn is over once no speech is
queued, no model request is open, and nothing has happened for ``SETTLE_S``.

``python -m unmute_core.testing simulate livekit`` runs it. See
``unmute_core.testing.simulate``.
"""

import asyncio
import contextlib
import time

from livekit.agents import AgentSession
from livekit.agents.voice.io import TextOutput

from unmute_core.call import CALLER_HUNG_UP
from unmute_core.livekit import UnmuteAgent, open_call
from unmute_core.testing.simulate._agent import AgentCall
from unmute_core.testing.simulate._record import Recorder

SETTLE_S = 0.8
"""How long nothing must happen before the agent's turn is over."""
POLL_S = 0.05


class Lines(TextOutput):
    """Collects each spoken segment as one line, and notes when text arrives."""

    def __init__(self, recorder: Recorder) -> None:
        """Start with nothing said."""
        super().__init__(label="simulate", next_in_chain=None)
        self.recorder = recorder
        self.lines: list[str] = []
        self._segment = ""

    async def capture_text(self, text: str) -> None:
        """Add text to the current line."""
        self._segment += text
        self.recorder.touch()

    def flush(self) -> None:
        """Close the current line."""
        if self._segment.strip():
            self.lines.append(" ".join(self._segment.split()))
        self._segment = ""
        self.recorder.touch()


class LiveKitCall:
    """One simulated call on LiveKit. See ``unmute_core.testing.simulate._call.TextCall``.

    Attributes:
        ended: True once the agent's side closed the session.
    """

    def __init__(self, call: AgentCall, recorder: Recorder) -> None:
        """Keep the call. Nothing runs until ``start``."""
        self.call = call
        self.recorder = recorder
        self.ended = False
        self._lines = Lines(recorder)
        self._stack = contextlib.AsyncExitStack()
        self._session: AgentSession | None = None
        self._leaving = False

    async def start(self) -> list[str]:
        """Start the session, greet, and return what the agent said."""
        session = await self._stack.enter_async_context(AgentSession())
        self._session = session
        session.output.transcription = self._lines
        session.on("close", self._closed)
        await session.start(UnmuteAgent(self.call.spec, call=self.call.call))
        handle = await open_call(session, self.call.greeting, self.call.call)
        if handle is not None:
            await handle.wait_for_playout()
        return await self._quiet(0)

    def _closed(self, event: object) -> None:
        """Note that the session closed. Unless the caller left, the agent ended the call."""
        if not self._leaving:
            self.ended = True

    async def say(self, line: str) -> list[str]:
        """Type one caller line, and return what the agent said once it is quiet."""
        assert self._session is not None, "start the call first"
        said = len(self._lines.lines)
        if not self.ended:
            await self._session.run(user_input=line)
        return await self._quiet(said)

    async def _quiet(self, said: int) -> list[str]:
        """Wait until the agent's turn is over, and return the lines said since ``said``."""
        assert self._session is not None
        while not self.ended:
            await asyncio.sleep(POLL_S)
            busy = self._session.current_speech is not None or self.recorder.open_requests
            if not busy and time.monotonic() - self.recorder.last_activity >= SETTLE_S:
                break
        self._lines.flush()
        return self._lines.lines[said:]

    async def hang_up(self) -> None:
        """Leave as the caller: the session closes, and the call ends with its held saves."""
        self._leaving = True
        self.call.call.caller_left()
        if self._session is not None:
            await self._session.aclose()
        await self.call.call.end(CALLER_HUNG_UP)

    async def close(self) -> None:
        """Close the session if it is still open, and end the call. Never raises."""
        self._leaving = True
        with contextlib.suppress(Exception):
            await self._stack.aclose()
        with contextlib.suppress(Exception):
            await self.call.call.end(CALLER_HUNG_UP)
