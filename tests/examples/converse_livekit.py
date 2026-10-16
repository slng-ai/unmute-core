"""Run whole scripted calls through a LiveKit example's own entrypoint, on its real model.

Run it inside the example's venv, from the project folder::

    .venv/bin/python -I ../../../tests/examples/converse_livekit.py agent ../scenarios/*.json

The entrypoint runs as written, on a fake job with no room. Its session drops speech in
and out, so each call is text only: the caller's lines go in as typed turns, and every line
the agent says comes back as text, its code-spoken lines included. The model is the
example's own, with the keys in your environment. See ``tests/examples/script.py`` for the
scenario format.

``CONVERSE_AUDIO=1`` keeps the session's speech instead. A microphone streams each caller
line, spoken by SLNG text to speech (``tests/examples/voices.py``), in real time, so the
example's own voice detector, turn detector and speech to text hear it. A speaker plays
the agent's audio on a clock and tells LiveKit when each line has played. Each turn then
also shows what speech to text heard.
"""

import asyncio
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

sys.path[:0] = [os.getcwd(), str(Path(__file__).resolve().parents[2])]

from livekit import rtc  # noqa: E402
from livekit.agents import AgentSession  # noqa: E402
from livekit.agents.testing import fake_job_context  # noqa: E402
from livekit.agents.utils import http_context  # noqa: E402
from livekit.agents.voice import io  # noqa: E402

from tests.examples import script, voices  # noqa: E402
from tests.examples.smoke_livekit import SmokeRoom, caller  # noqa: E402

SPEECH = ("stt", "tts", "vad", "turn_handling", "turn_detection")
TURN_TIMEOUT_S = 60
VERBOSE = bool(os.getenv("CONVERSE_VERBOSE"))
AUDIO = bool(os.getenv("CONVERSE_AUDIO"))
"""Keep the session's speech, with a caller who speaks."""
CALLER_LANGUAGE = os.getenv("CONVERSE_CALLER_LANGUAGE", "en")
QUIET_S = 2.5
ANSWER_S = 15.0
"""How long a caller line may go unanswered before the turn counts as one with no answer."""
CHUNK_S = 0.02


class Microphone(io.AudioInput):
    """A caller's microphone: their lines, then silence, in real time, 20 ms at a time."""

    def __init__(self) -> None:
        """Start with nothing to say."""
        super().__init__(label="converse-caller")
        self._audio = bytearray()
        self._start = 0.0
        self._sent = 0

    async def __anext__(self) -> rtc.AudioFrame:
        """Return the next 20 ms of the caller's audio, or of silence, on the wall clock."""
        if not self._start:
            self._start = time.monotonic()
        samples = int(voices.SAMPLE_RATE * CHUNK_S)
        chunk = bytes(self._audio[: samples * 2]).ljust(samples * 2, b"\x00")
        del self._audio[: samples * 2]
        self._sent += 1
        await asyncio.sleep(max(0.0, self._start + self._sent * CHUNK_S - time.monotonic()))
        return rtc.AudioFrame(chunk, voices.SAMPLE_RATE, 1, samples)

    async def say(self, line: str, language: str | None = None) -> None:
        """Speak one caller line: its audio is streamed from now on."""
        self._audio.extend(await voices.spoken(line, language or CALLER_LANGUAGE))

    async def done(self) -> None:
        """Wait until the caller's line has been streamed to its end."""
        while self._audio:
            await asyncio.sleep(CHUNK_S)


class Speaker(io.AudioOutput):
    """A speaker that plays the agent's audio on a clock, and says when each line played."""

    def __init__(self) -> None:
        """Start silent."""
        super().__init__(
            label="converse-speaker", capabilities=io.AudioOutputCapabilities(pause=False)
        )
        self._seconds = 0.0
        self._started = 0.0
        self._playing: asyncio.Task[None] | None = None
        self.until = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Take one frame of the current line, and start its playback on the first."""
        await super().capture_frame(frame)
        if not self._seconds:
            self._started = time.monotonic()
            self.on_playback_started(created_at=time.time())
        self._seconds += frame.duration
        self.until = self._started + self._seconds

    def flush(self) -> None:
        """End the line: report it played once its audio has had time to play."""
        super().flush()
        seconds, self._seconds = self._seconds, 0.0
        if not seconds:
            return
        wait = max(0.0, self._started + seconds - time.monotonic())

        async def played() -> None:
            await asyncio.sleep(wait)
            self.on_playback_finished(playback_position=seconds, interrupted=False)

        self._playing = asyncio.create_task(played())

    def clear_buffer(self) -> None:
        """Stop the line at once: report how much of it played."""
        if self._playing is not None and not self._playing.done():
            self._playing.cancel()
            self.on_playback_finished(
                playback_position=time.monotonic() - self._started, interrupted=True
            )
        elif self._seconds:
            super().flush()
            self.on_playback_finished(
                playback_position=time.monotonic() - self._started, interrupted=True
            )
        self._seconds = 0.0
        self.until = time.monotonic()


class TextSession(AgentSession):
    """An ``AgentSession`` with no speech and no room, that counts its model requests."""

    started: ClassVar[list["TextSession"]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Drop the speech plugins, and count each request the session's model gets."""
        if not AUDIO:
            for key in SPEECH:
                kwargs.pop(key, None)
        self.requests = 0
        llm = kwargs.get("llm")
        if llm is not None:
            chat = llm.chat

            def counted(*args: Any, **inner: Any) -> Any:
                self.requests += 1
                return chat(*args, **inner)

            llm.chat = counted
        super().__init__(**kwargs)
        self.said: list[str] = []
        self.called: list[str] = []

        self._spoken: list[str] = []
        self.heard: list[str] = []
        self.busy_at = time.monotonic()
        self.answers = 0

        @self.on("user_input_transcribed")
        def transcribed(event: Any) -> None:
            if event.is_final and event.transcript.strip():
                self.heard.append(event.transcript.strip())

        @self.on("agent_state_changed")
        def state_changed(event: Any) -> None:
            self.busy_at = time.monotonic()
            if event.new_state in ("thinking", "speaking"):
                self.answers += 1

        @self.on("conversation_item_added")
        def added(event: Any) -> None:
            item = event.item
            if getattr(item, "role", None) == "assistant" and (text := item.text_content):
                if text in self._spoken:
                    self._spoken.remove(text)
                else:
                    self.said.append(text)

        @self.on("function_tools_executed")
        def ran(event: Any) -> None:
            self.called.extend(
                f"{call.name}({call.arguments})" if VERBOSE else call.name
                for call in event.function_calls
            )
            if os.getenv("CONVERSE_TIMING"):
                for call, output in zip(
                    event.function_calls, event.function_call_outputs, strict=False
                ):
                    result = output.output if output is not None else ""
                    print(f"    call {call.name} {call.arguments} -> {result}"[:400], flush=True)

    def say(self, text: Any, **kwargs: Any) -> Any:
        """Note a line code says, whether or not it joins the chat, then say it."""
        if isinstance(text, str):
            self.said.append(text)
            if kwargs.get("add_to_chat_ctx", True):
                self._spoken.append(text)
        return super().say(text, **kwargs)

    async def start(self, agent: Any, **kwargs: Any) -> Any:
        """Start with no room. On an audio run, the microphone and speaker stand in for it."""
        kwargs.pop("room", None)
        kwargs.pop("room_options", None)
        if AUDIO:
            self.microphone = Microphone()
            self.speaker = Speaker()
            self.input.audio = self.microphone
            self.output.audio = self.speaker
        TextSession.started.append(self)
        return await super().start(agent, **kwargs)

    async def quiet(self, answers: int | None = None) -> None:
        """Wait until the agent has answered, said everything, and its audio has played.

        Args:
            answers: The agent's answer count before the caller's line. Speech to text and
                the turn detector take a moment after the line ends, so the agent is only
                quiet once it has started an answer after it, or ``ANSWER_S`` passed.
        """
        start = time.monotonic()
        deadline = start + TURN_TIMEOUT_S
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            now = time.monotonic()
            if answers is not None and self.answers == answers and now - start < ANSWER_S:
                continue
            listening = self.agent_state == "listening" and self.current_speech is None
            if (
                listening
                and now > self.speaker.until
                and now - max(self.busy_at, self.speaker.until) > QUIET_S
            ):
                return


def patch(module: Any) -> None:
    """Give the example, and the shared stack it runs on, the text session."""
    for name, loaded in list(sys.modules.items()):
        shared = loaded is module or name.startswith("voice_stack")
        if shared and getattr(loaded, "AgentSession", None) is AgentSession:
            setattr(loaded, "AgentSession", TextSession)  # noqa: B010


async def settle(session: TextSession) -> None:
    """Wait until the agent has said everything it queued, its code lines included."""
    for _ in range(100):
        await asyncio.sleep(0.1)
        current = session.current_speech
        if current is None:
            await asyncio.sleep(0.3)
            if session.current_speech is None:
                return
        else:
            await current.wait_for_playout()


async def converse(module: Any, scenario: dict[str, Any]) -> bool:
    """Run one scenario through the module's entrypoint. Return True when it passed."""
    script.STATES.clear()
    script.start_values(scenario)
    TextSession.started.clear()
    call = script.Call()
    metadata = json.dumps({"call_start": scenario["dispatch"]}) if "dispatch" in scenario else ""
    with fake_job_context(
        room=SmokeRoom(), job_id=f"converse-{scenario['name']}", job_metadata=metadata
    ) as ctx:
        setup = getattr(module.server, "setup_fnc", None)
        if setup is not None:
            setup(ctx.proc)
        ctx.wait_for_participant = caller  # ty: ignore[invalid-assignment]
        await module.entrypoint(ctx)
        call.problems.extend(script.refusal(scenario, bool(TextSession.started)))
        if not TextSession.started:
            return script.report("livekit", scenario, call)
        [session] = TextSession.started
        try:
            if AUDIO:
                await session.quiet()
            else:
                await settle(session)
            call.greeting = list(session.said)
            for want in scenario["turns"]:
                said, tools, requests = len(session.said), len(session.called), session.requests
                heard = len(session.heard)
                turn = script.Turn(want["caller"])
                try:
                    if AUDIO:
                        answers = session.answers
                        await session.microphone.say(want["caller"], want.get("language"))
                        await session.microphone.done()
                        session.busy_at = time.monotonic()
                        await session.quiet(answers)
                    else:
                        await asyncio.wait_for(
                            session.run(user_input=want["caller"]), TURN_TIMEOUT_S
                        )
                        await settle(session)
                except Exception as error:  # noqa: BLE001
                    call.problems.append(f"turn {len(call.turns) + 1} failed: {error!r}")
                turn.heard = session.heard[heard:]
                turn.said = session.said[said:]
                turn.tools = session.called[tools:]
                turn.requests = session.requests - requests
                call.turns.append(turn)
                if not session._started:  # the agent ended the call
                    break
        finally:
            await session.aclose()
    return script.report("livekit", scenario, call)


async def main() -> None:
    """Run every scenario named on the command line, one call each.

    The speech plugins need an HTTP session, which a worker gives each job; a script opens
    its own.
    """
    name, *paths = sys.argv[1:]
    module = importlib.import_module(name)
    patch(module)
    async with http_context.open():
        results = [await converse(module, scenario) for scenario in script.load(paths)]
    script.exit_code(results)


if __name__ == "__main__":
    if VERBOSE:
        import logging

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("    log %(name)s: %(message)s"))
        logging.getLogger("unmute_core").addHandler(handler)
        logging.getLogger("unmute_core").setLevel(logging.INFO)
    if not AUDIO:
        os.environ.setdefault("LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE", "0")
    asyncio.run(main())
