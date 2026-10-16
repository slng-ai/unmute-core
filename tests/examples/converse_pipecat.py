"""Run whole scripted calls through a Pipecat example's own ``bot()``, on its real model.

Run it inside the example's venv, from the project folder::

    .venv/bin/python -I ../../../tests/examples/converse_pipecat.py bot ../scenarios/*.json

``bot()`` runs as written, on a fake transport. The transport says a caller connected,
then types each caller line once the agent has gone quiet. Speech to text is a pass
through. Text to speech is a stand-in that turns text into a moment of silence, so the
pipeline sees the bot start and stop speaking as it does on a real call, and a confirm
line counts as heard only once it "played". The model is the example's own, with the
keys in your environment. See ``tests/examples/script.py`` for the scenario format.

``CONVERSE_AUDIO=1`` runs the whole voice pipeline instead: each caller line is spoken
by SLNG text to speech (``tests/examples/voices.py``) and streamed in real time, so the
example's own voice detector, turn detector and speech to text hear it; the agent's own
text to speech speaks every line, and the transport plays its audio on a clock. Each
turn then also shows what speech to text heard.
"""

import asyncio
import importlib
import logging
import os
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

sys.path[:0] = [os.getcwd(), str(Path(__file__).resolve().parents[2])]

from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData  # noqa: E402
from pipecat.processors.filters.identity_filter import IdentityFilter  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.runner.types import RunnerArguments  # noqa: E402
from pipecat.services.settings import TTSSettings  # noqa: E402
from pipecat.services.tts_service import TTSService  # noqa: E402
from pipecat.transports.base_transport import BaseTransport  # noqa: E402

from tests.examples import script, voices  # noqa: E402

QUIET_S = 2.5
"""How long the agent must stay quiet before the caller's next line."""
ANSWER_S = 15.0
"""How long a model response may stay open, or a tool result that asks for one may wait for
it, before the turn ends.

A slow model would otherwise look quiet, and the caller's next line would cut off what
the model says next, like a read-back. Gemini's first chunk can take over 3 s. LiveKit's
runner waits the same way while the agent is ``thinking``."""
TURN_TIMEOUT_S = 60
VERBOSE = bool(os.getenv("CONVERSE_VERBOSE"))
TIMING = bool(os.getenv("CONVERSE_TIMING"))
TIMELINE: list[tuple[float, str]] = []
"""Timed events of the call, when ``CONVERSE_TIMING`` is set."""


def mark(event: str) -> None:
    """Note an event, with the time, for the timing trace."""
    if TIMING:
        TIMELINE.append((time.monotonic(), event))


class TimingHandler(logging.Handler):
    """Puts the library's log lines on the timing trace."""

    def emit(self, record: logging.LogRecord) -> None:
        mark(f"log {record.name}: {record.getMessage()}")
REAL_VOICE = bool(os.getenv("CONVERSE_REAL_VOICE"))
"""Keep the example's own text to speech: each line is synthesised, and its audio dropped."""
AUDIO = bool(os.getenv("CONVERSE_AUDIO"))
"""Run the whole voice pipeline, with a caller who speaks."""
CALLER_LANGUAGE = os.getenv("CONVERSE_CALLER_LANGUAGE", "en")
CHUNK_S = 0.02
ACTIVITY = (
    TTSAudioRawFrame,
    TTSTextFrame,
    LLMTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    FunctionCallInProgressFrame,
)
"""The frames that show the agent is still answering."""
HEARD: list[str] = []
"""What speech to text heard, in order."""


def passthrough(*args: object, **kwargs: object) -> FrameProcessor:
    """Stand in for speech to text: the caller types, so frames pass through."""
    return IdentityFilter()


class TextTTS(TTSService):
    """Stand in for text to speech: each sentence becomes a moment of silence.

    The base service still pushes the sentence's text, so the context records what the
    agent said, and the start and stop of speech, so a line can be heard.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Take no settings: there is no voice to configure."""
        super().__init__(
            sample_rate=16000,
            push_start_frame=True,
            push_stop_frames=True,
            stop_frame_timeout_s=0.3,
            settings=TTSSettings(model=None, voice=None, language=None),
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None]:
        """Yield 20 ms of silence for the sentence."""
        yield TTSAudioRawFrame(b"\x00" * 640, 16000, 1, context_id=context_id)


class CallerInput(FrameProcessor):
    """Says a caller connected when the pipeline starts. ``say`` types a caller line."""

    def __init__(self, transport: "FakeTransport") -> None:
        """Keep the transport whose event this fires."""
        super().__init__()
        self.transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame on. After the start frame, the caller connects."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.transport.connected()

    async def say(self, line: str, language: str | None = None) -> None:
        """Type one caller line, as a finished user turn.

        The caller starts and stops speaking first, as on a real call, so the agent's last
        line joins the context before the caller's words.
        """
        await self.push_frame(UserStartedSpeakingFrame())
        await asyncio.sleep(0.2)
        await self.push_frame(UserStoppedSpeakingFrame())
        await self.push_frame(
            LLMMessagesAppendFrame([{"role": "user", "content": line}], run_llm=True)
        )

    async def done(self) -> None:
        """A typed line is whole at once."""


class AgentOutput(FrameProcessor):
    """Keeps what the agent says and does, and plays each piece of speech as a bot would.

    Like an output transport, it says the bot started speaking at a line's first audio,
    and stopped once its speech ends.
    """

    def __init__(self) -> None:
        """Start with nothing said."""
        super().__init__()
        self.said: list[str] = []
        self.tools: list[str] = []
        self.requests = 0
        self.last = time.monotonic()
        self._line: list[str] = []
        self._speaking = False
        self.ended = False
        self.answer_due = False
        self._texted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame on, and note the agent's speech, requests and tool calls."""
        await super().process_frame(frame, direction)
        if isinstance(frame, EndFrame):
            self.ended = True
        await self.push_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, ACTIVITY):
            self.last = time.monotonic()
        if TIMING:
            if isinstance(frame, LLMTextFrame):
                if not self._texted:
                    mark("first text")
                self._texted = True
            elif isinstance(frame, LLMFullResponseStartFrame | LLMFullResponseEndFrame):
                self._texted = False
                mark(type(frame).__name__)
            elif isinstance(frame, FunctionCallInProgressFrame):
                mark(f"call {frame.function_name} {frame.arguments}"[:300])
            elif isinstance(frame, FunctionCallResultFrame):
                run_llm = frame.properties.run_llm if frame.properties else None
                mark(f"result {frame.function_name} run_llm={run_llm} {frame.result}"[:300])
            elif isinstance(frame, MetricsFrame):
                for data in frame.data:
                    if isinstance(data, TTFBMetricsData) and "LLM" in data.processor:
                        mark(f"ttfb {data.processor} {data.value:.2f}s")
        if isinstance(frame, FunctionCallResultFrame):
            properties = frame.properties
            run = (
                frame.run_llm if frame.run_llm is not None else (properties and properties.run_llm)
            )
            self.answer_due = run is not False
        if isinstance(frame, LLMFullResponseStartFrame):
            self.answer_due = True
            self.requests += 1
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.answer_due = False
        elif isinstance(frame, FunctionCallInProgressFrame):
            name = frame.function_name
            self.tools.append(f"{name}({frame.arguments})" if VERBOSE else name)
        elif isinstance(frame, TTSAudioRawFrame) and not self._speaking:
            self._speaking = True
            await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            await self.push_frame(BotStartedSpeakingFrame())
        elif isinstance(frame, TTSTextFrame):
            self._line.append(frame.text)
        elif isinstance(frame, TTSStoppedFrame):
            if line := " ".join(" ".join(self._line).split()):
                self.said.append(line)
            self._line = []
            if self._speaking and not AUDIO:
                self._speaking = False
                await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
                await self.push_frame(BotStoppedSpeakingFrame())

    async def quiet(self) -> None:
        """Wait until nothing has come out of the agent for ``QUIET_S``, and no reply is due."""
        deadline = time.monotonic() + TURN_TIMEOUT_S
        while time.monotonic() < deadline and not self.ended:
            await asyncio.sleep(0.1)
            idle = time.monotonic() - self.last
            if idle > QUIET_S and (not self.answer_due or idle > ANSWER_S):
                return


class AudioCaller(CallerInput):
    """A caller who speaks: their audio, then silence, in real time, 20 ms at a time."""

    def __init__(self, transport: "FakeTransport") -> None:
        """Start with nothing to say."""
        super().__init__(transport)
        self._audio = bytearray()
        self._pump: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame on. After the start frame, start streaming audio."""
        if isinstance(frame, StartFrame) and self._pump is None:
            self._pump = asyncio.create_task(self._stream())
        await super().process_frame(frame, direction)

    async def _stream(self) -> None:
        """Push the caller's audio, or silence, every 20 ms, on the wall clock."""
        size = int(voices.SAMPLE_RATE * CHUNK_S) * 2
        start, sent = time.monotonic(), 0
        while True:
            chunk = bytes(self._audio[:size]).ljust(size, b"\x00")
            del self._audio[:size]
            await self.push_frame(
                InputAudioRawFrame(audio=chunk, sample_rate=voices.SAMPLE_RATE, num_channels=1)
            )
            sent += 1
            await asyncio.sleep(max(0.0, start + sent * CHUNK_S - time.monotonic()))

    async def say(self, line: str, language: str | None = None) -> None:
        """Speak one caller line, in its own language: its audio is streamed from now on."""
        self._audio.extend(await voices.spoken(line, language or CALLER_LANGUAGE))

    async def done(self) -> None:
        """Wait until the caller's line has been streamed to its end."""
        while self._audio:
            await asyncio.sleep(CHUNK_S)

    async def cleanup(self) -> None:
        """Stop streaming."""
        if self._pump is not None:
            self._pump.cancel()
        await super().cleanup()


class AudioOutput(AgentOutput):
    """Plays the agent's audio on a clock, as a transport would, and says when it stops."""

    def __init__(self) -> None:
        """Start silent."""
        super().__init__()
        self._until = 0.0
        self._clock: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Note what the agent says, and when its audio would finish playing."""
        if isinstance(frame, StartFrame) and self._clock is None:
            self._clock = asyncio.create_task(self._tick())
        if isinstance(frame, TTSAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            seconds = len(frame.audio) / (frame.sample_rate * 2 * frame.num_channels)
            self._until = max(self._until, time.monotonic()) + seconds
        await super().process_frame(frame, direction)

    async def _tick(self) -> None:
        """Say the bot stopped speaking once its audio has played out."""
        while True:
            await asyncio.sleep(0.05)
            if self._speaking and time.monotonic() > self._until + 0.2:
                self._speaking = False
                self.last = time.monotonic()
                await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
                await self.push_frame(BotStoppedSpeakingFrame())

    async def quiet(self) -> None:
        """Wait until the agent has said everything, and its audio has played out."""
        deadline = time.monotonic() + TURN_TIMEOUT_S
        while time.monotonic() < deadline and not self.ended:
            await asyncio.sleep(0.1)
            idle = time.monotonic() - self.last
            if not self._speaking and idle > QUIET_S and (not self.answer_due or idle > ANSWER_S):
                return

    async def cleanup(self) -> None:
        """Stop the clock."""
        if self._clock is not None:
            self._clock.cancel()
        await super().cleanup()


class FakeTransport(BaseTransport):
    """A transport with one caller who types instead of speaking."""

    def __init__(self) -> None:
        """Register the events a web transport fires."""
        super().__init__(name="converse")
        self.caller = AudioCaller(self) if AUDIO else CallerInput(self)
        self.agent = AudioOutput() if AUDIO else AgentOutput()
        self.ready = asyncio.Event()
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    async def connected(self) -> None:
        """Fire the caller-connected event, as a web transport does."""
        await self._call_event_handler("on_client_connected", "converse-client")
        self.ready.set()

    async def hang_up(self) -> None:
        """Fire the caller-disconnected event, as a web transport does."""
        await self._call_event_handler("on_client_disconnected", "converse-client")

    def input(self) -> FrameProcessor:
        """The caller's side."""
        return self.caller

    def output(self) -> FrameProcessor:
        """The agent's side."""
        return self.agent


def patch(module: Any, transport: FakeTransport) -> None:
    """Give the example, and the shared stack it runs on, the fake transport and speech."""

    async def create_transport(*args: object, **kwargs: object) -> FakeTransport:
        """Hand the bot the fake transport."""
        return transport

    for name, loaded in list(sys.modules.items()):
        if loaded is module or name.startswith("voice_stack"):
            if hasattr(loaded, "create_transport"):
                setattr(loaded, "create_transport", create_transport)  # noqa: B010
            if hasattr(loaded, "SlngSTTService") and not AUDIO:
                setattr(loaded, "SlngSTTService", passthrough)  # noqa: B010
            if hasattr(loaded, "SlngTTSService") and not (REAL_VOICE or AUDIO):
                setattr(loaded, "SlngTTSService", TextTTS)  # noqa: B010
            if AUDIO and (stt := getattr(loaded, "SlngSTTService", None)) is not None:
                hear(stt)


def hear(stt: type[FrameProcessor]) -> None:
    """Keep every final transcript the speech-to-text service pushes."""
    if getattr(stt, "_converse_heard", False):
        return
    push = stt.push_frame

    async def pushed(
        self: FrameProcessor, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            HEARD.append(frame.text.strip())
        await push(self, frame, direction)

    stt.push_frame = pushed  # ty: ignore[invalid-assignment]
    stt._converse_heard = True  # ty: ignore[unresolved-attribute]


async def converse(module: Any, scenario: dict[str, Any]) -> bool:
    """Run one scenario through the module's ``bot()``. Return True when it passed."""
    script.STATES.clear()
    script.start_values(scenario)
    transport = FakeTransport()
    patch(module, transport)
    agent = transport.agent
    call = script.Call()
    body = {"call_start": scenario["dispatch"]} if "dispatch" in scenario else None
    running = asyncio.create_task(module.bot(RunnerArguments(body=body)))
    try:
        ready = asyncio.create_task(transport.ready.wait())
        await asyncio.wait({ready, running}, timeout=30, return_when=asyncio.FIRST_COMPLETED)
        ready.cancel()
        call.problems.extend(script.refusal(scenario, transport.ready.is_set()))
        if not transport.ready.is_set():
            return script.report("pipecat", scenario, call)
        await agent.quiet()
        call.greeting = list(agent.said)
        for want in scenario["turns"]:
            if agent.ended or running.done():
                break
            said, tools, requests = len(agent.said), len(agent.tools), agent.requests
            heard = len(HEARD)
            started = time.monotonic()
            mark(f"caller: {want['caller']}")
            await transport.caller.say(want["caller"], want.get("language"))
            await transport.caller.done()
            agent.last = time.monotonic()
            await agent.quiet()
            turn = script.Turn(want["caller"])
            turn.heard = HEARD[heard:]
            turn.said = agent.said[said:]
            turn.tools = agent.tools[tools:]
            turn.requests = agent.requests - requests
            call.turns.append(turn)
            if TIMING:
                mark("quiet: next turn")
                print(f"  timing, turn {len(call.turns)}:")
                for at, event in TIMELINE:
                    if at >= started:
                        print(f"    {at - started:6.2f}  {event}")
                TIMELINE.clear()
    except Exception as error:  # noqa: BLE001
        call.problems.append(f"the call failed: {error!r}")
    finally:
        if not running.done():
            await transport.hang_up()
        try:
            await asyncio.wait_for(running, 15)
        except Exception as error:  # noqa: BLE001
            call.problems.append(f"the bot did not end cleanly: {error!r}")
    return script.report("pipecat", scenario, call)


async def main() -> None:
    """Run every scenario named on the command line, one call each."""
    name, *paths = sys.argv[1:]
    module = importlib.import_module(name)
    results = [await converse(module, scenario) for scenario in script.load(paths)]
    script.exit_code(results)


if __name__ == "__main__":
    if TIMING:
        logging.getLogger("unmute_core").addHandler(TimingHandler())
        logging.getLogger("unmute_core").setLevel(logging.INFO)
    if VERBOSE:
        import logging

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("    log %(name)s: %(message)s"))
        logging.getLogger("unmute_core").addHandler(handler)
        logging.getLogger("unmute_core").setLevel(logging.INFO)
    else:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    asyncio.run(main())
