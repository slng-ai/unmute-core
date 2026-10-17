"""Run a Pipecat project's own ``bot()`` through scripted calls, for tests.

``python -m unmute_core.testing`` runs it. A project fits when its module has
``bot(runner_args)`` and gets its transport from ``create_transport`` in a module the
run patches: the project's module, or one named with ``--patch``.

``converse`` runs whole calls from scenario files on the project's real model. ``bot()``
runs as written, on a fake transport. The transport says a caller connected, then types
each caller line once the agent has gone quiet. Speech to text is a pass through. Text to
speech is a stand-in that turns text into a moment of silence, so the pipeline sees the
bot start and stop speaking as it does on a real call, and a confirm line counts as heard
only once it "played".

``CONVERSE_AUDIO=1`` runs the whole voice pipeline instead: each caller line is spoken by
SLNG text to speech and streamed in real time, so the project's own voice detector, turn
detector and speech to text hear it. The agent's own text to speech speaks every line,
and the transport plays its audio on a clock. Each turn then also shows what speech to
text heard.

``smoke`` runs one text turn. Without ``live`` a scripted model answers and no request
leaves the machine.
"""

import asyncio
import importlib
import json
import logging
import sys
import time
from collections.abc import AsyncGenerator, Sequence
from types import ModuleType
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    EndWorkerFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_transport import BaseTransport

from unmute_core.testing import _offline, _scenario, _voices
from unmute_core.testing._scenario import ConverseSettings, Scenario

QUIET_S = 2.5
"""How long the agent must stay quiet before the caller's next line."""
ANSWER_S = 15.0
"""How long a model response may stay open, or a tool result that asks for one may wait for
it, before the turn ends.

A slow model would otherwise look quiet, and the caller's next line would cut off what
the model says next, like a read-back. Gemini's first chunk can take over 3 s. LiveKit's
runner waits the same way while the agent is ``thinking``."""
TURN_TIMEOUT_S = 60
SMOKE_TIMEOUT_S = 30
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
SPEECH = ("OpenAISTTService", "OpenAITTSService", "SlngSTTService", "SlngTTSService")
"""The speech services a smoke run swaps for a pass through."""
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

    def __init__(self, settings: ConverseSettings) -> None:
        """Start with nothing said."""
        super().__init__()
        self._settings = settings
        self.said: list[str] = []
        self.tools: list[str] = []
        self.requests = 0
        self.last = time.monotonic()
        self._line: list[str] = []
        self._speaking = False
        self.ended = False
        self.answer_due = False

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
            self.tools.append(f"{name}({frame.arguments})" if self._settings.verbose else name)
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
            if self._speaking and not self._settings.audio:
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

    def __init__(self, transport: "FakeTransport", settings: ConverseSettings) -> None:
        """Start with nothing to say."""
        super().__init__(transport)
        self._settings = settings
        self._audio = bytearray()
        self._pump: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame on. After the start frame, start streaming audio."""
        if isinstance(frame, StartFrame) and self._pump is None:
            self._pump = asyncio.create_task(self._stream())
        await super().process_frame(frame, direction)

    async def _stream(self) -> None:
        """Push the caller's audio, or silence, every 20 ms, on the wall clock."""
        size = int(_voices.SAMPLE_RATE * CHUNK_S) * 2
        start, sent = time.monotonic(), 0
        while True:
            chunk = bytes(self._audio[:size]).ljust(size, b"\x00")
            del self._audio[:size]
            await self.push_frame(
                InputAudioRawFrame(audio=chunk, sample_rate=_voices.SAMPLE_RATE, num_channels=1)
            )
            sent += 1
            await asyncio.sleep(max(0.0, start + sent * CHUNK_S - time.monotonic()))

    async def say(self, line: str, language: str | None = None) -> None:
        """Speak one caller line, in its own language: its audio is streamed from now on."""
        language = language or self._settings.caller_language
        self._audio.extend(await _voices.spoken(line, language, self._settings))

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

    def __init__(self, settings: ConverseSettings) -> None:
        """Start silent."""
        super().__init__(settings)
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

    def __init__(self, settings: ConverseSettings) -> None:
        """Register the events a web transport fires."""
        super().__init__(name="converse")
        audio = settings.audio
        self.caller = AudioCaller(self, settings) if audio else CallerInput(self)
        self.agent = AudioOutput(settings) if audio else AgentOutput(settings)
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


def _shared(module: ModuleType, prefixes: Sequence[str]) -> list[ModuleType]:
    """Return the project's module and every loaded module whose name has a prefix."""
    loaded = [m for n, m in list(sys.modules.items()) if n.startswith(tuple(prefixes))]
    return [module, *(m for m in loaded if m is not module)] if prefixes else [module]


def _patch(
    modules: Sequence[ModuleType], transport: BaseTransport, settings: ConverseSettings
) -> None:
    """Give the project, and the modules it builds its pipeline in, the fake transport."""

    async def create_transport(*args: object, **kwargs: object) -> BaseTransport:
        """Hand the bot the fake transport."""
        return transport

    for loaded in modules:
        if hasattr(loaded, "create_transport"):
            setattr(loaded, "create_transport", create_transport)  # noqa: B010
        if hasattr(loaded, "SlngSTTService") and not settings.audio:
            setattr(loaded, "SlngSTTService", passthrough)  # noqa: B010
        if hasattr(loaded, "SlngTTSService") and not (settings.real_voice or settings.audio):
            setattr(loaded, "SlngTTSService", TextTTS)  # noqa: B010
        if settings.audio and (stt := getattr(loaded, "SlngSTTService", None)) is not None:
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


async def _turn(
    transport: FakeTransport, call: _scenario.Call, caller: str, language: str | None
) -> None:
    """Say one caller line, wait for the agent, and add what it did to the call."""
    agent = transport.agent
    said, tools, requests = len(agent.said), len(agent.tools), agent.requests
    heard = len(HEARD)
    await transport.caller.say(caller, language)
    await transport.caller.done()
    agent.last = time.monotonic()
    await agent.quiet()
    turn = _scenario.Turn(caller)
    turn.heard = HEARD[heard:]
    turn.said = agent.said[said:]
    turn.tools = agent.tools[tools:]
    turn.requests = agent.requests - requests
    call.turns.append(turn)


async def _converse_one(
    module: ModuleType, scenario: Scenario, settings: ConverseSettings, patch: Sequence[str]
) -> bool:
    """Run one scenario through the module's ``bot()``. Return True when it passed."""
    _scenario.start_values(scenario)
    transport = FakeTransport(settings)
    _patch(_shared(module, patch), transport, settings)
    agent = transport.agent
    call = _scenario.Call()
    body = {"call_start": scenario.dispatch} if scenario.dispatch is not None else None
    running = asyncio.create_task(module.bot(RunnerArguments(body=body)))
    try:
        ready = asyncio.create_task(transport.ready.wait())
        await asyncio.wait({ready, running}, timeout=30, return_when=asyncio.FIRST_COMPLETED)
        ready.cancel()
        call.problems.extend(_scenario.refusal(scenario, transport.ready.is_set()))
        if not transport.ready.is_set():
            return _scenario.report("pipecat", scenario, call, settings)
        await agent.quiet()
        call.greeting = list(agent.said)
        for want in scenario.turns:
            if agent.ended or running.done():
                break
            await _turn(transport, call, want.caller, want.language)
    except Exception as error:  # noqa: BLE001
        call.problems.append(f"the call failed: {error!r}")
    finally:
        if not running.done():
            await transport.hang_up()
        try:
            await asyncio.wait_for(running, 15)
        except Exception as error:  # noqa: BLE001
            call.problems.append(f"the bot did not end cleanly: {error!r}")
    return _scenario.report("pipecat", scenario, call, settings)


async def converse(
    module: ModuleType,
    scenarios: Sequence[Scenario],
    settings: ConverseSettings,
    *,
    patch: Sequence[str] = (),
) -> list[bool]:
    """Run each scenario as one call through the module's ``bot()``.

    Args:
        module: The project's module, with ``bot``.
        scenarios: The calls to run, in order.
        settings: How to run them.
        patch: Name prefixes of other modules that build the pipeline.

    Returns:
        Whether each scenario passed.
    """
    if settings.verbose:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("    log %(name)s: %(message)s"))
        logging.getLogger("unmute_core").addHandler(handler)
        logging.getLogger("unmute_core").setLevel(logging.INFO)
    else:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    _scenario.watch()
    return [await _converse_one(module, scenario, settings, patch) for scenario in scenarios]


def openai_reply(request: Any) -> Any:
    """Keep the request's system text, then answer with the reply, streamed as OpenAI does."""
    import httpx2

    _offline.PROMPTS.append(_offline.system_text(json.loads(request.content)["messages"]))
    chunks = [{"role": "assistant", "content": _offline.REPLY}, {}]
    events = [
        {
            "id": "smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "scripted",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None if delta else "stop"}],
        }
        for delta in chunks
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})


def _scripted_client(self: object, *args: object, **kwargs: object) -> Any:
    """Give the project's real OpenAI service a client that never leaves the machine."""
    import httpx2
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key="smoke",
        base_url="https://openai.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(openai_reply)),
    )


def _scripted_llm() -> Any:
    """Stand in for a model built by a helper, like a Gemini on Vertex."""
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(settings=OpenAILLMService.Settings(model="scripted"))


class SmokeCaller(FrameProcessor):
    """Says a caller connected when the pipeline starts, then types the caller line."""

    def __init__(self, transport: "SmokeTransport") -> None:
        """Keep the transport whose event this fires."""
        super().__init__()
        self.transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame on. After the start frame, connect and speak."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.transport.connected()
            caller = LLMMessagesAppendFrame(
                [{"role": "user", "content": _offline.CALLER}], run_llm=True
            )
            await self.push_frame(caller)


class SmokeOutput(FrameProcessor):
    """Keeps what the agent says, and ends the call after its first full reply.

    A reply is the model's text, or a line code speaks after a tool, like the opening
    line of a step that listens. A line spoken before any tool, like a greeting, is not.
    """

    def __init__(self) -> None:
        """Start with nothing said."""
        super().__init__()
        self.replies: list[str] = []
        self._text = ""
        self._tool_answered = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame on. Collect reply text and end once a reply is complete."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self._text += frame.text
        elif isinstance(frame, FunctionCallResultFrame):
            self._tool_answered = True
        elif isinstance(frame, TTSSpeakFrame) and self._tool_answered:
            self._text += frame.text
            await self._reply()
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._reply()

    async def _reply(self) -> None:
        """Keep the reply so far and end the call, when there is one."""
        if self._text.strip():
            self.replies.append(self._text)
            self._text = ""
            await self.push_frame(EndWorkerFrame(reason="smoke test done"))


class SmokeTransport(BaseTransport):
    """A transport with one caller who types one line instead of speaking."""

    def __init__(self) -> None:
        """Register the events a web transport fires."""
        super().__init__(name="smoke")
        self.caller = SmokeCaller(self)
        self.agent = SmokeOutput()
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    async def connected(self) -> None:
        """Fire the caller-connected event, as a web transport does."""
        await self._call_event_handler("on_client_connected", "smoke-client")

    def input(self) -> FrameProcessor:
        """The caller's side."""
        return self.caller

    def output(self) -> FrameProcessor:
        """The agent's side."""
        return self.agent


async def _smoke(module: ModuleType, name: str, live: bool, patch: Sequence[str]) -> str:
    """Run the module's ``bot()`` until the agent answers the caller. Return the reply."""
    transport = SmokeTransport()

    async def create_transport(*args: object, **kwargs: object) -> SmokeTransport:
        """Hand the bot the fake transport."""
        return transport

    for loaded in _shared(module, patch):
        if hasattr(loaded, "create_transport"):
            setattr(loaded, "create_transport", create_transport)  # noqa: B010
        for service in SPEECH:
            if hasattr(loaded, service):
                setattr(loaded, service, passthrough)
        if not live and hasattr(loaded, "build_llm"):
            setattr(loaded, "build_llm", lambda *args, **kwargs: _scripted_llm())  # noqa: B010
    if not live:
        from pipecat.services.openai.llm import OpenAILLMService

        OpenAILLMService.create_client = _scripted_client
    await module.bot(RunnerArguments())
    return _offline.check(name, transport.agent.replies, live=live)


def smoke(name: str, *, live: bool = False, patch: Sequence[str] = ()) -> str:
    """Run one caller turn through a project's ``bot()``. Return the agent's reply.

    Args:
        name: The project's module, like ``bot``.
        live: Answer with the project's own model and keys. Otherwise a script answers,
            placeholder keys fill the environment, and no request leaves the machine.
        patch: Name prefixes of other modules that build the pipeline.

    Raises:
        SystemExit: The agent did not answer, or the library logged an error.
    """
    _offline.listen_for_errors()
    if not live:
        _offline.go_offline()
    module = importlib.import_module(name)
    return asyncio.run(asyncio.wait_for(_smoke(module, name, live, patch), SMOKE_TIMEOUT_S))
