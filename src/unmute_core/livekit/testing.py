"""Run a LiveKit project's own entrypoint through scripted calls, for tests.

``python -m unmute_core.testing`` runs it. A project fits when its module has
``entrypoint(ctx)`` and a ``server`` (an ``AgentServer``), and builds its ``AgentSession``
in a module the run patches: the project's module, or one named with ``--patch``.

``converse`` runs whole calls from scenario files on the project's real model. The
entrypoint runs as written, on a fake job with no room. Its session drops speech in and
out, so each call is text only: the caller's lines go in as typed turns, and every line
the agent says comes back as text, its code-spoken lines included.

``CONVERSE_AUDIO=1`` keeps the session's speech instead. A microphone streams each caller
line, spoken by SLNG text to speech, in real time, so the project's own voice detector,
turn detector and speech to text hear it. A speaker plays the agent's audio on a clock
and tells LiveKit when each line has played. Each turn then also shows what speech to
text heard.

``smoke`` runs one text turn. Without ``live`` a scripted model answers and no request
leaves the machine.
"""

import asyncio
import importlib
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

from livekit import rtc
from livekit.agents import AgentSession
from livekit.agents.testing import fake_job_context
from livekit.agents.utils import http_context
from livekit.agents.voice import io

from unmute_core.livekit._model import ModelCallLLM
from unmute_core.testing import _offline, _scenario, _voices
from unmute_core.testing._scenario import ConverseSettings, Scenario

SPEECH = ("stt", "tts", "vad", "turn_handling", "turn_detection")
TURN_TIMEOUT_S = 60
SMOKE_TIMEOUT_S = 30
QUIET_S = 2.5
ANSWER_S = 15.0
"""How long a caller line may go unanswered before the turn counts as one with no answer."""
CHUNK_S = 0.02


class Microphone(io.AudioInput):
    """A caller's microphone: their lines, then silence, in real time, 20 ms at a time."""

    def __init__(self, settings: ConverseSettings) -> None:
        """Start with nothing to say."""
        super().__init__(label="converse-caller")
        self._settings = settings
        self._audio = bytearray()
        self._start = 0.0
        self._sent = 0

    async def __anext__(self) -> rtc.AudioFrame:
        """Return the next 20 ms of the caller's audio, or of silence, on the wall clock."""
        if not self._start:
            self._start = time.monotonic()
        samples = int(_voices.SAMPLE_RATE * CHUNK_S)
        chunk = bytes(self._audio[: samples * 2]).ljust(samples * 2, b"\x00")
        del self._audio[: samples * 2]
        self._sent += 1
        await asyncio.sleep(max(0.0, self._start + self._sent * CHUNK_S - time.monotonic()))
        return rtc.AudioFrame(chunk, _voices.SAMPLE_RATE, 1, samples)

    async def say(self, line: str, language: str | None = None) -> None:
        """Speak one caller line: its audio is streamed from now on."""
        language = language or self._settings.caller_language
        self._audio.extend(await _voices.spoken(line, language, self._settings))

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
    """An ``AgentSession`` with no speech and no room, that counts its model requests.

    Attributes:
        settings: How the run runs. ``converse`` sets it before the entrypoint runs.
        started: Every session started since the last call began.
    """

    settings: ClassVar[ConverseSettings]
    started: ClassVar[list["TextSession"]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Drop the speech plugins, and count each request the session's model gets."""
        if not self.settings.audio:
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
                f"{call.name}({call.arguments})" if self.settings.verbose else call.name
                for call in event.function_calls
            )

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
        if self.settings.audio:
            self.microphone = Microphone(self.settings)
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


def _patch(modules: Sequence[ModuleType], session: type[AgentSession]) -> None:
    """Give each module that builds an ``AgentSession`` the session class to build."""
    for module in modules:
        if getattr(module, "AgentSession", None) is AgentSession:
            setattr(module, "AgentSession", session)  # noqa: B010


def _shared(module: ModuleType, prefixes: Sequence[str]) -> list[ModuleType]:
    """Return the project's module and every loaded module whose name has a prefix."""
    loaded = [m for n, m in list(sys.modules.items()) if n.startswith(tuple(prefixes))]
    return [module, *(m for m in loaded if m is not module)] if prefixes else [module]


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


async def _turn(
    session: TextSession, call: _scenario.Call, caller: str, language: str | None
) -> None:
    """Say one caller line, wait for the agent, and add what it did to the call."""
    said, tools, requests = len(session.said), len(session.called), session.requests
    heard = len(session.heard)
    turn = _scenario.Turn(caller)
    try:
        if session.settings.audio:
            answers = session.answers
            await session.microphone.say(caller, language)
            await session.microphone.done()
            session.busy_at = time.monotonic()
            await session.quiet(answers)
        else:
            await asyncio.wait_for(session.run(user_input=caller), TURN_TIMEOUT_S)
            await settle(session)
    except Exception as error:  # noqa: BLE001
        call.problems.append(f"turn {len(call.turns) + 1} failed: {error!r}")
    turn.heard = session.heard[heard:]
    turn.said = session.said[said:]
    turn.tools = session.called[tools:]
    turn.requests = session.requests - requests
    call.turns.append(turn)


async def _converse_one(module: ModuleType, scenario: Scenario, settings: ConverseSettings) -> bool:
    """Run one scenario through the module's entrypoint. Return True when it passed."""
    _scenario.start_values(scenario)
    TextSession.started.clear()
    call = _scenario.Call()
    dispatch = scenario.dispatch
    metadata = json.dumps({"call_start": dispatch}) if dispatch is not None else ""
    with fake_job_context(
        room=SmokeRoom(), job_id=f"converse-{scenario.name}", job_metadata=metadata
    ) as ctx:
        setup = getattr(module.server, "setup_fnc", None)
        if setup is not None:
            setup(ctx.proc)
        ctx.wait_for_participant = caller  # ty: ignore[invalid-assignment]
        await module.entrypoint(ctx)
        call.problems.extend(_scenario.refusal(scenario, bool(TextSession.started)))
        if not TextSession.started:
            return _scenario.report("livekit", scenario, call, settings)
        [session] = TextSession.started
        try:
            if settings.audio:
                await session.quiet()
            else:
                await settle(session)
            call.greeting = list(session.said)
            for want in scenario.turns:
                await _turn(session, call, want.caller, want.language)
                if not session._started:  # the agent ended the call
                    break
        finally:
            await session.aclose()
    return _scenario.report("livekit", scenario, call, settings)


async def converse(
    module: ModuleType,
    scenarios: Sequence[Scenario],
    settings: ConverseSettings,
    *,
    patch: Sequence[str] = (),
) -> list[bool]:
    """Run each scenario as one call through the module's entrypoint.

    The speech plugins need an HTTP session, which a worker gives each job. A run opens
    its own.

    Args:
        module: The project's module, with ``entrypoint`` and ``server``.
        scenarios: The calls to run, in order.
        settings: How to run them.
        patch: Name prefixes of other modules that build the session.

    Returns:
        Whether each scenario passed.
    """
    if settings.verbose:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("    log %(name)s: %(message)s"))
        logging.getLogger("unmute_core").addHandler(handler)
        logging.getLogger("unmute_core").setLevel(logging.INFO)
    if not settings.audio:
        os.environ.setdefault("LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE", "0")
    _scenario.watch()
    TextSession.settings = settings
    _patch(_shared(module, patch), TextSession)
    async with http_context.open():
        return [await _converse_one(module, scenario, settings) for scenario in scenarios]


class SmokeSession(AgentSession):
    """An ``AgentSession`` with no speech and no room. A script answers unless live."""

    live: ClassVar[bool] = False
    started: ClassVar[list["SmokeSession"]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Drop the speech plugins and, unless live, swap a framework model for a script."""
        for key in SPEECH:
            kwargs.pop(key, None)
        if not self.live and not isinstance(kwargs.get("llm"), ModelCallLLM):
            kwargs["llm"] = ModelCallLLM(_offline.Scripted())
        super().__init__(**kwargs)

    async def start(self, agent: Any, **kwargs: Any) -> Any:
        """Start with no room, so no audio is read or played."""
        kwargs.pop("room", None)
        kwargs.pop("room_options", None)
        SmokeSession.started.append(self)
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


async def _smoke(module: ModuleType, name: str, live: bool, patch: Sequence[str]) -> str:
    """Run the module's entrypoint and one caller turn. Return the agent's reply."""
    SmokeSession.live = live
    SmokeSession.started.clear()
    _patch(_shared(module, patch), SmokeSession)
    with fake_job_context(room=SmokeRoom(), job_id=f"smoke-{name}") as ctx:
        setup = getattr(module.server, "setup_fnc", None)
        if setup is not None:
            setup(ctx.proc)
        ctx.wait_for_participant = caller  # ty: ignore[invalid-assignment]
        await module.entrypoint(ctx)
        [session] = SmokeSession.started
        try:
            result = await session.run(user_input=_offline.CALLER)
        finally:
            await session.aclose()
    replies = [
        event.item.text_content or ""
        for event in result.events
        if event.type == "message" and event.item.role == "assistant"
    ]
    return _offline.check(name, replies, live=live)


def smoke(name: str, *, live: bool = False, patch: Sequence[str] = ()) -> str:
    """Run one caller turn through a project's entrypoint. Return the agent's reply.

    Args:
        name: The project's module, like ``agent``.
        live: Answer with the project's own model and keys. Otherwise a script answers,
            placeholder keys fill the environment, and no request leaves the machine.
        patch: Name prefixes of other modules that build the session.

    Raises:
        SystemExit: The agent did not answer, or the library logged an error.
    """
    _offline.listen_for_errors()
    if not live:
        _offline.go_offline()
    module = importlib.import_module(name)
    return asyncio.run(asyncio.wait_for(_smoke(module, name, live, patch), SMOKE_TIMEOUT_S))
