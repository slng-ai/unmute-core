"""LiveKit session events as the dev records of ``unmute_core.dev``.

The observer keeps the call's exchanges as the Pipecat observer does. An input exchange is
the caller speaking, or typing. A response exchange is everything the agent does after it:
its model requests, its tool calls, its speech, until the agent is idle again or the caller
speaks. Both hold the text of what was said.

Operations are one model request (``llm``), one transcript (``stt``), one piece of synthesis
(``tts``) and one tool call (``tool``). A task's handoff comes from ``DevRecorder.task_event``.
``UnmuteAgent`` reports each model request from its ``llm_node``. A plain agent's request,
and a realtime model's reply, come from LiveKit's metrics once they are over. A side
request, a handoff's summary or a step's finish check, gets no operation (``side_request``).

A response is linked to the caller turn the model request carries, or to the input it
follows. Only public session events are read.
"""

import contextlib
import weakref
from collections import deque
from collections.abc import Callable, Collection, Iterable
from contextvars import ContextVar
from typing import Any, Literal

from livekit.agents import llm, stt
from livekit.agents.metrics import LLMMetrics, RealtimeModelMetrics, STTMetrics, TTSMetrics
from livekit.agents.voice import (
    AgentSession,
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    MetricsCollectedEvent,
    SpeechCreatedEvent,
    ToolExecutionUpdatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.agents.voice.events import EventTypes
from livekit.agents.voice.io import AudioOutput, PlaybackStartedEvent

from unmute_core.call import CallHooks
from unmute_core.dev import DevRecorder, known_name
from unmute_core.livekit._dev_timings import Timings
from unmute_core.livekit._dev_tools import ToolRuns
from unmute_core.livekit._history import is_cue
from unmute_core.livekit._trace import end_reason
from unmute_core.tasks import CallTasks

_OBSERVERS: "weakref.WeakKeyDictionary[AgentSession, DevObserver]" = weakref.WeakKeyDictionary()
_REQUEST: ContextVar[str | None] = ContextVar("unmute_dev_request", default=None)
"""The ``llm`` operation of the model request that runs in this context, if any, or
``_SIDE`` for a side request."""
_SIDE = "side-request"
"""Marks a side request, like a summary or a finish check, which gets no operation."""
_BUSY = frozenset({"thinking", "speaking"})
_MAX_ID = 128
_UNCLAIMED_KEPT = 8
"""How many streamed answers wait for the chat message that repeats them. A reply LiveKit
never speaks, like one the caller talked over, leaves its answer waiting."""


def dev_observer(session: AgentSession | None) -> "DevObserver | None":
    """Return the observer attached to a session, or None."""
    return _OBSERVERS.get(session) if session is not None else None


def side_request() -> Callable[[], None]:
    """Mark the model request about to run in this context as a side request.

    A side request, a handoff's summary or a step's finish check, gets no ``llm``
    operation, as on Pipecat, where it never passes the pipeline. LiveKit still reports a
    plugin's metrics of it, and this tells the observer to skip them. A plugin's stream
    reads the mark as it opens.

    Returns:
        A function that takes the mark off once the request is over.
    """
    previous = _REQUEST.get()
    _REQUEST.set(_SIDE)

    def done() -> None:
        """Take the mark off, unless another request in this context replaced it."""
        if _REQUEST.get() == _SIDE:
            _REQUEST.set(previous)

    return done


def _printable(value: str | None) -> str | None:
    """Return a provider's id when it can sit in a record: short, printable, no spaces."""
    if value and len(value) <= _MAX_ID and all(33 <= ord(c) <= 126 for c in value):
        return value
    return None


class DevObserver:
    """Sends the dev records of one LiveKit call from its session's events.

    Attach it once: ``DevObserver(recorder).attach(session)``. It sends the exchanges, the
    caller's text and the agent's, the operations (``llm``, ``stt``, ``tts``, ``tool``) and
    these measurements: ``first_response`` and ``request_duration`` of each model request
    and each piece of synthesis, ``tool_duration`` of each tool call, ``reply_latency``,
    ``turn_detection``, ``transcription_delay``, ``speech_duration``, ``playback_delay`` and
    ``text_aggregation`` of each reply, and ``first_speech`` of the call. The session's close
    ends the call's records.

    With ``UnmuteAgent``, every agent of the call reports its model requests and the lines
    it says outside the chat, and the observer listens to each agent's tasks.
    """

    def __init__(
        self,
        recorder: DevRecorder,
        *,
        handoffs: Collection[str] = (),
        controls: Collection[str] = (),
    ) -> None:
        """Watch one call.

        Args:
            recorder: The call's recorder, built with ``target="livekit"``.
                ``recorder.watch_state`` is for the call's state.
            handoffs: The tools that start a task or a group. They get no ``tool``
                operation, because ``recorder.task_event`` draws the run as a ``handoff``.
                An ``UnmuteAgent`` adds its own as it enters.
            controls: The tools that hand the call over without a result: human transfers.
                They get a ``handoff`` operation and no ``tool_duration``. An
                ``UnmuteAgent`` adds its agent handoff tools as it enters.

        Raises:
            ValueError: If the recorder was built for another framework.
        """
        if (call := recorder.get("call", "call")) is not None and call["target"] != "livekit":
            raise ValueError(
                "DevObserver for LiveKit needs DevRecorder(call_id, target='livekit')."
            )
        self._rec = recorder
        self._tools = ToolRuns(recorder, handoffs=handoffs, controls=controls)
        self._timings = Timings(recorder)
        self._call: CallHooks | None = None
        self._session: AgentSession | None = None
        self._listeners: list[tuple[EventTypes, Callable[..., None]]] = []
        self._tasks: list[CallTasks] = []
        self._agent_state = "initializing"
        self._input: str | None = None
        self._unlinked: str | None = None
        self._stt: str | None = None
        self._callers: dict[str, str] = {}
        self._answers: dict[str, str] = {}
        self._waiting: dict[str, dict[str, Any]] = {}
        self._cut: str | None = None
        self._spoken: set[str] = set()
        self._interrupted: set[str] = set()
        self._speeches: dict[str, str] = {}
        self._running: dict[str, str] = {}
        self._answer: dict[str, str] = {}
        self._unclaimed: deque[str] = deque(maxlen=_UNCLAIMED_KEPT)
        self._last_llm: str | None = None
        self._requested: set[str] = set()
        self._audio: AudioOutput | None = None
        self._last_stt: str | None = None

    def attach(self, session: AgentSession) -> None:
        """Follow the session's events. Call it before or after ``session.start``.

        Raises:
            ValueError: If the observer already follows another session, or the session
                already has an observer.
        """
        if self._session is session:
            return
        if self._session is not None or dev_observer(session) is not None:
            raise ValueError("A session takes one DevObserver, and a DevObserver one session.")
        self._session = session
        self._timings.attached()
        _OBSERVERS[session] = self
        self._listeners = [
            ("user_state_changed", self._user_state),
            ("user_input_transcribed", self._transcribed),
            ("conversation_item_added", self._item_added),
            ("agent_state_changed", self._agent_state_changed),
            ("speech_created", self._speech_created),
            ("tool_execution_updated", self._tool_updated),
            ("metrics_collected", self._metrics),
            ("error", self._error),
            ("close", self._closed),
        ]
        for name, listener in self._listeners:
            session.on(name, listener)
        self._watch_audio()
        with contextlib.suppress(RuntimeError):
            self.entered(session.current_agent)

    def entered(self, agent: object) -> None:
        """Follow the tasks and the handoff tools of an agent that takes the call.

        Only this agent's handoff tools, and the ``controls`` the observer was built with,
        draw as handoffs from now on. ``UnmuteAgent`` calls it.
        """
        self._tools.set_controls(getattr(agent, "handoffs", ()))
        self._call = getattr(agent, "call", None) or self._call
        tasks = getattr(agent, "tasks", None)
        if isinstance(tasks, CallTasks) and not any(known is tasks for known in self._tasks):
            self._tasks.append(tasks)
            tasks.listen(self._rec.task_event)
            self._tools.add_handoffs(tool["name"] for tool in tasks.delegate_tools())

    def finish(self, *, error: bool = False, reason: str | None = None) -> None:
        """End the call's records. The session's close does it. Call it when none comes.

        Args:
            error: True when the call ended in an error.
            reason: What ended it.
        """
        self._rec.end_call(error=error, reason=reason)
        self._detach()

    def _detach(self) -> None:
        """Stop following the session."""
        if (session := self._session) is not None:
            for name, listener in self._listeners:
                session.off(name, listener)
            _OBSERVERS.pop(session, None)
        self._listeners.clear()
        if self._audio is not None:
            self._audio.off("playback_started", self._playback_started)
            self._audio = None

    def request_started(
        self, *, name: str, model: str, provider: str, caller: llm.ChatMessage | None
    ) -> str:
        """Open an ``llm`` operation for a model request, with its two timings pending.

        Args:
            name: What answers the request: the LLM plugin's label, or the model call's type.
            model: The model name, or ``unknown``.
            provider: Who serves it, or ``unknown``.
            caller: The newest caller message the request carries, which links the reply.

        Returns:
            The operation's id, for the other ``request_`` methods.
        """
        if caller is not None:
            self._caller_message(caller)
        response = self._response()
        self._requested.add(response)
        operation = self._rec.operation(
            "llm",
            name,
            exchange_id=response,
            model=known_name(model),
            provider=known_name(provider),
        )
        self._running[operation] = response
        self._last_llm = operation
        self._rec.pending_request(operation, response, "livekit-llm-node")
        _REQUEST.set(operation)
        return operation

    def request_text(self, operation: str, text: str) -> None:
        """Add the words a model request streamed to its answer's text row."""
        if (response := self._running.get(operation)) is None or not text:
            return
        row = self._answer.setdefault(operation, self._rec.new_id("answer-text"))
        self._rec.append_text(row, response, text)

    def request_calls(self, operation: str, call_ids: Iterable[str]) -> None:
        """Note the tool calls a model request chose, so each tool names its request."""
        self._tools.chose(operation, call_ids)

    def request_ended(
        self,
        operation: str,
        state: str,
        *,
        first: float | None,
        duration: float,
        model: str | None = None,
    ) -> None:
        """Settle a model request, its text and its two timings.

        Args:
            operation: The id ``request_started`` returned.
            state: ``ended``, ``cancelled`` or ``failed``.
            first: Seconds to the first chunk, or None when none came.
            duration: Seconds the whole request took.
            model: The model that answered, when a fallback chain moved to another one than
                the request started on. None keeps the name ``request_started`` wrote.
        """
        if _REQUEST.get() == operation:
            _REQUEST.set(None)
        if (response := self._running.pop(operation, None)) is None:
            return
        if known := known_name(model):
            self._rec.update("operation", operation, model=known)
        row = self._answer.pop(operation, None)
        if row is not None:
            if state == "ended":
                self._unclaimed.append(row)
            elif response in self._spoken:
                self._interrupted.add(response)
        self._rec.settle_request(
            operation,
            state,
            timings={"first_response": first, "request_duration": duration},
            source="livekit-llm-node",
            unavailable=f"The request {state} before this measurement arrived."
            if state != "ended"
            else "The model sent nothing.",
            answer=row,
        )

    def said(self, line: str) -> None:
        """Record a line the agent says outside the chat, like a step's receipt, as final."""
        if line.strip():
            self._rec.direct_text(self._target(), line)

    def _user_state(self, event: UserStateChangedEvent) -> None:
        """Open an input when the caller starts to speak."""
        if event.new_state == "speaking":
            self._caller_starts()

    def _caller_starts(self) -> None:
        """Open an input exchange for the caller's turn, unless one is open and unanswered.

        LiveKit says the caller speaks each time voice starts, also within one turn, so
        an open input keeps the words. An input a reply already answers, still open, is
        incomplete. The open response ends: at once when the agent is idle, else once it
        is, cut off when LiveKit says the caller interrupted it.
        """
        if self._rec.closed or (self._input is not None and self._input == self._unlinked):
            return
        if self._input is not None:
            self._rec.update("exchange", self._input, state="incomplete")
        if (response := self._rec.response) is not None:
            self._rec.close_response()
            if self._busy():
                self._settle(self._cut)
                self._cut = response
            else:
                self._settle(response)
        self._input = self._unlinked = self._rec.open_input()

    def _transcribed(self, event: UserInputTranscribedEvent) -> None:
        """Record what the caller said as a text row, provisional until final."""
        if not event.transcript.strip() or self._rec.closed:
            return
        self._caller_starts()
        if (current := self._input) is None:
            return
        self._rec.user_text(current, event.transcript, final=event.is_final)
        self._stt_operation(current, final=event.is_final)

    def _stt_operation(self, current: str, *, final: bool) -> None:
        """Keep an ``stt`` operation running from a transcript's first words to its end."""
        if (plugin := self._stt_plugin()) is None:
            return
        if self._stt is None:
            self._stt = self._last_stt = self._rec.operation(
                "stt",
                plugin.label,
                exchange_id=current,
                model=known_name(plugin.model),
                provider=known_name(plugin.provider),
            )
        if final:
            self._rec.update("operation", self._stt, state="ended")
            self._stt = None

    def _stt_plugin(self) -> stt.STT | None:
        """Return the agent's STT plugin, else the session's, or None, as on a realtime model."""
        session = self._session
        if session is None:
            return None
        with contextlib.suppress(RuntimeError):
            if isinstance(own := session.current_agent.stt, stt.STT):
                return own
        return session.stt if isinstance(session.stt, stt.STT) else None

    def _item_added(self, event: ConversationItemAddedEvent) -> None:
        """Take a caller message as the end of its turn, and a reply as the agent's text."""
        item = event.item
        if not isinstance(item, llm.ChatMessage) or self._rec.closed or is_cue(item):
            return
        if item.role == "user":
            current = self._caller_message(item)
            if (response := self._answers.get(current)) is not None:
                self._timings.turn(response, dict(item.metrics))
            else:
                self._waiting[current] = dict(item.metrics)
        elif item.role == "assistant":
            self._assistant_message(item)

    def _caller_message(self, message: llm.ChatMessage) -> str:
        """End the input a caller message closes, and return it. Typed text becomes its row."""
        if (known := self._callers.get(message.id)) is not None:
            return known
        if self._input is None:
            self._caller_starts()
        if (current := self._input) is None:
            return ""
        self._callers[message.id] = current
        self._rec.typed_text(current, message.text_content or "")
        self._rec.update("exchange", current, state="ended")
        self._input = None
        return current

    def _response(self) -> str:
        """Return the open response exchange, or open one linked to the input it answers."""
        rec = self._rec
        if rec.response is not None:
            return rec.response
        linked = self._unlinked
        identity = rec.open_response(linked)
        if linked is not None:
            self._answers[linked] = identity
            self._unlinked = None
        if linked is not None and (metrics := self._waiting.pop(linked, None)) is not None:
            self._timings.turn(identity, metrics)
        return identity

    def _target(self) -> str:
        """Return the response the agent's work now belongs to: a cut one, or the open one."""
        return self._rec.response or self._cut or self._response()

    def _busy(self) -> bool:
        """Return True while the agent thinks or speaks, or a request or a tool runs."""
        return self._agent_state in _BUSY or bool(self._running) or self._tools.running

    def _settle(self, response: str | None) -> None:
        """End a response: cut off when LiveKit said so, else ended."""
        if response is None:
            return
        self._timings.settled(response)
        self._rec.settle_exchange(
            response, cut=response in self._interrupted, spoke=response in self._spoken
        )
        if self._cut == response:
            self._cut = None

    def _watch_audio(self) -> None:
        """Follow the session's audio output for the start of playback, once one is set."""
        if self._session is None or (output := self._session.output.audio) is self._audio:
            return
        if self._audio is not None:
            self._audio.off("playback_started", self._playback_started)
        self._audio = output
        if output is not None:
            output.on("playback_started", self._playback_started)

    def _playback_started(self, event: PlaybackStartedEvent) -> None:
        """Take the output's playback start as the call's first speech."""
        self._timings.first_speech(event.created_at, "livekit-audio-output")

    def _speech_created(self, event: SpeechCreatedEvent) -> None:
        """Tie new speech to the open response, so its synthesis lands there."""
        self._watch_audio()
        if self._rec.response is not None:
            self._speeches.setdefault(event.speech_handle.id, self._rec.response)

    def _agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        """Follow the agent: thinking opens the reply, speaking plays it, idle ends it."""
        if self._rec.closed:
            return
        self._watch_audio()
        old, self._agent_state = self._agent_state, event.new_state
        if old == "initializing":
            self._timings.started(event.created_at)
        if event.new_state == "thinking":
            self._response()
        elif event.new_state == "speaking":
            response = self._target()
            if self._session is not None and (speech := self._session.current_speech):
                self._speeches.setdefault(speech.id, response)
            self._spoken.add(response)
            self._timings.speaking(response, event.created_at)
            self._rec.update("exchange", response, playback="speaking")
            if self._audio is None:
                self._timings.first_speech(event.created_at, "livekit-agent-state")
        elif not self._busy() and not self._tools.delegating:
            self._settle(self._cut)
            self._settle(self._rec.response)
            self._rec.close_response()

    def _assistant_message(self, item: llm.ChatMessage) -> None:
        """Take a reply as the agent's text, unless a model request already streamed it."""
        response = self._target()
        words = (item.text_content or "").strip()
        if words:
            if (row := self._claim(words)) is not None:
                if item.interrupted:
                    self._rec.update("text", row, state="incomplete")
            else:
                if response not in self._requested:
                    self._rec.limited(generated_text="limited")
                self._rec.direct_text(response, words, interrupted=item.interrupted)
        if item.interrupted:
            self._interrupted.add(response)
        self._timings.reply(response, item.id, dict(item.metrics))

    def _claim(self, words: str) -> str | None:
        """Return the streamed answer a reply repeats, and take it off the unclaimed list."""
        for row in self._unclaimed:
            streamed = ((self._rec.get("text", row) or {}).get("text") or "").strip()
            if streamed == words or streamed.startswith(words):
                self._unclaimed.remove(row)
                return row
        return None

    def _tool_updated(self, event: ToolExecutionUpdatedEvent) -> None:
        """Open a ``tool`` operation as a tool starts, and end and time it as it ends."""
        update = event.update
        if self._rec.closed:
            return
        if update.type == "tool_call_started":
            self._tools.started(
                update, event.created_at, target=self._target, last_request=self._last_llm
            )
        elif update.type == "tool_call_ended":
            self._tools.ended(
                update,
                event.created_at,
                closing=lambda: self._session is not None and self._session._is_closing(),
            )

    def _metrics(self, event: MetricsCollectedEvent) -> None:
        """Turn LiveKit's metrics into operations: synthesis, and requests nobody reported."""
        metrics = event.metrics
        if self._rec.closed:
            return
        if isinstance(metrics, LLMMetrics) and (operation := _REQUEST.get()) is not None:
            if operation != _SIDE and (request_id := _printable(metrics.request_id)):
                self._rec.update("operation", operation, source_request_id=request_id)
        elif isinstance(metrics, LLMMetrics | RealtimeModelMetrics):
            self._reported_request(metrics)
        elif isinstance(metrics, TTSMetrics):
            self._synthesis(metrics)
        elif isinstance(metrics, STTMetrics) and not metrics.streamed and self._last_stt:
            self._rec.measure_operation(
                self._last_stt,
                "request_duration",
                metrics.duration,
                "livekit-stt-metrics",
                exchange_id=(self._rec.get("operation", self._last_stt) or {}).get("exchange_id"),
            )

    def _speech_response(self, speech_id: str | None) -> str:
        """Return the response a speech belongs to, tying a new speech to the current one."""
        if speech_id is None:
            return self._target()
        if speech_id not in self._speeches:
            self._speeches[speech_id] = self._target()
        return self._speeches[speech_id]

    def _reported_operation(
        self,
        kind: Literal["llm", "tts"],
        metrics: LLMMetrics | RealtimeModelMetrics | TTSMetrics,
        response: str,
        *,
        first: float,
        source: str,
        silent: str,
    ) -> str:
        """Record an operation LiveKit reported once it was over, with its two timings.

        Args:
            kind: ``llm`` for a model request, ``tts`` for a piece of synthesis.
            metrics: What LiveKit reported.
            response: The response it belongs to.
            first: Seconds to the first token or audio, negative when none came.
            source: Where the timings came from.
            silent: Why the first timing is unavailable when none came.

        Returns:
            The operation's id.
        """
        about = metrics.metadata
        model = known_name(about.model_name) if about else None
        provider = known_name(about.model_provider) if about else None
        operation = self._rec.operation(
            kind,
            metrics.label or kind,
            exchange_id=response,
            state="cancelled" if metrics.cancelled else "ended",
            model=model,
            provider=provider,
            request_id=_printable(metrics.request_id),
        )
        for metric, value in (
            ("first_response", first if first >= 0 else None),
            ("request_duration", metrics.duration),
        ):
            self._rec.measure_operation(
                operation,
                metric,
                value,
                source,
                exchange_id=response,
                reason=None if value is not None else silent,
                model=model,
                provider=provider,
            )
        return operation

    def _reported_request(self, metrics: LLMMetrics | RealtimeModelMetrics) -> None:
        """Record a model request LiveKit reported once it was over, with its two timings."""
        speech = metrics.speech_id if isinstance(metrics, LLMMetrics) else None
        response = self._speech_response(speech)
        self._requested.add(response)
        self._rec.limited(model_calls="partial")
        self._last_llm = self._reported_operation(
            "llm",
            metrics,
            response,
            first=metrics.ttft,
            source="livekit-llm-metrics",
            silent="The model sent no first token.",
        )

    def _synthesis(self, metrics: TTSMetrics) -> None:
        """Record one piece of synthesis LiveKit reported once it was over, with its timings."""
        self._reported_operation(
            "tts",
            metrics,
            self._speech_response(metrics.speech_id),
            first=metrics.ttfb,
            source="livekit-tts-metrics",
            silent="The synthesis sent no audio.",
        )

    def _error(self, event: ErrorEvent) -> None:
        """Mark the call as an error when a part fails for good, with its name only."""
        if not event.error.recoverable:
            source = event.source
            name = getattr(source, "label", None) or type(source).__name__
            self._rec.mark_error(f"{name} reported a fatal error.")

    def _closed(self, event: CloseEvent) -> None:
        """End the replies that are done, then the call's records, with the call's end reason.

        A fatal error's reason, already on the call, stays.
        """
        if not self._busy():
            self._settle(self._cut)
            self._settle(self._rec.response)
            self._rec.close_response()
        failed = (self._rec.get("call", "call") or {}).get("reason")
        reason = end_reason(event.reason.value)
        if self._call is not None:
            reason = self._call.resolved_reason(reason)
        self.finish(error=event.error is not None, reason=None if failed else reason)
