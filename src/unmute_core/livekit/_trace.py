"""The call's spans, from LiveKit session events and the agents' ``tts_node``.

``conversation`` starts with the agent and ends when the session closes, with the transcript
and the call state. A ``turn`` opens when the agent starts a reply or the caller speaks,
whichever LiveKit reports first, and the agent's message ends it, with the latencies LiveKit
puts on each message. The caller's message also gives the ``stt`` span, from its own
timings. A tool run is an ``llm_tool_call``. The ``llm`` spans come from ``llm_node``, the
``tts`` spans from ``tts_node`` and the ``task_step`` spans from the step loop.

When the caller speaks over the agent, LiveKit's adaptive detector judges it an
interruption or a backchannel. Each verdict is a ``caller.over_agent`` event on the turn.

STT is read from the caller's message, not from ``stt_node``. Overriding ``stt_node`` stops
LiveKit reusing the STT stream when a step starts (voice/agent_activity.py:1091).
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterable, AsyncIterator

from livekit import rtc
from livekit.agents import inference, llm, tts
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.voice import (
    Agent,
    AgentSession,
    CloseEvent,
    ConversationItemAddedEvent,
    MetricsCollectedEvent,
    ModelSettings,
    ToolExecutionUpdatedEvent,
    UserInputTranscribedEvent,
)
from opentelemetry.trace import Span

from unmute_core.call import CALLER_HUNG_UP, ENDED, ERROR, CallHooks
from unmute_core.livekit._close import CloseGuard
from unmute_core.livekit._history import is_cue
from unmute_core.state import CallState
from unmute_core.tasks import ToolStatus
from unmute_core.telemetry import Telemetry, TurnTracker

logger = logging.getLogger(__name__)

_CLOSE_REASONS = {
    "participant_disconnected": CALLER_HUNG_UP,
    "error": ERROR,
}
"""LiveKit's close reasons that name why the call ended, as the shared end reasons."""

HANDED_OFF = json.dumps(ToolStatus(status="handed off").model_dump())
"""The output a handoff tool shows. Pipecat's handoff returns the same."""


class SessionTrace:
    """Turns one session's events into the call's spans.

    One turn is one caller message and everything the agent does until the next. A turn is
    the caller's until the agent does something in it. A new caller message after that
    starts the next turn.

    Attributes:
        telemetry: The call's telemetry.
        state: The call state, written on the call span when the session closes.
        turn: The turn that is open now, or None between turns.
        closing: The task that closes the telemetry once the session closes.
        results: Tool outputs to show in place of what LiveKit reports, by call ID. A step's
            closing tool puts its real result here, since the model reads only the status.
        handing_off: True from a handoff until its tool ends. LiveKit reports no output for
            a tool that returns an agent, so the span shows ``HANDED_OFF``.
        call: The call's hooks, or None. The close runs the call's end before the
            telemetry closes.
    """

    def __init__(
        self, telemetry: Telemetry, state: CallState | None = None, call: CallHooks | None = None
    ) -> None:
        """Keep the call's telemetry. Nothing is traced before ``attach``.

        Args:
            telemetry: The call's telemetry, new for this call.
            state: The call state, or None.
            call: The call's hooks, or None.
        """
        self.telemetry = telemetry
        self.state = state
        self.call = call
        self._turn = TurnTracker(telemetry)
        self.closing: asyncio.Task[bool] | None = None
        self.results: dict[str, str] = {}
        self.handing_off = False
        self._seen: set[str] = set()
        self._callers: dict[str, llm.ChatMessage] = {}
        self._replies: list[llm.ChatMessage] = []
        self._lines: list[str] = []
        self._language: str | None = None
        self._tools: dict[str, Span] = {}
        self._session: AgentSession | None = None

    @property
    def turn(self) -> Span | None:
        """The turn that is open now, or None between turns."""
        return self._turn.span

    def attach(self, session: AgentSession) -> None:
        """Start the call span and follow the session's events. A second call does nothing."""
        if self._session is not None:
            return
        self._session = session
        self.telemetry.start_call()
        self.telemetry.describe_call(framework="livekit")
        session.on("conversation_item_added", self._item_added)
        session.on("user_input_transcribed", self._transcribed)
        session.on("tool_execution_updated", self._tool_updated)
        session.on("overlapping_speech", self.over_agent)
        session.on("metrics_collected", self._metrics)
        session.on("close", self._closed)

    def _metrics(self, event: MetricsCollectedEvent) -> None:
        """Record a realtime model's reply as an ``llm`` span, once the reply is over.

        No request of a realtime model passes through ``ModelNode``, so this is its only
        span. The text the model said is in the turn.
        """
        metrics = event.metrics
        if not isinstance(metrics, RealtimeModelMetrics):
            return
        about = metrics.metadata
        span = self.telemetry.start_model_request(
            model=(about.model_name if about else None) or "unknown",
            provider=(about.model_provider if about else None) or "unknown",
            parent=self.open_turn(answering=False),
            started_at=metrics.timestamp,
        )
        self.telemetry.end_model_request(
            span,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            ttfb=metrics.ttft if metrics.ttft >= 0 else None,
            interrupted=metrics.cancelled,
        )

    def _transcribed(self, event: UserInputTranscribedEvent) -> None:
        """Keep the language of the caller's last final transcript for its ``stt`` span."""
        if event.is_final and event.language:
            self._language = str(event.language)

    def over_agent(self, event: inference.OverlappingSpeechEvent) -> None:
        """Record LiveKit's verdict on speech over the agent: an interruption or not.

        A backchannel does not stop the agent. With ``agent_ended``, the agent finished
        before the detector decided, so it is no verdict.
        """
        self.telemetry.event(
            "caller.over_agent",
            {
                "interruption": event.is_interruption,
                "agent_ended": event.agent_ended,
                "probability": round(event.probability, 2),
                "detection_delay": round(event.detection_delay, 2),
            },
            parent=self.turn,
        )

    def _item_added(self, event: ConversationItemAddedEvent) -> None:
        """Give a caller message to its turn, and add each line the agent says to the open one."""
        item = event.item
        if not isinstance(item, llm.ChatMessage):
            return
        if item.role == "user" and not is_cue(item):
            self.caller(item)
            self._stt(item)
        elif item.role == "assistant" and item.text_content:
            self.open_turn()
            self._replies.append(item)
            self._lines.append(item.text_content)

    def caller(self, message: llm.ChatMessage | None) -> None:
        """Start the turn of a caller message the trace has not seen yet.

        LiveKit adds the caller's message to the chat after the reply's request has started,
        so ``ModelNode`` hands the request's newest caller message here first. The open
        turn keeps the newest copy of its messages, which carries the speech metrics. A
        second message before the agent answers joins the open turn. One that starts with
        the words of an earlier one replaces it: LiveKit merges a turn whose reply was
        dropped into one new message.

        Args:
            message: The caller message, or None.
        """
        if message is None:
            return
        if message.id in self._callers:
            self._callers[message.id] = message
            return
        if message.id in self._seen:
            return
        self._seen.add(message.id)
        if not self._turn.caller_open:
            self._end_turn()
            self._turn.start()
        words = message.text_content or ""
        self._callers = {
            key: earlier
            for key, earlier in self._callers.items()
            if not (earlier.text_content and words.startswith(earlier.text_content))
        }
        self._callers[message.id] = message

    def said(self, text: str) -> None:
        """Add a line the agent says without adding it to the chat, like a step's receipt."""
        self.open_turn()
        self._lines.append(text)

    def open_turn(self, caller: llm.ChatMessage | None = None, *, answering: bool = True) -> Span:
        """Return the turn for something the agent does now, opening one if none is open.

        Args:
            caller: The newest caller message a model request carries. A new one starts
                the next turn first.
            answering: False for a model request. A request alone does not answer the
                caller: LiveKit cancels one when the caller goes on speaking, and the rest
                of their words belong to the same turn.
        """
        self.caller(caller)
        return self._turn.open(answering=answering)

    def _end_turn(self) -> None:
        """End the open turn with the caller's words, the agent's lines and the latencies."""
        if self.turn is None:
            return
        callers = list(self._callers.values())

        def first(messages: list[llm.ChatMessage], name: str) -> float | None:
            """The first value of one metric among the messages. A step line has none."""
            values = (m.metrics.get(name) for m in messages)
            return next((value for value in values if isinstance(value, float)), None)

        self._turn.end(
            user_text=" ".join(m.text_content or "" for m in callers).strip() or None,
            agent_lines=self._lines,
            interrupted=any(reply.interrupted for reply in self._replies),
            e2e=first(self._replies, "e2e_latency"),
            stt_final=first(callers, "transcription_delay"),
            eou=first(callers, "end_of_turn_delay"),
            llm_ttft=first(self._replies, "llm_node_ttft"),
            tts_ttfb=first(self._replies, "tts_node_ttfb"),
        )
        self._callers, self._replies, self._lines = {}, [], []

    def _stt(self, message: llm.ChatMessage) -> None:
        """Record the caller's message as an ``stt`` span, when STT heard it.

        A typed message has no speech timings and gives no span.
        """
        metrics = message.metrics
        started = metrics.get("started_speaking_at")
        heard = metrics.get("stt_metadata")
        if started is None or heard is None:
            return
        self.telemetry.stt(
            text=message.text_content or "",
            model=heard.get("model_name") or "unknown",
            provider=heard.get("model_provider") or "unknown",
            start_time=int(started * 1e9),
            language=self._language,
            ttfb=metrics.get("transcription_delay"),
            parent=self.turn,
        )

    def _tool_updated(self, event: ToolExecutionUpdatedEvent) -> None:
        """Start an ``llm_tool_call`` when a tool starts, and end it when it ends."""
        update = event.update
        if update.type == "tool_call_started":
            call = update.function_call
            self._tools[call.call_id] = self.telemetry.start_tool_call(
                name=call.name,
                call_id=call.call_id,
                arguments=call.arguments,
                parent=self.open_turn(),
            )
        elif update.type == "tool_call_ended" and (span := self._tools.pop(update.call_id, None)):
            error = None if update.status == "done" else update.status
            result = self.results.pop(update.call_id, update.message)
            if result is None and error is None and self.handing_off:
                self.handing_off, result = False, HANDED_OFF
            self.telemetry.end_tool_call(span, result=result, error=error)

    def show_result(self, call_id: str, result: str) -> None:
        """Show ``result`` as a tool call's output, in place of what LiveKit reports."""
        self.results[call_id] = result

    def start_handoff(self) -> None:
        """Note that a handoff tool runs: its span shows ``HANDED_OFF`` when it ends."""
        self.handing_off = True

    def tool_ran(self, *, name: str, call_id: str, arguments: str, result: str) -> None:
        """Record a tool that code ran, like a step's opening tool, as an ``llm_tool_call``."""
        span = self.telemetry.start_tool_call(
            name=name, call_id=call_id, arguments=arguments, parent=self.open_turn()
        )
        self.telemetry.end_tool_call(span, result=result)

    def _closed(self, event: CloseEvent) -> None:
        """Settle the call's held saves, end the open turn and close the telemetry."""
        self.closing = asyncio.create_task(self._close(end_reason(event.reason.value)))

    async def _close(self, reason: str) -> bool:
        """Settle the held saves in the last turn, then close the telemetry and its sinks.

        With ``call``, the call's end settles them, and its reason wins.
        """
        if self.call is not None:
            reason = (await self.call.end(reason)).reason
        elif self.state is not None:
            await self.state.settle_held()
        self._end_turn()
        for span in self._tools.values():
            span.end()
        self._tools.clear()
        return await self.telemetry.close(state=self.state, reason=reason)


def end_reason(close_reason: str) -> str:
    """Return the shared end reason for LiveKit's close reason.

    A caller who left is ``caller_hung_up`` and a failed session is ``error``. Any other
    close says nothing about why, so it is ``ended``: the call's own request, like the end
    tool or the max duration, wins over it in ``CallHooks.end``.
    """
    return _CLOSE_REASONS.get(close_reason, ENDED)


def trace_session(
    session: AgentSession,
    telemetry: Telemetry,
    state: CallState | None = None,
    *,
    call: CallHooks | None = None,
) -> SessionTrace:
    """Trace a session of your own: its turns, transcripts, tool calls and events.

    The job's shutdown waits until the last spans are flushed. ``llm`` spans come from
    ``ModelCallLLM(..., trace=trace)``. ``tts`` spans come from ``traced_tts`` in your
    agent's ``tts_node``. ``UnmuteAgent`` does all of this itself.

    Args:
        session: The call's session. Call this before or right after ``session.start``.
        telemetry: The call's telemetry, new for this call.
        state: The call state, written on the call span when the session closes.
        call: The call's hooks. The close runs their end before the telemetry closes.
            ``open_call`` attaches them to the session.

    Returns:
        The session's trace.
    """
    trace = SessionTrace(telemetry, state, call)
    trace.attach(session)
    guard = CloseGuard(lambda: [trace.closing], reraise=True)
    session.once("close", lambda _event: guard.closed())
    guard.hold_job()
    return trace


async def traced_tts(
    agent: Agent,
    trace: SessionTrace | None,
    text: AsyncIterable[str],
    model_settings: ModelSettings,
) -> AsyncIterator[rtc.AudioFrame]:
    """Run LiveKit's own ``tts_node`` and record each piece of speech as a ``tts`` span.

    The span opens with the first text, so its ttfb runs from there to the first audio.
    Speech cut off before its end is marked interrupted.

    Args:
        agent: The agent or step speaking.
        trace: The session's trace, or None for no span.
        text: The text LiveKit sends to TTS.
        model_settings: LiveKit's settings for this reply.
    """
    if trace is None:
        async for frame in Agent.default.tts_node(agent, text, model_settings):
            yield frame
        return
    spoken: list[str] = []
    span: Span | None = None
    first_text = 0.0
    ttfb: float | None = None
    done = False

    async def heard() -> AsyncIterator[str]:
        """Pass the text on, noting when the first chunk arrives."""
        nonlocal span, first_text
        async for chunk in text:
            if span is None:
                first_text = time.perf_counter()
                span = _start_tts(agent, trace.telemetry, trace.open_turn())
            spoken.append(chunk)
            yield chunk

    try:
        async for frame in Agent.default.tts_node(agent, heard(), model_settings):
            if ttfb is None and span is not None:
                ttfb = time.perf_counter() - first_text
            yield frame
        done = True
    finally:
        if span is not None:
            trace.telemetry.end_tts(span, text="".join(spoken), ttfb=ttfb, interrupted=not done)


def _start_tts(agent: Agent, telemetry: Telemetry, turn: Span) -> Span:
    """Start a ``tts`` span named after the agent's TTS plugin."""
    plugin = agent.tts if isinstance(agent.tts, tts.TTS) else agent.session.tts
    voice = plugin_voice(plugin)
    return telemetry.start_tts(
        model=plugin.model if plugin else "unknown",
        provider=plugin.provider if plugin else "unknown",
        voice=str(voice) if voice else None,
        parent=turn,
    )


def plugin_voice(plugin: tts.TTS | None) -> object | None:
    """Return the voice a TTS plugin speaks with, or None when it does not say.

    LiveKit's TTS base has no voice, so this reads the private options most plugins keep it
    in. A plugin that names it elsewhere gives None, and its span has no voice.
    """
    return getattr(getattr(plugin, "_opts", None), "voice", None)
