"""One call's tracing: its own TracerProvider, its sinks, and the spans both adapters emit.

Every span carries the call's attributes. Span names follow Coval and Pipecat. Attributes
follow the OpenTelemetry GenAI conventions, so Langfuse, Logfire and Coval all read them.
docs/core/observe/telemetry.mdx lists the contract.
"""

import asyncio
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from datetime import UTC, datetime
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.sdk.trace import SpanLimits, SpanProcessor, TracerProvider
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel, Field, ValidationError

from unmute_core import __version__
from unmute_core.state import CallState
from unmute_core.telemetry._sinks import Sink

logger = logging.getLogger(__name__)

CallOrigin = Literal["phone", "browser", "websocket"]
CALL_SPAN = "conversation"
TURN_SPAN = "turn"
MODEL_SPAN = "llm"
TOOL_SPAN = "llm_tool_call"
TASK_SPAN = "task_step"
STT_SPAN = "stt"
TTS_SPAN = "tts"
MAX_ATTRIBUTE_LENGTH = 256 * 1024
CLOSE_TIMEOUT_SECONDS = 10.0
OBSERVATION_TYPE = "langfuse.observation.type"
COMPLETION_START = "langfuse.observation.completion_start_time"
LEVEL = "langfuse.observation.level"
STATUS_MESSAGE = "langfuse.observation.status_message"
CUT_OFF = "cut off before the reply finished"


def _seconds(value: float | None) -> float | None:
    """Return a latency in seconds, or None when it is missing, negative or not finite."""
    return value if value is not None and math.isfinite(value) and value >= 0 else None


def _json(value: object) -> str | None:
    """Return ``value`` as JSON text, or None when it is empty."""
    return json.dumps(value, ensure_ascii=False, default=str) if value else None


def _completion_start(span: Span, ttfb: float | None) -> str | None:
    """Return the span's start plus ``ttfb`` as an ISO 8601 time, or None without a ttfb."""
    start = getattr(span, "start_time", None)
    if ttfb is None or start is None:
        return None
    return datetime.fromtimestamp(start / 1e9 + ttfb, UTC).isoformat()


def _ended(state: CallState | None, reason: str | None) -> dict[str, Any]:
    """Return how the call ended: the reason, the final values and the confirmed ones."""
    values = state.dump() if state is not None else None
    confirmed = [name for name in values or {} if state is not None and state.is_confirmed(name)]
    return {"end_reason": reason, "state": values, "confirmed": confirmed}


def _end(span: Span, attributes: Mapping[str, AttributeValue | None], error: str | None) -> None:
    """Set the attributes that have a value, mark an error, and end the span."""
    span.set_attributes({key: value for key, value in attributes.items() if value is not None})
    if error is not None:
        span.set_attributes({"error.type": error})
        span.set_status(Status(StatusCode.ERROR, error))
    span.end()


UPSTREAM_PROVIDER = "unmute.upstream.provider"
"""Who runs the model behind a gateway like SLNG, read from the model id."""
GATEWAYS = frozenset({"openrouter", "slng"})
"""Providers that pass a request on to another provider named in the model id."""


def upstream_provider(
    provider: str, model: str, gateways: AbstractSet[str] = GATEWAYS
) -> str | None:
    """Return who runs a gateway's model, like ``cartesia`` for SLNG's ``cartesia/sonic:3.5``.

    A gateway names its models ``[<gateway>/]<upstream>/<model>``. Any other provider, or a
    model id without an upstream part, gives None.

    Args:
        provider: Who serves the request.
        model: The model id.
        gateways: The providers that are gateways, in lowercase.
    """
    if provider.lower() not in gateways:
        return None
    parts = model.lower().removeprefix(f"{provider.lower()}/").split("/")
    return parts[0].lower() if len(parts) > 1 and parts[0] else None


class _Identity(BaseModel):
    """Who a call's telemetry is for: the agent and the call, neither empty."""

    agent: str = Field(min_length=1)
    call_id: str = Field(min_length=1)


class _CallAttributes(SpanProcessor):
    """Stamps the call's identity on every span, framework spans included."""

    def __init__(self, attributes: Mapping[str, AttributeValue]) -> None:
        """Keep the attributes to stamp."""
        self._attributes = attributes

    def on_start(self, span: SdkSpan, parent_context: Context | None = None) -> None:
        """Stamp the call's identity on a span that just started."""
        span.set_attributes(self._attributes)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Report the flush done: nothing is held here.

        Before OpenTelemetry 1.42 the base method returns None, which the provider reads as
        a failed flush.
        """
        return True


class Telemetry:
    """The tracing of one call: a TracerProvider of its own, sending to every sink.

    Build one per call, with new sinks. Start the call span first, then turns, model requests,
    tool calls, speech and task steps under it. ``event`` records each decision. ``close``
    ends the call with its transcript and flushes every sink.

    Attributes:
        agent: The agent's name. It is the trace name and the service name.
        call_id: The call's ID, stamped on every span as ``session.id``.
        provider: The call's TracerProvider. An adapter may hand it to the framework.
        tracer: The tracer the span helpers use.
    """

    def __init__(
        self,
        *sinks: Sink,
        agent: str,
        call_id: str,
        resource: Mapping[str, AttributeValue] | None = None,
        processors: Sequence[SpanProcessor] = (),
        gateways: AbstractSet[str] = GATEWAYS,
    ) -> None:
        """Build the call's TracerProvider with one processor per sink.

        Args:
            *sinks: Where the spans go.
            agent: The agent's name.
            call_id: The call's ID, like the room or session ID.
            resource: Extra resource attributes, like ``deployment.environment.name``.
            processors: Extra span processors that run before the sinks, for example one
                that adds your own attributes.
            gateways: The providers that pass a request on to another, named in the model
                id. A speech span then records that one as ``unmute.upstream.provider``.

        Raises:
            ValueError: If ``agent`` or ``call_id`` is empty.
        """
        try:
            _Identity(agent=agent, call_id=call_id)
        except ValidationError as error:
            raise ValueError("Telemetry needs an agent name and a call_id.") from error
        self.agent = agent
        self.call_id = call_id
        self.gateways = frozenset(name.lower() for name in gateways)
        self.provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: agent, **(resource or {})}),
            span_limits=SpanLimits(max_span_attribute_length=MAX_ATTRIBUTE_LENGTH),
        )
        identity = {"session.id": call_id, "gen_ai.conversation.id": call_id, "agent.name": agent}
        for processor in (_CallAttributes(identity), *processors):
            self.provider.add_span_processor(processor)
        for sink in sinks:
            self.provider.add_span_processor(sink.processor())
        self.tracer = self.provider.get_tracer("unmute_core", __version__)
        self._call: Span | None = None
        self._described: dict[str, str] = {}
        self._turn: Span | None = None
        self._transcript: list[dict[str, str]] = []
        self._requests: Counter[int] = Counter()
        self._cut: Counter[int] = Counter()

    def _start(
        self,
        name: str,
        parent: Span | None,
        attributes: Mapping[str, AttributeValue | None],
        start_time: int | None = None,
    ) -> Span:
        """Start a span under ``parent``, or under the call span when it is None."""
        parent = parent or self._call
        if parent is None:
            raise RuntimeError("Call telemetry.start_call() before any other span.")
        return self.tracer.start_span(
            name,
            context=trace.set_span_in_context(parent),
            attributes={key: value for key, value in attributes.items() if value is not None},
            start_time=start_time,
        )

    def start_call(self) -> Span:
        """Start the call's root span, ``conversation``, in a new trace.

        Raises:
            RuntimeError: If the call span was already started.
        """
        if self._call is not None:
            raise RuntimeError("The call span is already started. Build one Telemetry per call.")
        span = self.tracer.start_span(
            CALL_SPAN,
            context=Context(),
            attributes={
                "gen_ai.operation.name": "invoke_workflow",
                "langfuse.trace.name": self.agent,
                **self._described,
            },
        )
        self._call = span
        return span

    def describe_call(
        self, *, origin: CallOrigin | None = None, framework: str | None = None
    ) -> None:
        """Say how the caller reached the agent and which framework runs it.

        The values go on the call span as ``unmute.call.origin`` and ``unmute.call.framework``.
        Values given before ``start_call`` wait and are set when it starts the span. After the
        call span ended, nothing changes. A value left out keeps what an earlier call set.

        Args:
            origin: ``phone``, ``browser`` or ``websocket``.
            framework: The agent framework, like ``livekit`` or ``pipecat``.
        """
        values = {
            key: value
            for key, value in (("unmute.call.origin", origin), ("unmute.call.framework", framework))
            if value
        }
        self._described.update(values)
        if self._call is not None and self._call.is_recording():
            self._call.set_attributes(values)

    def start_turn(self, number: int, *, parent: Span | None = None) -> Span:
        """Start a ``turn``: one thing the caller says and the agent's answer.

        Until it ends, it is the default parent of ``stt``, ``tts`` and event spans.
        """
        self._turn = self._start(
            TURN_SPAN,
            parent,
            {
                "turn.number": number,
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": self.agent,
            },
        )
        return self._turn

    def end_turn(
        self,
        span: Span,
        *,
        user_text: str | None = None,
        agent_lines: Sequence[str] = (),
        interrupted: bool = False,
        e2e: float | None = None,
        stt_final: float | None = None,
        eou: float | None = None,
        llm_ttft: float | None = None,
        tts_ttfb: float | None = None,
        error: str | None = None,
    ) -> None:
        """End a turn. Latencies are in seconds. Missing or negative ones are left out.

        Args:
            span: The span ``start_turn`` returned.
            user_text: What the caller said.
            agent_lines: Each line the agent said in the turn, in order. The transcript keeps
                them apart. The turn's output joins them with a space.
            interrupted: Whether the caller cut the agent off.
            e2e: From the end of the caller's speech to the agent's first audio.
            stt_final: From the end of the caller's speech to the final transcript.
            eou: From the end of the caller's speech to the end of their turn: how long
                turn detection waited.
            llm_ttft: From the model request to its first token.
            tts_ttfb: From the first text sent to TTS to its first audio.
            error: The error type, when the turn failed.

        The turn also shows how many model requests it made, ``turn.llm_requests``, and how
        many of them were cut off, ``turn.llm_requests_cut``. A line code spoke needs none.
        """
        if span is self._turn:
            self._turn = None
        turn = span.get_span_context().span_id
        lines = [line for line in agent_lines if line]
        if user_text:
            self._transcript.append({"role": "user", "content": user_text})
        self._transcript.extend({"role": "assistant", "content": line} for line in lines)
        _end(
            span,
            {
                "input.value": user_text,
                "output.value": " ".join(lines) or None,
                "turn.interrupted": interrupted,
                "turn.e2e_seconds": _seconds(e2e),
                "turn.stt_final_seconds": _seconds(stt_final),
                "turn.eou_seconds": _seconds(eou),
                "turn.llm_ttft_seconds": _seconds(llm_ttft),
                "turn.tts_ttfb_seconds": _seconds(tts_ttfb),
                "turn.llm_requests": self._requests.pop(turn, 0),
                "turn.llm_requests_cut": self._cut.pop(turn, 0),
            },
            error,
        )

    def start_model_request(
        self,
        *,
        model: str,
        provider: str,
        site: str = "",
        messages: Sequence[Mapping[str, Any]] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        parent: Span | None = None,
        started_at: float | None = None,
    ) -> Span:
        """Start an ``llm`` span for one model request. Pass the turn as ``parent``.

        Args:
            model: The model name, like ``gpt-4.1``.
            provider: Who serves it, like ``openai`` or ``slng``.
            site: The prompt site, ``task:<name>`` while a task runs.
            messages: The chat sent, prompt filled, in OpenAI format. Stored as text.
            tools: The tools offered, as OpenAI function schemas. Stored as text.
            parent: The span to nest under, usually the turn.
            started_at: When the request started, in seconds since the epoch, for a
                request reported once it is over, like a realtime model's reply. None is now.
        """
        if parent is not None:
            self._requests[parent.get_span_context().span_id] += 1
        return self._start(
            MODEL_SPAN,
            parent,
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": provider.lower(),
                "gen_ai.request.model": model,
                "unmute.prompt.site": site or None,
                "input.value": _json(messages),
                "gen_ai.tool.definitions": _json(tools),
            },
            start_time=int(started_at * 1e9) if started_at is not None else None,
        )

    def end_model_request(
        self,
        span: Span,
        *,
        text: str | None = None,
        tool_calls: Sequence[Mapping[str, Any]] = (),
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        finish_reason: str | None = None,
        ttfb: float | None = None,
        error: str | None = None,
        interrupted: bool = False,
    ) -> None:
        """End a model request.

        The output is the text alone, or with tool calls the whole assistant message in
        OpenAI format, so Langfuse shows each call the model chose.

        Args:
            span: The span ``start_model_request`` returned.
            text: The text the model wrote.
            tool_calls: The calls the model chose, in OpenAI format: ``{"id", "type":
                "function", "function": {"name", "arguments"}}`` with arguments as JSON text.
            input_tokens: Tokens in the prompt.
            output_tokens: Tokens in the answer.
            finish_reason: Why the model stopped, like ``stop`` or ``tool_calls``.
            ttfb: Seconds to the first token.
            error: The error type, when the request failed.
            interrupted: True when the reply was cut off before it finished, like a reply
                the caller spoke over. Langfuse shows the span as a warning that says so.
        """
        if interrupted and (parent := getattr(span, "parent", None)) is not None:
            self._cut[parent.span_id] += 1
        output = (
            _json({"role": "assistant", "content": text, "tool_calls": list(tool_calls)})
            if tool_calls
            else text
        )
        _end(
            span,
            {
                "output.value": output,
                "gen_ai.usage.input_tokens": input_tokens,
                "gen_ai.usage.output_tokens": output_tokens,
                "gen_ai.response.finish_reasons": [finish_reason] if finish_reason else None,
                "metrics.ttfb": _seconds(ttfb),
                COMPLETION_START: _completion_start(span, _seconds(ttfb)),
                "llm.interrupted": interrupted or None,
                LEVEL: "WARNING" if interrupted else None,
                STATUS_MESSAGE: CUT_OFF if interrupted else None,
            },
            error,
        )

    def stt(
        self,
        *,
        text: str,
        model: str,
        provider: str,
        start_time: int | None = None,
        language: str | None = None,
        ttfb: float | None = None,
        parent: Span | None = None,
    ) -> None:
        """Record one final transcript as an ``stt`` span that ends now.

        Args:
            text: The transcript.
            model: The STT model.
            provider: Who serves it.
            start_time: When the caller started speaking, in ``time.time_ns()``. None is now.
            language: The language the STT heard.
            ttfb: Seconds from the end of speech to the transcript.
            parent: The span to nest under. The open turn, or the call, when None.
        """
        span = self._start(
            STT_SPAN,
            parent or self._turn,
            {
                OBSERVATION_TYPE: "generation",
                "gen_ai.operation.name": "stt",
                "gen_ai.provider.name": provider.lower(),
                "gen_ai.request.model": model,
                UPSTREAM_PROVIDER: upstream_provider(provider, model, self.gateways),
                "stt.language": language,
            },
            start_time,
        )
        _end(span, {"output.value": text, "metrics.ttfb": _seconds(ttfb)}, None)

    def start_tts(
        self, *, model: str, provider: str, voice: str | None = None, parent: Span | None = None
    ) -> Span:
        """Start a ``tts`` span for one piece of speech.

        Args:
            model: The TTS model.
            provider: Who serves it.
            voice: The voice ID, when the plugin has one.
            parent: The span to nest under. The open turn, or the call, when None.
        """
        return self._start(
            TTS_SPAN,
            parent or self._turn,
            {
                OBSERVATION_TYPE: "generation",
                "gen_ai.operation.name": "tts",
                "gen_ai.output.type": "speech",
                "gen_ai.provider.name": provider.lower(),
                "gen_ai.request.model": model,
                UPSTREAM_PROVIDER: upstream_provider(provider, model, self.gateways),
                "tts.voice": voice,
            },
        )

    def end_tts(
        self,
        span: Span,
        *,
        text: str,
        ttfb: float | None = None,
        interrupted: bool = False,
        error: str | None = None,
    ) -> None:
        """End a ``tts`` span with the text spoken and its character count as usage.

        Args:
            span: The span ``start_tts`` returned.
            text: The text sent to TTS.
            ttfb: Seconds from the first text to the first audio.
            interrupted: Whether the speech stopped before its end.
            error: The error type, when TTS failed.
        """
        _end(
            span,
            {
                "input.value": text,
                "metrics.character_count": len(text),
                "langfuse.observation.usage_details": json.dumps({"characters": len(text)}),
                "metrics.ttfb": _seconds(ttfb),
                COMPLETION_START: _completion_start(span, _seconds(ttfb)),
                "tts.interrupted": interrupted,
            },
            error,
        )

    def event(
        self, name: str, data: Mapping[str, Any] | None = None, *, parent: Span | None = None
    ) -> None:
        """Record one decision, like ``task.started`` or ``handoff``, as a zero-length span.

        Langfuse ignores OpenTelemetry span events, so each decision is its own span. The
        signature fits ``TaskEventSink``: pass it to ``CallTasks.listen``.

        Args:
            name: The decision. It is the span name.
            data: What the decision was about. Stored as text.
            parent: The span to nest under. The open turn, or the call, when None.
        """
        now = time.time_ns()
        span = self._start(
            name,
            parent or self._turn,
            {OBSERVATION_TYPE: "event", "unmute.event.name": name, "input.value": _json(data)},
            now,
        )
        span.end(end_time=now)

    def start_tool_call(
        self, *, name: str, call_id: str, arguments: str = "", parent: Span | None = None
    ) -> Span:
        """Start an ``llm_tool_call`` span. Pass the turn as ``parent``.

        Args:
            name: The tool's name.
            call_id: The ID the model gave this call.
            arguments: The arguments as the JSON text the model wrote.
            parent: The span to nest under, usually the turn.
        """
        return self._start(
            TOOL_SPAN,
            parent,
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": name,
                "gen_ai.tool.call.id": call_id,
                "gen_ai.tool.call.arguments": arguments or None,
                "function.name": name,
                "tool_call_id": call_id,
                "function.arguments": arguments or None,
            },
        )

    def end_tool_call(
        self, span: Span, *, result: str | None = None, error: str | None = None
    ) -> None:
        """End a tool call.

        Args:
            span: The span ``start_tool_call`` returned.
            result: What the tool returned, as text.
            error: The error type, when the tool failed.
        """
        _end(
            span,
            {"gen_ai.tool.call.result": result, "tool.error": int(error is not None)},
            error,
        )

    def start_task_step(self, *, task: str, step: str, parent: Span | None = None) -> Span:
        """Start a ``task_step`` span, under the call. It can last several turns.

        Args:
            task: The task's name.
            step: The step's name inside the task.
            parent: The span to nest under. The call span when None.
        """
        return self._start(
            TASK_SPAN,
            parent,
            {
                "unmute.task.name": task,
                "unmute.task.step": step,
                "input.value": _json({"task": task, "step": step}),
            },
        )

    def end_task_step(
        self,
        span: Span,
        *,
        outcome: str,
        saved: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """End a task step.

        Args:
            span: The span ``start_task_step`` returned.
            outcome: How the step ended, like ``completed``.
            saved: The values the step saved.
            error: The error type, when the step failed.
        """
        _end(
            span,
            {
                "unmute.task.outcome": outcome,
                "output.value": _json({"outcome": outcome, "saved": dict(saved or {})}),
            },
            error,
        )

    def headers(self, span: Span) -> dict[str, str]:
        """Return the ``traceparent`` header that ties an outgoing request to ``span``."""
        carrier: dict[str, str] = {}
        TraceContextTextMapPropagator().inject(carrier, context=trace.set_span_in_context(span))
        return carrier

    async def close(
        self,
        timeout: float = CLOSE_TIMEOUT_SECONDS,
        *,
        state: CallState | None = None,
        reason: str | None = None,
    ) -> bool:
        """End the call span, send what is left to every sink, and shut the provider down.

        The call span's input is the transcript of every ended turn, one entry per line. Its
        output is how the call ended: the reason, the final values and the confirmed ones.

        Args:
            timeout: Seconds to wait for the sinks. The default is ten, which fits a LiveKit
                job's shutdown window and leaves Coval's conversation submit room to finish.
            state: The call state. Its final values and what was confirmed go on the call.
            reason: Why the call ended, like ``user_initiated``.

        Returns:
            False if the sinks did not finish in time.
        """
        if self._call is not None and self._call.is_recording():
            _end(
                self._call,
                {
                    "input.value": _json(self._transcript),
                    "output.value": _json(_ended(state, reason)),
                    "unmute.call.end_reason": reason,
                },
                None,
            )
        try:
            flushed = await asyncio.wait_for(
                asyncio.to_thread(self._flush_and_shut_down, int(timeout * 1000)), timeout
            )
        except TimeoutError:
            flushed = False
        if not flushed:
            logger.warning(
                "Telemetry did not flush within %.1f s. Some spans may be lost. Raise the timeout "
                "of close(), or check that every sink's endpoint is reachable.",
                timeout,
            )
        return flushed

    def _flush_and_shut_down(self, timeout_millis: int) -> bool:
        """Send the ended spans, then shut every sink down. Runs in a thread."""
        flushed = self.provider.force_flush(timeout_millis)
        self.provider.shutdown()
        return flushed
