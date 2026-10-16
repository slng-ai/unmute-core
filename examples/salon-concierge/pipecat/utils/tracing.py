"""Generated tracing for salon-concierge: Pipecat's own spans, one trace per call.

Pipecat already nests `conversation` over `turn` over its service spans, so
nothing here re-parents anything. This module stamps each call's attributes on
every span, adds the tool spans Pipecat does not emit. Where the spans go, and under which
attribute names, is utils/telemetry.py.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span, SpanProcessor
from opentelemetry.util.types import AttributeValue
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.llm import LLMWorker

from utils.telemetry import (
    TRACE_NAME,
    Backend,
    Telemetry,
)

# pipecat's two structural spans: `conversation` wraps the call and `turn` wraps
# one exchange, the caller's line and the reply to it. A name that is not here
# still traces, it just does not get read back as the call or as a turn.
CALL_SPAN = "conversation"
TURN_SPAN = "turn"

# The process's one telemetry. Pipecat reads OpenTelemetry's global provider,
# which can be set once, so the first call builds it and later calls reuse it.
_TELEMETRY: Telemetry | None = None


class CallTrace(SpanProcessor):
    """One trace for the whole call, with pipecat's turn spans inside it.

    Two jobs, in one class because they are the same bookkeeping.

    It copies the call's correlating attributes onto every span, because a
    backend that filters and sums over spans, as Langfuse v4 does, cannot group
    the spans underneath an attribute that sits only on the top one. pipecat's
    own additional_span_attributes reach the conversation span alone, so they
    do not do this.

    And it records the conversation twice over: each line on the turn it
    belongs to, and the whole thing on the call's root, so opening the trace
    shows what was said rather than an empty envelope.

    The processor is one per process, because the tracer provider is, but the
    bookkeeping is one per call. A process can run several calls at once (the
    runner's Twilio route took three concurrent Coval calls on 2026-09-24), and
    a single shared record filed all three under the last call's session. Each
    call's record lives in a ContextVar set at the top of that call's coroutine,
    so every task its pipeline starts inherits it and sees only its own call.
    """

    def start_call(
        self, attributes: Mapping[str, AttributeValue], telemetry: Telemetry
    ) -> None:
        """Point tracing at this call, from inside the call's own coroutine."""
        _CURRENT_CALL.set(_Call(attributes, telemetry))

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Copy the current call's attributes onto a span as it opens."""
        call = _CURRENT_CALL.get()
        if call is not None:
            call.on_start(span)

    def said(self, role: str, text: str) -> None:
        """Record one line, on its own turn and on the call as a whole."""
        call = _CURRENT_CALL.get()
        if call is not None:
            call.said(role, text)


class _Call:
    """What one call's spans share: its attributes, its root, its current turn."""

    def __init__(
        self, attributes: Mapping[str, AttributeValue], telemetry: Telemetry
    ) -> None:
        self._attributes: dict[str, AttributeValue] = dict(attributes)
        self._backend = telemetry.backend
        self._call: Span | None = None
        self._turn: Span | None = None
        self._transcript: list[dict[str, str]] = []

    def on_start(self, span: Span) -> None:
        """Stamp the call's attributes on a span and remember the call and turn roots."""
        span.set_attributes(self._attributes)
        if span.name == CALL_SPAN:
            self._call = span
        elif span.name == TURN_SPAN:
            self._turn = span
            # Pipecat opens no agent span, and its `llm` spans sit right under
            # `turn`, so a turn is the agent run for a backend that groups by
            # one. ponytail: named after the package, not the worker that
            # answered, because the shared tracing context does not say which
            # worker spoke. Stamp the worker's name once each owns a context.
            span.set_attributes(self._backend.agent_run(TRACE_NAME))

    def said(self, role: str, text: str) -> None:
        """Record one line on the current turn and on the call's root."""
        backend = self._backend
        self._transcript.append({"role": role, "content": text})
        if self._turn is not None and self._turn.is_recording():
            line = json.dumps(text)
            self._turn.set_attributes(
                backend.observation(input_value=line)
                if role == "user"
                else backend.observation(output_value=line)
            )
        if self._call is not None and self._call.is_recording():
            # The whole conversation goes on the root, because that is the one
            # place a reader opens to see the call rather than one turn of it.
            self._call.set_attributes(
                backend.observation(
                    input_value=json.dumps(self._transcript),
                    output_value=None if role == "user" else json.dumps(text),
                )
            )


_CURRENT_CALL: ContextVar[_Call | None] = ContextVar("unmute_call", default=None)


CALL = CallTrace()


def _patch_pipecat_tracing(backend: Backend) -> None:
    """Record what Pipecat's speech and LLM spans heard and said, under the backend's names."""
    from pipecat.utils.tracing import service_decorators

    if getattr(service_decorators.add_stt_span_attributes, "__unmute_patch__", False):
        return

    def patch(original: Callable[..., Any], kind: str) -> Any:
        """Wrap one Pipecat span-attribute function so it also writes the backend's keys."""

        def patched(span: Span, *args: Any, **kwargs: Any) -> None:
            set_attribute = span.set_attribute
            # The SDK's own, which writes straight to the span and so never
            # comes back through the wrapper below.
            set_attributes = span.set_attributes

            # ponytail: Pipecat 1.9.0 emits speech data under generic keys that
            # no backend maps to input and output. Remove when Pipecat does.
            def mapped_set_attribute(key: str, value: Any) -> None:
                set_attribute(key, value)
                encoded = json.dumps(value, default=str)
                if kind == "stt" and key == "transcript":
                    set_attributes(backend.observation(output_value=encoded))
                    CALL.said("user", value)
                elif kind == "tts" and key == "text":
                    set_attributes(backend.observation(input_value=encoded))
                    CALL.said("assistant", value)
                else:
                    set_attributes(backend.speech_metric(kind, key, value))

            span.set_attribute = mapped_set_attribute  # ty: ignore[invalid-assignment]
            original(span, *args, **kwargs)
            audio = json.dumps("audio")
            set_attributes(
                backend.observation(input_value=audio)
                if kind == "stt"
                else backend.observation(output_value=audio)
            )

        # Plain assignment fails type checking: a function declares no attributes.
        setattr(patched, "__unmute_patch__", True)  # noqa: B010
        return patched

    service_decorators.add_stt_span_attributes = patch(
        service_decorators.add_stt_span_attributes, "stt"
    )
    service_decorators.add_tts_span_attributes = patch(
        service_decorators.add_tts_span_attributes, "tts"
    )

    original_add_llm_span_attributes = service_decorators.add_llm_span_attributes

    def patched_llm(span: Span, *args: Any, **kwargs: Any) -> None:
        original_add_llm_span_attributes(span, *args, **kwargs)
        messages = json.loads(kwargs.get("messages") or "[]")
        system_instruction = kwargs.get("system_instructions")
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})
        if messages:
            encoded = json.dumps(messages, default=str)
            span.set_attribute("input", encoded)
            span.set_attributes(backend.observation(input_value=encoded))

    setattr(patched_llm, "__unmute_patch__", True)  # noqa: B010 - a function declares no attributes
    # setattr: the replacement has a looser signature than the function it patches.
    setattr(service_decorators, "add_llm_span_attributes", patched_llm)  # noqa: B010


def setup_tracing() -> Telemetry:
    """The process's telemetry, built by the first call and shared by every later one."""
    global _TELEMETRY
    if _TELEMETRY is None:
        telemetry = Telemetry.install(resource=_resource())
        _patch_pipecat_tracing(telemetry.backend)
        telemetry.add(CALL)
        _TELEMETRY = telemetry
    return _TELEMETRY


def _resource() -> dict[str, AttributeValue]:
    """The process attributes Pipecat's own setup_tracing used to set."""
    return {
        "service.instance.id": os.getenv("HOSTNAME", "unknown"),
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    }


def start_call(telemetry: Telemetry, runner_args: Any) -> dict[str, AttributeValue]:
    """Begin tracing one call, and return the attributes for its conversation span."""
    attributes = telemetry.backend.call_attributes(
        getattr(runner_args, "session_id", None)
    )
    # Pipecat puts its attributes on the conversation span only. They have to
    # reach every span under it too, and this is also where the container
    # forgets the call it ran before.
    CALL.start_call(attributes, telemetry)
    return attributes


def enable_agent_tracing(main: PipelineWorker, agents: Sequence[LLMWorker]) -> None:
    """Point every agent worker at the main worker's tracing context."""
    # ponytail: Pipecat 1.9.0 LLMWorker exposes no tracing kwargs. Share the main
    # worker's context until its public constructor accepts them.
    for agent in agents:
        agent._enable_tracing = True
        agent._tracing_context = main._tracing_context


def flush_tracing(telemetry: Telemetry, session_id: str = "") -> None:
    """Send this call's trace."""
    # session_id is Coval's: it files a call no simulation claimed by it. Every
    # other backend already has the session on the spans.
    telemetry.flush()


# ponytail: Pipecat 1.9.0 has no standard-tool tracing hook. Remove this
# subclass when ordinary LLMWorker tools gain native spans.
class TracedLLMWorker(LLMWorker):
    """Add the tool spans missing from Pipecat's standard LLM tracing."""

    async def _trace_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        call: Callable[[trace.Span | None], Awaitable[Any]],
        tool_call_id: str | None = None,
    ) -> Any:
        """Run one tool call inside its own span, or bare when tracing is off."""
        context = self._tracing_context
        telemetry = _TELEMETRY
        if not self._enable_tracing or context is None or telemetry is None:
            return await call(None)
        backend = telemetry.backend
        parent = context.get_turn_context() or context.get_conversation_context()
        span_name, attributes = backend.tool_span(
            name, json.dumps(arguments, default=str), tool_call_id
        )
        with telemetry.tracer("unmute.tools").start_as_current_span(
            span_name, context=parent, attributes=attributes
        ) as span:
            try:
                result = await call(span)
            except asyncio.CancelledError:
                span.set_attributes(backend.tool_outcome("cancelled"))
                raise
            except Exception:
                span.set_attributes(backend.tool_outcome("error"))
                raise
            if result is not None:
                span.set_attributes(
                    backend.tool_result(json.dumps(result, default=str))
                )
            span.set_attributes(backend.tool_outcome("completed"))
            return result

    def _track_tool_call(self, method: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap Pipecat's tool tracker so each call gets a span."""
        tracked = super()._track_tool_call(method)

        @functools.wraps(tracked)
        async def wrapper(params: Any, *args: Any, **kwargs: Any) -> Any:
            async def call(span: trace.Span | None) -> Any:
                if span is not None and _TELEMETRY is not None:
                    backend = _TELEMETRY.backend
                    result_callback = params.result_callback

                    async def traced_result_callback(
                        result: Any, **callback_kwargs: Any
                    ) -> Any:
                        span.set_attributes(
                            backend.tool_result(json.dumps(result, default=str))
                        )
                        return await result_callback(result, **callback_kwargs)

                    params.result_callback = traced_result_callback
                return await tracked(params, *args, **kwargs)

            return await self._trace_tool(
                params.function_name,
                dict(params.arguments),
                call,
                tool_call_id=params.tool_call_id,
            )

        return wrapper

    def _trace_flow_tool(
        self, name: str, method: Callable[..., Awaitable[Any]]
    ) -> Callable[..., Any]:
        """Wrap one Pipecat Flows tool function so it gets a span."""

        @functools.wraps(method)
        async def wrapper(args: Mapping[str, Any], flow_manager: Any) -> Any:
            async def call(_span: trace.Span | None) -> Any:
                return await method(args, flow_manager)

            return await self._trace_tool(name, dict(args), call)

        return wrapper
