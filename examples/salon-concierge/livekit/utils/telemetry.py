"""Generated telemetry for salon-concierge.

One OpenTelemetry provider per call, and the one backend it exports to.

`utils/tracing.py` decides which spans a call produces. This module decides
where those spans go and which attribute names they carry. So the call code
never names a backend, and a new backend is one class here.
"""

from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
)
from opentelemetry.util.types import AttributeValue

AGENT_NAME = "concierge-salon-concierge-livekit"

TRACE_NAME = AGENT_NAME

# OpenTelemetry's exporter interface takes its flush budget in milliseconds.
FLUSH_TIMEOUT_MS = 30000


class Backend(ABC):
    """Where a call's spans go, and the attribute names that backend reads.

    The defaults are the OpenTelemetry GenAI names, which any OTLP backend can
    read. A subclass overrides only the names its backend reads differently.
    """

    __slots__ = ()

    @classmethod
    @abstractmethod
    def from_env(cls) -> Self:
        """Build the backend from this process's environment."""

    @abstractmethod
    def exporter(self) -> SpanExporter:
        """The exporter every span of the call goes out through."""

    def processor(self) -> SpanProcessor:
        """How spans reach the exporter: batched, so a slow backend never stalls a call."""
        return BatchSpanProcessor(self.exporter())

    def call_attributes(self, session_id: str | None) -> dict[str, AttributeValue]:
        """The attributes that tie every span of one call together."""
        attributes: dict[str, AttributeValue] = {"agent.name": TRACE_NAME}
        if session_id:
            attributes["session.id"] = session_id
        return attributes

    def agent_run(self, agent_name: str) -> dict[str, AttributeValue]:
        """Attributes that make a span one agent run. None by default."""
        return {}

    def observation(
        self,
        *,
        input_value: str | None = None,
        output_value: str | None = None,
        kind: str | None = None,
    ) -> dict[str, AttributeValue]:
        """What a span took in and gave back, under this backend's names."""
        pairs = (("input.value", input_value), ("output.value", output_value))
        return {key: value for key, value in pairs if value is not None}

    def speech_metric(
        self, kind: str, key: str, value: AttributeValue
    ) -> dict[str, AttributeValue]:
        """Extra attributes one speech metric maps to. None by default."""
        return {}

    def tool_span(
        self, name: str, arguments: str, call_id: str | None
    ) -> tuple[str, dict[str, AttributeValue]]:
        """The name and opening attributes of one tool call's span."""
        attributes: dict[str, AttributeValue] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "gen_ai.tool.call.arguments": arguments,
        }
        if call_id:
            attributes["gen_ai.tool.call.id"] = call_id
        return f"execute_tool {name}", attributes

    def tool_result(self, result: str) -> dict[str, AttributeValue]:
        """What a tool call returned."""
        return {"gen_ai.tool.call.result": result}

    def tool_outcome(self, status: str) -> dict[str, AttributeValue]:
        """How a tool call ended: `completed`, `cancelled` or `error`."""
        return {} if status == "completed" else {"error.type": status}


@dataclass(frozen=True, slots=True)
class Langfuse(Backend):
    """Langfuse's OTLP endpoint, on its v4 observations-first ingestion path."""

    public_key: str
    secret_key: str
    base_url: str

    @classmethod
    def from_env(cls) -> Self:
        """Read the three Langfuse values. All three are required."""
        values = [
            os.getenv(name, "")
            for name in (
                "LANGFUSE_PUBLIC_KEY",
                "LANGFUSE_SECRET_KEY",
                "LANGFUSE_BASE_URL",
            )
        ]
        if not all(values):
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL must all be set"
            )
        return cls(*values)

    def exporter(self) -> SpanExporter:
        """Plain OTLP with basic auth.

        The ingestion header is what routes an export to the v4 data model.
        Without it the spans still arrive, on the legacy path, and nothing
        fails or logs, so it is set here once for both targets.
        """
        auth = base64.b64encode(
            f"{self.public_key}:{self.secret_key}".encode()
        ).decode()
        return OTLPSpanExporter(
            endpoint=f"{self.base_url.rstrip('/')}/api/public/otel/v1/traces",
            headers={
                "Authorization": f"Basic {auth}",
                "x-langfuse-ingestion-version": "4",
            },
        )

    def call_attributes(self, session_id: str | None) -> dict[str, AttributeValue]:
        """Langfuse's own session and trace names, which its views group by."""
        attributes: dict[str, AttributeValue] = {"langfuse.trace.name": TRACE_NAME}
        if session_id:
            attributes["langfuse.session.id"] = session_id
        return attributes

    def observation(
        self,
        *,
        input_value: str | None = None,
        output_value: str | None = None,
        kind: str | None = None,
    ) -> dict[str, AttributeValue]:
        """Langfuse reads input and output off the observation, not the trace."""
        pairs = (
            ("langfuse.observation.type", kind),
            ("langfuse.observation.input", input_value),
            ("langfuse.observation.output", output_value),
        )
        return {key: value for key, value in pairs if value is not None}

    def speech_metric(
        self, kind: str, key: str, value: AttributeValue
    ) -> dict[str, AttributeValue]:
        """Map Pipecat's speech metrics onto the fields Langfuse charts."""
        if key == "metrics.ttfb":
            return {
                "langfuse.observation.metadata.ttfb_seconds": value,
                "langfuse.observation.completion_start_time": datetime.now(
                    UTC
                ).isoformat(),
            }
        if kind == "tts" and key == "metrics.character_count":
            return {
                "langfuse.observation.metadata.character_count": value,
                "langfuse.observation.usage_details": json.dumps({"characters": value}),
            }
        return {}

    def tool_span(
        self, name: str, arguments: str, call_id: str | None
    ) -> tuple[str, dict[str, AttributeValue]]:
        """A tool observation, named after the tool.

        Langfuse makes a span with these two GenAI keys a TOOL and names it from
        gen_ai.tool.name, whatever the span is called. livekit-agents sets the
        same pair, so one call reads the same way on both targets.
        """
        attributes: dict[str, AttributeValue] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "tool.function_name": name,
            "langfuse.observation.input": arguments,
        }
        if call_id:
            attributes["tool.call_id"] = call_id
        return name, attributes

    def tool_result(self, result: str) -> dict[str, AttributeValue]:
        """The tool's result is the observation's output."""
        return {"langfuse.observation.output": result}

    def tool_outcome(self, status: str) -> dict[str, AttributeValue]:
        """Every outcome is recorded, including a clean one."""
        return {"tool.result_status": status}


BACKEND: type[Backend] = Langfuse


class _Filtered(SpanProcessor):
    """Hand the backend every ended span except the ones `keep` refuses."""

    def __init__(
        self, inner: SpanProcessor, keep: Callable[[ReadableSpan], bool]
    ) -> None:
        """Wrap one processor with one export rule."""
        self._inner, self._keep = inner, keep

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Pass every span on as it starts."""
        self._inner.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        """Pass an ended span on only if it is kept."""
        if self._keep(span):
            self._inner.on_end(span)

    def shutdown(self) -> None:
        """Shut the wrapped processor down."""
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = FLUSH_TIMEOUT_MS) -> bool:
        """Flush the wrapped processor."""
        return self._inner.force_flush(timeout_millis)


class Telemetry:
    """The one tracer provider a call's spans go through, and its backend.

    Every target builds one of these and adds its own span processors to it.
    Nothing outside this module builds a provider or an exporter.
    """

    __slots__ = ("backend", "provider")

    def __init__(
        self,
        backend: Backend | None = None,
        *,
        resource: Mapping[str, AttributeValue] | None = None,
        keep: Callable[[ReadableSpan], bool] | None = None,
    ) -> None:
        """Build the provider and attach the backend's exporter to it.

        `backend` defaults to the one this package names in agent.yaml, read
        from the environment. `keep`, when given, decides which ended spans
        reach the backend at all.
        """
        self.backend = backend if backend is not None else BACKEND.from_env()
        # Without a service name every call lands as unknown_service.
        self.provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: TRACE_NAME, **(resource or {})})
        )
        processor = self.backend.processor()
        self.add(_Filtered(processor, keep) if keep else processor)

    @classmethod
    def install(
        cls,
        backend: Backend | None = None,
        *,
        resource: Mapping[str, AttributeValue] | None = None,
    ) -> Telemetry:
        """Build telemetry and set it as OpenTelemetry's global provider.

        For a framework that reads the global provider rather than being handed
        one. OpenTelemetry refuses a second global, so this fails loudly instead
        of leaving the call traced to a provider nobody exports from.
        """
        if isinstance(trace.get_tracer_provider(), TracerProvider):
            raise RuntimeError(
                "OpenTelemetry already has a TracerProvider; set up unmute telemetry first"
            )
        telemetry = cls(backend, resource=resource)
        trace.set_tracer_provider(telemetry.provider)
        return telemetry

    def tracer(self, name: str = "unmute") -> trace.Tracer:
        """A tracer on this provider."""
        return self.provider.get_tracer(name)

    def add(self, processor: SpanProcessor) -> None:
        """Run one more processor on every span."""
        self.provider.add_span_processor(processor)

    def flush(self) -> bool:
        """Send every span that has ended and not gone out yet."""
        return self.provider.force_flush()
