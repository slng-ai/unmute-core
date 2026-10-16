"""A filter that lets only metadata leave the process: no text, no events, no unknown names.

Callers add their own attribute names with
``MetadataOnly(attributes=DEFAULT_ATTRIBUTES | {...})``.
"""

import re
from collections.abc import Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Status
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.telemetry._shape import MappingExporter

SAFE_TOKEN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,254}")
RENAMED_SPAN = "operation"

DEFAULT_SPAN_NAMES = frozenset(
    {"conversation", "turn", "llm", "llm_tool_call", "task_step", "stt", "tts"}
)
DEFAULT_ATTRIBUTES = frozenset(
    {
        "session.id",
        "agent.name",
        "gen_ai.conversation.id",
        "gen_ai.operation.name",
        "gen_ai.agent.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
        "function.name",
        "tool_call_id",
        "tool.error",
        "error.type",
        "metrics.ttfb",
        "turn.number",
        "turn.interrupted",
        "turn.e2e_seconds",
        "turn.stt_final_seconds",
        "turn.eou_seconds",
        "turn.llm_ttft_seconds",
        "turn.tts_ttfb_seconds",
        "unmute.prompt.site",
        "unmute.upstream.provider",
        "unmute.task.name",
        "unmute.task.step",
        "unmute.task.outcome",
        "unmute.event.name",
        "unmute.call.end_reason",
        "unmute.call.origin",
        "unmute.call.framework",
        "langfuse.observation.type",
        "metrics.character_count",
        "stt.language",
        "tts.interrupted",
        "llm.interrupted",
        "langfuse.observation.level",
    }
)
DEFAULT_RESOURCE = frozenset(
    {
        "service.name",
        "service.instance.id",
        "service.version",
        "deployment.environment.name",
        "cloud.region",
    }
)


def _safe(value: object) -> bool:
    """Return True for a number, a bool, or a short string with no spaces or punctuation."""
    return isinstance(value, bool | int | float) or (
        isinstance(value, str) and SAFE_TOKEN.fullmatch(value) is not None
    )


def _token(value: str | None) -> str:
    """Return the value if it is a safe string, otherwise ``unknown``."""
    return value if value is not None and SAFE_TOKEN.fullmatch(value) else "unknown"


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class MetadataOnly:
    """Which span names, attributes and resource keys may leave the process.

    Any sink takes one as its ``filter``. A kept attribute must be on the list and hold a
    number, a bool or a safe string. Text like a transcript is never on the list.

    Attributes:
        attributes: Span attribute keys to keep.
        span_names: Span names to keep. Any other name becomes ``operation``.
        resource: Resource attribute keys to keep.
    """

    attributes: frozenset[str] = DEFAULT_ATTRIBUTES
    span_names: frozenset[str] = DEFAULT_SPAN_NAMES
    resource: frozenset[str] = DEFAULT_RESOURCE

    def wrap(self, exporter: SpanExporter) -> SpanExporter:
        """Return an exporter that filters each span, then hands it to ``exporter``."""
        return _MetadataOnlyExporter(self, exporter)

    def apply(self, span: ReadableSpan) -> ReadableSpan:
        """Return a copy of the span with only metadata left."""
        return self.apply_all([span])[0]

    def apply_all(self, spans: Sequence[ReadableSpan]) -> list[ReadableSpan]:
        """Return a copy of each span with only metadata left.

        The filtered resource and instrumentation scope are built once for each one the
        spans share, and not again for every span.
        """
        resources: dict[int, Resource] = {}
        scopes: dict[int, InstrumentationScope] = {}
        return [self._filtered(span, resources, scopes) for span in spans]

    def _filtered(
        self,
        span: ReadableSpan,
        resources: dict[int, Resource],
        scopes: dict[int, InstrumentationScope],
    ) -> ReadableSpan:
        """Return one span with only metadata left, reusing the resource and scope built before."""
        if id(span.resource) not in resources:
            resources[id(span.resource)] = Resource(
                {
                    key: value
                    for key, value in span.resource.attributes.items()
                    if key in self.resource and isinstance(value, str) and _safe(value)
                }
            )
        if id(span.instrumentation_scope) not in scopes:
            scope = span.instrumentation_scope
            scopes[id(scope)] = InstrumentationScope(
                name=_token(scope.name if scope else None),
                version=_token(scope.version if scope else None),
            )
        return ReadableSpan(
            name=span.name if span.name in self.span_names else RENAMED_SPAN,
            context=span.context,
            parent=span.parent,
            resource=resources[id(span.resource)],
            attributes={
                key: value
                for key, value in (span.attributes or {}).items()
                if key in self.attributes and _safe(value)
            },
            events=(),
            links=tuple(trace.Link(link.context) for link in span.links),
            kind=span.kind,
            status=Status(span.status.status_code),
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=scopes[id(span.instrumentation_scope)],
        )


class _MetadataOnlyExporter(MappingExporter):
    """Filters every span with a ``MetadataOnly`` before the real exporter sees it."""

    def __init__(self, rules: MetadataOnly, delegate: SpanExporter) -> None:
        """Keep the rules and the exporter that sends the filtered spans."""
        super().__init__(delegate)
        self._rules = rules

    def shape(self, span: ReadableSpan) -> ReadableSpan:
        """Return the span with only metadata left."""
        return self._rules.apply(span)

    def shape_all(self, spans: Sequence[ReadableSpan]) -> list[ReadableSpan]:
        """Return the spans with only metadata left, sharing what they have in common."""
        return self._rules.apply_all(spans)
