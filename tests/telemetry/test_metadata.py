"""MetadataOnly lets no text out: names, attributes, resource, events and status are filtered.

The first test checks the resource, events, names and status, using the library's own
attribute keys.
"""

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from tests.telemetry.conftest import SPANS, attrs, by_name, run_call
from unmute_core.telemetry import DEFAULT_ATTRIBUTES, MetadataOnly, Otlp, Telemetry

TEXT_KEYS = {
    "input.value",
    "output.value",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "function.arguments",
    "langfuse.trace.name",
    "gen_ai.tool.definitions",
}


def test_export_is_metadata_only_including_resource_events_names_and_status():
    exported = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource({"service.name": "salon", "host.name": "private-host"})
    )
    provider.add_span_processor(SimpleSpanProcessor(MetadataOnly().wrap(exported)))
    span = provider.get_tracer("test").start_span(
        "custom name with the caller's words",
        attributes={
            "session.id": "RM_abc",
            "turn.number": 3,
            "input.value": "my card is 4111",
            "gen_ai.input.messages": "[...]",
            "gen_ai.request.model": "has spaces, so not a safe token",
        },
    )
    span.add_event("transcript", {"text": "my card is 4111"})
    span.set_status(Status(StatusCode.ERROR, "card 4111 declined"))
    span.end()

    (safe,) = exported.get_finished_spans()
    assert safe.name == "operation"
    assert attrs(safe) == {"session.id": "RM_abc", "turn.number": 3}
    assert dict(safe.resource.attributes) == {"service.name": "salon"}
    assert safe.events == ()
    assert safe.status.status_code is StatusCode.ERROR
    assert safe.status.description is None


def test_extra_keys_extend_the_defaults():
    exported = InMemorySpanExporter()
    provider = TracerProvider()
    rules = MetadataOnly(attributes=DEFAULT_ATTRIBUTES | {"app.call_id"})
    provider.add_span_processor(SimpleSpanProcessor(rules.wrap(exported)))
    provider.get_tracer("test").start_span("turn", attributes={"app.call_id": "c1"}).end()

    assert attrs(exported.get_finished_spans()[0])["app.call_id"] == "c1"


async def test_a_filtered_sink_sends_no_text_from_a_real_call(exporters):
    await run_call(
        Telemetry(
            Otlp(endpoint="http://collector:4318/v1/traces", filter=MetadataOnly()),
            agent="salon",
            call_id="RM_abc",
        )
    )

    spans = by_name(exporters[0].get_finished_spans())
    assert set(spans) == (SPANS - {"task.started"}) | {"operation"}
    for span in spans.values():
        assert not TEXT_KEYS & set(attrs(span))
    assert attrs(spans["llm"])["gen_ai.usage.output_tokens"] == 4
    assert attrs(spans["llm_tool_call"])["function.name"] == "find_slots"
    assert attrs(spans["turn"])["turn.e2e_seconds"] == 1.1


def test_apply_all_builds_one_resource_for_the_spans_that_share_it():
    resource = Resource({"service.name": "salon"})
    spans = [ReadableSpan(name="turn", resource=resource) for _ in range(3)]

    filtered = MetadataOnly().apply_all(spans)

    assert len({id(span.resource) for span in filtered}) == 1
    assert filtered[0].resource.attributes["service.name"] == "salon"
