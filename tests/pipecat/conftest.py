"""A call's telemetry that keeps its spans in memory."""

from collections.abc import Callable

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from unmute_core.telemetry import Telemetry


@pytest.fixture
def spans() -> tuple[Telemetry, Callable[[], tuple[ReadableSpan, ...]]]:
    """A started call's telemetry, and a function that returns its finished spans."""
    exporter = InMemorySpanExporter()
    telemetry = Telemetry(
        agent="orders", call_id="call-1", processors=[SimpleSpanProcessor(exporter)]
    )
    telemetry.start_call()
    return telemetry, exporter.get_finished_spans
