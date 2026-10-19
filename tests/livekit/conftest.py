"""Skip the LiveKit tests when the ``livekit`` extra is not installed."""

from collections.abc import Iterator

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    """Record every span LiveKit finishes, for one test."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    yield exporter
    set_tracer_provider(TracerProvider())
