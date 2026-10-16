"""``correlate_coval`` finds the Coval simulation on a Pipecat call and the caller's origin."""

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from unmute_core.pipecat import correlate_coval
from unmute_core.telemetry import Coval, Sink, Telemetry


@dataclass(kw_only=True)
class Memory(Sink):
    """Keeps the spans in memory."""

    spans: InMemorySpanExporter = field(default_factory=InMemorySpanExporter)

    def exporter(self) -> SpanExporter:
        return self.spans


@dataclass
class Call:
    """A Coval sink that records its activations, and the telemetry of one call."""

    coval: Coval = field(default_factory=lambda: Coval(api_key="cv-key"))
    sink: Memory = field(default_factory=Memory)
    activations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.telemetry = Telemetry(self.sink, agent="salon", call_id="s1")
        cast(Any, self.coval).activate = lambda **kwargs: self.activations.append(kwargs)

    def correlate(self, **runner_args: Any) -> None:
        correlate_coval(cast(Any, SimpleNamespace(**runner_args)), self.coval, self.telemetry)

    async def origin(self) -> object:
        """Return the origin on the call span, ending the call to read it."""
        self.telemetry.start_call()
        await self.telemetry.close()
        (call,) = self.sink.spans.get_finished_spans()
        return dict(call.attributes or {}).get("unmute.call.origin")


KEY = "x-coval-simulation-id"


def test_websocket_headers_come_first() -> None:
    call = Call()
    call.coval.simulation_id = "from-env"
    call.correlate(
        websocket=SimpleNamespace(headers={"X-Coval-Simulation-Id": "from-header"}),
        body={"dialin_settings": {"sip_headers": {KEY: "from-sip"}}, KEY: "from-body"},
    )
    assert call.activations == [{"simulation_id": "from-header", "method": "websocket_header"}]


def test_sip_headers_come_before_the_body() -> None:
    call = Call()
    call.correlate(body={"dialin_settings": {"sip_headers": {KEY: "from-sip"}}, KEY: "from-body"})
    assert call.activations == [{"simulation_id": "from-sip", "method": "sip_header"}]


def test_the_body_comes_before_call_data() -> None:
    call = Call()
    call.correlate(
        body={"coval.simulation_id": "from-body"},
        call_data=SimpleNamespace(body={KEY: "from-carrier"}),
    )
    assert call.activations == [{"simulation_id": "from-body", "method": "call_body"}]


@pytest.mark.parametrize(
    "call_data", [SimpleNamespace(body={KEY: "from-carrier"}), {"body": {KEY: "from-carrier"}}]
)
def test_call_data_body_comes_before_the_environment(call_data: Any) -> None:
    call = Call()
    call.coval.simulation_id = "from-env"
    call.correlate(call_data=call_data)
    assert call.activations == [{"simulation_id": "from-carrier", "method": "carrier_parameter"}]


def test_the_environment_is_the_last_source() -> None:
    call = Call()
    call.coval.simulation_id = "from-env"
    call.correlate(body={})
    assert call.activations == [{"simulation_id": "from-env", "method": "environment"}]


def test_a_call_no_simulation_placed_says_so_and_activates_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    call = Call()
    with caplog.at_level(logging.INFO):
        call.correlate(room_url="https://x.daily.co/r")
    assert call.activations == []
    assert "No Coval simulation placed this browser call" in caplog.text


def test_an_id_without_a_key_warns(caplog: pytest.LogCaptureFixture) -> None:
    call = Call(coval=Coval(api_key=None))
    with caplog.at_level(logging.WARNING):
        call.correlate(body={KEY: "sim"})
    assert call.activations == []
    assert "COVAL_API_KEY is unset" in caplog.text


@pytest.mark.parametrize(
    ("runner_args", "origin"),
    [
        ({"body": {"dialin_settings": {"call_id": "c"}}}, "phone"),
        ({"websocket": object()}, "websocket"),
        ({"websocket": object(), "transport_type": "websocket"}, "websocket"),
        ({"websocket": object(), "transport_type": "twilio"}, "phone"),
        ({"webrtc_connection": object()}, "browser"),
        ({"room_url": "https://x.daily.co/r"}, "browser"),
        ({}, None),
    ],
)
async def test_origin_is_on_the_call_span(runner_args: dict[str, Any], origin: str | None) -> None:
    call = Call()
    call.correlate(**runner_args)
    assert await call.origin() == origin
