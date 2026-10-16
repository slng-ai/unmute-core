"""Each sink builds the right OTLP exporter and gets the whole call.

Endpoints and headers match unmute@d0c67f1 internal/generate/templates/telemetry/telemetry.py.tmpl.
"""

import base64
import logging

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from tests.telemetry.conftest import SPANS, attrs, by_name, run_call
from unmute_core.telemetry import (
    Coval,
    CovalSettings,
    Langfuse,
    LangfuseSettings,
    Logfire,
    LogfireSettings,
    Otlp,
    Telemetry,
    TelemetryConfigError,
)
from unmute_core.telemetry._coval import MAX_HELD_SPANS

SPAN_NAMES = SPANS
LANGFUSE = LangfuseSettings(
    public_key="pk-lf-1", secret_key="sk-lf-2", base_url="https://cloud.langfuse.com/"
)


@pytest.mark.parametrize(
    ("sink", "endpoint", "headers"),
    [
        pytest.param(
            Otlp(endpoint="http://collector:4318/v1/traces", headers={"x-key": "k"}),
            "http://collector:4318/v1/traces",
            {"x-key": "k"},
            id="otlp",
        ),
        pytest.param(
            Langfuse.from_settings(LANGFUSE),
            "https://cloud.langfuse.com/api/public/otel/v1/traces",
            {
                "Authorization": "Basic " + base64.b64encode(b"pk-lf-1:sk-lf-2").decode(),
                "x-langfuse-ingestion-version": "4",
            },
            id="langfuse",
        ),
        pytest.param(
            Logfire.from_settings(LogfireSettings(token="pylf_v1_eu_abc")),
            "https://logfire-eu.pydantic.dev/v1/traces",
            {"Authorization": "pylf_v1_eu_abc"},
            id="logfire",
        ),
    ],
)
async def test_sink_exports_the_whole_call(exporters, sink, endpoint, headers):
    await run_call(Telemetry(sink, agent="salon", call_id="RM_abc"))

    (exporter,) = exporters
    assert exporter.settings["endpoint"] == endpoint
    assert exporter.settings["headers"] == headers
    spans = by_name(exporter.get_finished_spans())
    assert set(spans) == SPAN_NAMES
    assert attrs(spans["llm"])["gen_ai.request.model"] == "gpt-4.1"
    assert attrs(spans["turn"])["input.value"] == "My number is 555 0100"


async def test_otlp_without_endpoint_leaves_the_otel_env_to_the_sdk(exporters):
    Otlp().exporter()
    assert exporters[0].settings == {"endpoint": None, "headers": None}


@pytest.mark.parametrize(
    ("token", "region"),
    [("pylf_v1_us_abc", "us"), ("pylf_v1_eu_abc", "eu"), ("pylf_v2_xx_abc", "us"), ("old", "us")],
)
def test_logfire_region_comes_from_the_token(token, region):
    assert Logfire(token=token).endpoint == f"https://logfire-{region}.pydantic.dev/v1/traces"


def test_from_env_names_every_missing_variable(monkeypatch):
    for name in ("LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_HOST", "LOGFIRE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    with pytest.raises(TelemetryConfigError) as raised:
        Langfuse.from_env()
    assert str(raised.value) == (
        "Set LANGFUSE_SECRET_KEY in the environment. "
        "Set LANGFUSE_BASE_URL or LANGFUSE_HOST in the environment."
    )
    with pytest.raises(TelemetryConfigError, match="LOGFIRE_TOKEN"):
        Logfire.from_env()


def test_from_env_reads_the_environment(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-2")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    monkeypatch.setenv("LOGFIRE_TOKEN", "pylf_v1_eu_abc")

    assert Langfuse.from_env() == Langfuse.from_settings(
        LangfuseSettings(
            public_key="pk-lf-1", secret_key="sk-lf-2", base_url="https://us.cloud.langfuse.com"
        )
    )
    assert Logfire.from_env().token == "pylf_v1_eu_abc"


async def test_coval_holds_spans_until_the_simulation_id(exporters):
    coval = Coval.from_settings(CovalSettings(api_key="cv-key"))
    telemetry = Telemetry(coval, agent="salon", call_id="RM_abc")
    telemetry.start_call()
    telemetry.end_turn(telemetry.start_turn(1))
    telemetry.provider.force_flush()
    assert exporters == []

    coval.activate(simulation_id="sim-1")
    await run_call_rest(telemetry)

    (exporter,) = exporters
    assert exporter.settings["endpoint"] == "https://api.coval.dev/v1/traces"
    assert exporter.settings["headers"] == {"x-api-key": "cv-key", "X-Simulation-Id": "sim-1"}
    assert exporter.settings["timeout"] == 30
    assert [span.name for span in exporter.get_finished_spans()] == ["turn", "llm", "conversation"]


async def run_call_rest(telemetry: Telemetry) -> None:
    telemetry.end_model_request(telemetry.start_model_request(model="m", provider="p"))
    await telemetry.close()


async def test_coval_conversation_id_uses_its_own_header(exporters):
    coval = Coval(api_key="cv-key")
    telemetry = Telemetry(coval, agent="salon", call_id="RM_abc")
    coval.activate(conversation_id="conv-9")
    await run_call(telemetry)

    assert exporters[0].settings["headers"] == {
        "x-api-key": "cv-key",
        "X-Conversation-Id": "conv-9",
    }
    assert set(by_name(exporters[0].get_finished_spans())) == SPAN_NAMES


def test_coval_activate_takes_exactly_one_id():
    coval = Coval(api_key="cv-key")
    Telemetry(coval, agent="salon", call_id="RM_abc")
    with pytest.raises(ValueError, match="exactly one"):
        coval.activate()
    with pytest.raises(ValueError, match="exactly one"):
        coval.activate(simulation_id="s", conversation_id="c")


def test_coval_activate_needs_its_telemetry():
    with pytest.raises(RuntimeError, match="Telemetry"):
        Coval(api_key="cv-key").activate(simulation_id="s")


def test_coval_keeps_only_the_newest_spans_before_activation(exporters, caplog):
    coval = Coval(api_key="cv-key")
    held = coval.exporter()
    for _ in range(MAX_HELD_SPANS + 5):
        held.export([ReadableSpan(name="turn")])

    with caplog.at_level(logging.WARNING):
        coval.activate(simulation_id="sim-1")
    assert len(exporters[0].get_finished_spans()) == MAX_HELD_SPANS
    assert "Dropped 5 span(s)" in caplog.text


async def test_coval_without_a_key_warns_and_sends_nothing(exporters, caplog, monkeypatch):
    monkeypatch.delenv("COVAL_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        coval = Coval.from_env()
        telemetry = Telemetry(coval, agent="salon", call_id="RM_abc")
        coval.activate(simulation_id="sim-1")
        await run_call(telemetry)

    assert exporters == []
    assert "COVAL_API_KEY is not set" in caplog.text
