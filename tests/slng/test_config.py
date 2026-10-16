"""``RouterConfig`` checks a config by the router's rules and sends it with keys in clear."""

import base64
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import Secret, ValidationError

from unmute_core.slng import (
    CONFIG_MAX_BYTES,
    AzureEndpoint,
    BedrockEndpoint,
    OpenAIEndpoint,
    RouterConfig,
    TierEntry,
    VertexEndpoint,
    read_vertex_key,
)

KEY = {"type": "service_account", "private_key": "PRIVATE", "client_email": "bot@x.test"}
SECRET = "sk-very-secret"


def entry(model: str, weight: int, priority: int = 0, **endpoint: Any) -> dict[str, Any]:
    """One tier entry as JSON data, with an endpoint when given."""
    return {"model": model, "weight": weight, "priority": priority} | (
        {"endpoint": endpoint} if endpoint else {}
    )


def refused(config: dict[str, Any]) -> str:
    """Return the error a config raises, checking that it never shows a key."""
    with pytest.raises(ValidationError) as raised:
        RouterConfig.model_validate(config)
    text = str(raised.value)
    assert SECRET not in text
    assert "PRIVATE" not in text
    return text


def test_single_sends_unmutes_shape() -> None:
    config = RouterConfig.single(
        "gpt-4.1-mini", OpenAIEndpoint(url="https://api.openai.com/v1", api_key=SECRET)
    )
    assert config.wire() == {
        "tiers": {
            "1": [
                {
                    "model": "gpt-4.1-mini",
                    "weight": 100,
                    "priority": 0,
                    "endpoint": {
                        "provider": "openai-compat",
                        "url": "https://api.openai.com/v1",
                        "api_key": SECRET,
                    },
                }
            ]
        }
    }
    assert config.models == {"gpt-4.1-mini"}
    assert SECRET not in repr(config)


def test_a_missing_provider_means_openai_compat_and_each_provider_is_its_own_class() -> None:
    config = RouterConfig.model_validate(
        {
            "tiers": {
                "1": [
                    entry("a", 50, url="https://a.test/v1", api_key=SECRET),
                    entry(
                        "b",
                        50,
                        provider="azure",
                        url="https://r.openai.azure.com",
                        api_key=SECRET,
                        azure_deployment="d",
                        api_version="2024-10-21",
                    ),
                ],
                "2": [
                    entry(
                        "c",
                        100,
                        provider="bedrock",
                        aws_access_key_id="AKIA",
                        aws_secret_access_key=SECRET,
                        aws_region="eu-central-1",
                        model_id="anthropic.x",
                    ),
                    entry(
                        "d", 100, 1, provider="vertex", vertex_credentials=KEY, vertex_location="eu"
                    ),
                ],
                "3": [entry("e", 100)],
            },
            "default_tier": 1,
        }
    )
    tiers = config.tiers or {}
    kinds = [type(item.endpoint) for items in tiers.values() for item in items]
    assert kinds == [OpenAIEndpoint, AzureEndpoint, BedrockEndpoint, VertexEndpoint, type(None)]
    wire = config.wire()
    assert wire["default_tier"] == 1
    assert wire["tiers"]["2"][1]["endpoint"]["vertex_credentials"] == KEY
    assert wire["tiers"]["2"][0]["endpoint"]["aws_secret_access_key"] == SECRET
    assert "cache_enabled" not in wire


def test_unset_fields_stay_out_so_the_stored_config_keeps_them() -> None:
    assert RouterConfig(cache_enabled=False).wire() == {"cache_enabled": False}
    assert RouterConfig().wire() == {}


def test_the_router_switches_appear_in_the_wire_config_when_set() -> None:
    config = RouterConfig(
        template_vars_strict=True,
        disable_reasoning=False,
        cache_skip_numeric_enabled=True,
        pii_detection_enabled=True,
        pii_regex_packs=("global", "us"),
        pii_ner_models=("gliner_pii_en",),
        slng_flow={"salon": {"steps": []}},
    )
    assert config.wire() == {
        "template_vars_strict": True,
        "disable_reasoning": False,
        "cache_skip_numeric_enabled": True,
        "pii_detection_enabled": True,
        "pii_regex_packs": ["global", "us"],
        "pii_ner_models": ["gliner_pii_en"],
        "slng_flow": {"salon": {"steps": []}},
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"tiers": {"1": [entry("a", 60), entry("b", 30)]}}, "weights sum to 90, not 100"),
        ({"tiers": {"1": [entry("a", 100, 1)]}}, "tier 1 needs an entry with priority 0"),
        ({"tiers": {"1": [entry("a", 50), entry("a", 50)]}}, "names model 'a' twice"),
        ({"tiers": {"4": [entry("a", 100)]}}, "'1', '2' or '3'"),
        ({"pii_regex_packs": ["mars"]}, "global"),
        ({"slng_flow": {"bad id": {}}}, "slng_flow must not contain a comma, whitespace"),
        (
            {
                "tiers": {
                    "1": [
                        entry(
                            "a",
                            100,
                            url="https://a.test",
                            api_key=SECRET,
                            auth_header="Authorization",
                        )
                    ]
                }
            },
            "must not be authorization",
        ),
        (
            {
                "tiers": {
                    "1": [
                        entry("a", 100, url="https://a.test", api_key=SECRET, auth_header="X Key")
                    ]
                }
            },
            "auth_header",
        ),
        (
            {
                "tiers": {
                    "1": [entry("a", 100, provider="azure", url="https://a.test", api_key=SECRET)]
                }
            },
            "azure_deployment",
        ),
        (
            {
                "tiers": {
                    "1": [
                        entry("a", 100, url="https://a.test", api_key=SECRET, vertex_location="eu")
                    ]
                }
            },
            "vertex_location",
        ),
        (
            {"tiers": {"1": [entry("a", 100, provider="gemini", api_key=SECRET)]}},
            "gemini",
        ),
        (
            {
                "tiers": {
                    "1": [
                        entry(
                            "a",
                            100,
                            provider="vertex",
                            vertex_credentials=SECRET,
                            vertex_location="eu",
                        )
                    ]
                }
            },
            "the key JSON, that JSON base64 encoded, or a path",
        ),
        (
            {
                "tiers": {
                    "1": [
                        entry(
                            "a",
                            100,
                            provider="vertex",
                            vertex_credentials={"type": "service_account", "private_key": SECRET},
                            vertex_location="eu",
                        )
                    ]
                }
            },
            "needs client_email",
        ),
    ],
)
def test_the_routers_rules_refuse_a_bad_config(config: dict[str, Any], message: str) -> None:
    assert message in refused(config)


def test_a_config_over_the_routers_limit_is_refused_when_sent() -> None:
    big = {"x": "y" * CONFIG_MAX_BYTES}
    config = RouterConfig(tiers={"1": [TierEntry(model="a", weight=100, kwargs=big)]})
    with pytest.raises(ValueError, match=f"at most {CONFIG_MAX_BYTES}"):
        config.wire()


def test_the_vertex_key_reads_as_json_base64_or_a_path(tmp_path: Path) -> None:
    text = json.dumps(KEY)
    path = tmp_path / "key.json"
    path.write_text(text)
    for form in (text, base64.b64encode(text.encode()).decode(), str(path), f"  {text}\n"):
        assert read_vertex_key(form) == KEY
    endpoint = VertexEndpoint(vertex_credentials=str(path), vertex_location="eu")
    assert endpoint.vertex_credentials.get_secret_value() == KEY
    assert "PRIVATE" not in repr(endpoint)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{not json", "not valid JSON"),
        (base64.b64encode(b"[1]").decode(), "path to the key file"),
        (base64.b64encode(b"{1").decode(), "not valid JSON"),
        ("[1, 2]", "path to the key file"),
        ("/no/such/key.json", "path to the key file"),
    ],
)
def test_a_vertex_key_in_no_known_form_says_which_forms_work(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_vertex_key(text)


def test_a_wrapped_key_is_still_checked_and_null_provider_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="needs a type"):
        VertexEndpoint(vertex_credentials=Secret({"x": "PRIVATE"}), vertex_location="eu")
    assert "'None'" in refused({"tiers": {"1": [entry("a", 100, provider=None, api_key=SECRET)]}})
    unreadable = tmp_path / "key.json"
    unreadable.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="cannot be read: UnicodeDecodeError"):
        read_vertex_key(str(unreadable))


def test_a_vertex_and_an_azure_endpoint_carry_a_model_id() -> None:
    config = RouterConfig.model_validate(
        {
            "tiers": {
                "1": [
                    entry(
                        "a",
                        50,
                        provider="vertex",
                        vertex_credentials=KEY,
                        vertex_location="eu",
                        model_id="gemini-x",
                    ),
                    entry(
                        "b",
                        50,
                        provider="azure",
                        url="https://r.openai.azure.com",
                        api_key=SECRET,
                        azure_deployment="d",
                        api_version="v",
                        model_id="ignored",
                    ),
                ]
            }
        }
    )
    endpoints = [item["endpoint"] for item in config.wire()["tiers"]["1"]]
    assert [endpoint["model_id"] for endpoint in endpoints] == ["gemini-x", "ignored"]


def test_a_plain_dump_keeps_every_key_hidden() -> None:
    config = RouterConfig.model_validate(
        {
            "tiers": {
                "1": [
                    entry(
                        "a", 50, provider="openai-compat", url="https://x.test/v1", api_key=SECRET
                    ),
                    entry("b", 50, provider="vertex", vertex_credentials=KEY, vertex_location="eu"),
                ]
            }
        }
    )
    for dumped in (
        config.model_dump_json(),
        json.dumps(config.model_dump(mode="json")),
        repr(config.model_dump()),
    ):
        assert SECRET not in dumped
        assert "PRIVATE" not in dumped
    assert SECRET in json.dumps(config.wire())


@pytest.mark.parametrize(("value", "ok"), [(0, False), (1, True), (10_000, True), (10_001, False)])
def test_slice_assistant_max_chars_takes_1_to_10000(value: int, ok: bool) -> None:
    if ok:
        assert RouterConfig(slice_assistant_max_chars=value).wire() == {
            "slice_assistant_max_chars": value
        }
    else:
        with pytest.raises(ValidationError, match="slice_assistant_max_chars"):
            RouterConfig(slice_assistant_max_chars=value)


def test_prewarm_cache_enabled_is_sent_when_set() -> None:
    assert RouterConfig(prewarm_cache_enabled=False).wire() == {"prewarm_cache_enabled": False}


@pytest.mark.parametrize(
    "key",
    ["api_base", "base_url", "metadata", "extra_headers", "reasoning", "aws_region", "Vertex_x"],
)
def test_a_tier_entry_keeps_a_default_field_the_router_drops_with_a_warning(
    key: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from unmute_core.slng import _blocked

    monkeypatch.setattr(_blocked, "_WARNED", set())
    with caplog.at_level(logging.WARNING):
        entry = TierEntry(model="a", weight=100, kwargs={key: "x"})
    assert entry.kwargs == {key: "x"}
    assert [r.getMessage() for r in caplog.records] == [
        f"a tier entry's kwargs sets {key!r}, which the SLNG Context Router drops: it never "
        "reaches the model. Leave it out, or set it with the router config's endpoint"
    ]


def test_a_tier_entry_keeps_reasoning_effort_and_sampling_defaults() -> None:
    kwargs = {"reasoning_effort": "low", "temperature": 0}
    assert TierEntry(model="a", weight=100, kwargs=kwargs).kwargs == kwargs
