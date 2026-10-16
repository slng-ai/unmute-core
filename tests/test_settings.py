"""Settings read the environment only when built, and errors name every variable to fix."""

import re
from pathlib import Path
from typing import Any

import pytest

from unmute_core.settings import (
    SettingsError,
    check_settings,
    env_example,
    env_flag,
    env_value,
)
from unmute_core.slng import SlngConfigError, SlngSettings
from unmute_core.telemetry import (
    CovalSettings,
    LangfuseSettings,
    LogfireSettings,
    TelemetryConfigError,
)

SETTINGS_DOC = Path(__file__).parents[1] / "docs" / "core" / "reference" / "settings.mdx"
ALL_VARIABLES = (
    "SLNG_API_KEY",
    "SLNG_ROUTER_URL",
    "SLNG_MODEL",
    "SLNG_WORLD_PART",
    "SLNG_LLM_CONFIG_ID",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
    "LOGFIRE_TOKEN",
    "COVAL_API_KEY",
)


@pytest.fixture(autouse=True)
def empty_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with none of the library's variables set."""
    for name in ALL_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_check_settings_names_every_missing_variable_at_once() -> None:
    with pytest.raises(SettingsError) as raised:
        check_settings(SlngSettings, LogfireSettings, CovalSettings)
    assert str(raised.value) == (
        "Set SLNG_API_KEY in the environment. "
        "Set SLNG_ROUTER_URL or SLNG_WORLD_PART in the environment. "
        "Set LOGFIRE_TOKEN in the environment."
    )


def test_check_settings_returns_the_loaded_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLNG_API_KEY", "key")
    monkeypatch.setenv("SLNG_ROUTER_URL", "https://router.test/v1")
    slng, coval = check_settings(SlngSettings, CovalSettings)
    assert isinstance(slng, SlngSettings)
    assert slng.model == "slng/auto"
    assert coval == CovalSettings(api_key=None)


def test_each_class_raises_its_own_error() -> None:
    with pytest.raises(SlngConfigError):
        SlngSettings()
    with pytest.raises(TelemetryConfigError):
        LogfireSettings()


def test_constructor_values_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLNG_API_KEY", "env")
    monkeypatch.setenv("SLNG_ROUTER_URL", "https://env.test/v1")
    settings = SlngSettings(api_key="arg", model="m")
    assert settings.api_key.get_secret_value() == "arg"
    assert settings.router_url == "https://env.test/v1"
    assert settings.model == "m"


@pytest.mark.parametrize("field", ["api_key", "router_url", "model"])
def test_an_empty_constructor_value_is_refused(field: str) -> None:
    values: dict[str, Any] = {"api_key": "k", "router_url": "https://router.test/v1", field: ""}
    name = f"SLNG_{field.upper()}"
    with pytest.raises(SlngConfigError) as raised:
        SlngSettings(**values)
    assert str(raised.value) == f"Set {name} in the environment."


def test_an_empty_variable_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGFIRE_TOKEN", "")
    with pytest.raises(TelemetryConfigError, match="Set LOGFIRE_TOKEN"):
        LogfireSettings()


def test_langfuse_host_is_an_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://host.test")
    assert LangfuseSettings().base_url == "https://host.test"
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://base.test")
    assert LangfuseSettings().base_url == "https://base.test"


def test_a_bad_value_names_its_variable() -> None:
    with pytest.raises(SlngConfigError, match=r"^SLNG_MODEL: Input should be a valid string"):
        SlngSettings(api_key="k", router_url="https://router.test/v1", model=3)  # ty: ignore[invalid-argument-type]


def test_secrets_never_show(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-very-secret")
    settings = LangfuseSettings(public_key="pk", base_url="https://host.test")
    assert "sk-very-secret" not in repr(settings)
    assert "sk-very-secret" not in str(settings.model_dump())


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "y", "on", "t"])
def test_env_flag_on(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SOME_FLAG", value)
    assert env_flag("SOME_FLAG") is True


@pytest.mark.parametrize("value", ["0", "false", "No", "n", "off", "f"])
def test_env_flag_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SOME_FLAG", value)
    assert env_flag("SOME_FLAG", default=True) is False


@pytest.mark.parametrize("value", [None, "", "  "])
def test_env_flag_unset_gives_the_default(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("SOME_FLAG", raising=False)
    else:
        monkeypatch.setenv("SOME_FLAG", value)
    assert env_flag("SOME_FLAG") is False
    assert env_flag("SOME_FLAG", default=True) is True


def test_env_flag_refuses_other_words(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_FLAG", "maybe")
    with pytest.raises(SettingsError, match=r"SOME_FLAG must be one of .*, not 'maybe'"):
        env_flag("SOME_FLAG")


def test_the_doc_lists_every_variable() -> None:
    expected = env_example(SlngSettings, LangfuseSettings, LogfireSettings, CovalSettings)
    block = re.search(r"```bash\n(# The SLNG API key\..*?)```", SETTINGS_DOC.read_text(), re.DOTALL)
    assert block is not None, "docs/core/reference/settings.mdx lost its env block"
    assert block[1] == expected


def test_a_world_part_names_the_router_and_a_url_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLNG_API_KEY", "key")
    assert SlngSettings(world_part="in").url == "https://in.context-router.slng.ai/v1"
    both = SlngSettings(world_part="in", router_url="https://router.test/v1")
    assert both.url == "https://router.test/v1"
    with pytest.raises(SlngConfigError) as raised:
        SlngSettings(world_part="EU North")
    assert str(raised.value) == (
        "SLNG_WORLD_PART: Value error, 'eu north' is no world part: write one like eu-north or us."
    )


def test_a_world_part_in_capitals_is_read_in_lower_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLNG_API_KEY", "key")
    assert SlngSettings(world_part="EU-North").url == "https://eu-north.context-router.slng.ai/v1"


def test_env_value_reads_the_named_variable_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ORDERS_TOKEN", raising=False)
    assert env_value("ORDERS_TOKEN", str) is None
    monkeypatch.setenv("ORDERS_TOKEN", "abc")
    assert env_value("ORDERS_TOKEN", str) == "abc"
    monkeypatch.setenv("ORDERS_TOKEN", "")
    assert env_value("ORDERS_TOKEN", str) is None


def test_env_value_is_case_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORDERS_TOKEN", raising=False)
    monkeypatch.setenv("orders_token", "abc")
    assert env_value("ORDERS_TOKEN", str) is None
    monkeypatch.setenv("ORDERS_TOKEN", "xyz")
    assert env_value("ORDERS_TOKEN", str) == "xyz"


def test_env_value_refuses_a_value_of_the_wrong_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORDERS_LIMIT", "many")
    with pytest.raises(SettingsError, match="ORDERS_LIMIT"):
        env_value("ORDERS_LIMIT", int)
