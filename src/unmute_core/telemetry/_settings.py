"""The trace sinks' settings, read from each vendor's own variables.

There is no OTLP class: the OpenTelemetry SDK reads ``OTEL_EXPORTER_OTLP_*`` itself.
"""

from typing import ClassVar

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import SettingsConfigDict

from unmute_core.settings import EnvSettings, SettingsError


class TelemetryConfigError(SettingsError):
    """A sink is missing a setting it needs, like an API key."""


class LangfuseSettings(EnvSettings):
    """A Langfuse project's keys and host.

    Attributes:
        public_key: The project's public key, ``pk-lf-...``.
        secret_key: The project's secret key, ``sk-lf-...``.
        base_url: Like ``https://cloud.langfuse.com``. ``LANGFUSE_HOST`` works too.
    """

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")
    config_error: ClassVar[type[SettingsError]] = TelemetryConfigError

    public_key: str = Field(min_length=1, description="The Langfuse public key, pk-lf-...")
    secret_key: SecretStr = Field(min_length=1, description="The Langfuse secret key, sk-lf-...")
    base_url: str = Field(
        min_length=1,
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
        description="The Langfuse host, like https://cloud.langfuse.com.",
    )


class LogfireSettings(EnvSettings):
    """A Logfire write token.

    Attributes:
        token: Like ``pylf_v1_eu_...``. The token names the region.
    """

    model_config = SettingsConfigDict(env_prefix="LOGFIRE_")
    config_error: ClassVar[type[SettingsError]] = TelemetryConfigError

    token: SecretStr = Field(
        min_length=1, description="The Logfire write token, like pylf_v1_eu_..."
    )


class CovalSettings(EnvSettings):
    """A Coval API key and, for a local run, the simulation the call belongs to.

    Both may be unset: an evaluation key must not break a live call.

    Attributes:
        api_key: The Coval API key, or None to export nothing.
        simulation_id: The Coval simulation to file the call under. A simulation that
            places a call names itself in the call, so this is for a local run.
    """

    model_config = SettingsConfigDict(env_prefix="COVAL_")
    config_error: ClassVar[type[SettingsError]] = TelemetryConfigError

    api_key: SecretStr | None = Field(
        default=None, description="The Coval API key. Unset sends no spans to Coval."
    )
    simulation_id: str | None = Field(
        default=None,
        description="The Coval simulation this call belongs to. Unset reads it from the call.",
    )
