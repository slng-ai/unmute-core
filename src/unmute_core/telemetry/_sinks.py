"""Where spans go: one sink per provider, all plain OTLP over HTTP, no vendor SDK.

Endpoints, headers and env variables from unmute@d0c67f1
internal/generate/templates/telemetry/telemetry.py.tmpl (Langfuse, Logfire). Coval is in
``_coval.py``. Unmute picks one backend when it compiles. Here a call can send to several
sinks at once.
"""

import base64
import re
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Annotated, Self

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from pydantic import StringConstraints
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.telemetry._logfire import LogfireExporter
from unmute_core.telemetry._metadata import MetadataOnly
from unmute_core.telemetry._settings import LangfuseSettings, LogfireSettings

LOGFIRE_TOKEN_REGION = re.compile(r"pylf_v[0-9]+_(?P<region>[a-z]+)_")
LOGFIRE_REGIONS = frozenset({"us", "eu"})


@original_errors
@dataclass(kw_only=True, config=VALUE)
class Sink(ABC):
    """One place spans go: an exporter, wrapped in a batch processor.

    This is the plug point for a new provider. Build sinks once per call, together with the
    call's ``Telemetry``, because a sink like ``Coval`` holds that call's state.

    Attributes:
        filter: Lets only metadata through when set, like ``MetadataOnly()``.
    """

    filter: MetadataOnly | None = None

    @abstractmethod
    def exporter(self) -> SpanExporter:
        """Build the exporter that sends spans to this provider."""

    def processor(self) -> SpanProcessor:
        """Build the span processor the call's TracerProvider runs for this sink."""
        exporter = self.exporter()
        return BatchSpanProcessor(self.filter.wrap(exporter) if self.filter else exporter)


@original_errors
@dataclass(kw_only=True, config=VALUE)
class Otlp(Sink):
    """Any OTLP/HTTP endpoint, like an OpenTelemetry Collector.

    With no endpoint, the OpenTelemetry SDK reads ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``,
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` and ``OTEL_EXPORTER_OTLP_HEADERS`` itself.

    Attributes:
        endpoint: The full traces URL, like ``http://collector:4318/v1/traces``.
        headers: Extra HTTP headers, like an API key.
    """

    endpoint: Annotated[str, StringConstraints(pattern=r"^https?://\S+$")] | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def exporter(self) -> SpanExporter:
        """Build an OTLP/HTTP exporter for the endpoint."""
        return OTLPSpanExporter(endpoint=self.endpoint, headers=self.headers or None)


@original_errors
@dataclass(kw_only=True, config=VALUE)
class Langfuse(Sink):
    """Langfuse's OTLP endpoint, on the v4 data model.

    Attributes:
        public_key: The project's public key, ``pk-lf-...``.
        secret_key: The project's secret key, ``sk-lf-...``.
        base_url: Like ``https://cloud.langfuse.com`` or ``https://us.cloud.langfuse.com``.
    """

    public_key: str
    secret_key: str
    base_url: str

    @classmethod
    def from_settings(
        cls,
        settings: LangfuseSettings,
        *,
        filter: MetadataOnly | None = None,
    ) -> Self:
        """Build the sink from loaded settings. ``filter`` lets only metadata through."""
        return cls(
            public_key=settings.public_key,
            secret_key=settings.secret_key.get_secret_value(),
            base_url=settings.base_url,
            filter=filter,
        )

    @classmethod
    def from_env(
        cls,
        *,
        filter: MetadataOnly | None = None,
    ) -> Self:
        """Read ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_BASE_URL``.

        Raises:
            TelemetryConfigError: Naming every one that is unset.
        """
        return cls.from_settings(LangfuseSettings(), filter=filter)

    def exporter(self) -> SpanExporter:
        """Build the exporter. Without the ingestion header, traces show up minutes late."""
        auth = base64.b64encode(f"{self.public_key}:{self.secret_key}".encode()).decode()
        return OTLPSpanExporter(
            endpoint=f"{self.base_url.rstrip('/')}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
        )


@original_errors
@dataclass(kw_only=True, config=VALUE)
class Logfire(Sink):
    """Pydantic Logfire's OTLP endpoint, in the region the write token names.

    Attributes:
        token: A Logfire write token, like ``pylf_v1_eu_...``.
    """

    token: str

    @classmethod
    def from_settings(
        cls,
        settings: LogfireSettings,
        *,
        filter: MetadataOnly | None = None,
    ) -> Self:
        """Build the sink from loaded settings. ``filter`` lets only metadata through."""
        return cls(token=settings.token.get_secret_value(), filter=filter)

    @classmethod
    def from_env(
        cls,
        *,
        filter: MetadataOnly | None = None,
    ) -> Self:
        """Read ``LOGFIRE_TOKEN``.

        Raises:
            TelemetryConfigError: If it is unset.
        """
        return cls.from_settings(LogfireSettings(), filter=filter)

    @property
    def endpoint(self) -> str:
        """Return the traces URL for the token's region. US when the token names none."""
        match = LOGFIRE_TOKEN_REGION.match(self.token)
        region = match["region"] if match and match["region"] in LOGFIRE_REGIONS else "us"
        return f"https://logfire-{region}.pydantic.dev/v1/traces"

    def exporter(self) -> SpanExporter:
        """Build the exporter. Logfire takes the raw token, with no ``Bearer``.

        It labels each row and maps the model messages to the GenAI names, on a copy of the span.
        """
        return LogfireExporter(
            OTLPSpanExporter(endpoint=self.endpoint, headers={"Authorization": self.token})
        )
