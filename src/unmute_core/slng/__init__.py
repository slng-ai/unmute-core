"""The SLNG Context Router as a model call.

``RouterHeaders`` and ``router_body`` build what the router reads. ``tracking_ids`` gives
SLNG speech-to-text and text-to-speech the call's agent and session ids. ``RouterConfig`` is a
typed router config a request carries inline. ``SlngSettings`` reads the ``SLNG_*``
variables. None of these need the ``[slng]`` extra. ``RouterModelCall``
sends the request and needs it: it is imported on first use. Native framework LLMs need
none of this.
"""

from typing import TYPE_CHECKING

from unmute_core.slng._config import (
    CONFIG_MAX_BYTES,
    AzureEndpoint,
    BedrockEndpoint,
    Endpoint,
    OpenAIEndpoint,
    RouterConfig,
    TierEntry,
    VertexEndpoint,
    read_vertex_key,
)
from unmute_core.slng._headers import (
    REQUEST_ID_HEADER,
    RouterHeaders,
    TrackingIds,
    router_body,
    tracking_ids,
)
from unmute_core.slng._settings import DEFAULT_MODEL, SlngConfigError, SlngSettings, router_url

if TYPE_CHECKING:
    from unmute_core.slng._router import RouterModelCall


def __getattr__(name: str) -> "type[RouterModelCall]":
    """Import ``RouterModelCall`` on first use, so the rest works without ``openai``."""
    if name == "RouterModelCall":
        from unmute_core.slng._router import RouterModelCall

        return RouterModelCall
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CONFIG_MAX_BYTES",
    "DEFAULT_MODEL",
    "REQUEST_ID_HEADER",
    "AzureEndpoint",
    "BedrockEndpoint",
    "Endpoint",
    "OpenAIEndpoint",
    "RouterConfig",
    "RouterHeaders",
    "RouterModelCall",
    "SlngConfigError",
    "SlngSettings",
    "TierEntry",
    "TrackingIds",
    "VertexEndpoint",
    "read_vertex_key",
    "router_body",
    "router_url",
    "tracking_ids",
]
