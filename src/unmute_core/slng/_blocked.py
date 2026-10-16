"""The request fields the SLNG Context Router drops, so a caller learns it from its own logs.

The router forwards every extra request field except the ones that could change the route,
the credentials or the headers, and the ones it builds itself. A blocked key never reaches
the model: the router drops it and logs a warning on its side. The library sends the
request as it is and logs one warning per key here too.
"""

import logging
from collections.abc import Iterable

logger = logging.getLogger(__name__)

BLOCKED_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "timeout",
        "num_retries",
        "max_retries",
        "strip_think",
        "api_key",
        "api_base",
        "base_url",
        "api_version",
        "azure_ad_token",
        "organization",
        "custom_llm_provider",
        "deployment_id",
        "extra_headers",
        "headers",
        "default_headers",
        "client",
        "mock_response",
        "fallbacks",
        "model_list",
        "metadata",
        "caching",
        "input",
        "instructions",
        "previous_response_id",
        "max_output_tokens",
        "text",
        "include",
        "truncation",
        "reasoning",
        "slng_reasoning_items",
    }
)
"""The exact names the router drops. ``reasoning_effort`` is not among them."""
BLOCKED_PREFIXES = ("aws_", "vertex_", "azure_", "watsonx_")
"""A name that starts with one of these is dropped too."""
NATIVE_ONLY = frozenset({"include", "slng_reasoning_items"})
"""Dropped from a tier entry, but a request body may carry them: a route that speaks the
Responses protocol natively forwards them."""


_WARNED: set[str] = set()
"""The keys already warned about in this process."""


def warn_dropped(keys: Iterable[str], where: str, *, request: bool = False) -> None:
    """Log one warning per key the router would drop. The key is still sent.

    Args:
        keys: The request fields to check.
        where: What holds them, for the message, like ``a tier entry's kwargs``.
        request: True for a request body or ``params``, where the Responses-only keys pass.
    """
    for key in keys:
        lowered = key.lower()
        if request and lowered in NATIVE_ONLY:
            continue
        if lowered in _WARNED:
            continue
        if lowered in BLOCKED_KEYS or lowered.startswith(BLOCKED_PREFIXES):
            _WARNED.add(lowered)
            logger.warning(
                "%s sets %r, which the SLNG Context Router drops: it never reaches the model. "
                "Leave it out, or set it with the router config's endpoint",
                where,
                key,
            )
