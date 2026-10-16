"""The SLNG Context Router: its model configuration, client and template variables."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.http import connection_limits

from call_state import CallState
from settings import ROUTER_KEEPALIVE_CONNECTIONS, ROUTER_MAX_CONNECTIONS

# --- SLNG Context Router ----------------------------------------------------
# The router answers a repeated turn from its cache instead of calling the model.
# That is why a router-bound system prompt below keeps its placeholders instead of
# being rendered here: the router substitutes them from template_variables, so the
# prompt it sees is identical on every call, which is what lets a turn repeat. A
# first turn never caches, and the router decides which later turns are
# repeatable, so a repeat served by the model is expected rather than a fault.
#
# The agent id header scopes that cache, and it is one value per prompt rather
# than one for the package: each agent and each task sends the authored agent_id,
# a colon, then its own name: `<agent_id>:<agent>` for an agent and
# `<agent_id>:task.<task>` for a task. The cache key is the last exchange and carries no
# system prompt, so two prompts under one scope get served each other's answers.
# A task runs on its owner's service here, so entering one swaps its scope in and
# every way out swaps the owner's back.


def _slng_config_reasoning() -> dict[str, Any]:
    """Build the model configuration sent inline in every think request.

    Credentials are read here rather than at import, so a missing one is reported
    by require_env() alongside every other name instead of raising a KeyError
    before anything has checked. No value is written into this file: each is the
    name of an environment variable.

    Returns:
        The configuration body for the router's think request.
    """
    return {
        "tiers": {
            "1": [
                {
                    "endpoint": {
                        "api_key": os.environ["OPENAI_API_KEY"],
                        "url": "https://api.openai.com/v1",
                    },
                    "model": "gpt-5.6-luna",
                    "weight": 100,
                }
            ]
        }
    }


async def _slng_log_provenance(response: Any) -> None:
    """Say where this answer came from, once per think request.

    The router states this only in response headers, and neither framework hands
    them to us, so a hook on the client is the one place that sees them. Without
    it the question an operator actually asks about a cache, whether it is
    working, has no answer in this run's own log.

    Three rules, from httpx's own event-hooks documentation, each of which would
    be a live-call defect if broken. It has to be async, because a sync callable
    on an AsyncClient is never awaited. It reads headers only: the hook runs
    before the body is read, so touching the body would consume the stream the
    framework is about to iterate. And it cannot raise, because a raising
    response hook fails the request it was only meant to describe.

    It also logs only a router think request. The scope header is what lets the
    line name a scope at all, so a request without one is a request this hook
    could not describe.

    Args:
        response: The httpx response, whose body has not been read yet.
    """
    try:
        scope = response.request.headers.get("X-Slng-Agent-Id")
        if not scope:
            return
        headers = response.headers
        # Field order is the contract, and the gate reads it off this line.
        fields = [
            "scope=" + scope,
            "source=" + headers.get("x-slng-response-source", "unknown"),
        ]
        if layer := headers.get("x-slng-cache-layer"):
            fields.append("layer=" + layer)
        if model := headers.get("x-slng-model"):
            fields.append("model=" + model)
        fields.append("request_id=" + headers.get("x-slng-request-id", "unknown"))
        logger.info("slng router: " + " ".join(fields))
    except Exception:  # noqa: BLE001 - a log line must never end a call
        logger.debug("could not read the router's provenance headers", exc_info=True)


class _SlngRouterLLMService(OpenAILLMService):
    """The router's LLM service: a response hook, and per-request variables.

    Two overrides, two different seams, and the second one is why this class
    holds any state.

    create_client is the only seam for reading the response headers: the service
    builds its own AsyncOpenAI and hands us neither the raw response nor its
    headers, and the router states where an answer came from only in a header.
    The connection limits restate the base class's own (pipecat
    services/openai/base_llm.py create_client at the pinned version). Restating
    them is deliberate. Anything different here would change connection reuse,
    which is a latency change nobody asked for and nothing would report.

    build_chat_completion_params is the seam for the request body. The base
    class's last statement before returning is params.update(self._settings.extra),
    which reads a settings snapshot mutated only by an explicit update: nothing
    re-evaluates the expression that filled it, so a value written part way
    through a step reached the model one turn late. This override reads the live
    state instead, on the streaming path and the one-shot path both, which is
    what makes the two targets refresh at the same point.
    """

    def __init__(self, *args: Any, slng_state: Any = None, **kwargs: Any) -> None:
        """Build the service and keep the call state its request body reads.

        Args:
            *args: Passed to the base service.
            slng_state: The live call state, read on every request.
            **kwargs: Passed to the base service.
        """
        super().__init__(*args, **kwargs)
        self._slng_state = slng_state

    def build_chat_completion_params(self, params_from_context: Any) -> dict[str, Any]:
        """Add the live template variables to the request body.

        Args:
            params_from_context: The invocation parameters the base service built.

        Returns:
            The request parameters with `template_variables` filled from state.
        """
        params = super().build_chat_completion_params(params_from_context)
        scope = (params.get("extra_headers") or {}).get("X-Slng-Agent-Id", "")
        body = dict(params.get("extra_body") or {})
        body["template_variables"] = _slng_template_variables(
            self._slng_state, _SLNG_TEMPLATE_PATHS.get(scope, ()), scope=scope
        )
        params["extra_body"] = body
        return params

    def create_client(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncOpenAI:
        """Build the client with a hook that logs where each answer came from.

        Args:
            api_key: The router's API key.
            base_url: The router's base URL.
            organization: Accepted for the base signature and not forwarded.
            project: Accepted for the base signature and not forwarded.
            default_headers: Accepted for the base signature and not forwarded.
            **kwargs: Accepted for the base signature and not forwarded.

        Returns:
            An AsyncOpenAI client with the provenance response hook installed.
        """
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(
                limits=connection_limits(
                    max_keepalive_connections=ROUTER_KEEPALIVE_CONNECTIONS,
                    max_connections=ROUTER_MAX_CONNECTIONS,
                    keepalive_expiry=None,
                ),
                event_hooks={"response": [_slng_log_provenance]},
            ),
        )


_SLNG_VARIABLE_LIMIT = 4000
_SLNG_TEMPLATE_PATHS = {
    "salon-concierge-slng-v1:complaint_specialist": [
        "customer_verified",
        "appointment",
    ],
    "salon-concierge-slng-v1:concierge": ["appointment"],
    "salon-concierge-slng-v1:task.manage_booking": [
        "booking_weekday",
        "booking_date",
        "salon_local_time",
        "appointment",
    ],
    "salon-concierge-slng-v1:task.verify_customer": ["customer_phone"],
}
_SLNG_SCOPE_SITES = {
    "salon-concierge-slng-v1:task.manage_booking": "task:manage_booking",
    "salon-concierge-slng-v1:task.verify_customer": "task:verify_customer",
}


def _slng_template_variables(
    state: Any, names: Sequence[str], *, scope: str = ""
) -> dict[str, str]:
    """Build the values the router substitutes into the prompt's placeholders.

    A name with no value yet sends the empty string, never None and never the text
    "None". A value over the router's limit is truncated with a warning rather
    than dropped: an over-long value must not end a live call.

    Args:
        state: The call state, or None before one exists.
        names: The placeholder names the prompt at this scope reads.
        scope: The cache scope, which picks the prompt site for a path lookup.

    Returns:
        Each name mapped to its text, bounded by the router's limit.
    """
    values = {}
    for name in names:
        # The bound is measured on the JSON rendering, not on the repr of the
        # object: they are different lengths, and the one the router receives is
        # the one that has to fit. CallState.render carries its own bound and its
        # own warning, so a declared value is already shortened by here. A path
        # arrives as one flat name, and CallState.lookup walks it.
        text = CallState.render(state, name, _SLNG_SCOPE_SITES.get(scope, ""))
        if len(text) > _SLNG_VARIABLE_LIMIT:
            logger.warning(
                "template variable {} is {} characters; truncating to {} for the router",
                name,
                len(text),
                _SLNG_VARIABLE_LIMIT,
            )
            text = text[:_SLNG_VARIABLE_LIMIT]
        values[name] = text
    return values
