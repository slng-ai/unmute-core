"""The model call through the SLNG Context Router.

The router fills the prompt itself, from ``template_variables``. So this call sends the
prompt unfilled, the variables, and the headers that scope the router's cache.
"""

import logging
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

from unmute_core.model import (
    ModelEvent,
    ModelRequest,
    is_gemini,
    router_messages,
    signed_tool_calls,
)
from unmute_core.model._chunks import chunk_events
from unmute_core.model._openai import (
    DEFAULT_KEEPALIVE_S,
    check_client_arguments,
    check_keepalive_without_client,
    join_options,
    keepalive_http_client,
    merged_headers,
    split_options,
)
from unmute_core.slng._blocked import warn_dropped
from unmute_core.slng._config import RouterConfig
from unmute_core.slng._headers import (
    AGENT_ID_HEADER,
    REQUEST_ID_HEADER,
    RouterHeaders,
    router_body,
    shortened,
)
from unmute_core.slng._settings import DEFAULT_MODEL, SlngSettings, router_url
from unmute_core.state._call_info import CallInfo

try:
    from openai import AsyncOpenAI, Timeout, omit
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionToolUnionParam,
    )
except ImportError as error:
    raise ImportError(
        'RouterModelCall needs the slng extra. Install it: uv add "unmute-core[slng]"'
    ) from error

logger = logging.getLogger(__name__)

TIMEOUT = Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0)
"""Unmute's router timeout, which is LiveKit's OpenAI plugin default. ``read`` bounds each
wait for the next bytes, the first token included."""
RESERVED_BODY = frozenset(
    {"slng_config", "slng_pure_proxy", "stream_options", "template_variables"}
)
"""Body fields ``RouterModelCall`` owns. Set them with its arguments, not ``body`` or ``params``."""


def answer_line(scope: str, headers: Mapping[str, str]) -> str:
    """Say where one router answer came from, in a fixed field order.

    The order is ``scope``, ``source``, ``layer``, ``model``, ``request_id``. ``layer`` and
    ``model`` appear only when the router sent them. Unmute's emitted code logs the same line.

    Args:
        scope: The agent id the request sent.
        headers: The ``x-slng-*`` response headers, with lowercase names.

    Returns:
        The fields joined by spaces.
    """
    fields = [f"scope={scope}", f"source={headers.get('x-slng-response-source', 'unknown')}"]
    if layer := headers.get("x-slng-cache-layer"):
        fields.append(f"layer={layer}")
    if model := headers.get("x-slng-model"):
        fields.append(f"model={model}")
    fields.append(f"request_id={headers.get('x-slng-request-id', 'unknown')}")
    return " ".join(fields)


class RouterModelCall:
    """A ``ModelCall`` that sends each request to the SLNG Context Router.

    Attributes:
        provider: ``slng``, the name telemetry shows for who serves the model.
        last_response: The ``x-slng-*`` headers of the latest response, for telemetry: the
            request id, the response source, the cache layer and the model that answered.
    """

    provider = "slng"
    stop_codes = frozenset({"project_mode_forbidden"})
    """The router refuses a key whose project mode forbids the request with 403
    ``project_mode_forbidden``. No other model fixes it, so ``FallbackModelCall`` stops."""
    variable_key_max = 64
    """The longest template-variable name the router takes."""

    def __init__(
        self,
        call: CallInfo,
        *,
        llm_config_id: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        world_part: str | None = None,
        api_key: str | None = None,
        config: RouterConfig | None = None,
        pure_proxy: bool = False,
        body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: Timeout | float = TIMEOUT,
        client: AsyncOpenAI | None = None,
        keepalive_s: float | None = DEFAULT_KEEPALIVE_S,
    ) -> None:
        """Set up the router client. Arguments left out come from ``SlngSettings``.

        Args:
            call: What the call carries, like ``state.call``. Every request sends its
                ``agent_id`` and ``session_id``.
            llm_config_id: The stored router config to use, when the key does not fix
                one. Defaults to ``SLNG_LLM_CONFIG_ID``, unless ``client`` is given.
            model: The model name. Defaults to ``SLNG_MODEL``, else ``slng/auto``.
            base_url: The router's OpenAI-compatible URL, ending in ``/v1``. Defaults to
                ``SLNG_ROUTER_URL``.
            world_part: The world part whose router to use, like ``eu-north``, when there
                is no ``base_url``. It wins over the environment's URL and world part.
            api_key: The SLNG key. Defaults to ``SLNG_API_KEY``.
            config: A router config sent with every request as ``slng_config``. The router
                lays it over the config it stores.
            pure_proxy: Send ``slng_pure_proxy``: no cache, no prompt edits, the default
                tier only.
            body: Extra body fields sent on every request, like ``reasoning_effort``. A
                request's ``params`` win over it. Its ``template_variables`` join each
                request's own, which win on a shared name. The openai request options in
                ``SDK_OPTIONS`` act as in ``create(**body)``: ``timeout``, ``max_retries``,
                ``extra_query`` and ``extra_headers`` are not body fields, and
                ``extra_body`` joins the body.
            extra_headers: Headers sent on every request. They never replace the identity
                headers.
            timeout: The client's timeout. ``TIMEOUT`` by default.
            client: A ready client, for tests. Then the environment is not read, and
                ``model`` defaults to ``slng/auto``. The default client never retries,
                because the failover rules decide that.
            keepalive_s: Seconds an idle connection to the router stays open for the next
                request. None keeps it open with no limit. The default, 120 s, covers the gap
                between two turns of a call. openai's own client closes it after 5 s, and the
                next turn then pays a new TCP and TLS handshake before its first token. Only
                for the client this call builds.

        Raises:
            TypeError: If ``call`` is no ``CallInfo``.
            SlngConfigError: If the key, or both the URL and the world part, are neither
                passed nor set.
            ValueError: If ``model`` names no model of ``config``, or ``body`` sets a field
                in ``RESERVED_BODY`` other than ``template_variables``, or one of its request
                options is not valid, or ``keepalive_s``
                is set together with ``client``. A field the router
                drops, like ``api_base`` or ``metadata``, is sent with one warning.
            ValidationError: If ``llm_config_id`` is empty, too long, or holds a comma,
                whitespace, a brace or a bar, or ``keepalive_s`` is not above zero.
        """
        # an argument of the wrong type, named with the fix: RouterHeaders was the old one
        if not isinstance(call, CallInfo):
            raise TypeError(
                f"RouterModelCall takes the call's CallInfo, like state.call, got "
                f"{type(call).__name__}. Pass llm_config_id= on its own"
            )
        # a router rule at the plug point: which body fields the router owns
        fields, _ = split_options(body or {})
        if reserved := sorted(set(fields) & (RESERVED_BODY - {"template_variables"})):
            raise ValueError(f"set {reserved[0]} with RouterModelCall's own argument, not body")
        if client is None:
            check_client_arguments(
                "RouterModelCall",
                model=model,
                base_url=base_url,
                timeout=timeout,
                keepalive_s=keepalive_s,
            )
        else:
            check_keepalive_without_client("RouterModelCall", keepalive_s)
        warn_dropped(fields, "RouterModelCall's body", request=True)
        self.body = dict(body or {})
        self.config = config
        self.pure_proxy = pure_proxy
        self.extra_headers = dict(extra_headers or {})
        if client is not None:
            self.model = model or DEFAULT_MODEL
            self.client = client
            self._owns_client = False
        else:
            given = {
                "api_key": api_key,
                "router_url": base_url or (router_url(world_part) if world_part else None),
                "world_part": world_part,
                "model": model,
            }
            passed: dict[str, Any] = {name: value for name, value in given.items() if value}
            settings = SlngSettings(**passed)
            self.model = settings.model
            self._owns_client = True
            self.client = AsyncOpenAI(
                base_url=settings.url,
                api_key=settings.api_key.get_secret_value(),
                max_retries=0,
                timeout=timeout,
                http_client=keepalive_http_client(keepalive_s, timeout),
            )
            llm_config_id = llm_config_id or settings.llm_config_id
        # needs the router config and the pinned model together
        if (
            config is not None
            and config.tiers is not None
            and self.model != DEFAULT_MODEL
            and self.model not in config.models
        ):
            raise ValueError(
                f"model {self.model!r} is not in the router config, which has "
                f"{', '.join(sorted(config.models)) or 'no models'}: pin one of those, or use "
                f"{DEFAULT_MODEL}"
            )
        self.headers = RouterHeaders(call.agent_id, call.session_id, llm_config_id)
        self._slng_config = config.wire() if config is not None else None
        self._gemini = is_gemini(self.model) or any(
            is_gemini(name) for name in (config.models if config is not None else ())
        )
        self.last_response: dict[str, str] = {}

    async def aclose(self) -> None:
        """Close the client when this call built it, not when ``client`` was passed in."""
        if self._owns_client:
            await self.client.close()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Send the request unfilled and stream the answer.

        When the request or the call's ``body`` carries template variables, the router fills
        the prompt, so each ``{{ a.b }}`` goes out as ``{{a__b}}``, the only form it reads.
        When there are none, nothing is filled and the messages go out exactly as written, so
        a prompt that shows a literal ``{{ name }}`` reaches the model unchanged.

        The answer ends with its ``Usage``. Each request carries a new ``X-Slng-Request-Id``,
        unless its headers set one, and a failure logs it.

        Each replayed tool call keeps its ``extra_content``, like a Gemini thought
        signature. When the model, or a model of the config, is a Gemini model, an
        assistant message whose first call has no signature gets Google's placeholder,
        ``skip_thought_signature_validator``.

        Raises:
            openai.APIStatusError: When the router answers with an error status.
                ``unmute_core.model.http_status`` reads it.
            ValueError: If the request's ``params`` set a field in ``RESERVED_BODY``, or one of
                its request options is not valid. A field the router drops is sent with one
                warning.
        """
        fields, requested = split_options(request.params)
        # a router rule at the plug point: the request must not set the fields the router owns
        if reserved := sorted(set(fields) & RESERVED_BODY):
            raise ValueError(f"a request's params must not set {reserved[0]}: RouterModelCall does")
        warn_dropped(fields, "a request's params", request=True)
        own_fields, own = split_options(self.body)
        options = join_options(own, requested)
        overlay = merged_headers(
            self.extra_headers, options.pop("extra_headers", {}), request.headers
        )
        if all(name.lower() != REQUEST_ID_HEADER.lower() for name in overlay):
            overlay[REQUEST_ID_HEADER] = uuid.uuid4().hex
        headers = self.headers.build(request.site, overlay)
        owned: dict[str, Any] = {}
        if self._slng_config is not None:
            owned["slng_config"] = self._slng_config
        if self.pure_proxy:
            owned["slng_pure_proxy"] = True
        variables = shortened(own_fields.get("template_variables", {}) | dict(request.variables))
        try:
            async for event in self._stream(
                request,
                headers,
                own_fields | fields | owned | router_body(variables),
                options,
                rewrite=bool(variables),
            ):
                yield event
        except Exception:
            request_id = next(
                (v for k, v in headers.items() if k.lower() == REQUEST_ID_HEADER.lower()), None
            )
            logger.warning("router request %s failed", request_id)
            raise

    async def _stream(
        self,
        request: ModelRequest,
        headers: dict[str, str],
        body: dict[str, Any],
        options: dict[str, Any],
        *,
        rewrite: bool,
    ) -> AsyncIterator[ModelEvent]:
        """Send one request and turn the router's chunks into events.

        Args:
            request: The request to send.
            headers: The headers to send.
            body: The extra body fields to send, the router's own among them.
            options: The openai request options: ``timeout``, ``max_retries`` and
                ``extra_query``.
            rewrite: True when the router fills variables, so placeholders are written in
                the form it reads. False sends the messages as written.
        """
        messages = router_messages(request.messages) if rewrite else request.messages
        client = self.client
        if "max_retries" in options:
            client = client.with_options(max_retries=options.pop("max_retries"))
        stream = await client.chat.completions.create(
            model=self.model,
            messages=cast(
                "list[ChatCompletionMessageParam]",
                signed_tool_calls(messages, gemini=self._gemini),
            ),
            tools=cast("list[ChatCompletionToolUnionParam]", request.tools)
            if request.tools
            else omit,
            tool_choice=request.tool_choice if request.tools and request.tool_choice else omit,
            stream=True,
            stream_options={"include_usage": True},
            extra_headers=headers,
            extra_body=body,
            **options,
        )
        async with stream:
            response = stream.response
            self.last_response = {
                name: value
                for name, value in response.headers.items()
                if name.startswith("x-slng-")
            }
            scope = next(
                (v for k, v in headers.items() if k.lower() == AGENT_ID_HEADER.lower()), None
            )
            if scope:
                logger.info("slng router: %s", answer_line(scope, self.last_response))
            async for event in chunk_events(
                stream, request_id=response.headers.get("x-request-id")
            ):
                yield event
