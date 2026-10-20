"""A native model call for any OpenAI-compatible chat completions endpoint."""

import logging
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Annotated, Any, cast
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    StringConstraints,
    ValidationError,
)

from unmute_core._pydantic import retitled
from unmute_core.model._call import ModelEvent, ModelRequest, NativeModelCall
from unmute_core.model._chunks import chunk_events
from unmute_core.model._prompt import folded_system
from unmute_core.model._signatures import is_gemini, signed_tool_calls, takes_one_system_message

if TYPE_CHECKING:
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient, Timeout

logger = logging.getLogger(__name__)

RESERVED_PARAMS = frozenset({"stream_options"})
"""Body fields ``OpenAIModelCall`` owns."""

DEFAULT_KEEPALIVE_S = 120.0
"""How long a call's own client keeps an idle connection open. openai's own is 5 s, shorter
than the gap between two turns of a voice call, so every turn would open a new connection."""


SDK_OPTIONS = frozenset({"timeout", "max_retries", "extra_query", "extra_headers", "extra_body"})
"""Names in ``params`` that are openai request options, not body fields. They act as in
``client.chat.completions.create(**params)``: ``timeout`` is the request's timeout,
``max_retries`` its retry count, ``extra_query`` its URL query, ``extra_headers`` its
headers, and ``extra_body`` fields join the JSON body."""


def _seconds_or_timeout(value: object) -> object:
    """Pass None, seconds above zero, or a timeout object like ``openai.Timeout``."""
    if value is None or hasattr(value, "connect"):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError("timeout must be seconds above zero or an openai.Timeout")


def split_options(params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``params`` into JSON body fields and openai request options.

    Args:
        params: Body fields and ``SDK_OPTIONS``, like ``{"temperature": 0, "timeout": 3}``.

    Returns:
        The body fields, with the fields of ``extra_body`` joined in, and the options that
        were set: ``timeout``, ``max_retries``, ``extra_query`` and ``extra_headers``.

    Raises:
        ValueError: If ``timeout`` is not seconds above zero, None or an ``openai.Timeout``,
            ``max_retries`` is not a whole number of 0 or more, or ``extra_query``,
            ``extra_headers`` or ``extra_body`` is not a mapping.
    """
    for name in ("extra_query", "extra_headers", "extra_body"):
        if name in params and not isinstance(params[name], Mapping):
            raise ValueError(f"{name} must be a mapping, got {type(params[name]).__name__}")
    if "timeout" in params:
        _seconds_or_timeout(params["timeout"])
    retries = params.get("max_retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("max_retries must be a whole number of 0 or more")
    body = {name: value for name, value in params.items() if name not in SDK_OPTIONS}
    body |= params.get("extra_body", {})
    options = {name: params[name] for name in SDK_OPTIONS - {"extra_body"} if name in params}
    return body, options


def merged_headers(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """Join header layers, a later layer winning on a name in any letter case."""
    kept: dict[str, tuple[str, Any]] = {}
    for layer in layers:
        kept |= {name.lower(): (name, value) for name, value in layer.items()}
    return dict(kept.values())


def join_options(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Lay one set of request options over another.

    ``timeout`` and ``max_retries`` in ``over`` replace those in ``base``. ``extra_query``
    and ``extra_headers`` join, and ``over`` wins on a shared name.

    Args:
        base: Options from ``split_options``, like the call's own.
        over: Options from ``split_options``, like one request's.

    Returns:
        A new dict.
    """
    joined = dict(base) | dict(over)
    if "extra_query" in base and "extra_query" in over:
        joined["extra_query"] = {**base["extra_query"], **over["extra_query"]}
    if "extra_headers" in base and "extra_headers" in over:
        joined["extra_headers"] = merged_headers(base["extra_headers"], over["extra_headers"])
    return joined


class ClientArguments(BaseModel):
    """The plain arguments of an OpenAI-compatible model call, checked when it is built."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | None = None
    base_url: Annotated[str, StringConstraints(pattern=r"^https?://\S+$")] | None = None
    timeout: Annotated[object, PlainValidator(_seconds_or_timeout)] = None
    keepalive_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = DEFAULT_KEEPALIVE_S


def check_client_arguments(
    title: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: object = None,
    keepalive_s: float | None = DEFAULT_KEEPALIVE_S,
) -> None:
    """Check a model call's model name, URL, timeout and keep-alive, naming the public class.

    Args:
        title: The public class, like ``OpenAIModelCall``.
        model: The model name.
        base_url: The endpoint's URL.
        timeout: Seconds, or a timeout object like ``openai.Timeout``.
        keepalive_s: Seconds above zero, or None for no limit.

    Raises:
        ValidationError: Naming the class and the argument.
    """
    try:
        ClientArguments(model=model, base_url=base_url, timeout=timeout, keepalive_s=keepalive_s)
    except ValidationError as error:
        raise retitled(error, title) from None


def check_keepalive_without_client(title: str, keepalive_s: float | None) -> None:
    """Refuse a ``keepalive_s`` that a ready client would ignore.

    Args:
        title: The public class, like ``OpenAIModelCall``.
        keepalive_s: The value the caller passed next to ``client=``.

    Raises:
        ValueError: If it is not the default. Set the keep-alive on the client itself.
    """
    if keepalive_s != DEFAULT_KEEPALIVE_S:
        raise ValueError(
            f"{title}: keepalive_s only applies to a client {title} builds. "
            "Drop client=, or set the keep-alive on the client you pass"
        )


def keepalive_http_client(
    keepalive_s: float | None, timeout: "Timeout | float | None"
) -> "DefaultAsyncHttpxClient":
    """Build openai's default HTTP client with an idle keep-alive of your choice.

    The connection limits are openai's own, except how long an idle connection stays open.

    Args:
        keepalive_s: Seconds an idle connection stays open. None keeps it open with no limit.
        timeout: The client's timeout.

    Returns:
        The HTTP client to pass to ``AsyncOpenAI(http_client=...)``.
    """
    import openai

    defaults = openai.DEFAULT_CONNECTION_LIMITS
    limits = type(defaults)(
        max_connections=defaults.max_connections,
        max_keepalive_connections=defaults.max_keepalive_connections,
        keepalive_expiry=keepalive_s,
    )
    return openai.DefaultAsyncHttpxClient(limits=limits, timeout=timeout)


class OpenAIModelCall(NativeModelCall):
    """Stream OpenAI-compatible chat completions with the ``openai`` package.

    The library fills the prompt, as with any native call. The request carries no SLNG
    header and no SLNG body field. The answer streams the same events as ``RouterModelCall``.

    Attributes:
        model: The model name sent.
        provider: ``openai``, or the host name of ``base_url`` when one is set.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: "AsyncOpenAI | None" = None,
        params: Mapping[str, Any] | None = None,
        timeout: "Timeout | float | None" = None,
        keepalive_s: float | None = DEFAULT_KEEPALIVE_S,
        tool_call_starts: bool = False,
    ) -> None:
        """Set up the client.

        Args:
            model: The model name.
            base_url: The endpoint's URL, ending in ``/v1``. None means OpenAI.
            api_key: The key. None reads ``OPENAI_API_KEY``.
            client: A ready client, for tests. Then ``base_url``, ``api_key`` and ``timeout``
                are not used, and ``aclose`` leaves the client open.
            params: Body fields sent on every request, like ``{"temperature": 0}``. A
                request's own ``params`` win. The openai request options in ``SDK_OPTIONS``
                act as in ``create(**params)``: ``timeout``, ``max_retries``,
                ``extra_query`` and ``extra_headers`` are not body fields, and ``extra_body``
                joins the body.
            timeout: The client's timeout. None waits 15 s to connect and 5 s for each read,
                as ``RouterModelCall`` does. The default client never retries: the failover
                rules decide that.
            keepalive_s: Seconds an idle connection to the provider stays open for the next
                request. None keeps it open with no limit. The default, 120 s, covers the gap
                between two turns of a call. openai's own client closes it after 5 s, and the
                next turn then pays a new TCP and TLS handshake before its first token. Only
                for the client this call builds.
            tool_call_starts: Stream a ``ToolCallStarted`` at the first piece of each tool
                call, before its ``ToolCall``. Off by default.

        Raises:
            ImportError: If the ``openai`` package is missing.
            ValueError: If ``params`` sets a field in ``RESERVED_PARAMS``, or one of its
                request options is not valid, or ``model``,
                ``base_url``, ``timeout`` or ``keepalive_s`` is not one the client can use, or
                ``keepalive_s`` is set together with ``client``.
        """
        try:
            from openai import AsyncOpenAI, Timeout
        except ImportError as error:
            raise ImportError(
                "OpenAIModelCall needs the openai package: install unmute-core[slng], or add openai"
            ) from error
        fields, _ = split_options(params or {})
        if reserved := sorted(set(fields) & RESERVED_PARAMS):  # a plain dict, not a model
            raise ValueError(f"set {reserved[0]} with OpenAIModelCall's own behaviour, not params")
        if client is None:
            check_client_arguments(
                "OpenAIModelCall",
                model=model,
                base_url=base_url,
                timeout=timeout,
                keepalive_s=keepalive_s,
            )
        else:
            check_client_arguments("OpenAIModelCall", model=model)
            check_keepalive_without_client("OpenAIModelCall", keepalive_s)
        super().__init__(self._send)
        self.model = model
        self.provider = (urlparse(base_url).hostname if base_url else None) or "openai"
        self.params = dict(params or {})
        self._owns_client = client is None
        if client is None:
            timeout = (
                timeout
                if timeout is not None
                else Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0)
            )
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                max_retries=0,
                timeout=timeout,
                http_client=keepalive_http_client(keepalive_s, timeout),
            )
        self.client = client
        self._one_system_message = takes_one_system_message(model)
        self._tool_call_starts = tool_call_starts

    async def _send(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Send one filled request and stream the events of the answer.

        Each replayed tool call keeps its ``extra_content``. For a Gemini model, an
        assistant message whose first call has no signature gets Google's placeholder.
        For a Gemini model, which takes one system message, the system and developer
        messages go out folded into one.

        Raises:
            openai.APIStatusError: When the endpoint answers with an error status.
            ValueError: If the request's ``params`` set a field in ``RESERVED_PARAMS``, or one
                of its request options is not valid.
        """
        fields, requested = split_options(request.params)
        if reserved := sorted(set(fields) & RESERVED_PARAMS):  # per request, at send
            raise ValueError(f"a request's params must not set {reserved[0]}: OpenAIModelCall does")
        from openai import omit

        own_fields, own = split_options(self.params)
        options = join_options(own, requested)
        headers = merged_headers(options.pop("extra_headers", {}), request.headers)
        client = self.client
        if "max_retries" in options:
            client = client.with_options(max_retries=options.pop("max_retries"))

        messages = request.messages
        if self._one_system_message:
            messages = folded_system(messages)
        stream = await client.chat.completions.create(
            model=self.model,
            messages=cast("Any", signed_tool_calls(messages, gemini=is_gemini(self.model))),
            tools=cast("Any", request.tools) if request.tools else omit,
            tool_choice=request.tool_choice if request.tools and request.tool_choice else omit,
            stream=True,
            stream_options={"include_usage": True},
            extra_headers=headers or None,
            extra_body=own_fields | fields,
            **options,
        )
        response = getattr(stream, "response", None)
        request_id = response.headers.get("x-request-id") if response is not None else None
        async for event in chunk_events(
            stream, request_id=request_id, tool_call_starts=self._tool_call_starts
        ):
            yield event

    async def aclose(self) -> None:
        """Close the client when this call built it. Closing twice is safe."""
        if self._owns_client:
            await self.client.close()
