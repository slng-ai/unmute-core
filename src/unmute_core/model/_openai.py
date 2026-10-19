"""A native model call for any OpenAI-compatible chat completions endpoint."""

import logging
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Annotated, Any, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, PlainValidator, StringConstraints, ValidationError

from unmute_core._pydantic import retitled
from unmute_core.model._call import ModelEvent, ModelRequest, NativeModelCall
from unmute_core.model._chunks import chunk_events
from unmute_core.model._prompt import folded_system
from unmute_core.model._signatures import is_gemini, signed_tool_calls, takes_one_system_message

if TYPE_CHECKING:
    from openai import AsyncOpenAI, Timeout

logger = logging.getLogger(__name__)

RESERVED_PARAMS = frozenset({"stream_options"})
"""Body fields ``OpenAIModelCall`` owns."""


def _seconds_or_timeout(value: object) -> object:
    """Pass None, seconds above zero, or a timeout object like ``openai.Timeout``."""
    if value is None or hasattr(value, "connect"):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError("timeout must be seconds above zero or an openai.Timeout")


class ClientArguments(BaseModel):
    """The plain arguments of an OpenAI-compatible model call, checked when it is built."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | None = None
    base_url: Annotated[str, StringConstraints(pattern=r"^https?://\S+$")] | None = None
    timeout: Annotated[object, PlainValidator(_seconds_or_timeout)] = None


def check_client_arguments(
    title: str, *, model: str | None = None, base_url: str | None = None, timeout: object = None
) -> None:
    """Check a model call's model name, URL and timeout, naming the public class.

    Args:
        title: The public class, like ``OpenAIModelCall``.
        model: The model name.
        base_url: The endpoint's URL.
        timeout: Seconds, or a timeout object like ``openai.Timeout``.

    Raises:
        ValidationError: Naming the class and the argument.
    """
    try:
        ClientArguments(model=model, base_url=base_url, timeout=timeout)
    except ValidationError as error:
        raise retitled(error, title) from None


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
    ) -> None:
        """Set up the client.

        Args:
            model: The model name.
            base_url: The endpoint's URL, ending in ``/v1``. None means OpenAI.
            api_key: The key. None reads ``OPENAI_API_KEY``.
            client: A ready client, for tests. Then ``base_url``, ``api_key`` and ``timeout``
                are not used, and ``aclose`` leaves the client open.
            params: Body fields sent on every request, like ``{"temperature": 0}``. A
                request's own ``params`` win.
            timeout: The client's timeout. None waits 15 s to connect and 5 s for each read,
                as ``RouterModelCall`` does. The default client never retries: the failover
                rules decide that.

        Raises:
            ImportError: If the ``openai`` package is missing.
            ValueError: If ``params`` sets a field in ``RESERVED_PARAMS``, or ``model``,
                ``base_url`` or ``timeout`` is not one the client can use.
        """
        try:
            from openai import AsyncOpenAI, Timeout
        except ImportError as error:
            raise ImportError(
                "OpenAIModelCall needs the openai package: install unmute-core[slng], or add openai"
            ) from error
        if reserved := sorted(set(params or {}) & RESERVED_PARAMS):  # a plain dict, not a model
            raise ValueError(f"set {reserved[0]} with OpenAIModelCall's own behaviour, not params")
        if client is None:
            check_client_arguments(
                "OpenAIModelCall", model=model, base_url=base_url, timeout=timeout
            )
        else:
            check_client_arguments("OpenAIModelCall", model=model)
        super().__init__(self._send)
        self.model = model
        self.provider = (urlparse(base_url).hostname if base_url else None) or "openai"
        self.params = dict(params or {})
        self._owns_client = client is None
        self.client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=timeout
            if timeout is not None
            else Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        )
        self._one_system_message = takes_one_system_message(model)

    async def _send(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Send one filled request and stream the events of the answer.

        Each replayed tool call keeps its ``extra_content``. For a Gemini model, an
        assistant message whose first call has no signature gets Google's placeholder.
        For a Gemini model, which takes one system message, the system and developer
        messages go out folded into one.

        Raises:
            openai.APIStatusError: When the endpoint answers with an error status.
            ValueError: If the request's ``params`` set a field in ``RESERVED_PARAMS``.
        """
        if reserved := sorted(set(request.params) & RESERVED_PARAMS):  # per request, at send
            raise ValueError(f"a request's params must not set {reserved[0]}: OpenAIModelCall does")
        from openai import omit

        messages = request.messages
        if self._one_system_message:
            messages = folded_system(messages)
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=cast("Any", signed_tool_calls(messages, gemini=is_gemini(self.model))),
            tools=cast("Any", request.tools) if request.tools else omit,
            tool_choice=request.tool_choice if request.tools and request.tool_choice else omit,
            stream=True,
            stream_options={"include_usage": True},
            extra_headers=request.headers or None,
            extra_body=self.params | request.params,
        )
        response = getattr(stream, "response", None)
        request_id = response.headers.get("x-request-id") if response is not None else None
        async for event in chunk_events(stream, request_id=request_id):
            yield event

    async def aclose(self) -> None:
        """Close the client when this call built it. Closing twice is safe."""
        if self._owns_client:
            await self.client.close()
