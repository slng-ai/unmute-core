"""The tool kinds the library runs: a webhook and a local Python handler.

The webhook runs over the hardened HTTP client in ``unmute_core.tools._http``, with strict
rules for methods, size caps, HMAC, protected headers and results. It can wrap the arguments
in an envelope body and send without waiting for the answer. Its auth from the environment
is Unmute's (unmute@868ae23 templates/livekit_v1/utils_auth.py.tmpl). The
handlers follow Unmute's emitted code (unmute@d0c67f1 templates/livekit_v1/
imports.py.tmpl): a local handler takes the arguments as keywords, or one ``Input``
model.
"""

import asyncio
import hashlib
import hmac
import inspect
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import cached_property
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, TypedDict, Unpack, get_args, get_type_hints
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

import httpx
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    create_model,
    model_validator,
)
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.settings import env_value
from unmute_core.state import CallState
from unmute_core.tools._http import HardenedHttpError, check_destination, send_pinned_request
from unmute_core.tools._spec import (
    ToolEffect,
    ToolError,
    ToolInterruption,
    ToolSpec,
    fill_text,
)

Method = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
METHODS = frozenset(get_args(Method))
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_REDIRECTS = 3
PROTECTED_HEADERS = frozenset(
    {
        "proxy-authorization",
        "content-length",
        "host",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "upgrade",
        "cookie",
        "x-signature-256",
    }
)
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
ACCEPTED = MappingProxyType({"ok": True, "accepted": True})
"""What the model reads from a webhook that does not wait for its answer."""

logger = logging.getLogger(__name__)


TokenProvider = Callable[[], str | Awaitable[str]]
"""Returns a webhook's token at each call, like a lookup in a secret store."""


def _env_name(name: str) -> str:
    """Refuse an environment variable name that is not UPPER_SNAKE: it may be a secret value."""
    if name and not _ENV_NAME.fullmatch(name):
        raise ValueError(
            "token_env must be an UPPER_SNAKE environment variable name, never a secret value"
        )
    return name


def _auth_header(name: str, info: ValidationInfo) -> str:
    """Refuse a header a bearer credential cannot use, or one the library sets itself."""
    if info.data.get("kind") == "bearer" and name != "Authorization":
        raise ValueError("header is not a bearer field: bearer always sends Authorization")
    if not _HEADER_NAME.fullmatch(name) or name.lower() in PROTECTED_HEADERS:
        raise ValueError(f"webhook auth header {name!r} cannot be used")
    return name


@original_errors
@dataclass(frozen=True, config=VALUE)
class WebhookAuth:
    """A webhook's credential, read at each call: from an environment variable, or a provider.

    The agent's code never holds the secret: a rotated token takes effect without a
    restart. Build one with ``WebhookAuth.bearer`` or ``WebhookAuth.api_key``.

    Attributes:
        kind: ``bearer`` sends ``Authorization: Bearer <token>``. ``api_key`` sends the
            token as it is, in ``header``.
        token_env: The environment variable that holds the token, in UPPER_SNAKE case.
        header: The header ``api_key`` sends.
        token: Returns the token at each call, in place of ``token_env``.
    """

    kind: Literal["bearer", "api_key"]
    token_env: Annotated[str, AfterValidator(_env_name)] = ""
    header: Annotated[str, AfterValidator(_auth_header)] = "Authorization"
    token: TokenProvider | None = None

    @model_validator(mode="after")
    def _one_token_source(self) -> Self:
        """Refuse a credential with no token source, or with two.

        Raises:
            ValueError: If neither or both of ``token_env`` and ``token`` are set.
        """
        if (self.token is None) == (not self.token_env):
            raise ValueError("webhook auth needs exactly one of token_env and token")
        return self

    @classmethod
    def bearer(cls, token_env: str = "", *, token: TokenProvider | None = None) -> Self:
        """Send ``Authorization: Bearer <token>``, from ``token_env`` or the ``token`` provider."""
        return cls("bearer", token_env, token=token)

    @classmethod
    def api_key(
        cls, token_env: str = "", header: str = "X-API-Key", *, token: TokenProvider | None = None
    ) -> Self:
        """Send the token as it is, in ``header``, from ``token_env`` or the ``token`` provider."""
        return cls("api_key", token_env, header, token)

    async def headers(self) -> dict[str, str]:
        """Return the header to send, with the token read now.

        Raises:
            ToolError: If the token is unset or empty, or the provider fails. The log says
                which.
        """
        if self.token is None:
            token = env_value(self.token_env, str) or ""
            if not token:
                logger.error("webhook auth: the environment variable %s is not set", self.token_env)
        else:
            try:
                token = self.token()
                if inspect.isawaitable(token):
                    token = await token
            except Exception:
                logger.exception("webhook auth: the token provider failed")
                token = ""
            if not token:
                logger.error("webhook auth: the token provider gave no token")
        if not token:
            raise ToolError("the webhook could not authenticate. Do not claim success.")
        return {self.header: f"Bearer {token}" if self.kind == "bearer" else token}


_DOC_SECTION = re.compile(r"\n\s*(?:Args|Arguments|Returns|Raises|Yields|Examples?):")


def _checked_url(url: str) -> str:
    """Refuse a URL with no scheme or host, or a placeholder in either."""
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"webhook url {url!r} needs a scheme and a host")
    if "{{" in f"{parts.scheme}://{parts.netloc}":
        raise ValueError(f"webhook url {url!r} may hold placeholders in its path and query only")
    return url


def _checked_headers(headers: dict[str, str]) -> dict[str, str]:
    """Refuse a header the library sets itself, or with a character that splits a request."""
    for name, value in headers.items():
        if not _HEADER_NAME.fullmatch(name) or any(c in value for c in "\r\n\0"):
            raise ValueError(f"webhook header {name!r} has a forbidden character")
        if name.lower() in PROTECTED_HEADERS:
            raise ValueError(f"webhook header {name!r} is set by the library")
    return headers


def _upper(method: object) -> object:
    """Read a method in any case: ``post`` is ``POST``. Refuse one that is not a method.

    Raises:
        ValueError: Naming the method and the ones allowed.
    """
    if not isinstance(method, str):
        return method
    if method.upper() not in METHODS:
        raise ValueError(f"webhook method {method!r} must be one of {sorted(METHODS)}")
    return method.upper()


class _WebhookConfig(BaseModel):
    """The options of one webhook, checked once when the agent starts."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", hide_input_in_errors=True, title="Webhook"
    )

    url: Annotated[str, AfterValidator(_checked_url)]
    method: Annotated[Method, BeforeValidator(_upper)] = "POST"
    headers: Annotated[
        dict[str, str], BeforeValidator(lambda v: v or {}), AfterValidator(_checked_headers)
    ] = {}
    hmac_secret: str | None = None
    timeout: Annotated[float, Field(ge=1, le=60)] = 10.0
    allow_private_network: bool = False
    auth: WebhookAuth | None = None
    body: Literal["raw", "envelope"] = "raw"
    tool_name: str = ""
    description: str = ""
    wait_for_response: bool = True

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Refuse options that cannot work together.

        Raises:
            ValueError: Naming the options and what to change.
        """
        if self.hmac_secret is not None and self.method == "GET":
            raise ValueError("a GET webhook has no body to sign. Use POST with hmac_secret")
        configured = {name.lower(): name for name in self.headers}
        if self.auth is not None and self.auth.header.lower() in configured:
            raise ValueError(f"webhook header {self.auth.header!r} is set by auth; remove it")
        if reason := check_destination(self.url, allow_http=self.allow_private_network):
            raise ValueError(f"webhook url {self.url!r}: {reason}")
        if self.auth is not None and "authorization" in configured:
            raise ValueError(
                f"webhook header {configured['authorization']!r} would send a second credential "
                "beside auth; remove it"
            )
        if self.body == "envelope" and self.method == "GET":
            raise ValueError("a GET webhook sends its arguments as the query: use body='raw'")
        if self.body == "envelope" and not self.tool_name:
            raise ValueError("an envelope webhook needs tool_name: build it with webhook_tool")
        return self

    @cached_property
    def query_names(self) -> frozenset[str]:
        """The query values the URL already sends."""
        return frozenset(
            name for name, _ in parse_qsl(urlsplit(self.url).query, keep_blank_values=True)
        )


class WebhookOptions(TypedDict, total=False):
    """The keyword options ``webhook_tool`` passes on to ``Webhook``.

    ``Webhook`` also takes ``description``, which ``webhook_tool`` sets from its own.
    """

    method: str
    headers: Mapping[str, str] | None
    hmac_secret: str | None
    timeout: float
    allow_private_network: bool
    auth: WebhookAuth | None
    body: Literal["raw", "envelope"]
    tool_name: str
    wait_for_response: bool


class Webhook:
    """Calls an HTTP endpoint with the arguments, and returns its JSON answer.

    Private, loopback and link-local addresses and plain ``http`` are refused unless
    ``allow_private_network`` is set. Redirects stay on the same origin, at most three.

    A webhook that does not wait sends in the background and answers the model at once.
    Its sends still in flight finish with ``drain``.
    """

    def __init__(
        self,
        url: str,
        *,
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
        hmac_secret: str | None = None,
        timeout: float = 10.0,
        allow_private_network: bool = False,
        auth: WebhookAuth | None = None,
        body: Literal["raw", "envelope"] = "raw",
        tool_name: str = "",
        description: str = "",
        wait_for_response: bool = True,
    ) -> None:
        """Check the endpoint once, when the agent starts.

        Args:
            url: The endpoint. ``{{name}}`` placeholders may fill the path and query values
                from the call state, never the scheme or host.
            method: GET sends the arguments as the query. The others send them as JSON.
            headers: Fixed headers. Put a secret in ``auth`` instead.
            hmac_secret: Signs the body: ``X-Signature-256`` is its HMAC-SHA256 in hex.
            timeout: Seconds for the whole call: DNS, connect, redirects and the answer.
            allow_private_network: Allow an internal service, or ``localhost`` while
                developing. Leave it off when someone else writes the URL.
            auth: The credential, read at each call.
            body: ``raw`` sends the arguments as the JSON object, as Unmute does.
                ``envelope`` wraps them: ``tool_name``, ``description``,
                ``arguments``, ``state.call`` as ``context``, and a ``timestamp``.
            tool_name: The tool's name, for the envelope. ``webhook_tool`` sets it.
            description: The tool's description, for the envelope.
            wait_for_response: False sends in the background, and the model reads
                ``{"ok": true, "accepted": true}`` at once: for a notification that
                should not hold the conversation. A failed send is only logged.

        Raises:
            ValueError: If an argument is unusable.
        """
        try:
            config = _WebhookConfig(
                url=url,
                method=method,  # ty: ignore[invalid-argument-type]
                headers=dict(headers or {}),
                hmac_secret=hmac_secret,
                timeout=timeout,
                allow_private_network=allow_private_network,
                auth=auth,
                body=body,
                tool_name=tool_name,
                description=description,
                wait_for_response=wait_for_response,
            )
        except ValidationError as error:
            raise next(
                (
                    own
                    for detail in error.errors()
                    if isinstance(own := detail.get("ctx", {}).get("error"), Exception)
                ),
                error,
            ) from None
        self.method: str = config.method
        self.headers = dict(config.headers)
        self.url = config.url
        self.query_names = config.query_names
        self.hmac_secret = config.hmac_secret
        self.timeout = config.timeout
        self.allow_private_network = config.allow_private_network
        self.auth = config.auth
        self.body = config.body
        self.tool_name = config.tool_name
        self.description = config.description
        self.wait_for_response = config.wait_for_response
        self._sending: set[asyncio.Task[None]] = set()

    def check_ready(self, state: CallState | None) -> None:
        """Refuse the call before anything is said or sent while the URL reads an unset value.

        Raises:
            ToolRefused: Naming the values that are not set.
        """
        fill_text(self.url, state)

    def _payload(
        self,
        arguments: dict[str, Any],
        state: CallState | None,
        context: Mapping[str, object] | None,
    ) -> Mapping[str, Any]:
        """Return the JSON body: the arguments, or the envelope around them.

        The envelope's ``context`` is what the call carries (``state.call``), and on an
        event tool its ``trigger_event`` and ``transcript``.
        """
        if self.body == "raw":
            return arguments
        call = state.call.model_dump(mode="json", exclude_none=True) if state is not None else {}
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "arguments": arguments,
            "context": call | dict(context or {}),
            "timestamp": time.time(),
        }

    def _request(
        self,
        arguments: dict[str, Any],
        state: CallState | None,
        context: Mapping[str, object] | None = None,
    ) -> tuple[str, bytes]:
        """Return the URL and body to send. GET puts the arguments in the query."""
        url = fill_text(self.url, state, lambda text: quote(text, safe=""))
        if self.method != "GET":
            try:
                body = json.dumps(
                    self._payload(arguments, state, context),
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                    default=str,
                )
            except ValueError:
                raise ToolError("webhook arguments could not be sent as JSON") from None
            return url, body.encode()
        if clash := sorted(self.query_names & arguments.keys()):
            raise ToolError(
                f"the argument {clash[0]!r} clashes with the {clash[0]!r} value the webhook "
                "URL already sends"
            )
        query = urlencode(
            sorted(
                (name, value if isinstance(value, str) else json.dumps(value))
                for name, value in arguments.items()
            )
        )
        if query:
            url += ("&" if urlsplit(url).query else "?") + query
        return url, b""

    async def run(
        self,
        arguments: dict[str, Any],
        state: CallState | None,
        *,
        context: Mapping[str, object] | None = None,
    ) -> object:
        """Send the request and return the parsed JSON answer.

        An empty answer reads ``{"ok": true}``. A webhook that does not wait returns
        ``ACCEPTED`` once the request is built, and sends it in the background.

        Args:
            arguments: The arguments, locked values filled.
            state: The call state, for the URL's placeholders.
            context: An event tool's event and transcript, which the envelope carries.

        Raises:
            ToolError: If the request is refused, fails, times out, answers with a status
                outside 2xx, or answers with something other than JSON.
        """
        url, body = self._request(arguments, state, context)
        if self.wait_for_response:
            return await self._send(url, body)
        task = asyncio.create_task(self._deliver(url, body))
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)
        return dict(ACCEPTED)

    async def _deliver(self, url: str, body: bytes) -> None:
        """Send a request nobody waits for. A failure is logged, never raised."""
        try:
            await self._send(url, body)
        except ToolError as error:
            logger.warning("webhook %s: a send that did not wait failed: %s", self.tool_name, error)

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for the sends still in flight, at most ``timeout`` seconds. Call it at the end."""
        if self._sending:
            await asyncio.wait(set(self._sending), timeout=timeout)

    async def _send(self, url: str, body: bytes) -> object:
        """Send one request and return the parsed JSON answer."""
        headers = dict(self.headers)
        if self.auth is not None:
            headers |= await self.auth.headers()
        if body:
            headers["Content-Type"] = "application/json"
        if self.hmac_secret is not None:
            digest = hmac.new(self.hmac_secret.encode(), body, hashlib.sha256)
            headers["X-Signature-256"] = digest.hexdigest()
        if len(url) + len(body) + sum(len(k) + len(v) for k, v in headers.items()) > (
            MAX_REQUEST_BYTES
        ):
            raise ToolError("webhook request exceeded the size limit. Send fewer or shorter values")
        # ponytail: one client per call, so no pool outlives an event loop. A shared client
        # saves loading the TLS trust store if tool latency ever shows it.
        async with httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as client:
            try:
                response, answer = await send_pinned_request(
                    client,
                    method=self.method,
                    url=url,
                    headers=headers,
                    content=body,
                    allow_http=self.allow_private_network,
                    allow_private=self.allow_private_network,
                    timeout_seconds=self.timeout,
                    max_redirects=MAX_REDIRECTS,
                    max_response_bytes=MAX_RESPONSE_BYTES,
                )
            except HardenedHttpError as error:
                raise ToolError(f"webhook {error}") from None
        if not response.is_success:
            short = f"webhook returned HTTP {response.status_code}"
            excerpt = _error_excerpt(answer)
            told = f"{short}: {excerpt}" if excerpt else short
            raise ToolError(f"{told}. Do not claim success", event_text=short)
        if not answer.strip():
            return {"ok": True}
        try:
            return json.loads(answer)
        except ValueError:
            raise ToolError("webhook returned invalid JSON. Do not claim success") from None


def _error_excerpt(answer: bytes) -> str:
    """Return what a failed webhook said, at most 300 characters, for the model to read."""
    text = answer.decode("utf-8", errors="replace")
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    if isinstance(body, dict):
        error = body.get("error")
        sources = [error, body] if isinstance(error, dict) else [body]
        parts = [
            str(value)
            for source in sources
            for key in ("error", "message", "code")
            if isinstance(value := source.get(key), str | int) and not isinstance(value, bool)
        ]
        text = "; ".join(dict.fromkeys(parts))
    return " ".join(text.split())[:300]


async def _call(handler: Callable[..., Any], *args: Any, **kwargs: Any) -> object:
    """Await an async handler. Run a plain one in a thread, so it cannot block the audio."""
    if inspect.iscoroutinefunction(handler):
        result = await handler(*args, **kwargs)
    else:
        result = await asyncio.to_thread(handler, *args, **kwargs)
    return result.model_dump(mode="json") if isinstance(result, BaseModel) else result


def _validated[M: BaseModel](model: type[M], arguments: dict[str, Any]) -> M:
    """Validate arguments into a model, worded for the model like ``day: Input should ...``."""
    try:
        return model.model_validate(arguments)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "arguments"
        raise ToolError(f"{where}: {first['msg']}") from None


class Done(BaseModel):
    """The output of a tool that returns no data, only that it ran: ``{"ok": true}``.

    A webhook that answers with an empty body returns it.
    """

    ok: Literal[True] = True


class Accepted(Done):
    """The output of a webhook that does not wait for its answer: ``{"ok": true, "accepted": true}``."""

    accepted: Literal[True] = True


def _is_model(annotation: object) -> bool:
    """Return True when an annotation is a Pydantic model class."""
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _argument_field(
    handler: str, parameter: inspect.Parameter, hints: Mapping[str, Any]
) -> tuple[Any, Any]:
    """Return the ``create_model`` field of one handler parameter: its type and default.

    Raises:
        TypeError: If the parameter is ``*args`` or ``**kwargs``, or has no type hint.
    """
    if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
        raise TypeError(f"{handler}: *args and **kwargs cannot be tool arguments")
    if parameter.name not in hints:
        raise TypeError(
            f"{handler}: give the parameter {parameter.name!r} a type, like "
            f"`{parameter.name}: Annotated[str, Field(description=...)]`. The model reads it"
        )
    default = ... if parameter.default is parameter.empty else parameter.default
    return hints[parameter.name], default


def _output_of(handler: str, returns: object, output: type[BaseModel] | None) -> type[BaseModel]:
    """Return a handler's output model: its return annotation, or the one given.

    Raises:
        TypeError: If neither names a Pydantic model, or the two differ.
    """
    if output is not None and _is_model(returns) and returns is not output:
        raise TypeError(
            f"{handler}: returns {getattr(returns, '__name__', returns)} but output= is "
            f"{output.__name__}. Keep one"
        )
    found = output if output is not None else returns
    if not _is_model(found):
        raise TypeError(
            f"{handler}: name its output: return a Pydantic model, like `-> Order`, or pass "
            "output=Order when it returns a dict"
        )
    return found  # ty: ignore[invalid-return-type]


class LocalHandler:
    """Calls a Python function with the checked arguments, as Unmute's local tools do.

    A function takes its arguments as keywords, or as one Pydantic model when that model
    is its only parameter.
    """

    def __init__(
        self, handler: Callable[..., Any], *, output: type[BaseModel] | None = None
    ) -> None:
        """Read the function's input and output from its signature.

        Args:
            handler: A function, sync or async. Every parameter has a type hint. A lone
                parameter typed as a Pydantic model is the whole input.
            output: The output model, for a function that returns a dict. By default its
                return annotation.

        Raises:
            TypeError: If a parameter has no type hint, the function takes ``*args`` or
                ``**kwargs``, or it names no output model.
        """
        self.name: str = getattr(handler, "__name__", "handler")
        hints = get_type_hints(handler, include_extras=True)
        parameters = list(inspect.signature(handler).parameters.values())
        self.handler = handler
        self.output = _output_of(self.name, hints.get("return"), output)
        lone = hints.get(parameters[0].name) if len(parameters) == 1 else None
        self._whole = _is_model(lone)
        if self._whole:
            self.arguments: type[BaseModel] = lone  # ty: ignore[invalid-assignment]
        else:
            fields: dict[str, Any] = {
                parameter.name: _argument_field(self.name, parameter, hints)
                for parameter in parameters
            }
            self.arguments = create_model(f"{self.name}_arguments", **fields)

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return the arguments as the function reads them, as JSON data."""
        return _validated(self.arguments, arguments).model_dump(mode="json")

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        """Validate the arguments into their Python types and call the function."""
        values = _validated(self.arguments, arguments)
        if self._whole:
            return await _call(self.handler, values)
        return await _call(self.handler, **dict(values))


def webhook_tool(
    name: str,
    description: str,
    url: str,
    *,
    input: type[BaseModel],
    output: type[BaseModel],
    locked: Mapping[str, object] | None = None,
    announce: str | Sequence[str] = (),
    effect: ToolEffect = "returns_data",
    interruption: ToolInterruption = "provider_default",
    announce_wait: bool = False,
    show_result: bool = True,
    result_instructions: str = "",
    hidden: Sequence[str] = (),
    **webhook: Unpack[WebhookOptions],
) -> ToolSpec:
    """Build a tool that calls a webhook.

    Args:
        name: The tool name.
        description: What it does, for the model.
        url: The endpoint. See ``Webhook``.
        input: The Pydantic model of the arguments. Its JSON Schema is what the model sees,
            and a ``Field(description=...)`` tells the model what an argument is.
        output: The Pydantic model of the answer. ``Done`` for a webhook that answers with
            an empty body, ``Accepted`` for one that does not wait for its answer.
        locked: Arguments filled from the call state, hidden from the model.
        announce: A line said as the tool starts. See ``ToolSpec``.
        effect: ``ends_conversation`` ends the call once the tool succeeds.
        interruption: What the caller's speech does to a running call.
        announce_wait: Run the tool once the announce line has played. See ``ToolSpec``.
        show_result: False hides a result that succeeded from the model. See ``ToolSpec``.
        result_instructions: How the model should use the result. See ``ToolSpec``.
        hidden: Result fields the model never reads. See ``ToolSpec``.
        **webhook: ``method``, ``headers``, ``hmac_secret``, ``timeout``,
            ``allow_private_network``, ``auth``, ``body`` and ``wait_for_response``, as
            ``Webhook`` takes them.

    Returns:
        The tool.

    Raises:
        ValueError: If a GET webhook's argument has the name of a query value in its URL,
            or a webhook that does not wait for its answer has an output other than
            ``Accepted``.
    """
    if webhook.get("wait_for_response") is False and not (
        _is_model(output) and issubclass(output, Accepted)
    ):
        raise ValueError(
            f"{name}: a webhook that does not wait for its answer returns "
            '{"ok": true, "accepted": true}. Pass output=Accepted'
        )
    parameters = _schema_of(input)
    if webhook.get("body") == "envelope":
        webhook.setdefault("tool_name", name)
        endpoint = Webhook(url, description=description, **webhook)
    else:
        endpoint = Webhook(url, **webhook)
    if endpoint.method == "GET" and (
        clash := sorted(
            endpoint.query_names & (parameters["properties"].keys() | (locked or {}).keys())
        )
    ):
        raise ValueError(
            f"{name}: the argument {clash[0]!r} is also a query value of the webhook URL. "
            "Rename it or drop it from the URL"
        )
    return ToolSpec(
        name,
        description,
        parameters,
        output,
        endpoint,
        dict(locked or {}),
        announce,
        effect,
        interruption,
        announce_wait=announce_wait,
        show_result=show_result,
        result_instructions=result_instructions,
        hidden=tuple(hidden),
    )


def local_tool(
    handler: Callable[..., Any],
    *,
    output: type[BaseModel] | None = None,
    name: str | None = None,
    description: str | None = None,
    locked: Mapping[str, object] | None = None,
    announce: str | Sequence[str] = (),
    effect: ToolEffect = "returns_data",
    interruption: ToolInterruption = "provider_default",
    announce_wait: bool = False,
    show_result: bool = True,
    result_instructions: str = "",
    hidden: Sequence[str] = (),
) -> ToolSpec:
    """Build a tool from a Python function. Works as a plain ``@local_tool`` decorator.

    The input schema comes from the signature, and every parameter needs a type. Describe
    an argument with ``Annotated[str, Field(description=...)]``. A function whose only
    parameter is a Pydantic model takes that model as its whole input. A locked argument
    is a parameter too.

    The output is the Pydantic model the function returns, from its return annotation
    (``-> Order``). A function that returns a dict names it with ``output=``. A result
    that does not fit fails the call. Raise ``ToolError`` for a failure the model should
    read.

    Args:
        handler: The function.
        output: The output model, for a function that returns a dict.
        name: The tool name. Defaults to the function name.
        description: What it does. Defaults to the docstring, up to its ``Args:`` section.
        locked: Arguments filled from the call state, hidden from the model.
        announce: A line said as the tool starts. See ``ToolSpec``.
        effect: ``ends_conversation`` ends the call once the tool succeeds.
        interruption: What the caller's speech does to a running call.
        announce_wait: Run the tool once the announce line has played. See ``ToolSpec``.
        show_result: False hides a result that succeeded from the model. See ``ToolSpec``.
        result_instructions: How the model should use the result. See ``ToolSpec``.
        hidden: Result fields the model never reads. See ``ToolSpec``.

    Returns:
        The tool.

    Raises:
        TypeError: If a parameter has no type, or the function names no output model.
    """
    executor = LocalHandler(handler, output=output)
    if description is None:
        description = _DOC_SECTION.split(inspect.getdoc(handler) or "")[0].strip()
    return ToolSpec(
        name or executor.name,
        description,
        _schema_of(executor.arguments),
        executor.output,
        executor,
        dict(locked or {}),
        announce,
        effect,
        interruption,
        announce_wait=announce_wait,
        show_result=show_result,
        result_instructions=result_instructions,
        hidden=tuple(hidden),
    )


def _schema_of(model: type[BaseModel]) -> dict[str, Any]:
    """Return a model's JSON Schema, with ``properties`` even when it has none."""
    return {"properties": {}, **model.model_json_schema()}
