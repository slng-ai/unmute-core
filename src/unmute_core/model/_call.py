"""The one plug point for a model request, and its native implementation."""

from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from dataclasses import field, replace
from typing import Annotated, Any, Literal, Protocol

from pydantic import PlainValidator, SkipValidation, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.model._prompt import fill_messages

ToolChoice = Literal["auto", "required", "none"]
"""Whether the model may, must, or must not call a tool."""

TOOL_CHOICES: tuple[str, ...] = ("auto", "required", "none")
_TOOL_CHOICE = TypeAdapter(ToolChoice)

SUMMARY_SITE = "summary"
"""The prompt site of a summary request."""
FINISH_TOOL_NAME = "finish"
"""The tool a task step saves its values with. Its calls never cross to another agent."""
STEP_START_CUE = "(The call continues.)"
"""The caller line a fresh step opens on when it has nothing else. Gemini refuses a request
with no contents. Adapters never count it as the caller's, carry it, or pass it on."""


def tool_choice(value: object) -> ToolChoice | None:
    """Read a framework's tool choice setting, keeping only the three plain ones."""
    try:
        return _TOOL_CHOICE.validate_python(value, strict=True)
    except ValidationError:
        return None


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class ModelRequest:
    """One request to the model, with the prompt still unfilled.

    Attributes:
        messages: The chat in OpenAI chat format. Placeholders like ``{{name}}`` are unfilled.
        tools: The tools the model may call, as OpenAI function tool schemas.
        site: ``""`` for the main prompt, or ``task:<name>`` while a task runs.
        variables: Placeholder name to its text, from ``template_variables``.
        headers: Extra HTTP headers for this request only, like ``traceparent``.
        tool_choice: ``required`` makes the model call a tool, ``none`` forbids one. None
            leaves it to the model.
        model: The name of the model that answers, like ``gpt-4.1-mini``, for a
            ``before_model`` hook to read. Under ``FallbackModelCall`` it is the first
            call's. ``""`` when unknown.
        params: Extra body fields for this request only, like ``{"temperature": 0}``.
            ``RouterModelCall`` sends them. A framework's own LLM gets them as
            ``extra_body`` when it is OpenAI compatible.
    """

    messages: SkipValidation[list[dict[str, Any]]]
    tools: SkipValidation[list[dict[str, Any]]] = field(default_factory=list)
    site: str = ""
    variables: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    tool_choice: ToolChoice | None = None
    model: str = ""
    params: SkipValidation[dict[str, Any]] = field(default_factory=dict)


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class TextDelta:
    """The next piece of the answer's text.

    Attributes:
        text: The piece of text.
        response_id: The id the provider gave the answer, like ``chatcmpl-...``, or None.
        request_id: The provider's ``x-request-id`` for the HTTP request, or None.
    """

    text: str
    response_id: str | None = None
    request_id: str | None = None


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class ToolCall:
    """One complete tool call. Arguments are the JSON text the model wrote.

    Attributes:
        id: The call id the model gave.
        name: The tool's name.
        arguments: The JSON text the model wrote.
        extra: Provider data sent with the call, in the OpenAI-compatible
            ``extra_content`` form, like Gemini's
            ``{"google": {"thought_signature": "..."}}``. Adapters keep it in the history
            and send it back with the call, as the provider asks. Empty when there is none.
        response_id: The id the provider gave the answer, like ``chatcmpl-...``, or None.
        request_id: The provider's ``x-request-id`` for the HTTP request, or None.
    """

    id: str
    name: str
    arguments: str
    extra: Mapping[str, Any] = field(default_factory=dict, hash=False)
    response_id: str | None = None
    request_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        """Return the call as an OpenAI ``tool_calls`` entry, without ``extra``.

        The dict is what a span records as a call the model chose.
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class Usage:
    """How many tokens a request used, sent once at the end of the answer, when known.

    The counts pass through unchecked, as a provider sends them: a provider that leaves
    one null must not fail the stream at its last chunk.

    Attributes:
        input_tokens: The prompt's tokens, cached ones included.
        output_tokens: The answer's tokens.
        cached_input_tokens: How many input tokens the provider served from its prompt
            cache, or None when it did not say.
        response_id: The id the provider gave the answer, like ``chatcmpl-...``, or None.
        request_id: The provider's id for the HTTP request, its ``x-request-id``
            response header, or None. Support tickets and provider logs use it.
        reasoning_tokens: How many output tokens the provider spent on hidden reasoning,
            or None when it did not say.
        total_tokens: The provider's own total, or None when it did not say. It can be
            more than input plus output, for example when a provider counts thinking
            separately.
    """

    input_tokens: SkipValidation[int]
    output_tokens: SkipValidation[int]
    cached_input_tokens: SkipValidation[int | None] = None
    response_id: str | None = None
    request_id: str | None = None
    reasoning_tokens: SkipValidation[int | None] = None
    total_tokens: SkipValidation[int | None] = None


ModelEvent = TextDelta | ToolCall | Usage
"""What a model call streams. A consumer skips a ``Usage`` it has no use for."""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class ModelAnswer:
    """What a whole model request answered.

    Attributes:
        text: Every text piece, joined.
        calls: The tool calls, in the order the model made them.
        usage: The token counts, when the model gave them.
    """

    text: str = ""
    calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None


async def collect(events: AsyncIterable[ModelEvent]) -> ModelAnswer:
    """Read a model call's events to the end and return the whole answer.

    Args:
        events: What a ``ModelCall`` streams, or the same through the ``wrap_model`` hooks.
    """
    text: list[str] = []
    calls: list[ToolCall] = []
    usage: Usage | None = None
    async for event in events:
        if isinstance(event, TextDelta):
            text.append(event.text)
        elif isinstance(event, ToolCall):
            calls.append(event)
        elif isinstance(event, Usage):
            usage = event
        else:  # own text: a ModelCall streamed something that is not a model event
            raise TypeError(
                f"a model call streamed {type(event).__name__}; stream TextDelta, ToolCall "
                "or Usage events"
            )
    return ModelAnswer("".join(text), tuple(calls), usage)


class ModelCall(Protocol):
    """Sends a request to a model and streams the answer back.

    This is the one plug point for model requests. Adapters call it. A new way to reach a
    model is a new implementation of it. A call may also have the members of
    ``NamedModelCall``, and ``aclose``. All of them are optional.
    """

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Stream the model's text and tool calls for one request."""
        ...


def _streams(value: object) -> object:
    """Pass a value that has a ``stream`` method, as a ``ModelCall`` does."""
    if not callable(getattr(value, "stream", None)):
        raise ValueError(f"needs a ModelCall with a stream method, got {type(value).__name__}")
    return value


ModelCallArg = Annotated[ModelCall, PlainValidator(_streams)]
"""A ``ModelCall`` in a validated signature: it keeps its identity and must have ``stream``."""


class NativeModelCall:
    """A framework's own LLM. The library fills the prompt, the LLM does the rest.

    The adapter turns the framework LLM into a stream function, for example around LiveKit's
    ``llm.chat()``. The LLM keeps its own client, settings and retries.
    """

    def __init__(self, stream: Callable[[ModelRequest], AsyncIterator[ModelEvent]]) -> None:
        """Wrap the adapter's stream function.

        Args:
            stream: Sends a filled request to the framework LLM and yields its events.
        """
        self._stream = stream

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Fill the prompt with the request's variables, then stream the framework LLM."""
        filled = replace(
            request, messages=fill_messages(request.messages, request.variables), variables={}
        )
        async for event in self._stream(filled):
            yield event


class NamedModelCall(Protocol):
    """What a model call may tell about itself: the model, who serves it, its limits.

    Every member is optional. A framework LLM carries the same names. ``model_name``,
    ``provider_name``, ``retries_itself`` and ``variable_key_max`` read each one when it is
    there and fall back when it is not.
    """

    @property
    def model(self) -> str:
        """The model name, like ``gpt-4.1-mini``."""
        ...

    @property
    def provider(self) -> str:
        """Who serves the model, like ``slng``."""
        ...

    @property
    def retries_itself(self) -> bool:
        """True when the call retries and fails over on its own."""
        ...

    @property
    def variable_key_max(self) -> int:
        """The longest placeholder key the call takes, when it fills the prompt itself."""
        ...

    @property
    def stop_codes(self) -> frozenset[str]:
        """Error codes after which ``FallbackModelCall`` tries no other call."""
        ...


def model_name(model: object) -> str:
    """Return the model name a model call or a framework LLM sends, or ``unknown``."""
    return str(getattr(model, "model", "") or "unknown")


def answered_model(model: object) -> str:
    """Return the model name that answered the last request, read after it ends.

    A model call that may hand a request to another model, like ``FallbackModelCall``, names
    the one that answered in ``answered``. Any other names itself.
    """
    answered = getattr(model, "answered", "")
    return answered if isinstance(answered, str) and answered else model_name(model)


def provider_name(model: object) -> str:
    """Return who serves a model call or a framework LLM, like ``slng``, or ``unknown``."""
    return str(getattr(model, "provider", "") or "unknown")


def retries_itself(model: object) -> bool:
    """Return True when a model call retries and fails over on its own.

    An adapter that retries too would walk the whole chain again, so it skips its own retry.
    """
    return bool(getattr(model, "retries_itself", False))


def variable_key_max(*models: object) -> int | None:
    """Return the tightest placeholder key limit of the model calls, or None for no limit.

    A call that fills the prompt on its own side, like the router, names its limit in
    ``variable_key_max``. A call the library fills for has none.
    """
    limits = [
        limit
        for model in models
        if isinstance(limit := getattr(model, "variable_key_max", None), int)
    ]
    return min(limits, default=None)


async def close_model(model: object) -> None:
    """Await ``model.aclose()`` when it has one, and do nothing otherwise."""
    if callable(close := getattr(model, "aclose", None)):
        await close()
