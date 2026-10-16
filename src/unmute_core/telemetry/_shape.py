"""What the vendor sinks share: rewrite a copy of a span, and read the chat text a span carries."""

import json
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Self

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


def with_attributes(span: ReadableSpan, attributes: Mapping[str, Any]) -> ReadableSpan:
    """Return a copy of an ended span that carries ``attributes`` in place of its own.

    The span itself is not touched, so another sink still gets the original.
    """
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=dict(attributes),
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class MappingExporter(SpanExporter):
    """An exporter that rewrites a copy of every span, then hands the copies to another one.

    A subclass says how in ``shape``. Until a delegate is set, nothing is flushed or shut down.
    """

    def __init__(self, delegate: SpanExporter | None = None) -> None:
        """Keep the exporter that sends the rewritten spans."""
        self._delegate = delegate

    @abstractmethod
    def shape(self, span: ReadableSpan) -> ReadableSpan:
        """Return the copy of one ended span that goes out."""

    def shape_all(self, spans: Sequence[ReadableSpan]) -> list[ReadableSpan]:
        """Return the copy of each span that goes out. A subclass may do it in one pass."""
        return [self.shape(span) for span in spans]

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Rewrite a copy of every span, then export the copies."""
        if self._delegate is None:
            return SpanExportResult.SUCCESS
        return self._delegate.export(self.shape_all(spans))

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Flush the wrapped exporter, if there is one."""
        return self._delegate.force_flush(timeout_millis) if self._delegate else True

    def shutdown(self) -> None:
        """Shut the wrapped exporter down, if there is one."""
        if self._delegate:
            self._delegate.shutdown()


class ChatMessage(BaseModel):
    """One OpenAI-format chat message, read back from the JSON a span carries.

    Every key is kept in the order it came in, so ``as_dict`` writes the message as it was.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Return one key of the message, or ``default`` when it has none."""
        return (self.model_extra or {}).get(key, default)

    def as_dict(self) -> dict[str, Any]:
        """Return the message as the dict it was read from."""
        return dict(self.model_extra or {})

    @property
    def role(self) -> Any:
        """The message's ``role``, or None."""
        return self.get("role")

    @property
    def content(self) -> Any:
        """The message's ``content``: text, a list of parts, or None."""
        return self.get("content")

    @property
    def tool_call_id(self) -> Any:
        """The ``tool_call_id`` of a tool result, or None."""
        return self.get("tool_call_id")

    @property
    def tool_functions(self) -> list[Mapping[str, Any]]:
        """The ``function`` of each tool call the message made, ``{}`` for a call without one."""
        calls = self.get("tool_calls") or []
        return [call.get("function") or {} for call in calls]

    @property
    def text(self) -> str:
        """The plain text, whether the content is a string or a list of parts."""
        content = self.content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type", "text") == "text"
            )
        return str(content or "")


class AssistantReply(ChatMessage):
    """A chat message whose role is ``assistant``: the model's reply, with its tool calls."""

    @model_validator(mode="after")
    def _is_assistant(self) -> Self:
        """Refuse a message from any other role."""
        if self.role != "assistant":
            raise ValueError(f"expected an assistant message, not role {self.role!r}")
        return self


_Entry = Annotated[ChatMessage | object, Field(union_mode="left_to_right")]
_MESSAGES = TypeAdapter(list[_Entry])


def read_json_list[T](adapter: TypeAdapter[list[T]], value: object) -> list[T]:
    """Read a JSON list with ``adapter``, or return ``[]`` when the text is not one.

    Pydantic's JSON parser refuses text that Python's accepts, like a string holding a lone
    surrogate escape. That text is read with ``json.loads`` and checked as Python data.
    """
    text = str(value or "[]")
    try:
        return adapter.validate_json(text)
    except ValueError:
        pass
    try:
        return adapter.validate_python(json.loads(text))
    except ValueError:
        return []


def chat_messages(value: object) -> list[ChatMessage]:
    """Return the OpenAI-format messages in a span's ``input.value``, or none when it is not."""
    entries = read_json_list(_MESSAGES, value)
    return [entry for entry in entries if isinstance(entry, ChatMessage)]


def assistant_reply(value: object) -> AssistantReply | None:
    """Return a model's output as an assistant message, or None when it is plain text."""
    text = str(value)
    if not text.startswith("{"):
        return None
    try:
        return AssistantReply.model_validate_json(text)
    except ValueError:
        return None
