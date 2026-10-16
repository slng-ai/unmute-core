"""Logfire's names for a span: a readable row label and the GenAI message attributes.

Ported from unmute@d70247b internal/generate/templates/telemetry/telemetry.py.tmpl:239-375.
The shared span contract does not carry these names. ``LogfireExporter`` adds them to a copy of
each span, so no other sink sees them.
"""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.util.types import AttributeValue

from unmute_core.telemetry._shape import (
    ChatMessage,
    MappingExporter,
    assistant_reply,
    chat_messages,
    with_attributes,
)

LOGFIRE_LABELS = MappingProxyType(
    {
        "chat": ("chat", "gen_ai.request.model"),
        "execute_tool": ("tool", "gen_ai.tool.name"),
        "stt": ("stt", "gen_ai.request.model"),
        "tts": ("tts", "gen_ai.request.model"),
    }
)
LINE_CHARS = 80


def _plain(value: object) -> str:
    """Return an attribute as text, unwrapping a JSON-encoded string."""
    text = "" if value is None else str(value)
    if text.startswith('"'):
        try:
            return str(json.loads(text))
        except ValueError:
            return text
    return text


def logfire_message(name: str, attributes: Mapping[str, Any]) -> str | None:
    """Return the line Logfire shows for a span, or None to keep the span's own name.

    A turn reads as what the caller said. A model request, tool call, transcript or speech
    reads as its operation and what it ran on, like Logfire's own integrations write
    ``chat gpt-4o``. The call reads as ``call <session id>``.

    Args:
        name: The span's name.
        attributes: The span's attributes.
    """
    if name == "turn":
        line = _plain(attributes.get("input.value"))
        return f"turn: {line[:LINE_CHARS]}" if line else "turn"
    label = LOGFIRE_LABELS.get(str(attributes.get("gen_ai.operation.name")))
    if label is not None:
        return f"{label[0]} {_plain(attributes.get(label[1]))}".strip()
    if name == "conversation":
        session = attributes.get("session.id") or attributes.get("gen_ai.conversation.id")
        return f"call {session}" if session else "call"
    return None


def _parts(message: ChatMessage) -> list[dict[str, Any]]:
    """Return one OpenAI-format message as GenAI message parts, tool calls and replies too."""
    if message.role == "tool":
        return [
            {
                "type": "tool_call_response",
                "id": message.tool_call_id,
                "response": message.content,
            }
        ]
    text = message.text
    texts = [{"type": "text", "content": text}] if text else []
    return texts + [
        {"type": "tool_call", "name": call.get("name"), "arguments": call.get("arguments")}
        for call in message.tool_functions
    ]


def _reply(value: object) -> dict[str, Any]:
    """Return the model's output as one assistant message: parsed when it is JSON with calls."""
    if parsed := assistant_reply(value):
        return {"role": "assistant", "parts": _parts(parsed)}
    return {"role": "assistant", "parts": [{"type": "text", "content": str(value)}]}


def genai_messages(attributes: Mapping[str, Any]) -> dict[str, AttributeValue]:
    """Return a model request's messages under the GenAI names Logfire's LLM panel reads.

    System messages go to ``gen_ai.system_instructions``. The other messages go to
    ``gen_ai.input.messages`` and the output to ``gen_ai.output.messages``. A span that is
    not a model request, or that has the messages already, gets nothing.

    Args:
        attributes: The span's attributes, with the neutral ``input.value`` and ``output.value``.
    """
    if attributes.get("gen_ai.operation.name") != "chat" or "gen_ai.input.messages" in attributes:
        return {}
    mapped: dict[str, AttributeValue] = {}
    messages = chat_messages(attributes.get("input.value"))
    system = [message for message in messages if message.role == "system"]
    if system:
        mapped["gen_ai.system_instructions"] = json.dumps(
            [{"type": "text", "content": message.text} for message in system]
        )
    others = [message for message in messages if message.role != "system"]
    if others:
        mapped["gen_ai.input.messages"] = json.dumps(
            [{"role": message.get("role", "user"), "parts": _parts(message)} for message in others]
        )
    if attributes.get("output.value"):
        mapped["gen_ai.output.messages"] = json.dumps([_reply(attributes["output.value"])])
    return mapped


def logfire_shape(span: ReadableSpan) -> ReadableSpan:
    """Return a copy of an ended span, labelled and mapped for Logfire. Nothing else moves."""
    attributes: dict[str, Any] = dict(span.attributes or {})
    attributes.update(genai_messages(attributes))
    if message := logfire_message(span.name, attributes):
        attributes["logfire.msg"] = message
    return with_attributes(span, attributes)


class LogfireExporter(MappingExporter):
    """Wraps the OTLP exporter and shapes a copy of each span the way Logfire reads it."""

    def shape(self, span: ReadableSpan) -> ReadableSpan:
        """Return the span, labelled and mapped for Logfire."""
        return logfire_shape(span)
