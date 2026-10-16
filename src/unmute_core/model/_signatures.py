"""Gemini thought signatures on the tool calls a request replays.

Gemini 3 signs the first tool call of each reply with an ``extra_content`` of
``{"google": {"thought_signature": "..."}}``. It refuses a request with a 400 when a
function call step of the current turn comes back without one. For a call Gemini did not
make, Google documents the placeholder ``skip_thought_signature_validator``.
"""

from collections.abc import Mapping
from typing import Any

SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"
"""Google's documented placeholder signature, for a call Gemini did not sign."""
EXTRA_CONTENT = "extra_content"
"""The key of a tool call's provider data in the OpenAI-compatible format."""
REPLAYED_PROVIDERS = ("google", "livekit", "xai")
"""The ``extra_content`` keys a replayed tool call carries back, on both adapters.

LiveKit 1.8 sends back only these keys (llm/_provider_format/openai.py:11), so Pipecat's
``ModelCallLLMService`` keeps to the same rule.
"""


def is_gemini(model: str) -> bool:
    """Return whether a model name names a Gemini model, like ``google/gemini-3-flash``."""
    return "gemini" in model.lower()


def takes_one_system_message(model: str) -> bool:
    """Return whether a model takes one system message: a Gemini id, after any prefix.

    The id after any ``openai/`` and vendor prefix starts with ``gemini``, like
    ``google/gemini-3-flash``. The host is no sign of it: a Google endpoint serves other
    models too.
    """
    return model.removeprefix("openai/").rsplit("/", 1)[-1].lower().startswith("gemini")


def has_signature(extra: object) -> bool:
    """Return whether a tool call's ``extra_content`` holds a Google thought signature."""
    google = extra.get("google") if isinstance(extra, Mapping) else None
    return isinstance(google, Mapping) and bool(google.get("thought_signature"))


def replayed_extra(extra: Mapping[str, Any]) -> dict[str, Any]:
    """Return the provider data a replayed tool call sends back.

    Only the keys of ``REPLAYED_PROVIDERS`` that hold something go back. A call with none
    of them sends no ``extra_content`` at all.

    Args:
        extra: The provider data the call came with, as ``ToolCall.extra`` holds it.
    """
    return {key: extra[key] for key in REPLAYED_PROVIDERS if extra.get(key)}


def signed_tool_calls(messages: list[dict[str, Any]], *, gemini: bool) -> list[dict[str, Any]]:
    """Return the messages with a signature on the first tool call of each assistant message.

    A first call with no Google signature gets ``SKIP_THOUGHT_SIGNATURE``, its other
    provider data kept. Nothing changes unless ``gemini`` is true or a call already holds
    a Google signature, so a request to any other provider goes as it was.

    Args:
        messages: The chat in OpenAI format.
        gemini: The request goes to a Gemini model.

    Returns:
        A new list. Messages that need no placeholder are the same objects.
    """
    calls = (call for message in messages for call in _tool_calls(message))
    if not gemini and not any(has_signature(call.get(EXTRA_CONTENT)) for call in calls):
        return messages
    return [_signed(message) for message in messages]


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Return an assistant message's tool calls, or none for any other message."""
    calls = message.get("tool_calls") if message.get("role") == "assistant" else None
    return [call for call in calls or [] if isinstance(call, dict)]


def _signed(message: dict[str, Any]) -> dict[str, Any]:
    """Return a message whose first tool call carries a signature, the placeholder if need be."""
    calls = _tool_calls(message)
    if not calls or has_signature(calls[0].get(EXTRA_CONTENT)):
        return message
    first, *rest = calls
    extra = first.get(EXTRA_CONTENT)
    extra = dict(extra) if isinstance(extra, Mapping) else {}
    google = extra.get("google")
    extra["google"] = {
        **(google if isinstance(google, Mapping) else {}),
        "thought_signature": SKIP_THOUGHT_SIGNATURE,
    }
    return {**message, "tool_calls": [{**first, EXTRA_CONTENT: extra}, *rest]}
