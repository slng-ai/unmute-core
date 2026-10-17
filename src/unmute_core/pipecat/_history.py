"""What a task step hears of the call, on Pipecat's OpenAI-format message dicts.

It follows unmute@4a738ad templates/pipecat_v1/agents.py.tmpl. A tool
call and its reply always go together: a provider refuses a call with no reply.
"""

import copy
from collections.abc import Sequence
from typing import Any

from unmute_core.handoff import (
    STEP_START_MESSAGE,
    call_name,
    last_n,
    newest_caller,
    shaped,
    speech_only,
    without_calls_to,
    without_instructions,
    without_text_signatures,
)
from unmute_core.tasks import TaskSpec

__all__ = [
    "STEP_START_MESSAGE",
    "below_replies",
    "caller_turns",
    "ends_with_tool_result",
    "last_n",
    "made_other_calls",
    "newest_caller_message",
    "set_tool_reply",
    "shape_history",
    "shaped",
    "speech_only",
    "tool_pair",
    "trailing_results",
    "without_call",
    "without_calls_to",
    "without_instructions",
    "without_text_signatures",
]


def ends_with_tool_result(messages: Sequence[Any]) -> bool:
    """Return True when a request answers a tool's result: its newest message past prompts."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") in ("system", "developer"):
            continue
        return message.get("role") == "tool"
    return False


def caller_turns(messages: Sequence[Any]) -> int:
    """Return how many caller turns the context holds.

    A provider's own message, like Gemini's thought signature (an ``LLMSpecificMessage``),
    is not a dict and never a caller turn.
    """
    return sum(
        1 for message in messages if isinstance(message, dict) and message.get("role") == "user"
    )


def shape_history(task: TaskSpec, messages: Sequence[Any]) -> list[Any]:
    """Return the messages a step opens on, from its ``history``.

    ``reset`` and ``caller_turn`` (core ``entry_history``) are the adapter's to apply.
    """
    return shaped(
        messages,
        task.history,
        max_messages=task.max_messages,
        include_tool_calls=task.include_tool_calls,
    )


def newest_caller_message(messages: Sequence[Any]) -> dict[str, Any] | None:
    """Return a copy of the caller's newest turn, or None."""
    newest = newest_caller(messages, skip_cue=False)
    return dict(newest) if newest is not None else None


def _is_tool_reply(message: Any) -> bool:
    """Return True for a tool's reply message."""
    return isinstance(message, dict) and message.get("role") == "tool"


def below_replies(messages: Sequence[Any], call_id: str, carried: Sequence[Any]) -> list[Any]:
    """Return ``messages`` with ``carried`` right after the tool replies to ``call_id``'s turn.

    This is LiveKit's order: the call, its result, then the caller's turn the run answered.
    A user message cannot sit between a call and its reply.
    """
    for index, message in enumerate(messages):
        if isinstance(message, dict) and any(
            call.get("id") == call_id for call in message.get("tool_calls") or []
        ):
            end = index + 1
            while end < len(messages) and _is_tool_reply(messages[end]):
                end += 1
            return [*messages[:end], *carried, *messages[end:]]
    return [*messages, *carried]


def without_call(messages: Sequence[Any], call_id: str) -> list[Any]:
    """Return the messages without the call ``call_id`` and its reply. Calls beside it stay."""
    kept: list[Any] = []
    for message in messages:
        if _is_tool_reply(message) and message.get("tool_call_id") == call_id:
            continue
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not calls or all(call.get("id") != call_id for call in calls):
            kept.append(message)
            continue
        others = [call for call in calls if call.get("id") != call_id]
        rest = {key: value for key, value in message.items() if key != "tool_calls"}
        if others:
            kept.append({**rest, "tool_calls": others})
        elif rest.get("content"):
            kept.append(rest)
    return kept


def set_tool_reply(messages: Sequence[Any], call_id: str, content: str) -> None:
    """Write ``content`` into the tool reply of ``call_id``, in place."""
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") == "tool"
            and message.get("tool_call_id") == call_id
        ):
            message["content"] = content


def made_other_calls(messages: Sequence[Any], call_id: str) -> bool:
    """Return True when the assistant turn that made ``call_id`` made other calls too."""
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if calls and any(call.get("id") == call_id for call in calls):
            return len(calls) > 1
    return False


def tool_pair(messages: Sequence[Any], call_id: str) -> list[dict[str, Any]]:
    """Return copies of the call ``call_id`` and its reply, with no other call beside it."""
    call = next(
        (
            message
            for message in messages
            if isinstance(message, dict)
            and any(item.get("id") == call_id for item in message.get("tool_calls") or [])
        ),
        None,
    )
    reply = next(
        (
            message
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "tool"
            and message.get("tool_call_id") == call_id
        ),
        None,
    )
    if call is None or reply is None:
        return []
    only = copy.deepcopy(call)
    only["tool_calls"] = [item for item in only["tool_calls"] if item.get("id") == call_id]
    return [only, copy.deepcopy(reply)]


def trailing_results(messages: Sequence[Any]) -> list[tuple[str, str]]:
    """Return the tool name and reply text of each tool reply a request ends on.

    System and developer messages after the replies are passed over.
    """
    names = {
        call.get("id"): call_name(call) or ""
        for message in messages
        if isinstance(message, dict)
        for call in message.get("tool_calls") or []
    }
    results: list[tuple[str, str]] = []
    for message in reversed(messages):
        role = message.get("role") if isinstance(message, dict) else None
        if role in ("system", "developer"):
            continue
        if role != "tool":
            break
        content = message.get("content")
        text = content if isinstance(content, str) else str(content)
        results.append((names.get(message.get("tool_call_id"), ""), text))
    return results[::-1]
