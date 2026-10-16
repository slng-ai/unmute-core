"""How much of the call the next agent or step hears, on OpenAI-format message dicts.

``shape_history`` applies a ``HandoffContext`` to a chat in OpenAI's message format, so any
framework that keeps one can use it. A tool call and its reply always go together: a
provider refuses a call with no reply.
"""

import copy
from collections.abc import Collection, Sequence
from typing import Any

from unmute_core.handoff._context import HandoffContext
from unmute_core.model import FINISH_TOOL_NAME, STEP_START_CUE

STEP_START_MESSAGE = {"role": "user", "content": STEP_START_CUE}


def newest_caller(messages: Sequence[Any], *, skip_cue: bool) -> dict[str, Any] | None:
    """Return the caller's newest message in an OpenAI-format chat, or None.

    Args:
        messages: The chat. Anything that is not a dict is skipped.
        skip_cue: True passes over the step-start cue, which the caller never said, and
            returns the turn before it. False returns the cue when it is the newest.

    Returns:
        The message itself, not a copy.
    """
    return next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, dict)
            and message.get("role") == "user"
            and not (skip_cue and message.get("content") == STEP_START_CUE)
        ),
        None,
    )


def shape_history(
    messages: Sequence[Any],
    context: HandoffContext,
    *,
    tools: Collection[str] | None = None,
) -> list[Any]:
    """Return what the next agent hears of the call, by the handoff's ``history``.

    ``summary`` returns what was said. Put ``summary_message(text)`` in its place when a
    summary came back.

    Args:
        messages: The call so far, as OpenAI chat messages. Other items, like a provider's
            own message object, pass through.
        context: How much of the call crosses.
        tools: The names of the tools the next agent has. A call to any other tool and its
            reply are dropped. ``None`` keeps every call.

    Returns:
        A new list. The input is not changed.
    """
    return shaped(
        messages,
        context.history,
        max_messages=context.max_messages,
        include_tool_calls=context.include_tool_calls,
        tools=tools,
    )


def shaped(
    messages: Sequence[Any],
    history: str,
    *,
    max_messages: int | None = None,
    include_tool_calls: bool = True,
    tools: Collection[str] | None = None,
) -> list[Any]:
    """Return what an agent or a step hears of the call, by a ``History`` value.

    No system message crosses, no step-start cue and no earlier step's finish, and earlier
    replies' provider data stays behind, as on LiveKit. ``summary`` gives what was said. The
    caller puts the summary in its place when there is one.

    Args:
        messages: The call's messages.
        history: ``messages``, ``full``, ``last_n``, ``summary`` or ``reset``.
        max_messages: How many messages ``last_n`` keeps.
        include_tool_calls: ``full`` and ``last_n`` keep tool calls and their results.
        tools: The names of the tools the reader has. After the ``last_n`` window, a call to
            any other tool and its reply are dropped, as LiveKit does. ``None`` keeps every
            call.

    Returns:
        A new list. The input is not changed.
    """
    if history == "reset":
        return []
    kept = [
        message
        for message in without_calls_to(without_instructions(messages), {FINISH_TOOL_NAME})
        if message != STEP_START_MESSAGE
    ]
    if history in ("messages", "summary") or not include_tool_calls:
        kept = speech_only(kept)
    kept = last_n(kept, max_messages or 0) if history == "last_n" else copy.deepcopy(kept)
    if tools is not None:
        kept = without_unknown_calls(kept, tools)
    return without_text_signatures(kept)


def without_instructions(messages: Sequence[Any]) -> list[Any]:
    """Return the call without system and developer messages. A step reads its own prompt."""
    return [
        message
        for message in messages
        if not isinstance(message, dict) or message.get("role") in ("user", "assistant", "tool")
    ]


def speech_only(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Return what was said out loud: no tool call and no tool reply.

    An assistant turn that spoke and called a tool keeps what it said.
    """
    return [spoken for message in messages if (spoken := _spoken(message)) is not None]


def _without_key(message: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a copy of a message without one key."""
    return {k: value for k, value in message.items() if k != key}


def _spoken(message: Any) -> dict[str, Any] | None:
    """Return what a message said out loud, or None when it said nothing."""
    if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
        return None
    if not message.get("tool_calls"):
        return message
    spoken = _without_key(message, "tool_calls")
    return spoken if spoken.get("content") else None


def last_n(messages: Sequence[Any], limit: int) -> list[Any]:
    """Return the newest ``limit`` messages, without a tool reply whose call fell outside."""
    window = list(messages[-limit:]) if limit > 0 else []
    return next(
        (
            window[index:]
            for index, message in enumerate(window)
            if not (isinstance(message, dict) and message.get("role") == "tool")
        ),
        [],
    )


def without_text_signatures(messages: Sequence[Any]) -> list[Any]:
    """Drop provider data, like a Gemini thought signature, from earlier spoken turns.

    The same rule as the LiveKit adapter's: an earlier reply's reasoning carried into a step
    that never made it handed a verify step's caller back 13 of 32 times, against 0 of 32
    without. Only an assistant message's own ``extra_content`` goes. A tool call keeps its
    signature, because Gemini refuses a call replayed without it.

    """
    return [
        {key: value for key, value in message.items() if key != "extra_content"}
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("extra_content")
        else message
        for message in messages
    ]


def without_calls_to(messages: Sequence[Any], names: Collection[str]) -> list[Any]:
    """Return the messages without any call to ``names`` and its reply.

    A step that opens on the step before's finish reads it as its own work, already done.
    """
    dropped = {
        call.get("id")
        for message in messages
        if isinstance(message, dict)
        for call in message.get("tool_calls") or []
        if call_name(call) in names
    }
    return [
        message
        for item in messages
        for message in _without_calls(item, names)
        if not (
            isinstance(message, dict)
            and message.get("role") == "tool"
            and message.get("tool_call_id") in dropped
        )
    ]


def _without_calls(message: Any, names: Collection[str]) -> list[Any]:
    """Return the message with no call to ``names``, or nothing when it only made those."""
    if not isinstance(message, dict) or not message.get("tool_calls"):
        return [message]
    others = [call for call in message["tool_calls"] if call_name(call) not in names]
    if others:
        return [{**message, "tool_calls": others}]
    return [_without_key(message, "tool_calls")] if message.get("content") else []


def without_unknown_calls(messages: Sequence[Any], tools: Collection[str]) -> list[Any]:
    """Return the messages without a call to a tool outside ``tools``, and its reply.

    A provider refuses a call to a tool the request does not offer. An assistant turn that
    spoke and called an unknown tool keeps what it said.

    Args:
        messages: The chat, as OpenAI messages.
        tools: The names of the tools the reader has.
    """
    called = {
        name
        for message in messages
        if isinstance(message, dict)
        for call in message.get("tool_calls") or []
        if (name := call_name(call))
    }
    return without_calls_to(messages, called - set(tools))


def call_name(call: dict[str, Any]) -> str | None:
    """Return the function name of one ``tool_calls`` entry."""
    return (call.get("function") or {}).get("name")
