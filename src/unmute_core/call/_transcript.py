"""The call's transcript, read from either framework's chat history."""

from collections.abc import Iterable
from typing import Any


def transcript_messages(
    messages: Iterable[object], *, drop_empty: bool = False
) -> list[dict[str, str]]:
    """Return what the caller and the agent said, as ``{"role", "content"}`` messages.

    Args:
        messages: A chat history in OpenAI format. Anything that is not a dict, any other
            role, and any content that is not text is left out.
        drop_empty: Also leave out a message whose text is empty.
    """
    kept: list[dict[str, Any]] = [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in ("user", "assistant")
        and isinstance(message.get("content"), str)
    ]
    return [
        {"role": message["role"], "content": message["content"]}
        for message in kept
        if message["content"] or not drop_empty
    ]
