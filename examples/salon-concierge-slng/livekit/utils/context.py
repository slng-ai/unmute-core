"""Shape the conversation a step or an agent is handed."""

from __future__ import annotations

from livekit.agents import RunContext, llm

from settings import logger


def _caller_turns(chat_ctx: llm.ChatContext) -> int:
    """Count the turns the caller has taken in this context.

    The one signal that separates "they asked again" from "the model re-read its
    own finished work". A step that has already run and returned cannot have a
    new request in front of it unless somebody spoke, so this is what each
    delegate's re-entry guard compares.

    Args:
        chat_ctx: The conversation to count in.

    Returns:
        The number of caller messages.
    """
    return sum(1 for message in chat_ctx.messages() if message.role == "user")


def _newest_caller_turn(chat_ctx: llm.ChatContext) -> llm.ChatMessage | None:
    """Find the caller turn a terminal tool call answered.

    Carried into the owner's restored context when the step ends on its tool,
    because the owner never saw it: the step consumed the turn and the owner's
    snapshot was taken before it arrived. Without it the owner reads a completed
    status with no request in front of it, and a compound request loses the half
    the step did not own.

    Args:
        chat_ctx: The step's conversation.

    Returns:
        The newest caller message, or None when the caller has not spoken.
    """
    return next(
        (
            item
            for item in reversed(chat_ctx.items)
            if isinstance(item, llm.ChatMessage) and item.role == "user"
        ),
        None,
    )


def _insert_carried_turn(
    owner_ctx: llm.ChatContext, carried: list[llm.ChatMessage | None]
) -> None:
    """Put the caller turn a step consumed back in front of the owner, once.

    Only the newest one: a group whose steps each carried a turn has already
    answered the earlier ones. Skipped when the owner already holds the item,
    which is what a shared group leaves behind.

    Args:
        owner_ctx: The owner's restored conversation.
        carried: The turn each step carried, oldest first, None where a step had none.
    """
    item = next((item for item in reversed(carried) if item is not None), None)
    if item is not None and owner_ctx.get_by_id(item.id) is None:
        owner_ctx.insert(item)
        logger.info("carried the caller's turn back to the owner")


def _sibling_call(ctx: RunContext, names: set[str]) -> bool:
    """Say whether this response also called one of the named tools.

    The framework adds every function call of a response to the speech handle
    before it starts any of their bodies, so a tool can see what it was called
    beside. Its own call is skipped by id.

    Args:
        ctx: The calling tool's run context.
        names: The tool names to look for.

    Returns:
        True when a different call in this response names one of them.
    """
    # Read by the type field, not isinstance: a call is anything the speech
    # handle lists with type "function_call", which is what the group smoke
    # hands it, and what this helper matched before it was a comprehension.
    return any(
        getattr(item, "type", None) == "function_call"
        and getattr(item, "call_id", None) != ctx.function_call.call_id
        and getattr(item, "name", None) in names
        for item in ctx.speech_handle.chat_items
    )
