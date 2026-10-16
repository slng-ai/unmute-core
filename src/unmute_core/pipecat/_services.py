"""What the observers read from a Pipecat service and from the frames that name one."""

import weakref
from collections.abc import Callable, Hashable
from types import CellType
from typing import Any, cast

from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.ai_service import AIService
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from unmute_core.model import STEP_START_CUE


def llm_settings(service: LLMService[Any]) -> LLMSettings:
    """Return an LLM service's settings, typed as the LLM's: ``settings`` names the base class."""
    return cast(LLMSettings, service.settings)


_KEEPS_RUNNING: weakref.WeakSet[Callable[..., object]] = weakref.WeakSet()
"""The handlers of the library's tools that run on through the caller's speech."""


def mark_keeps_running[H: Callable[..., object]](handler: H) -> H:
    """Mark a library tool handler as one that keeps running when the caller speaks."""
    _KEEPS_RUNNING.add(handler)
    return handler


def _cell_value(cell: CellType) -> object:
    """Return what a closure cell holds, or None for a cell not yet filled."""
    try:
        return cell.cell_contents
    except ValueError:
        return None


def _is_library_handler(handler: object) -> bool:
    """Return True for a library handler, or a Flows transition function that wraps one."""
    cells = (handler, *(_cell_value(cell) for cell in getattr(handler, "__closure__", None) or ()))
    return any(isinstance(cell, Hashable) and cell in _KEEPS_RUNNING for cell in cells)


def hide_async_guidance(llm: object) -> None:
    """Keep Pipecat from adding its "ASYNC TOOLS" text for the library's own tools.

    Pipecat adds the text when a tool runs with ``cancel_on_interruption=False``, which a
    library tool with ``interruption="continue"`` does, a transfer among them. The tool
    still runs on after the caller speaks. The text teaches the model a protocol of its
    own, and the same agent on LiveKit reads the plain prompt. So the service skips those
    tools when it asks whether an async tool is registered, and its instructions are
    composed again. A tool of the user's own that keeps running still adds the text.

    Args:
        llm: The call's LLM service. Calling it again changes nothing. Anything that is
            not a Pipecat ``LLMService`` is left alone.
    """
    if not isinstance(llm, LLMService):
        return

    def has_async_tools() -> bool:
        """Return True when a tool that is not the library's keeps running."""
        return any(
            not item.cancel_on_interruption and not _is_library_handler(item.handler)
            for item in llm._functions.values()
        )

    llm._has_async_tools = has_async_tools  # ty: ignore[invalid-assignment]
    llm._compose_system_instruction()


def service_setting(service: AIService, name: str) -> str | None:
    """Return one of a service's settings when it is set to a string."""
    value = getattr(service.settings, name, None)
    return value if isinstance(value, str) and value else None


def service_provider(service: AIService) -> str:
    """Return the provider a service class names, like ``slng`` for ``SlngTTSService``."""
    name = type(service).__name__
    return name.removesuffix("TTSService").removesuffix("STTService").lower() or "unknown"


def typed_text(frame: LLMMessagesAppendFrame) -> list[str]:
    """Return the text of each caller message an append carries. The step cue is none."""
    return [
        content
        for message in frame.messages
        if isinstance(message, dict) and message.get("role") == "user"
        if isinstance(content := message.get("content"), str)
        and content.strip()
        and content != STEP_START_CUE
    ]
