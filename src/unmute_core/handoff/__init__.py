"""Handoffs between agents: what the next agent hears, and the summary that can replace it.

Each adapter has its own ``hand_off``. It takes a ``HandoffContext``, and shapes its
framework's history by it. ``shape_history`` applies one to OpenAI-format messages, for any
framework. A task step's ``history`` takes the same values.
"""

from unmute_core.handoff._context import (
    SUMMARY_PREFIX,
    SUMMARY_PROMPT,
    SUMMARY_SITE,
    SUMMARY_TIMEOUT_SECONDS,
    HandoffContext,
    History,
    summary_message,
    summary_request,
    transcript,
)
from unmute_core.handoff._history import (
    STEP_START_MESSAGE,
    call_name,
    last_n,
    newest_caller,
    shape_history,
    shaped,
    speech_only,
    without_calls_to,
    without_instructions,
    without_text_signatures,
    without_unknown_calls,
)

__all__ = [
    "STEP_START_MESSAGE",
    "SUMMARY_PREFIX",
    "SUMMARY_PROMPT",
    "SUMMARY_SITE",
    "SUMMARY_TIMEOUT_SECONDS",
    "HandoffContext",
    "History",
    "call_name",
    "last_n",
    "newest_caller",
    "shape_history",
    "shaped",
    "speech_only",
    "summary_message",
    "summary_request",
    "transcript",
    "without_calls_to",
    "without_instructions",
    "without_text_signatures",
    "without_unknown_calls",
]
