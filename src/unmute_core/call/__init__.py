"""What happens around a call: values it starts with, its hooks, failure and the end."""

from unmute_core.call._goodbye import GOODBYE_LINE
from unmute_core.call._hooks import (
    AGENT_ENDED,
    CALLER_HUNG_UP,
    ENDED,
    ERROR,
    FAILURE_LINE,
    IDLE,
    MAX_DURATION,
    NOTES_HEADER,
    TRANSFERRED,
    CallEnd,
    CallHooks,
)
from unmute_core.call._start import CallStartError, CallStartSettings, save_call_start
from unmute_core.call._transcript import transcript_messages
from unmute_core.hooks import EventTool

__all__ = [
    "AGENT_ENDED",
    "CALLER_HUNG_UP",
    "ENDED",
    "ERROR",
    "FAILURE_LINE",
    "GOODBYE_LINE",
    "IDLE",
    "MAX_DURATION",
    "NOTES_HEADER",
    "TRANSFERRED",
    "CallEnd",
    "CallHooks",
    "CallStartError",
    "CallStartSettings",
    "EventTool",
    "save_call_start",
    "transcript_messages",
]
