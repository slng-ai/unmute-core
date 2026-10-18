"""Tools the model can call, the same on every framework.

A ``ToolSpec`` holds what the model sees, the Pydantic model of its result, and a
``ToolExecutor`` that runs the call. ``ToolRegistry`` gives adapters the schemas for the
model request and one ``call`` for every tool call. A result that does not fit its output
model fails the call. A failure the model should hear is a ``ToolError``, which each
adapter maps to its framework's error. A call that reads a value not set yet, or leaves a
required argument out or blank, is refused with ``{"refused": ...}`` before anything is
said or sent. ``replies_to`` says when no model reply follows such a refusal.

What a result does on the main agent sits on the registry: ``Assign`` rows save its values
into the State, and a ``ToolReply`` is a line code says in place of a model reply.
"""

from unmute_core.tools._announce import LinePicker
from unmute_core.tools._builtins import current_datetime_tool, user_phone_number_tool
from unmute_core.tools._executors import (
    Accepted,
    Done,
    LocalHandler,
    Webhook,
    WebhookAuth,
    WebhookOptions,
    local_tool,
    webhook_tool,
)
from unmute_core.tools._lines import Assign, SpokenLines, ToolReply
from unmute_core.tools._mcp import McpServer, McpServerError, McpTransport
from unmute_core.tools._save import (
    SAVE_VALUES_TOOL,
    free_values,
    is_values,
    values_block,
    values_tool,
)
from unmute_core.tools._schema import ParameterSchemaError, check_parameters
from unmute_core.tools._spec import (
    TRANSCRIPT_MAX_BYTES,
    ArgumentsMissing,
    ToolEffect,
    ToolError,
    ToolEventSink,
    ToolExecutor,
    ToolHooks,
    ToolInterruption,
    ToolRefused,
    ToolRegistry,
    ToolRun,
    ToolSpec,
    ToolStarted,
    Transcript,
    check_tool_names,
    replies_to,
)
from unmute_core.tools._spoken import spoken

__all__ = [
    "SAVE_VALUES_TOOL",
    "TRANSCRIPT_MAX_BYTES",
    "Accepted",
    "ArgumentsMissing",
    "Assign",
    "Done",
    "LinePicker",
    "LocalHandler",
    "McpServer",
    "McpServerError",
    "McpTransport",
    "ParameterSchemaError",
    "SpokenLines",
    "ToolEffect",
    "ToolError",
    "ToolEventSink",
    "ToolExecutor",
    "ToolHooks",
    "ToolInterruption",
    "ToolRefused",
    "ToolRegistry",
    "ToolReply",
    "ToolRun",
    "ToolSpec",
    "ToolStarted",
    "Transcript",
    "Webhook",
    "WebhookAuth",
    "WebhookOptions",
    "check_parameters",
    "check_tool_names",
    "current_datetime_tool",
    "free_values",
    "is_values",
    "local_tool",
    "replies_to",
    "spoken",
    "user_phone_number_tool",
    "values_block",
    "values_tool",
    "webhook_tool",
]
