"""A ``ToolRegistry`` as LiveKit tools, for an agent of your own.

``UnmuteAgent`` builds its tools itself. ``function_tools`` gives the same tools to any
LiveKit ``Agent``: each one runs through the registry, says its announce line as it starts,
saves what the registry's ``assign`` names, speaks a registry ``ToolReply`` in place of a
model reply, and ends the call when its result says so.
"""

from collections.abc import Awaitable
from typing import Any

from livekit.agents import llm
from livekit.agents.voice import RunContext

from unmute_core.call import AGENT_ENDED, CallHooks
from unmute_core.livekit._calls import (
    announce,
    asks_no_reply,
    end_call_now,
    raw_tool,
    say,
    tool_text,
)
from unmute_core.state import CallState
from unmute_core.tools import LinePicker, SpokenLines, ToolHooks, ToolRegistry, ToolSpec


def function_tools(
    registry: ToolRegistry,
    state: CallState | None = None,
    *,
    hooks: ToolHooks | None = None,
    language: str = "en",
    timezone: str | None = None,
) -> list[llm.RawFunctionTool]:
    """Return the registry's tools as LiveKit tools. Build them per call.

    Pass them to ``Agent(tools=...)``.

    Args:
        registry: The agent's tools.
        state: The call state a locked argument reads. None takes the state of ``hooks``
            when it is a ``CallHooks``.
        hooks: The tool hooks to run around each call, like the call's ``CallHooks``. A
            ``CallHooks`` also drains the registry's webhooks at the call's end.
        language: The call's language, a BCP-47 tag like ``es-MX``. A reply line says the
            result's dates and times in it.
        timezone: Where the business is, an IANA name like ``Europe/Madrid``. A reply line
            then names a day from today: "tomorrow, October 9".

    Raises:
        ValueError: When the timezone is unknown.
    """
    if isinstance(hooks, CallHooks):
        hooks.watch(registry)
        state = state if state is not None else hooks.state
    pick = LinePicker()
    lines = SpokenLines(language, timezone)

    async def handler(arguments: dict[str, Any], context: RunContext) -> object:
        """Run one call. A tool that ends the call, or one a reply line answers, asks for no reply.

        So does a refusal for a missing argument the reply already asked for.
        """
        session = context.session

        def started(tool: ToolSpec) -> Awaitable[None] | None:
            """Say the tool's announce line as it starts."""
            if (line := pick.pick(f"tool:{tool.name}", tool.announce)) is not None:
                return announce(session, tool, line)
            return None

        name = context.function_call.name
        result = await registry.call(name, arguments, state, started=started, hooks=hooks)
        shown = tool_text(registry.for_model(name, result))
        if registry.ends_call(name, result):
            if isinstance(hooks, CallHooks):
                hooks.end_requested(AGENT_ENDED)
            end_call_now(session)
            return llm.ToolResult(shown, reply_required=False)
        if await asks_no_reply(result, context):
            return llm.ToolResult(shown, reply_required=False)
        if (line := registry.use_result(name, result, state, lines, hooks=hooks)) is not None:
            say(session, line)
            return llm.ToolResult(shown, reply_required=False)
        return shown

    return [
        raw_tool(
            schema["function"],
            handler,
            keep_running=registry.keeps_running(schema["function"]["name"]),
        )
        for schema in registry.schemas()
    ]
