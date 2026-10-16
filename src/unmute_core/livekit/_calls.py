"""The pieces a LiveKit agent and its task steps share to say lines and run tools."""

import contextlib
import json
import logging
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from livekit.agents import NOT_GIVEN, AgentSession, NotGivenOr, llm
from livekit.agents.voice import FunctionToolsExecutedEvent, RunContext, SpeechHandle

from unmute_core.livekit._realtime import speak, speaks_itself
from unmute_core.tools import ToolError, ToolSpec

logger = logging.getLogger(__name__)


def tool_text(result: object) -> str:
    """Return a tool result as the model reads it: JSON, unless it is text already.

    LiveKit turns any other value into ``str(value)`` (llm/utils.py:1001), which is a
    Python repr for a dict.
    """
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def say(
    session: AgentSession,
    text: str,
    *,
    add_to_chat_ctx: bool = True,
    allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
) -> SpeechHandle | None:
    """Say a line without waiting for it. A line that cannot be spoken only logs.

    Waiting would hold the step until the line has played, which is the delay the line
    exists to cover.

    Args:
        session: The call's session.
        text: The line.
        add_to_chat_ctx: Whether the line joins the chat history.
        allow_interruptions: False for a line the caller may not cut off.

    Returns:
        The line's speech, to learn later whether it played to its end, or None.
    """
    try:
        return speak(
            session,
            text,
            add_to_chat_ctx=add_to_chat_ctx,
            allow_interruptions=allow_interruptions,
        )
    except Exception:
        logger.warning("task line could not be spoken", exc_info=True)
        return None


def end_call_now(session: AgentSession) -> None:
    """Close the session for a tool that ends the call, and ask for no reply to the reply's tools.

    A reply with a second tool beside the ending one would get a follow-up request from
    LiveKit once every tool ran: the ending tool's own ``reply_required=False`` does not
    cover its neighbour. The call's last tool event cancels the reply for all of them.
    """

    def cancel(event: FunctionToolsExecutedEvent) -> None:
        """Ask LiveKit for no reply to this round of tools."""
        event.cancel_tool_reply()

    session.once("function_tools_executed", cancel)
    session.shutdown()


def announce(session: AgentSession, tool: ToolSpec, line: str) -> Awaitable[None] | None:
    """Say a tool's announce line. With ``announce_wait``, return what waits for it to play.

    LiveKit marks the reply that called the tool as generated before the tool runs, so
    the line plays while the tool waits for it. A realtime model that speaks itself says
    nothing: it words its own line as it calls the tool, and a second reply with the
    line would come while the tool's result is still out.
    """
    if speaks_itself(session):
        return None
    waits = tool.announce_wait
    speech = say(session, line, allow_interruptions=False if waits else NOT_GIVEN)
    if speech is None or not waits:
        return None
    return speech.wait_for_playout()


def raw_tool(
    schema: dict[str, Any],
    handler: Callable[[dict[str, Any], RunContext], Coroutine[Any, Any, object]],
    *,
    flags: llm.ToolFlag = llm.ToolFlag.NONE,
    keep_running: bool = False,
) -> llm.RawFunctionTool:
    """Return a LiveKit tool with the schema as written and a core error as LiveKit's.

    Args:
        schema: ``name``, ``description`` and ``parameters``.
        handler: Runs one call with the model's arguments and LiveKit's run context.
        flags: LiveKit tool flags.
        keep_running: The caller's speech never cuts off the reply that called the tool.
            LiveKit lets a running tool finish when that reply is cut off, and once the
            interruption timeout passes (voice/speech_handle.py:16) it answers the model with
            a placeholder and delivers the result later (voice/generation.py:1255). The call
            turns interruptions off for the reply instead, so the result is the answer.
    """

    async def call(raw_arguments: dict[str, Any], context: RunContext) -> object:
        """Run the call. A core ``ToolError`` reaches the model as LiveKit's."""
        if keep_running:
            with contextlib.suppress(RuntimeError):
                context.disallow_interruptions()
        try:
            return await handler(raw_arguments, context)
        except ToolError as error:
            raise llm.ToolError(error.message) from None

    return llm.function_tool(call, raw_schema=schema, flags=flags)
