"""An end-call tool whose goodbye rides in the reply that ends the call.

The model says its goodbye and calls the tool in one reply, so the call ends after one
model request. A second request only for the goodbye would cost the caller a second wait.
"""

from collections.abc import Callable
from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.services.llm_service import FunctionCallParams, LLMService

from unmute_core.call import AGENT_ENDED, GOODBYE_LINE, CallHooks
from unmute_core.pipecat._realtime import speak, speaks_itself
from unmute_core.pipecat._tools import answer_ending, mark_ender

END_CALL = "end_call"

END_CALL_DESCRIPTION = """
Ends the current call and disconnects immediately.

Call when:
- The user clearly indicates they are done (e.g., “that’s all, bye”).

Do not call when:
- The user asks to pause, hold, or transfer.
- Intent is unclear.

This is the final action the agent can take.
Once called, no further interaction is possible with the user.
Don't generate any other text or response when the tool is called.
"""  # noqa: RUF001 - LiveKit's own curly quotes, kept so the model reads the same bytes
"""What LiveKit's ``EndCallTool`` tells the model first, word for word (LiveKit Agents 1.8.7,
``livekit/agents/beta/tools/end_call.py``). The same text on both frameworks gives the same
choice: without it, Gemini says its goodbye and never calls the tool.
"""


def reply_text(llm: LLMService[Any]) -> Callable[[], str]:
    """Follow the LLM's replies, and return what reads the newest one's text.

    ``end_call_tool`` needs it to know whether the reply that ends the call said its
    goodbye. ``UnmuteFlows`` follows replies itself: pass ``lambda: flows.reply_text``.

    Args:
        llm: The call's LLM service.

    Returns:
        A function that returns the text of the reply streaming now, or of the last one.
    """
    reply: list[str] = []

    async def follow(_llm: object, frame: Frame) -> None:
        """Start over at each reply, and keep its text."""
        if isinstance(frame, LLMFullResponseStartFrame):
            reply.clear()
        elif isinstance(frame, LLMTextFrame):
            reply.append(frame.text)

    llm.add_event_handler("on_before_push_frame", follow)
    return lambda: "".join(reply).strip()


def end_call_tool(
    extra_description: str,
    goodbye: str | None,
    reply_text: Callable[[], str],
    call: CallHooks | None = None,
) -> FunctionSchema:
    """Return an ``end_call`` tool that ends the call without asking the model again.

    Give the tool to ``UnmuteFlows`` in ``native_tools``. The call ends once the goodbye has
    played: the worker drains the pipeline before it stops.

    Args:
        extra_description: What the model reads after ``END_CALL_DESCRIPTION``, as in
            LiveKit's ``EndCallTool``: when this agent ends a call, and to say its goodbye in
            the same reply as the call.
        goodbye: The line spoken when the reply that ends the call has no words, in the
            call's language. None says ``GOODBYE_LINE``, in English.
        reply_text: Returns the text of the reply that called the tool. Pass
            ``lambda: flows.reply_text``, which works before ``flows`` is built, or
            ``reply_text(llm)`` without ``UnmuteFlows``.
        call: The call's hooks. The end is noted as the agent's own (``AGENT_ENDED``), so
            the call ends with that reason and a later failure plays no line.
    """
    line = goodbye if goodbye is not None else GOODBYE_LINE

    async def handler(params: FunctionCallParams) -> None:
        """Answer with no model reply, say the goodbye if the reply had none, then end."""
        await answer_ending(params, {"ended": True})
        if not reply_text() and speaks_itself(params.llm):
            await speak(params.llm, line, wait=True)
        elif not reply_text():
            await params.llm.push_frame(TTSSpeakFrame(line))
        if call is not None:
            call.end_requested(AGENT_ENDED)
        await params.pipeline_worker.end()

    description = f"{END_CALL_DESCRIPTION}\n{extra_description}"
    return FunctionSchema(END_CALL, description, {}, [], handler=mark_ender(handler))
