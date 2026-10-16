"""A tool the model calls when an outbound call reaches voicemail.

The model hears the greeting of a voicemail system and calls the tool. The tool ends the
call, after the message it was set to leave.
"""

import asyncio
import logging

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.llm_service import FunctionCallParams

from unmute_core.call import AGENT_ENDED, CallHooks
from unmute_core.pipecat._idle import IdleNudgeObserver
from unmute_core.pipecat._realtime import ask, speaks_itself
from unmute_core.pipecat._tools import answer_ending, played_after
from unmute_core.telephony import VOICEMAIL_MESSAGE, OnVoicemail

logger = logging.getLogger(__name__)

VOICEMAIL_TOOL = "detected_answering_machine"
VOICEMAIL_DESCRIPTION = (
    "Call this if you detect a voicemail system, after hearing the voicemail greeting."
)
VOICEMAIL_SETTLE_SECONDS = 0.5
"""The pause between the last word of the message and the hang-up."""


def voicemail_tool(
    *,
    on_voicemail: OnVoicemail = "hangup",
    message: str = VOICEMAIL_MESSAGE,
    idle: IdleNudgeObserver | None = None,
    call: CallHooks | None = None,
) -> FunctionSchema:
    """Return the ``detected_answering_machine`` tool, which ends the call at a voicemail.

    Give the tool to ``UnmuteFlows`` in ``native_tools``, or to the context's tools on a
    pipeline without it. The call ends as the agent's own (``AGENT_ENDED``), after the
    message with ``leave_message``, at once with ``hangup``. The tool asks for no reply of
    its own.

    Args:
        on_voicemail: ``hangup`` ends the call, or ``leave_message`` has the model say
            ``message`` first and wait until it has played, for at most 30 seconds. A
            message that fails to play is logged and the call still ends.
        message: What the model is told to say with ``leave_message``.
        idle: The call's idle nudges, held while the tool speaks, so a quiet answering
            machine draws no nudge.
        call: The call's hooks, which note the end as the agent's own.
    """

    async def handler(params: FunctionCallParams) -> None:
        """Answer with no reply, leave the message when asked, then end the call."""
        if idle is not None:
            idle.controller.suspend("voicemail")
        try:
            await answer_ending(params, {"status": "completed"})
            if on_voicemail == "leave_message":
                await _leave(params, message)
            await asyncio.sleep(VOICEMAIL_SETTLE_SECONDS)
            if call is not None:
                call.end_requested(AGENT_ENDED)
            await params.pipeline_worker.end()
        finally:
            if idle is not None:
                idle.controller.resume("voicemail")

    return FunctionSchema(VOICEMAIL_TOOL, VOICEMAIL_DESCRIPTION, {}, [], handler=handler)


async def _leave(params: FunctionCallParams, message: str) -> None:
    """Have the model say ``message`` and wait until it has played. A failure is logged."""
    try:
        if speaks_itself(params.llm):
            await ask(params.llm, message, wait=True)
            return
        frame = LLMMessagesAppendFrame(
            messages=[{"role": "system", "content": message}], run_llm=True
        )
        await played_after(params.llm, params.pipeline_worker.queue_frame(frame))
    except Exception:
        logger.exception("the voicemail message could not be left")
