"""The agents and the task steps they delegate to, with their tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
import time
from collections.abc import AsyncIterable
from typing import Annotated, Literal

from livekit import api, rtc
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentTask,
    ChatContext,
    FlushSentinel,
    ModelSettings,
    NotGivenOr,
    RunContext,
    function_tool,
    get_job_context,
    llm,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import TaskCompletedEvent, TaskGroup
from livekit.agents.llm import ChatChunk, Tool
from pydantic import Field

import tools.find_or_create_customer
import tools.find_slots
import tools.record_complaint
import tools.save_booking
from call_state import (
    STEP_RESULTS,
    CallState,
    StateRefused,
    _group_status,
    _task_status,
    _terminal_success,
)
from prompts import (
    COMPLAINT_SPECIALIST_PROMPT,
    CONCIERGE_PROMPT,
    MANAGE_BOOKING_PROMPT,
    VERIFY_CUSTOMER_PROMPT,
)
from session import _refusal, _render
from settings import _TASK_MAX_ATTEMPTS, _TERMINAL_WAIT_SECS, logger
from utils import dev_metrics, knowledge
from utils.context import (
    _caller_turns,
    _insert_carried_turn,
    _newest_caller_turn,
    _sibling_call,
)
from utils.dev_metrics import dev_llm_node, dev_say
from utils.telephony import _refer_uri

# --- announcements ---------------------------------------------------------

# A line a package writes as `announce:` is spoken by this code, not written by
# the model, so one sentence is the same sentence every time it fires. A live
# call on 2026-09-16 entered the booking flow twice inside a minute and played
# "Sure, let me get that sorted for you." both times. An author writing several
# alternatives gets one of them per firing, and never the one that site used
# last.
_ANNOUNCE_LAST: dict[str, str] = {}


def _announce(key: str, lines: list[str]) -> str:
    """Pick one of these lines, never the one this key used last.

    Args:
        key: The announcement site, so each site remembers its own last line.
        lines: The alternatives the author wrote for that site.

    Returns:
        The line to speak.
    """
    # ponytail: process wide, not per call. Two concurrent callers can only
    # affect each other's variety, which is the whole thing this is for.
    choices = [line for line in lines if line != _ANNOUNCE_LAST.get(key)] or lines
    pick = random.choice(choices)
    _ANNOUNCE_LAST[key] = pick
    return pick


class _GroupStop(Exception):
    """Internal signal that a group step ended unserved, so the group stops.

    Raised from the completion callback, which the framework runs inside its own
    try: the group completes with this exception and it comes back out of
    `await group`, with the results the group had reached.
    """

    def __init__(self, results: dict) -> None:
        """Carry the results the group reached.

        Args:
            results: Each finished step's result, by step id.
        """
        super().__init__()
        self.results = results


class _TaskTransfer(Exception):
    """Internal signal that a task handed the caller to another agent."""

    def __init__(self, agent: Agent) -> None:
        """Carry the agent the caller moves to.

        Args:
            agent: The target of the handoff.
        """
        super().__init__()
        self.agent = agent


async def _share_task_result(
    group: TaskGroup, event: TaskCompletedEvent, flow: dict
) -> None:
    """Fill LiveKit's empty task tool output before the next shared task.

    `flow` is the delegate's own record of the run: the results it has reached,
    and any caller turn a step consumed. The group's own results are not
    readable from here, and a group that stops early has to hand back what it
    did do.

    Args:
        group: The running group.
        event: The step that just completed.
        flow: The delegate's record of the run.

    Raises:
        RuntimeError: The completed step left no usable finish output.
    """
    flow["results"][event.task_id] = event.result
    flow["carried"].append(getattr(event.agent_task, "_carried_turn", None))
    finish_call_id = getattr(event.agent_task, "_finish_call_id", None)
    if finish_call_id is None:
        # A step can end without a finish call at all: a transfer out of it, or
        # a tool that ended it and could not record an id. There is no output of
        # ours to fill, and nothing to fail about.
        return
    task_output = next(
        (
            item
            for item in reversed(event.agent_task.chat_ctx.items)
            if isinstance(item, llm.FunctionCallOutput)
            and not item.is_error
            # Matched on the call id alone. The name used to be checked too, and
            # the only ending was `finish`; a step that ends on its own tool
            # produces an output under that tool's name and the same id.
            and item.call_id == finish_call_id
        ),
        None,
    )
    if task_output is None:
        raise RuntimeError("completed task has no matching successful finish output")

    shared_ctx = group.chat_ctx.copy()
    shared_output = shared_ctx.get_by_id(task_output.id)
    if shared_output is None:
        raise RuntimeError("completed task finish output is absent from its group")
    if not isinstance(shared_output, llm.FunctionCallOutput):
        raise RuntimeError("completed task finish output has an invalid type")
    if (
        shared_output.name != task_output.name
        or shared_output.call_id != task_output.call_id
        or shared_output.is_error
    ):
        raise RuntimeError("completed task finish output changed identity")

    shared_ctx.remove(shared_output)
    shared_ctx.insert(
        shared_output.model_copy(
            update={
                "output": json.dumps(
                    _task_status(event.result),
                    sort_keys=True,
                )
            }
        )
    )

    await group.update_chat_ctx(
        shared_ctx,
        exclude_invalid_function_calls=False,
    )
    if isinstance(event.result, dict) and event.result.get("unserved_request"):
        # The step could not serve what the caller asked. Running the next one
        # would answer a question nobody asked, so the group stops here and the
        # owner is handed the unserved status with the results so far.
        logger.info("group stopped: " + event.task_id + " ended unserved")
        raise _GroupStop(dict(flow["results"]))


# --- agents ----------------------------------------------------------------


class ComplaintSpecialist(Agent):
    """The complaint_specialist agent, compiled from its entry in agent.yaml."""

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Synthesize speech through the dev feed's wrapper.

        Args:
            text: The text to speak.
            model_settings: The framework's per-turn model settings.

        Returns:
            The synthesized audio frames.
        """
        return dev_metrics.dev_tts_node(self, text, model_settings)

    def llm_node(
        self, chat_ctx: ChatContext, tools: list[Tool], model_settings: ModelSettings
    ) -> AsyncIterable[ChatChunk | str | FlushSentinel]:
        """Ask the model, through the dev feed's wrapper.

        Args:
            chat_ctx: The conversation to send.
            tools: The tools the model may call.
            model_settings: The framework's per-turn model settings.

        Returns:
            The model's chunks.
        """
        return dev_llm_node(
            self, Agent.default.llm_node, chat_ctx, tools, model_settings
        )

    def __init__(self, chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN) -> None:
        """Build the agent.

        Args:
            chat_ctx: The conversation a handoff carries in.
        """
        super().__init__(
            instructions=COMPLAINT_SPECIALIST_PROMPT,
            chat_ctx=chat_ctx,
            tools=[
                *EndCallTool(
                    extra_description='End the call once the caller\'s own words say the call is over: "no, that\'s all", "I\'m good, thanks", "goodbye". A question is never one of those, however tired it sounds. On a live call on 2026-09-16 the caller asked "can we just recap when I have the next appointment?" and this ended the call on them. Answer what they asked and wait for them to say they are done.',
                    end_instructions="Say one short goodbye in your own words, then end the call. Do not offer anything else first.",
                ).tools
            ],
        )
        if isinstance(chat_ctx, llm.ChatContext):
            self._chat_ctx = chat_ctx.copy()

    async def _refresh_prompt(self) -> None:
        """Render explicit saved-value references after a task writes state."""
        await self.update_instructions(
            _render(
                COMPLAINT_SPECIALIST_PROMPT,
                self.session.userdata,
                site="agent:complaint_specialist",
            )
        )

    async def on_enter(self) -> None:
        """Open this agent's turn as it takes over the call."""
        # Rendered on the way in, and again by _refresh_prompt whenever a step
        # writes declared state: a call variable never changes, a declared value
        # does, and both reach this prompt through the same placeholders.
        await self.update_instructions(
            _render(
                COMPLAINT_SPECIALIST_PROMPT,
                self.session.userdata,
                site="agent:complaint_specialist",
            )
        )
        # This agent took over via handoff; let its own instructions drive the
        # opening (the prompt already says not to re-greet).
        # This opening turn withholds the agent's own handoffs (B3: an agent that can
        # hand the call back before it has said anything ping-pongs). The framework's
        # on-enter tool flag would hide them for the rest of the call instead, because
        # its filter follows the context of everything this reply starts: a live call
        # offered one specialist nothing but its delegate for ten turns while the
        # caller asked for another one (B: salon handoffs, 2026-08-20).
        self.session.generate_reply(
            tools=[t.id for t in self.tools if t.id != "to_concierge"]
        )

    @function_tool
    async def record_complaint(
        self,
        ctx: RunContext,
        requested_resolution: Annotated[
            str,
            Field(
                description="What the customer wants the salon to do, or an empty string"
            ),
        ],
        summary: Annotated[
            str, Field(description="Short factual summary of the complaint")
        ],
    ) -> dict:
        """Record the verified customer's complaint after enough facts are known. This does not replace a requested manager transfer."""
        refusal = _refusal(
            "record_complaint",
            ctx.userdata,
            [("customer_phone", "run verify_customer first.")],
        )
        if refusal:
            return {"refused": refusal}
        # Not awaited, on purpose: see the same line in webhook_tool.
        dev_say(self.session, "Let me get that written down.")
        result = tools.record_complaint.record_complaint(
            requested_resolution=requested_resolution,
            summary=summary,
            customer_phone=ctx.userdata.plain("customer_phone"),
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    @function_tool
    async def look_up_refund_policy(
        self,
        ctx: RunContext,
        query: Annotated[
            str, Field(description="What to look up. Use the caller's own words.")
        ],
    ) -> dict:
        """Look up the salon's refund and complaints policy. Use this before you state any refund, redo, timescale, or goodwill offer, so you quote the policy instead of guessing it. It covers the redo window, the three refund tiers, colour corrections, retail returns, and how a complaint is handled.

        The results may not answer the question. They are ordered best first, and each carries a relevance score where a higher number means a closer match. If nothing returned actually answers what was asked, say you do not have that information rather than offering the closest result.
        """
        return await knowledge.look_up("refunds", query)

    @function_tool
    async def to_concierge(self, ctx: RunContext) -> Agent:
        """The caller changes to a request owned by another specialist."""
        return Concierge(
            chat_ctx=llm.ChatContext(
                items=[
                    m
                    for m in self.chat_ctx.messages()
                    if m.role in ("user", "assistant")
                ]
            )
        )

    @function_tool(on_duplicate="reject")
    async def to_manager(self, ctx: RunContext) -> str | None:
        """The caller explicitly asks for a manager or is clearly and strongly frustrated."""
        # One line per firing, so a live test can see which control the model
        # picked and when — the transfer that ran is otherwise invisible in the
        # logs until the carrier request goes out.
        #
        # This tool returns a string only on the paths where the caller is still
        # here and the conversation continues. A function tool's return value is
        # fed back to the LLM, which then takes another turn: harmless while the
        # caller is listening, wrong once the session is over. After a warm merge
        # that turn would speak into a room holding the caller and the person we
        # just handed them to, right after saying goodbye.
        logger.info("human transfer fired: to_manager (cold)")
        # cold: SIP REFER through the job context. The caller leaves the room.
        job_ctx = get_job_context()
        # Pick the SIP caller by kind: identities are assigned at dispatch, and a
        # room can hold more than one participant.
        caller = next(
            (
                participant
                for participant in job_ctx.room.remote_participants.values()
                if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
            ),
            None,
        )
        if caller is None:
            # No SIP leg in the room, so there is nothing to refer. The usual
            # cause is a session that never arrived by phone: an Agent Console
            # run, a browser session, or a dispatch rule pointing at a different
            # agent. Cold acts on the caller's own leg, so unlike warm it cannot
            # be tested without a real inbound call.
            #
            # This branch logged nothing until 2026-08-12. On a live test that
            # day it fired, spoke a vague reply, and left an operator reading a
            # log that showed the tool firing and then simply stopping. The
            # earlier reasoning was that line 1 plus the absence of line 2 says
            # this on its own; absence is only a signal to somebody who already
            # knows to look for it.
            logger.info("cold transfer skipped: no phone caller in the room")
            return (
                "There is no phone caller to transfer, because this session did not "
                "arrive by phone. Tell the caller you cannot put them through and "
                "keep helping them."
            )
        # The tool speaks the announcement itself (V4/B4): leaving it to the
        # prompt lets a model say "putting you through" and never call this,
        # which a caller cannot tell apart from a transfer that failed.
        await dev_say(
            ctx.session,
            "Putting you through now, one moment.",
            allow_interruptions=False,
        )
        request = api.TransferSIPParticipantRequest(
            room_name=job_ctx.room.name,
            participant_identity=caller.identity,
            transfer_to=_refer_uri(os.environ["MANAGER_PHONE_NUMBER"]),
            play_dialtone=True,
        )
        request.ringing_timeout.FromNanoseconds(int(30 * 1e9))
        # The REFER goes out here. Logged before the call, so a request that never
        # returns is still visible as a phase the transfer reached.
        logger.info("cold transfer referring the caller out")
        started = time.monotonic()
        try:
            await job_ctx.api.sip.transfer_sip_participant(request)
        except Exception as error:  # noqa: BLE001 - any carrier failure is reported, never fatal
            # on_unavailable: hangup. Attempt the goodbye here, and always shut
            # down even if generating it fails or is cancelled.
            logger.info(
                "cold transfer failed after %ds: %s",
                int(time.monotonic() - started),
                error,
            )
            try:
                await ctx.session.generate_reply(
                    instructions="Tell the caller the transfer failed and say goodbye."
                )
            finally:
                ctx.session.shutdown()
            return None
        # A completed REFER takes the caller out of the room, which ends this
        # session on its own. Returning a value here would buy one LLM turn
        # spoken to nobody, racing the teardown.
        logger.info(
            "cold transfer completed after %ds", int(time.monotonic() - started)
        )
        return None

    @function_tool
    async def verify_customer(self, ctx: RunContext) -> dict:
        """Run this when the caller explicitly corrects their phone number, and when a complaint is about to be written down. Never before a booking: the `book` flow runs this step itself when the saved number is not confirmed. A change of date, time or service is not a phone correction. When this flow finishes it returns a status. Continue with the caller. Do not run this flow again for the same request. A completed status means the step finished. Read any saved values through your own prompt references. An unserved status means the step could not help. Ask the caller what they need, then use your tools or a handoff. The caller's words right above the status are what the step answered; it served the part it owns, so act only on what remains."""
        # Already ran, and nobody has spoken since. Then this is not a second
        # request, it is the model reading its own completed result as though it
        # were one, and running the whole thing again is the worst possible
        # answer: on a live call on 2026-09-16 (trace 917975e9) a booking saved,
        # the flow re-entered on the caller's own "3 o'clock" still sitting
        # above the result, read the diary twice more, and the last thing the
        # caller heard was an agent saying it was off to check the diary.
        #
        # Refused here rather than asked for in the prompt, which this package
        # did twice and which failed twice. The announcement is below this, so a
        # refused call speaks nothing at all.
        if _caller_turns(self.chat_ctx) == getattr(
            self, "_ran_at_verify_customer", None
        ):
            return {
                "refused": "This already ran and finished, and the caller has "
                "not spoken since, so its result is the answer to what they "
                "asked. Reply to them from the values in your prompt. Run it "
                "again only once they have asked for something new."
            }
        # Not awaited, for the same reason the tool announcement is not: awaiting
        # the handle holds the step until playout finishes, which is the delay
        # this line exists to hide.
        dev_say(
            self.session,
            _announce(
                "delegate:verify_customer",
                [
                    "Let me confirm who I'm speaking to.",
                    "One second, let me check who I've got.",
                    "Right, let me make sure I have you.",
                ],
            ),
        )
        owner_ctx = self.chat_ctx.copy()
        # Filled by a step that ended on its own tool, and read in the finally
        # below. A list because a group runs several steps and the last one to
        # carry a turn is the one the owner has not seen.
        _carried: list[llm.ChatMessage | None] = []
        try:
            _step = VerifyCustomer(
                chat_ctx=owner_ctx.copy(
                    exclude_instructions=True,
                    exclude_config_update=True,
                    exclude_handoff=True,
                )
            )
            result = await _step
            _carried.append(getattr(_step, "_carried_turn", None))
        finally:
            _insert_carried_turn(owner_ctx, _carried)
            await self.update_chat_ctx(owner_ctx, exclude_invalid_function_calls=False)
            await self._refresh_prompt()
            # Counted after the restore, so a turn a step consumed and carried
            # back is included: that turn is the one the owner is about to read,
            # and it is exactly the turn that must not be mistaken for a new
            # request. Set in the finally so a step that raised still blocks an
            # immediate retry the caller did not ask for.
            self._ran_at_verify_customer = _caller_turns(self.chat_ctx)
        dev_metrics.dev_task_returned(ctx, result)
        return _task_status(result)


class Concierge(Agent):
    """The concierge agent, compiled from its entry in agent.yaml."""

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Synthesize speech through the dev feed's wrapper.

        Args:
            text: The text to speak.
            model_settings: The framework's per-turn model settings.

        Returns:
            The synthesized audio frames.
        """
        return dev_metrics.dev_tts_node(self, text, model_settings)

    def llm_node(
        self, chat_ctx: ChatContext, tools: list[Tool], model_settings: ModelSettings
    ) -> AsyncIterable[ChatChunk | str | FlushSentinel]:
        """Ask the model, through the dev feed's wrapper.

        Args:
            chat_ctx: The conversation to send.
            tools: The tools the model may call.
            model_settings: The framework's per-turn model settings.

        Returns:
            The model's chunks.
        """
        return dev_llm_node(
            self, Agent.default.llm_node, chat_ctx, tools, model_settings
        )

    def __init__(
        self, chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN, initial: bool = False
    ) -> None:
        """Build the agent.

        Args:
            chat_ctx: The conversation a handoff carries in.
            initial: True for the call's first agent, which greets the caller.
        """
        self._initial = initial
        super().__init__(
            instructions=CONCIERGE_PROMPT,
            chat_ctx=chat_ctx,
            tools=[
                *EndCallTool(
                    extra_description='End the call once the caller\'s own words say the call is over: "no, that\'s all", "I\'m good, thanks", "goodbye". A question is never one of those, however tired it sounds. On a live call on 2026-09-16 the caller asked "can we just recap when I have the next appointment?" and this ended the call on them. Answer what they asked and wait for them to say they are done.',
                    end_instructions="Say one short goodbye in your own words, then end the call. Do not offer anything else first.",
                ).tools
            ],
        )
        if isinstance(chat_ctx, llm.ChatContext):
            self._chat_ctx = chat_ctx.copy()

    async def _refresh_prompt(self) -> None:
        """Render explicit saved-value references after a task writes state."""
        await self.update_instructions(
            _render(CONCIERGE_PROMPT, self.session.userdata, site="agent:concierge")
        )

    async def on_enter(self) -> None:
        """Open this agent's turn as it takes over the call."""
        # Rendered on the way in, and again by _refresh_prompt whenever a step
        # writes declared state: a call variable never changes, a declared value
        # does, and both reach this prompt through the same placeholders.
        await self.update_instructions(
            _render(CONCIERGE_PROMPT, self.session.userdata, site="agent:concierge")
        )
        if not self._initial:
            # This opening turn withholds the agent's own handoffs (B3: an agent that can
            # hand the call back before it has said anything ping-pongs). The framework's
            # on-enter tool flag would hide them for the rest of the call instead, because
            # its filter follows the context of everything this reply starts: a live call
            # offered one specialist nothing but its delegate for ten turns while the
            # caller asked for another one (B: salon handoffs, 2026-08-20).
            self.session.generate_reply(
                tools=[t.id for t in self.tools if t.id != "to_complaints"]
            )
            return
        pass  # the caller speaks first

    @function_tool
    async def look_up_salon_info(
        self,
        ctx: RunContext,
        query: Annotated[
            str, Field(description="What to look up. Use the caller's own words.")
        ],
    ) -> dict:
        """Look up anything about the salon itself: services and prices, how long things take, opening hours, the stylists and what each one does, booking, deposits, cancellation and lateness terms, payment methods, parking, and accessibility. Use this whenever the caller asks a question about how the salon works, not only about price. Quote the document rather than estimating.
        Prefer this over asking the caller for anything. A question about the salon is not a booking request: look it up and answer it, and only start verification if the caller actually wants to make, move, or cancel an appointment.

        The results may not answer the question. They are ordered best first, and each carries a relevance score where a higher number means a closer match. If nothing returned actually answers what was asked, say you do not have that information rather than offering the closest result.
        """  # noqa: D205
        return await knowledge.look_up("services", query)

    @function_tool
    async def to_complaints(self, ctx: RunContext) -> Agent:
        """The caller has a complaint or service problem."""
        return ComplaintSpecialist(
            chat_ctx=llm.ChatContext(
                items=[
                    m
                    for m in self.chat_ctx.messages()
                    if m.role in ("user", "assistant")
                ]
            )
        )

    @function_tool(on_duplicate="reject")
    async def to_manager(self, ctx: RunContext) -> str | None:
        """The caller explicitly asks for a manager or is clearly and strongly frustrated."""
        # One line per firing, so a live test can see which control the model
        # picked and when — the transfer that ran is otherwise invisible in the
        # logs until the carrier request goes out.
        #
        # This tool returns a string only on the paths where the caller is still
        # here and the conversation continues. A function tool's return value is
        # fed back to the LLM, which then takes another turn: harmless while the
        # caller is listening, wrong once the session is over. After a warm merge
        # that turn would speak into a room holding the caller and the person we
        # just handed them to, right after saying goodbye.
        logger.info("human transfer fired: to_manager (cold)")
        # cold: SIP REFER through the job context. The caller leaves the room.
        job_ctx = get_job_context()
        # Pick the SIP caller by kind: identities are assigned at dispatch, and a
        # room can hold more than one participant.
        caller = next(
            (
                participant
                for participant in job_ctx.room.remote_participants.values()
                if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
            ),
            None,
        )
        if caller is None:
            # No SIP leg in the room, so there is nothing to refer. The usual
            # cause is a session that never arrived by phone: an Agent Console
            # run, a browser session, or a dispatch rule pointing at a different
            # agent. Cold acts on the caller's own leg, so unlike warm it cannot
            # be tested without a real inbound call.
            #
            # This branch logged nothing until 2026-08-12. On a live test that
            # day it fired, spoke a vague reply, and left an operator reading a
            # log that showed the tool firing and then simply stopping. The
            # earlier reasoning was that line 1 plus the absence of line 2 says
            # this on its own; absence is only a signal to somebody who already
            # knows to look for it.
            logger.info("cold transfer skipped: no phone caller in the room")
            return (
                "There is no phone caller to transfer, because this session did not "
                "arrive by phone. Tell the caller you cannot put them through and "
                "keep helping them."
            )
        # The tool speaks the announcement itself (V4/B4): leaving it to the
        # prompt lets a model say "putting you through" and never call this,
        # which a caller cannot tell apart from a transfer that failed.
        await dev_say(
            ctx.session,
            "Putting you through now, one moment.",
            allow_interruptions=False,
        )
        request = api.TransferSIPParticipantRequest(
            room_name=job_ctx.room.name,
            participant_identity=caller.identity,
            transfer_to=_refer_uri(os.environ["MANAGER_PHONE_NUMBER"]),
            play_dialtone=True,
        )
        request.ringing_timeout.FromNanoseconds(int(30 * 1e9))
        # The REFER goes out here. Logged before the call, so a request that never
        # returns is still visible as a phase the transfer reached.
        logger.info("cold transfer referring the caller out")
        started = time.monotonic()
        try:
            await job_ctx.api.sip.transfer_sip_participant(request)
        except Exception as error:  # noqa: BLE001 - any carrier failure is reported, never fatal
            # on_unavailable: hangup. Attempt the goodbye here, and always shut
            # down even if generating it fails or is cancelled.
            logger.info(
                "cold transfer failed after %ds: %s",
                int(time.monotonic() - started),
                error,
            )
            try:
                await ctx.session.generate_reply(
                    instructions="Tell the caller the transfer failed and say goodbye."
                )
            finally:
                ctx.session.shutdown()
            return None
        # A completed REFER takes the caller out of the room, which ends this
        # session on its own. Returning a value here would buy one LLM turn
        # spoken to nobody, racing the teardown.
        logger.info(
            "cold transfer completed after %ds", int(time.monotonic() - started)
        )
        return None

    @function_tool
    async def verify_customer(self, ctx: RunContext) -> dict:
        """Run this when the caller explicitly corrects their phone number, and when a complaint is about to be written down. Never before a booking: the `book` flow runs this step itself when the saved number is not confirmed. A change of date, time or service is not a phone correction. When this flow finishes it returns a status. Continue with the caller. Do not run this flow again for the same request. A completed status means the step finished. Read any saved values through your own prompt references. An unserved status means the step could not help. Ask the caller what they need, then use your tools or a handoff. The caller's words right above the status are what the step answered; it served the part it owns, so act only on what remains."""
        # Already ran, and nobody has spoken since. Then this is not a second
        # request, it is the model reading its own completed result as though it
        # were one, and running the whole thing again is the worst possible
        # answer: on a live call on 2026-09-16 (trace 917975e9) a booking saved,
        # the flow re-entered on the caller's own "3 o'clock" still sitting
        # above the result, read the diary twice more, and the last thing the
        # caller heard was an agent saying it was off to check the diary.
        #
        # Refused here rather than asked for in the prompt, which this package
        # did twice and which failed twice. The announcement is below this, so a
        # refused call speaks nothing at all.
        if _caller_turns(self.chat_ctx) == getattr(
            self, "_ran_at_verify_customer", None
        ):
            return {
                "refused": "This already ran and finished, and the caller has "
                "not spoken since, so its result is the answer to what they "
                "asked. Reply to them from the values in your prompt. Run it "
                "again only once they have asked for something new."
            }
        # Not awaited, for the same reason the tool announcement is not: awaiting
        # the handle holds the step until playout finishes, which is the delay
        # this line exists to hide.
        dev_say(
            self.session,
            _announce(
                "delegate:verify_customer",
                [
                    "Let me confirm who I'm speaking to.",
                    "One second, let me check who I've got.",
                    "Right, let me make sure I have you.",
                ],
            ),
        )
        owner_ctx = self.chat_ctx.copy()
        # Filled by a step that ended on its own tool, and read in the finally
        # below. A list because a group runs several steps and the last one to
        # carry a turn is the one the owner has not seen.
        _carried: list[llm.ChatMessage | None] = []
        try:
            _step = VerifyCustomer(
                chat_ctx=owner_ctx.copy(
                    exclude_instructions=True,
                    exclude_config_update=True,
                    exclude_handoff=True,
                )
            )
            result = await _step
            _carried.append(getattr(_step, "_carried_turn", None))
        finally:
            _insert_carried_turn(owner_ctx, _carried)
            await self.update_chat_ctx(owner_ctx, exclude_invalid_function_calls=False)
            await self._refresh_prompt()
            # Counted after the restore, so a turn a step consumed and carried
            # back is included: that turn is the one the owner is about to read,
            # and it is exactly the turn that must not be mistaken for a new
            # request. Set in the finally so a step that raised still blocks an
            # immediate retry the caller did not ask for.
            self._ran_at_verify_customer = _caller_turns(self.chat_ctx)
        dev_metrics.dev_task_returned(ctx, result)
        return _task_status(result)

    @function_tool
    async def book(self, ctx: RunContext) -> dict | Agent:
        """The caller wants to create, move or cancel a booking. Includes a change to an appointment just made. When this flow finishes it returns a status. Continue with the caller. Do not run this flow again for the same request. A completed status means the step finished. Read any saved values through your own prompt references. An unserved status means the step could not help. Ask the caller what they need, then use your tools or a handoff. The caller's words right above the status are what the step answered; it served the part it owns, so act only on what remains."""
        # Already ran, and nobody has spoken since. Then this is not a second
        # request, it is the model reading its own completed result as though it
        # were one, and running the whole thing again is the worst possible
        # answer: on a live call on 2026-09-16 (trace 917975e9) a booking saved,
        # the flow re-entered on the caller's own "3 o'clock" still sitting
        # above the result, read the diary twice more, and the last thing the
        # caller heard was an agent saying it was off to check the diary.
        #
        # Refused here rather than asked for in the prompt, which this package
        # did twice and which failed twice. The announcement is below this, so a
        # refused call speaks nothing at all.
        if _caller_turns(self.chat_ctx) == getattr(self, "_ran_at_book", None):
            return {
                "refused": "This already ran and finished, and the caller has "
                "not spoken since, so its result is the answer to what they "
                "asked. Reply to them from the values in your prompt. Run it "
                "again only once they have asked for something new."
            }
        owner_ctx = self.chat_ctx.copy()
        # Filled by a step that ended on its own tool, and read in the finally
        # below. A list because a group runs several steps and the last one to
        # carry a turn is the one the owner has not seen.
        _carried: list[llm.ChatMessage | None] = []
        # What the group reached, recorded as it goes: a group that stops early
        # still owes the owner the steps it did run.
        _flow = {"results": {}, "carried": _carried}
        try:
            group = TaskGroup(
                summarize_chat_ctx=False,
                on_task_completed=lambda event: _share_task_result(group, event, _flow),
            )
            await group.update_chat_ctx(
                owner_ctx.copy(
                    exclude_instructions=True,
                    exclude_config_update=True,
                    exclude_handoff=True,
                ),
                exclude_invalid_function_calls=False,
            )
            # Decided once, as the group starts, which is what the key says. A
            # step whose confirmation lapses mid-flow is the next invocation's
            # question, not this one's.
            if ctx.userdata.is_confirmed("customer_phone"):
                logger.info(
                    "skipped verify_customer: customer_phone is already confirmed"
                )
            else:
                group.add(
                    lambda: VerifyCustomer(speak_opening=True),
                    id="verify_customer",
                    description="verify customer",
                )
            group.add(
                lambda: ManageBooking(speak_opening=True),
                id="manage_booking",
                description="manage booking",
            )
            result = await group
            task_results = result.task_results
        except _TaskTransfer as transfer:
            return transfer.agent
        except _GroupStop as stop:
            task_results = stop.results
        finally:
            _insert_carried_turn(owner_ctx, _carried)
            await self.update_chat_ctx(owner_ctx, exclude_invalid_function_calls=False)
            await self._refresh_prompt()
            # Counted after the restore, so a turn a step consumed and carried
            # back is included: that turn is the one the owner is about to read,
            # and it is exactly the turn that must not be mistaken for a new
            # request. Set in the finally so a step that raised still blocks an
            # immediate retry the caller did not ask for.
            self._ran_at_book = _caller_turns(self.chat_ctx)
        dev_metrics.dev_task_returned(ctx, *task_results.values())
        return _group_status(task_results)


# --- tasks -----------------------------------------------------------------


def _task_result(values: dict, unserved_request: str) -> dict:
    """Add the unserved request to a step's result, if it named one.

    A step that could not serve a request names it on the way out, so the agent
    that owns the step reads it off the result and takes it from there.

    Args:
        values: The values the step recorded.
        unserved_request: What the step could not serve, or an empty string.

    Returns:
        The values, with the unserved request added when there is one.
    """
    if not unserved_request:
        return values
    return {**values, "unserved_request": unserved_request}


class _RetryEmptyTaskResponseMixin:
    """Retry a task's model request when it comes back empty."""

    _response_tool_call_ids: set[str]

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Synthesize speech through the dev feed's wrapper.

        Args:
            text: The text to speak.
            model_settings: The framework's per-turn model settings.

        Returns:
            The synthesized audio frames.
        """
        return dev_metrics.dev_tts_node(self, text, model_settings)

    async def update_chat_ctx(
        self, chat_ctx: ChatContext, *, exclude_invalid_function_calls: bool = False
    ) -> None:
        """Replace the task's context, keeping tool records the task cannot run.

        Args:
            chat_ctx: The context to install.
            exclude_invalid_function_calls: Drop calls whose tool is gone.
        """
        # History policy owns old tool records, including tools this task cannot run.
        assert isinstance(self, Agent)
        await Agent.update_chat_ctx(
            self,
            chat_ctx,
            exclude_invalid_function_calls=exclude_invalid_function_calls,
        )

    async def update_tools(self, tools: list[Tool | llm.Toolset]) -> None:
        """Replace the task's tools without losing its context.

        Args:
            tools: The tools the task may call from now on.
        """
        assert isinstance(self, Agent)
        history = self.chat_ctx.copy()
        await Agent.update_tools(self, tools)
        await self.update_chat_ctx(history, exclude_invalid_function_calls=False)

    async def llm_node(
        self, chat_ctx: ChatContext, tools: list[Tool], model_settings: ModelSettings
    ) -> AsyncIterable[ChatChunk | str | FlushSentinel]:
        """Ask the model, retrying up to twice when it answers with nothing.

        Args:
            chat_ctx: The conversation to send.
            tools: The tools the model may call.
            model_settings: The framework's per-turn model settings.

        Yields:
            The model's chunks, or a short apology when every attempt was empty.
        """
        # Every generated user of this mixin is an AgentTask; narrow that
        # invariant here so the emitted project type-checks without a new base.
        assert isinstance(self, Agent)
        # ponytail: keyed on the SDK's own marker rather than the placeholder
        # wording, and the emitted-code test is what catches it changing.
        #
        # The delegate that started this task is still running, so the framework
        # injects that call plus a placeholder output reading "The tool call is
        # still in progress." into the context (voice/generation.py,
        # _inject_running_tool_calls). For the agent that made the call that is
        # right: it stops the model re-issuing a call already in flight. A task
        # is a different agent with a different prompt, and its opening turn
        # reads an unfinished tool call it never made: it then answers with
        # nothing, or apologises for a failure that did not happen, which the
        # caller hears as the agent breaking. Reproduced on 3 of 3 scripted
        # salon calls (B: task opened silent after delegate, 2026-08-21).
        running_placeholders = {
            item.call_id
            for item in chat_ctx.items
            if isinstance(item, llm.FunctionCall)
            and item.extra.get("__lk_running_placeholder__")
        }
        if running_placeholders:
            chat_ctx = chat_ctx.copy()
            chat_ctx.items = [
                item
                for item in chat_ctx.items
                if getattr(item, "call_id", None) not in running_placeholders
            ]
        completed_tool_call_ids = {
            item.call_id
            for item in chat_ctx.items
            if isinstance(item, llm.FunctionCallOutput)
            and item.call_id in self._response_tool_call_ids
        }
        post_tool = bool(completed_tool_call_ids)
        finish_tool = llm.ToolContext(tools).get_function_tool("finish")
        if finish_tool is None:
            raise RuntimeError("task retry has no finish tool")
        finish_only = False
        request_tools: list[llm.Tool] = tools
        request_chat_ctx = chat_ctx
        for attempt in range(_TASK_MAX_ATTEMPTS):
            has_response = False
            async for chunk in dev_llm_node(
                self,
                Agent.default.llm_node,
                request_chat_ctx,
                request_tools,
                model_settings,
            ):
                if isinstance(chunk, str):
                    has_response = has_response or bool(chunk.strip())
                elif isinstance(chunk, llm.ChatChunk) and chunk.delta is not None:
                    delta = chunk.delta
                    tool_calls = delta.tool_calls or []
                    if finish_only and tool_calls:
                        allowed = [call for call in tool_calls if call.name == "finish"]
                        if len(allowed) != len(tool_calls):
                            logger.warning(
                                "task post-tool reply tried another non-finish tool; "
                                "ignoring it"
                            )
                            chunk = chunk.model_copy(
                                update={
                                    "delta": delta.model_copy(
                                        update={"tool_calls": allowed}
                                    )
                                }
                            )
                            delta = chunk.delta
                            if delta is None:
                                raise RuntimeError("task retry lost its response delta")
                            tool_calls = allowed
                    self._response_tool_call_ids.update(
                        call.call_id for call in tool_calls
                    )
                    has_response = has_response or bool(
                        tool_calls or (delta.content or "").strip()
                    )
                yield chunk
            if has_response:
                self._response_tool_call_ids.difference_update(completed_tool_call_ids)
                return
            if attempt < _TASK_MAX_ATTEMPTS - 1:
                logger.warning(
                    "task response was empty; retrying %d/%d",
                    attempt + 1,
                    _TASK_MAX_ATTEMPTS - 1,
                )

                if attempt == 0 and post_tool:
                    finish_only = True
                    request_tools = [finish_tool]

                # Copy the original each time. Responses API reuses a previous
                # response when the context is unchanged, so each retry needs
                # a distinct instruction as well as a fresh context object.
                request_chat_ctx = chat_ctx.copy()
                instructions_index = request_chat_ctx.index_by_id(
                    "lk.agent_task.instructions"
                )
                if instructions_index is None:
                    raise RuntimeError("task retry has no instruction message")
                instructions = request_chat_ctx.items[instructions_index]
                if not isinstance(instructions, llm.ChatMessage):
                    raise RuntimeError("task retry instruction has an invalid type")
                instruction_text = instructions.raw_text_content
                if not instruction_text:
                    raise RuntimeError("task retry instruction is empty")
                if finish_only:
                    recovery = (
                        "The response after the tool result was empty. Use the tool "
                        "result already in context. Do not repeat that operation or "
                        "call another operation. Produce the task's next valid "
                        "response now, using what the caller has already told you "
                        "rather than asking again; call finish only if the task is "
                        "complete."
                        if attempt == 0
                        else "The finish retry was also empty. This is the second "
                        "retry. Use the existing tool result without repeating any "
                        "operation. Produce the task's next valid response now; call "
                        "finish only if the task is complete."
                    )
                else:
                    # "Do not ask again" is the load-bearing half.
                    #
                    # Without it a retry points the model at the task instructions
                    # and it restarts the script: observed on a live call, the
                    # caller answered "tomorrow if possible", the response came
                    # back empty, and the retry asked "what day would you like?"
                    # again. The caller's turn is in the copied context the whole
                    # time; the instruction just has to say to use it.
                    recovery = (
                        "The previous response was empty. Follow the current task "
                        "instructions and produce its next valid response now. The "
                        "caller's turns are already in context: use what they have "
                        "already told you, and never ask again for something they "
                        "have answered."
                        if attempt == 0
                        else "The prior retry was also empty. This is the second "
                        "retry. Follow the current task instructions and produce "
                        "its next non-empty valid response now, using what the "
                        "caller has already said rather than asking again."
                    )
                request_chat_ctx.items[instructions_index] = instructions.model_copy(
                    update={"content": [instruction_text + "\n\n" + recovery]}
                )

        logger.warning("task response stayed empty after two retries")
        if finish_only:
            yield (
                "I couldn't finish after the last tool result. Please ask me to "
                "check the current state before trying again."
            )
        else:
            yield "Sorry, I couldn't complete that. Please try again."


class ManageBooking(_RetryEmptyTaskResponseMixin, AgentTask[dict]):
    """The manage_booking step, compiled from its entry in agent.yaml."""

    def __init__(
        self,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        speak_opening: bool = False,
    ) -> None:
        """Build the step.

        Args:
            chat_ctx: The conversation the step starts from.
            speak_opening: Say the step's announcement as it is entered.
        """
        super().__init__(instructions=MANAGE_BOOKING_PROMPT, chat_ctx=chat_ctx)
        if isinstance(chat_ctx, llm.ChatContext):
            self._chat_ctx = chat_ctx.copy()
        # A group step speaks its own `announce:` as it is entered. A delegate
        # says it at the seam instead, before this class is even built, so the
        # caller of a standalone entry leaves this False and nothing says it
        # twice.
        self._speak_opening = speak_opening
        self._response_tool_call_ids: set[str] = set()
        # The call id of the ending that committed. A group reads it to fill its
        # own tool output, and a handoff called beside a terminal tool reads it
        # to know whether the step's work actually landed.
        self._finish_call_id: str | None = None
        # Per invocation, because a task instance is built per invocation: the
        # tool that ended the step and what it returned, whether one is still
        # running, an event a sibling handoff waits on, and the caller turn that
        # arrived with the tool call.
        self._terminal: tuple[str, dict] | None = None
        self._terminal_pending = False
        self._terminal_settled = asyncio.Event()
        self._carried_turn: llm.ChatMessage | None = None
        self._terminal_claimed = False
        # A handoff that arrived while a tool of this step was running. It owns
        # the ending once that tool settles, so the tool must not complete the
        # step under it.
        self._transfer_waiting = False

    def _claim_terminal(self) -> bool:
        """Claim the step's ending for a handoff.

        Returns:
            False when a handoff has already claimed it.
        """
        if self._terminal_claimed:
            return False
        self._terminal_claimed = True
        return True

    async def on_enter(self) -> None:
        """Open this step's turn as it is entered."""
        await self.update_chat_ctx(
            llm.ChatContext(
                items=[
                    m
                    for m in self.chat_ctx.messages()
                    if m.role in ("user", "assistant")
                ]
            )
        )
        await self.update_instructions(
            _render(
                MANAGE_BOOKING_PROMPT, self.session.userdata, site="task:manage_booking"
            )
        )
        # Said here, not at the seam, so it lands in this step's own context and
        # the model can see the sentence it just spoke. Not awaited: the opening
        # request runs while it plays.
        if self._speak_opening:
            dev_say(
                self.session,
                _announce(
                    "task:manage_booking",
                    ["One sec.", "Let me have a look.", "One moment, let me look."],
                ),
            )
        # The task's own instructions describe this step; let them drive the opening.
        # This opening turn withholds the agent's own handoffs (B3: an agent that can
        # hand the call back before it has said anything ping-pongs). The framework's
        # on-enter tool flag would hide them for the rest of the call instead, because
        # its filter follows the context of everything this reply starts: a live call
        # offered one specialist nothing but its delegate for ten turns while the
        # caller asked for another one (B: salon handoffs, 2026-08-20).
        self.session.generate_reply(
            tools=[t.id for t in self.tools if t.id != "to_complaints"]
        )

    def _begin_terminal(self) -> None:
        """Mark a tool that ends this step as in flight."""
        self._terminal_pending = True
        self._terminal_settled.clear()

    def _settle_terminal(self) -> None:
        """Mark the step's ending tool as settled and wake anything waiting."""
        self._terminal_pending = False
        self._terminal_settled.set()

    async def _end_on_save_booking(self, ctx: RunContext, result: dict) -> dict | None:
        """End this step on a validated result from save_booking.

        The result never reaches the model: a tool that said what it did has
        already made the decision the model would have been asked to confirm.
        Anything that is not a success goes back as an ordinary result, which is
        what keeps a failed action a conversation rather than a saved one.

        Args:
            ctx: The tool call's run context.
            result: What the tool returned.

        Returns:
            None when the step ended, otherwise the result to hand the model.
        """
        if self.done():
            # Delivered again after this step completed. There is nothing left
            # to save, and saving would append a second entry onto a list.
            self._settle_terminal()
            return None
        if not _terminal_success(
            result,
            {
                "status": (
                    "booked",
                    "moved",
                    "cancelled",
                )
            },
        ):
            # A failed action has nothing to save, and it still settles: a
            # handoff waiting beside it stops waiting and leaves the caller here.
            self._settle_terminal()
            return result
        self._terminal = ("save_booking", result)
        self._carried_turn = _newest_caller_turn(self.chat_ctx)
        try:
            _values = ctx.userdata.save_result("manage_booking", result)
        except StateRefused as refused:
            # The action happened and the save did not. The step stays open so
            # the model can repair it, and its mutations stay closed so the
            # repair cannot run the action a second time.
            self._settle_terminal()
            logger.warning("terminal save_booking: not recorded: " + refused.message)
            return {
                **result,
                "refused": "Not recorded: "
                + refused.message
                + ". Call finish with the values this tool returned; recover a missing one with a read tool of this step, or finish with unserved_request saying the action happened and its details could not be saved.",
            }
        self._finish_call_id = ctx.function_call.call_id
        self._settle_terminal()
        logger.info("step manage_booking ended on save_booking")
        if self._transfer_waiting or _sibling_call(ctx, {"to_complaints"}):
            # A handoff was called in the same response, or arrived while this
            # tool ran and is waiting for it. It owns the ending: the save is
            # committed, and completing here would race it.
            return None
        dev_metrics.dev_task_finished(ctx, _values)
        self.complete(_values)
        # None, so the framework asks this step for no reply. The next step
        # opens the next turn, or the owner answers when the delegate returns
        # its status: the framework holds the owner's turn open for the whole
        # step and generates that reply itself. A value returned here made the
        # step speak first and the owner speak again, one request and one
        # acknowledgement more than the caller needed.
        return None

    @function_tool
    async def find_slots(
        self,
        ctx: RunContext,
        date: Annotated[
            str,
            Field(
                description='The day to look at, YYYY-MM-DD. Always a real day, never an empty string: the caller\'s own bookings come back whatever day you ask for, so an empty date buys nothing and answers nothing.\nSend the day the caller named, and **never a day they did not name**. Today is not a default. A caller who asks for an appointment without saying when needs the question "which day?", not a guess: a guess reads the wrong day, comes back with nothing, and costs them the wait twice over. A live call on 2026-09-16 guessed today, got `full`, called again for tomorrow, and the caller heard two announcements in three seconds.'
            ),
        ],
        service: Annotated[
            Literal["any", "haircut", "hair-color", "blowout"],
            Field(
                description="Narrow the free times to one service, or `any` to see all three. Use `any` when the caller has not said which service they want."
            ),
        ],
    ) -> dict:
        """Read the diary: what the caller already has booked, and what times are free. Call this before offering any time and before changing any booking. It never writes anything, so call it whenever you are unsure what the caller holds.
        Give a date to see free times on that date. Leave the service out to see every service. The caller's own bookings come back every time, whatever you asked, so you never need a second call to find out what they hold.
        Ask the caller which day they want before you call this, and never guess one. A call for a day nobody named answers nothing, and the second call you then have to make speaks a second announcement the caller did not need to hear.
        """  # noqa: D205
        refusal = _refusal(
            "find_slots",
            ctx.userdata,
            [("customer_phone", "run verify_customer first.")],
        )
        if refusal:
            return {"refused": refusal}
        result = tools.find_slots.find_slots(
            date=date,
            service=service,
            customer_phone=ctx.userdata.plain("customer_phone"),
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    @function_tool
    async def save_booking(
        self,
        ctx: RunContext,
        action: Annotated[
            Literal["book", "move", "cancel"],
            Field(
                description="book a new time, move an existing booking, or cancel one"
            ),
        ],
        additional: Annotated[
            bool,
            Field(
                description="True only when the caller asked for a second appointment on top of one they already hold. A caller who holds a booking is changing it, so this is false for an ordinary booking and the tool refuses with has_booking."
            ),
        ],
        booking_id: Annotated[
            str,
            Field(
                description="The exact booking_id from find_slots when moving or cancelling. An empty string when booking, which has no booking yet."
            ),
        ],
        confirmed: Annotated[
            bool,
            Field(
                description="True only after the caller said yes to the sentence you read back."
            ),
        ],
        slot_id: Annotated[
            str,
            Field(
                description="The exact slot_id from find_slots when booking or moving. An empty string when cancelling, which needs no slot."
            ),
        ],
    ) -> dict | None:
        """Write the diary: book a new time, move an existing booking, or cancel one. This is the only tool that changes anything, so nothing is booked, moved or cancelled until it runs and says so.
        Call it only after the caller has heard the whole thing back in one sentence, the service, the day, the time and their phone number, and has said yes to that sentence. Choosing a time is not a yes. Use a slot_id and a booking_id exactly as find_slots returned them; never invent either.
        """  # noqa: D205
        refusal = _refusal(
            "save_booking",
            ctx.userdata,
            [("customer_phone", "run verify_customer first.")],
        )
        if refusal:
            return {"refused": refusal}
        if self._terminal is not None:
            logger.info(
                "terminal tool refused: save_booking cannot run again in this step"
            )
            return {
                "refused": "save_booking already succeeded in this step. Call finish to record it; a new action is a new request from the caller."
            }
        # In flight from here until _end_on_ settles it, whichever response it
        # was called in: a handoff or a finish arriving while it runs waits for
        # it rather than moving the caller past an action nobody recorded.
        self._begin_terminal()
        try:
            result = tools.save_booking.save_booking(
                action=action,
                additional=additional,
                booking_id=booking_id,
                confirmed=confirmed,
                slot_id=slot_id,
                customer_phone=ctx.userdata.plain("customer_phone"),
            )
            if inspect.isawaitable(result):
                result = await result
        except BaseException:
            self._settle_terminal()
            raise
        return await self._end_on_save_booking(ctx, result)

    @function_tool
    async def to_complaints(self, ctx: RunContext) -> str | None:
        """The caller has a complaint or service problem."""
        if self._terminal_pending or _sibling_call(ctx, {"save_booking"}):
            # A tool that ends this step is running, or was called in the same
            # response and has not started yet. The caller asked for two things
            # at once, and the booking has to be committed before the caller is
            # moved: a transfer that went first would leave the step's own work
            # unrecorded.
            self._transfer_waiting = True
            try:
                await asyncio.wait_for(
                    self._terminal_settled.wait(), timeout=_TERMINAL_WAIT_SECS
                )
            except TimeoutError:
                self._settle_terminal()
            finally:
                self._transfer_waiting = False
            if self._finish_call_id is None:
                logger.info("handoff held: the step's own action did not complete")
                return "The action did not complete, so the caller was not moved. Read the tool result and continue."
            if self.done():
                # The step ended while this handoff waited. There is nothing
                # left to move the caller out of; the owner has its own handoffs.
                return "This step already ended. Hand the caller over from where you are now."
        if not self._claim_terminal():
            return

        try:
            self.complete(
                _TaskTransfer(
                    ComplaintSpecialist(
                        chat_ctx=llm.ChatContext(
                            items=[
                                m
                                for m in self.chat_ctx.messages()
                                if m.role in ("user", "assistant")
                            ]
                        )
                    )
                )
            )
        except BaseException:
            self._terminal_claimed = False
            raise

    # The schema is StepResult's, built from State's own annotations, so the
    # model is asked for exactly what the save accepts. Raw, so LiveKit sends it
    # as written rather than rebuilding it from a signature.
    @function_tool(
        raw_schema={
            "name": "finish",
            "description": "Save this step's values and finish, or return unserved without saving.",
            "parameters": STEP_RESULTS["manage_booking"].tool_parameters(),
        }
    )
    async def finish(
        self, raw_arguments: dict[str, object], ctx: RunContext[CallState]
    ) -> str | None:
        """Save this step's values and finish, or return unserved without saving."""
        if self.done():
            return
        if self._terminal_claimed:
            return
        if self._terminal_pending:
            # A tool that ends this step is still running, and its result
            # decides the step. Wait for it rather than ending the step under it
            # and leaving what it did unrecorded.
            try:
                await asyncio.wait_for(
                    self._terminal_settled.wait(), timeout=_TERMINAL_WAIT_SECS
                )
            except TimeoutError:
                self._settle_terminal()
            if self.done():
                return
        try:
            _args = dict(raw_arguments)
            if self._terminal is not None:
                # Repairing a save the tool already made. The tool's own values
                # win where they validate: the model cannot manufacture a
                # reference, and a repair that retyped one would record an
                # action nobody took.
                _args = STEP_RESULTS["manage_booking"].retain(_args, self._terminal[1])
            _values = ctx.userdata.save_result("manage_booking", _args)
        except StateRefused as refused:
            logger.warning("finish %s: %s", "manage_booking", refused.message)
            return f"Not recorded: {refused.message}. Ask again, then call finish with a value that fits."
        self._terminal_claimed = True
        dev_metrics.dev_task_finished(ctx, _values)
        self.complete(_values)
        self._finish_call_id = ctx.function_call.call_id


class VerifyCustomer(_RetryEmptyTaskResponseMixin, AgentTask[dict]):
    """The verify_customer step, compiled from its entry in agent.yaml."""

    def __init__(
        self,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        speak_opening: bool = False,
    ) -> None:
        """Build the step.

        Args:
            chat_ctx: The conversation the step starts from.
            speak_opening: Say the step's announcement as it is entered.
        """
        super().__init__(instructions=VERIFY_CUSTOMER_PROMPT, chat_ctx=chat_ctx)
        if isinstance(chat_ctx, llm.ChatContext):
            self._chat_ctx = chat_ctx.copy()
        # A group step speaks its own `announce:` as it is entered. A delegate
        # says it at the seam instead, before this class is even built, so the
        # caller of a standalone entry leaves this False and nothing says it
        # twice.
        self._speak_opening = speak_opening
        self._response_tool_call_ids: set[str] = set()
        # The call id of the ending that committed. A group reads it to fill its
        # own tool output, and a handoff called beside a terminal tool reads it
        # to know whether the step's work actually landed.
        self._finish_call_id: str | None = None
        # Per invocation, because a task instance is built per invocation: the
        # tool that ended the step and what it returned, whether one is still
        # running, an event a sibling handoff waits on, and the caller turn that
        # arrived with the tool call.
        self._terminal: tuple[str, dict] | None = None
        self._terminal_pending = False
        self._terminal_settled = asyncio.Event()
        self._carried_turn: llm.ChatMessage | None = None

    async def on_enter(self) -> None:
        """Open this step's turn as it is entered."""
        # Entered, so whatever it confirms is being decided again. Withdrawn
        # before the prompt renders, so this step never reads a confirmation it
        # is about to replace.
        self.session.userdata.withdraw_confirmation("verify_customer")
        await self.update_chat_ctx(
            llm.ChatContext(
                items=[
                    m
                    for m in self.chat_ctx.messages()
                    if m.role in ("user", "assistant")
                ]
            )
        )
        await self.update_instructions(
            _render(
                VERIFY_CUSTOMER_PROMPT,
                self.session.userdata,
                site="task:verify_customer",
            )
        )
        # Said here, not at the seam, so it lands in this step's own context and
        # the model can see the sentence it just spoke. Not awaited: the opening
        # request runs while it plays.
        if self._speak_opening:
            dev_say(
                self.session,
                _announce(
                    "task:verify_customer",
                    [
                        "Let me confirm who I'm speaking to.",
                        "One second, let me check who I've got.",
                        "Right, let me make sure I have you.",
                    ],
                ),
            )
        # The task's own instructions describe this step; let them drive the opening.
        self.session.generate_reply()

    def _begin_terminal(self) -> None:
        """Mark a tool that ends this step as in flight."""
        self._terminal_pending = True
        self._terminal_settled.clear()

    def _settle_terminal(self) -> None:
        """Mark the step's ending tool as settled and wake anything waiting."""
        self._terminal_pending = False
        self._terminal_settled.set()

    async def _end_on_find_or_create_customer(
        self, ctx: RunContext, result: dict
    ) -> dict | None:
        """End this step on a validated result from find_or_create_customer.

        The result never reaches the model: a tool that said what it did has
        already made the decision the model would have been asked to confirm.
        Anything that is not a success goes back as an ordinary result, which is
        what keeps a failed action a conversation rather than a saved one.

        Args:
            ctx: The tool call's run context.
            result: What the tool returned.

        Returns:
            None when the step ended, otherwise the result to hand the model.
        """
        if self.done():
            # Delivered again after this step completed. There is nothing left
            # to save, and saving would append a second entry onto a list.
            self._settle_terminal()
            return None
        if not _terminal_success(
            result,
            {
                "customer_status": (
                    "existing",
                    "created",
                )
            },
        ):
            # A failed action has nothing to save, and it still settles: a
            # handoff waiting beside it stops waiting and leaves the caller here.
            self._settle_terminal()
            return result
        self._terminal = ("find_or_create_customer", result)
        self._carried_turn = _newest_caller_turn(self.chat_ctx)
        try:
            _values = ctx.userdata.save_result("verify_customer", result)
        except StateRefused as refused:
            # The action happened and the save did not. The step stays open so
            # the model can repair it, and its mutations stay closed so the
            # repair cannot run the action a second time.
            self._settle_terminal()
            logger.warning(
                "terminal find_or_create_customer: not recorded: " + refused.message
            )
            return {
                **result,
                "refused": "Not recorded: "
                + refused.message
                + ". Call finish with the values this tool returned; recover a missing one with a read tool of this step, or finish with unserved_request saying the action happened and its details could not be saved.",
            }
        self._finish_call_id = ctx.function_call.call_id
        self._settle_terminal()
        logger.info("step verify_customer ended on find_or_create_customer")
        dev_metrics.dev_task_finished(ctx, _values)
        self.complete(_values)
        # None, so the framework asks this step for no reply. The next step
        # opens the next turn, or the owner answers when the delegate returns
        # its status: the framework holds the owner's turn open for the whole
        # step and generates that reply itself. A value returned here made the
        # step speak first and the owner speak again, one request and one
        # acknowledgement more than the caller needed.
        return None

    @function_tool
    async def find_or_create_customer(
        self,
        ctx: RunContext,
        phone: Annotated[
            str,
            Field(
                description="The exact confirmed number in E.164: a plus sign, then 10 to 15 digits, with no spaces, brackets or dashes. The plus sign is part of the number and is never dropped; a text run sent the digits alone and the lookup could not tell which country it was. Never infer a country code the caller did not say."
            ),
        ],
    ) -> dict | None:
        """Look up or create one salon customer from the exact confirmed phone number. Use only after the digit readback got a clear yes. Reuse the record that already owns the number, or create one for a number that is new. Never guess a number or pass one the caller has not confirmed. A caller giving a replacement number is not confirmation. Ask the readback question first; call this tool only after a later caller message agrees to it."""
        if self._terminal is not None:
            logger.info(
                "terminal tool refused: find_or_create_customer cannot run again in this step"
            )
            return {
                "refused": "find_or_create_customer already succeeded in this step. Call finish to record it; a new action is a new request from the caller."
            }
        # In flight from here until _end_on_ settles it, whichever response it
        # was called in: a handoff or a finish arriving while it runs waits for
        # it rather than moving the caller past an action nobody recorded.
        self._begin_terminal()
        try:
            result = tools.find_or_create_customer.find_or_create_customer(phone=phone)
            if inspect.isawaitable(result):
                result = await result
        except BaseException:
            self._settle_terminal()
            raise
        return await self._end_on_find_or_create_customer(ctx, result)

    # The schema is StepResult's, built from State's own annotations, so the
    # model is asked for exactly what the save accepts. Raw, so LiveKit sends it
    # as written rather than rebuilding it from a signature.
    @function_tool(
        raw_schema={
            "name": "finish",
            "description": "Save this step's values and finish, or return unserved without saving.",
            "parameters": STEP_RESULTS["verify_customer"].tool_parameters(),
        }
    )
    async def finish(
        self, raw_arguments: dict[str, object], ctx: RunContext[CallState]
    ) -> str | None:
        """Save this step's values and finish, or return unserved without saving."""
        if self.done():
            return
        if self._terminal_pending:
            # A tool that ends this step is still running, and its result
            # decides the step. Wait for it rather than ending the step under it
            # and leaving what it did unrecorded.
            try:
                await asyncio.wait_for(
                    self._terminal_settled.wait(), timeout=_TERMINAL_WAIT_SECS
                )
            except TimeoutError:
                self._settle_terminal()
            if self.done():
                return
        try:
            _args = dict(raw_arguments)
            if self._terminal is not None:
                # Repairing a save the tool already made. The tool's own values
                # win where they validate: the model cannot manufacture a
                # reference, and a repair that retyped one would record an
                # action nobody took.
                _args = STEP_RESULTS["verify_customer"].retain(_args, self._terminal[1])
            _values = ctx.userdata.save_result("verify_customer", _args)
        except StateRefused as refused:
            logger.warning("finish %s: %s", "verify_customer", refused.message)
            return f"Not recorded: {refused.message}. Ask again, then call finish with a value that fits."
        dev_metrics.dev_task_finished(ctx, _values)
        self.complete(_values)
        self._finish_call_id = ctx.function_call.call_id
