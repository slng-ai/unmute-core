"""The agents, their tools, and the task steps they run as flows."""

from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import json
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any, cast

from loguru import logger
from pipecat.flows import NO_RESPONSE, FlowManager, FlowsFunctionSchema, NodeConfig
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMUpdateSettingsFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.workers.llm import LLMWorkerActivationArgs, tool
from pipecat_slng import SlngTTSService

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
    BOOK_MANAGE_BOOKING_TASK_PROMPT,
    BOOK_VERIFY_CUSTOMER_TASK_PROMPT,
    COMPLAINT_SPECIALIST_PROMPT,
    CONCIERGE_PROMPT,
    VERIFY_CUSTOMER_TASK_PROMPT,
)
from session import _refusal, _render
from settings import TERMINAL_SETTLE_TIMEOUT_SECS
from utils import knowledge
from utils.context import (
    _caller_turns,
    _carried_messages,
    _newest_caller_message,
    _settle_task_call,
    _speech_only,
)
from utils.dev_metrics import DevReporter
from utils.google import _GoogleVertexLLM
from utils.telephony import _transfer_twiml, _update_carrier_call
from utils.tracing import TracedLLMWorker


def _direct_tool(
    fn: Callable[..., Any] | None = None,
    *,
    cancel_on_interruption: bool = True,
    timeout_secs: float | None = None,
    announce: str | None = None,
) -> Any:
    """Keep every direct function call terminal, even on malformed input.

    Args:
        fn: The handler, when used as a bare `@_direct_tool`.
        cancel_on_interruption: Whether the caller speaking cancels the call.
        timeout_secs: How long the framework waits for the handler.
        announce: A line spoken as the tool starts, or None for silence.

    Returns:
        The decorated handler, or a decorator when called with options.
    """

    def decorate(handler: Callable[..., Any]) -> Any:
        """Wrap one handler so a bad or failing call still resolves."""
        signature = inspect.signature(handler)
        declared = set(signature.parameters) - {"self", "params"}

        @functools.wraps(handler)
        async def guarded(*args: Any, **kwargs: Any) -> Any:
            """Run the handler, resolving the call itself on bad input or failure."""
            params = kwargs.get("params")
            if params is None:
                params_index = 1 if "self" in signature.parameters else 0
                params = args[params_index]

            unexpected = sorted(set(kwargs) - declared - {"params"})
            if unexpected:
                allowed = ", ".join(sorted(declared)) or "none"
                await params.result_callback(
                    {
                        "error": (
                            f"Unexpected arguments for {params.function_name}: "
                            f"{', '.join(unexpected)}. Allowed arguments: {allowed}. "
                            "Retry with only allowed arguments."
                        )
                    }
                )
                return

            if announce:
                # Cover the wait: TTSSpeakFrame is a DataFrame, so it is queued
                # and speech starts while the handler body runs. Nothing waits
                # for playout here, which is the point of the line. Queued after
                # the argument check, so a rejected call stays silent, and an
                # interruption discards it in line with cancel_on_interruption.
                await params.llm.push_frame(TTSSpeakFrame(announce))

            resolved = False
            original_result_callback = params.result_callback

            async def resolve(result: Any, **callback_kwargs: Any) -> Any:
                """Record that the call was resolved, then hand the result on."""
                nonlocal resolved
                resolved = True
                return await original_result_callback(result, **callback_kwargs)

            params.result_callback = resolve
            try:
                return await handler(*args, **kwargs)
            except Exception:
                logger.exception(
                    "tool failed before completing: {}", params.function_name
                )
                if not resolved:
                    await original_result_callback(
                        {
                            "error": (
                                f"{params.function_name} failed before completing. "
                                "Do not claim success; retry only with corrected input."
                            )
                        }
                    )
                    return
                raise
            finally:
                params.result_callback = original_result_callback

        return tool(
            cancel_on_interruption=cancel_on_interruption,
            timeout_secs=timeout_secs,
        )(guarded)

    if fn is not None:
        return decorate(fn)
    return decorate


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
        key: The announcement site, which remembers its own last line.
        lines: The alternatives the author wrote.

    Returns:
        The line to speak.
    """
    # ponytail: process wide, not per call. Two concurrent callers can only
    # affect each other's variety, which is the whole thing this is for.
    choices = [line for line in lines if line != _ANNOUNCE_LAST.get(key)] or lines
    pick = random.choice(choices)
    _ANNOUNCE_LAST[key] = pick
    return pick


def build_complaint_specialist_llm(state: CallState | None = None) -> LLMService[Any]:
    """Build the language model for this agent.

    Args:
        state: The call state, read by the router for its template variables.

    Returns:
        The configured LLM service.
    """
    return _GoogleVertexLLM(
        location="eu",
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GoogleLLMService.Settings(
            model="gemini-3.1-flash-lite",
            system_instruction=_render(
                COMPLAINT_SPECIALIST_PROMPT, state, site="agent:complaint_specialist"
            ),
            thinking=GoogleLLMService.ThinkingConfig(**{"thinking_level": "MINIMAL"}),
        ),
    )


def build_complaint_specialist_tts(slng_session_id: str) -> TTSService:
    """Build the voice for this agent.

    Args:
        slng_session_id: This call's SLNG session id, sent with the agent id.

    Returns:
        The configured TTS service.
    """
    return SlngTTSService(
        api_key=os.environ["SLNG_API_KEY"],
        voice="62ae83ad-4f6a-430b-af41-a9bede9286ca",
        model="cartesia/sonic:3.5",
        language=Language("en"),
        warm_standby_enabled=True,
        world_part="eu-north",
        external_agent_id="salon-concierge",
        external_session_id=slng_session_id,
    )


class ComplaintSpecialistAgent(TracedLLMWorker):
    """Agent: complaint_specialist."""

    def __init__(
        self,
        state: CallState | None = None,
        context: LLMContext | None = None,
        call_context: dict[str, Any] | None = None,
        dev_metrics: DevReporter | None = None,
        *,
        slng_session_id: str,
    ) -> None:
        """Build this agent's model and voice and join the call's shared context.

        Args:
            state: The call's typed variables, shared by every agent.
            context: The LLM context every agent on the call shares.
            call_context: The call's own facts and transport handles.
            dev_metrics: The dev-page reporter, or None outside `unmute dev`.
            slng_session_id: This call's SLNG session id, sent with the agent id.
        """
        # Always supplied when the package keeps state; None only in a test.
        self.state: CallState = cast(CallState, state)
        if context is not None:
            self.context = context
        self.call_context = call_context if call_context is not None else {}

        llm = build_complaint_specialist_llm(state)
        tts = build_complaint_specialist_tts(slng_session_id)
        self._dev = dev_metrics
        if self._dev:
            self._dev.observe_llm(llm)
            self._dev.observe_tts(tts)
        super().__init__(
            "complaint_specialist", llm=llm, pipeline=Pipeline([llm, tts]), bridged=()
        )
        # Every tool the model called in one response, recorded before any of
        # their handlers run. A step that ends on its own tool has to know
        # whether a handoff was called beside it: this framework runs both
        # handlers and whichever finishes first would otherwise close the step
        # under the other one.
        self._response_calls: tuple[str, ...] = ()

        @llm.event_handler("on_function_calls_started")
        async def _record_calls(service: Any, function_calls: list[Any]) -> None:
            """Remember the tools the model called in this response."""
            self._response_calls = tuple(call.function_name for call in function_calls)

    async def queue_frame(self, frame: Any, *args: Any, **kwargs: Any) -> None:
        """Stamp the frame for the dev page, then queue it as usual.

        Args:
            frame: The frame to queue.
            *args: Passed to the base worker.
            **kwargs: Passed to the base worker.
        """
        if self._dev:
            self._dev.stamp(frame)
        await super().queue_frame(frame, *args, **kwargs)

    async def on_activated(self, args: dict[str, Any] | None) -> None:
        """Set this agent's prompt and scope, then ask for its first reply when told to.

        Args:
            args: The activation arguments the previous agent or the entry passed.
        """
        token = self._dev.enter_activation(args) if self._dev else None
        try:
            await self.queue_frame(
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(
                        system_instruction=_render(
                            COMPLAINT_SPECIALIST_PROMPT,
                            self.state,
                            site="agent:complaint_specialist",
                        )
                    ),
                )
            )
            await super().on_activated(args)
            # Pipecat 1.8 and 1.9 only run on activation when messages are nonempty.
            # A handoff already shaped the shared context; request the reply
            # without adding a synthetic message or waiting for the caller.
            if args and args.get("run_llm") and not args.get("messages"):
                await self.queue_frame(LLMRunFrame())
        finally:
            if self._dev:
                self._dev.exit_activation(token)

    def _bind_state(
        self, handler: Callable[..., Awaitable[Any]]
    ) -> Callable[[Any, FlowManager], Awaitable[Any]]:
        """Give a module-level flows handler this worker's call state.

        A task tool can then read injected variables the same way an agent tool does.

        Args:
            handler: The module-level flows handler.

        Returns:
            A handler with the call state already bound.
        """

        async def _bound(args: Any, flow_manager: FlowManager) -> Any:
            """Call the handler with this worker's state."""
            return await handler(args, flow_manager, state=self.state)

        return _bound

    @_direct_tool(cancel_on_interruption=False)
    async def to_concierge(self, params: FunctionCallParams) -> None:
        """The caller changes to a request owned by another specialist.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
        """
        # context.history on this handoff. One LLMContext is shared for the whole
        # call, so the receiver is given the shaped list rather than a copy.
        self.context.set_messages(_speech_only(self.context.get_messages()))
        await self.activate_worker(
            "concierge",
            args=LLMWorkerActivationArgs(
                metadata=self._dev.activation_metadata() if self._dev else None,
                messages=[],
                run_llm=True,
            ),
            deactivate_self=True,
            result_callback=params.result_callback,
        )

    @_direct_tool(announce="Let me get that written down.")
    async def record_complaint(
        self, params: FunctionCallParams, requested_resolution: str, summary: str
    ) -> None:
        """Record the verified customer's complaint after enough facts are known. This does not replace a requested manager transfer.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
            requested_resolution (str): What the customer wants the salon to do, or an empty string
            summary (str): Short factual summary of the complaint
        """
        state = self.state
        refusal = _refusal(
            "record_complaint",
            state,
            [("customer_phone", "run verify_customer first.")],
        )
        if refusal:
            await params.result_callback({"refused": refusal})
            return
        result = tools.record_complaint.record_complaint(
            requested_resolution=requested_resolution,
            summary=summary,
            customer_phone=state.plain("customer_phone"),
        )
        if inspect.isawaitable(result):
            result = await result
        await params.result_callback(result)

    @_direct_tool
    async def look_up_refund_policy(
        self, params: FunctionCallParams, query: str
    ) -> None:
        """Look up the salon's refund and complaints policy. Use this before you state any refund, redo, timescale, or goodwill offer, so you quote the policy instead of guessing it. It covers the redo window, the three refund tiers, colour corrections, retail returns, and how a complaint is handled.

        The results may not answer the question. They are ordered best first, and each carries a relevance score where a higher number means a closer match. If nothing returned actually answers what was asked, say you do not have that information rather than offering the closest result.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
            query (str): What to look up. Use the caller's own words.
        """
        await params.result_callback(await knowledge.look_up("refunds", query))

    @_direct_tool
    async def end_call(self, params: FunctionCallParams) -> None:
        """End the call once the caller's own words say the call is over: "no, that's all", "I'm good, thanks", "goodbye". A question is never one of those, however tired it sounds. On a live call on 2026-09-16 the caller asked "can we just recap when I have the next appointment?" and this ended the call on them. Answer what they asked and wait for them to say they are done.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
        """
        await params.llm.push_frame(
            LLMMessagesAppendFrame(
                [
                    {
                        "role": "developer",
                        "content": "Say one short goodbye in your own words, then end the call. Do not offer anything else first.",
                    }
                ],
                run_llm=False,
            )
        )
        await params.result_callback({"ended": True})
        await params.pipeline_worker.end()

    @_direct_tool(cancel_on_interruption=False)
    async def to_manager(self, params: FunctionCallParams) -> None:
        """The caller explicitly asks for a manager or is clearly and strongly frustrated.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
        """
        logger.info("human transfer fired: to_manager (cold)")
        transfer_result = self.call_context.get("_transfer_result")
        if transfer_result is not None:
            # Asked twice. Replay the first answer rather than dial again.
            await params.result_callback(transfer_result)
            return
        phone_call = self.call_context.get("_phone_call")
        if not phone_call or not phone_call.get("call_id"):
            # No phone call, so nothing to hand over: the browser and console
            # transports have no call for your carrier to redirect. Saying so beats
            # raising, and beats telling the model the caller was transferred.
            self.call_context["_transfer_result"] = {
                "failed": "this session is not a phone call, so it cannot be transferred"
            }
            await params.result_callback(self.call_context["_transfer_result"])
            return
        # Claimed before any transfer work awaits: a request arriving while
        # the first one is in flight must not slip past the guard.
        self.call_context["_transfer_result"] = {
            "in_progress": "A transfer is already under way; do not start another."
        }

        # The announcement is the first verb of the document below, spoken by your
        # carrier rather than by this agent. Applying the update replaces the call's
        # document, which tears this media stream down, so a line the agent started
        # speaking would be cut off by its own transfer.
        try:
            await _update_carrier_call(
                phone_call["call_id"],
                _transfer_twiml(os.environ["MANAGER_PHONE_NUMBER"], 30),
            )
        except Exception as exc:  # noqa: BLE001 - any failure leaves the call untouched
            # The request failed, so the call is untouched and the caller is still
            # here listening. Tell them, and never report a transfer that did not
            # happen.
            logger.warning("cold transfer failed: {}", exc)
            self.call_context["_transfer_result"] = {
                "failed": "The transfer could not be completed; the call is ending."
            }
            failure_error = None
            try:
                await params.llm.push_frame(
                    LLMMessagesAppendFrame(
                        [
                            {
                                "role": "developer",
                                "content": "Tell the caller nobody can take the call right now, apologize, and say goodbye.",
                            }
                        ],
                        run_llm=False,
                    )
                )
                await params.result_callback(self.call_context["_transfer_result"])
            except BaseException as exc:  # noqa: BLE001 - held, then re-raised once the call has ended
                failure_error = exc
            try:
                await params.pipeline_worker.end()
            except BaseException:
                if failure_error is None:
                    raise
                logger.exception("failed to end call after transfer failure")
            if failure_error is not None:
                raise failure_error  # noqa: B904 - the transfer error is logged above
            return
        # The stream ends as your carrier applies the document. Recorded first so a
        # request that arrives in that window replays this rather than re-firing.
        self.call_context["_transfer_result"] = {"transfer_started": True}
        await params.result_callback(self.call_context["_transfer_result"])

    @_direct_tool
    async def verify_customer(self, params: FunctionCallParams) -> None:
        """Run this when the caller explicitly corrects their phone number, and when a complaint is about to be written down. Never before a booking: the `book` flow runs this step itself when the saved number is not confirmed. A change of date, time or service is not a phone correction. The caller's words right above the status are what the step answered; it served the part it owns, so act only on what remains.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
        """
        # Already ran, and nobody has spoken since. Then this is not a second
        # request, it is the model reading its own completed result as though it
        # were one. On a live call on 2026-09-16 (trace 917975e9, LiveKit, same
        # package) a booking saved, the flow re-entered on the caller's own
        # "3 o'clock" still sitting above the result, and the last thing the
        # caller heard was an agent saying it was off to check the diary.
        #
        # Refused here rather than asked for in the prompt, which this package
        # did twice and which failed twice. Before the announcement below, so a
        # refused call speaks nothing at all.
        if _caller_turns(self.context.get_messages()) == getattr(
            self, "_ran_at_verify_customer", None
        ):
            await params.result_callback(
                {
                    "refused": "This already ran and finished, and the caller has "
                    "not spoken since, so its result is the answer to what they "
                    "asked. Reply to them from the values in your prompt. Run it "
                    "again only once they have asked for something new."
                }
            )
            return
        # Cover the entry: two model requests pass between the caller finishing and
        # the step's first word, and this puts speech in them. TTSSpeakFrame is a
        # DataFrame, so it is queued and speech starts while the flow is built.
        # Queued at the very start of the method, before anything else runs.
        await self.queue_frame(
            TTSSpeakFrame(
                _announce(
                    "delegate:verify_customer",
                    [
                        "Let me confirm who I'm speaking to.",
                        "One second, let me check who I've got.",
                        "Right, let me make sure I have you.",
                    ],
                )
            )
        )
        self._verify_customer_visit = object()
        self._verify_customer_results = {}
        # Per invocation: the tool that ended a step and what it returned, an
        # event a sibling handoff waits on, and the caller turn the step
        # consumed on its way out.
        self._verify_customer_terminal = None
        self._verify_customer_pending = False
        self._verify_customer_settled = asyncio.Event()
        self._verify_customer_carried_turn = None
        self._verify_customer_active_step = "verify_customer"
        flow = FlowManager(
            llm=self.llm,
            context_aggregator=LLMContextAggregatorPair(self.context),
            worker=self,
        )
        # Resolve this tool call so it never dangles (V3/B1), but with
        # run_llm=False: the flow's first node (respond_immediately) is the sole
        # responder, so the owner never runs a second completion and the caller
        # hears the task's opening line once, not twice (V7/B4).
        await params.result_callback(
            {"status": "running the verify_customer task"},
            properties=FunctionCallResultProperties(run_llm=False),
        )
        # V2/B2: drain the resolved owner call before snapshotting. Otherwise
        # restoration erases that call and the unchanged request delegates again.
        await self.flush_pipeline()
        self._verify_customer_snapshot = (
            copy.deepcopy(self.context.get_messages()),
            self.context.tools,
        )
        # How many caller turns the owner has seen, so a step that ends on its
        # tool can tell whether the turn it consumed is one the owner never saw.
        self._verify_customer_snapshot_turns = _caller_turns(
            self.context.get_messages()
        )
        # context.history on this task. Shaped after the snapshot above, so the
        # finish path restores the owner's own context whatever this step saw.
        self.context.set_messages(_speech_only(self.context.get_messages()))
        await flow.initialize(await self._verify_customer_node_verify_customer())

    def _verify_customer_settle(self) -> None:
        """Mark the running mutation over, one way or another."""
        self._verify_customer_pending = False
        self._verify_customer_settled.set()

    async def _verify_customer_node_verify_customer(self) -> NodeConfig:
        """Build the flow node for the verify_customer step.

        Returns:
            The node that runs this step's prompt and tools.
        """
        # This step's own ending, whatever the step before it did. Held on the
        # flow rather than the step, so the tool that ended the previous step
        # closed this one's mutations too: a verified caller was told their
        # first booking had already succeeded.
        self._verify_customer_terminal = None
        self._verify_customer_pending = False
        self._verify_customer_settled.clear()
        self._verify_customer_carried_turn = None
        # Entered, so whatever it confirms is being decided again. Withdrawn
        # before role_message renders, so this step never reads a confirmation
        # it is about to replace.
        self.state.withdraw_confirmation("verify_customer")
        self.context.set_messages(_speech_only(self.context.get_messages()))
        return NodeConfig(
            name="verify_customer",
            role_message=_render(
                VERIFY_CUSTOMER_TASK_PROMPT, self.state, site="task:verify_customer"
            ),
            task_messages=[
                {
                    "role": "developer",
                    "content": "Begin this step. Work from what the caller has already said.",
                }
            ],
            functions=[
                FlowsFunctionSchema(
                    name="find_or_create_customer",
                    description="Look up or create one salon customer from the exact confirmed phone number. Use only after the digit readback got a clear yes. Reuse the record that already owns the number, or create one for a number that is new. Never guess a number or pass one the caller has not confirmed. A caller giving a replacement number is not confirmation. Ask the readback question first; call this tool only after a later caller message agrees to it.",
                    properties={
                        "phone": {
                            "description": "The exact confirmed number in E.164: a plus sign, then 10 to 15 digits, with no spaces, brackets or dashes. The plus sign is part of the number and is never dropped; a text run sent the digits alone and the lookup could not tell which country it was. Never infer a country code the caller did not say.",
                            "type": "string",
                        }
                    },
                    required=["phone"],
                    handler=_flow_visit(
                        self,
                        "verify_customer",
                        self._trace_flow_tool(
                            "find_or_create_customer",
                            self._verify_customer_terminal_verify_customer_find_or_create_customer,
                        ),
                    ),
                ),
                FlowsFunctionSchema(
                    name="finish_verify_customer_verify_customer",
                    description="Record the result of this step and finish.",
                    properties=STEP_RESULTS["verify_customer"].tool_parameters()[
                        "properties"
                    ],
                    required=[],
                    handler=_flow_visit(
                        self,
                        "verify_customer",
                        self._trace_flow_tool(
                            "finish_verify_customer_verify_customer",
                            self._verify_customer_finish_verify_customer,
                        ),
                    ),
                ),
            ],
        )

    async def _verify_customer_terminal_verify_customer_find_or_create_customer(
        self, args: Any, flow_manager: FlowManager
    ) -> tuple[dict[str, Any], Any]:
        """Run find_or_create_customer and end this step when it says it did the work.

        The result never reaches the model on a success: the tool has already
        made the decision the model would have been asked to confirm. Anything
        else goes back as an ordinary result and the step stays open.

        Args:
            args: The arguments the model supplied.
            flow_manager: The flow this step runs in.

        Returns:
            The result the model sees, and whether a reply is generated.
        """
        if self._verify_customer_active_step != "verify_customer":
            return {"status": "already handled"}, NO_RESPONSE
        if self._verify_customer_terminal is not None:
            logger.info(
                "terminal tool refused: find_or_create_customer cannot run again in this step"
            )
            return {
                "refused": "find_or_create_customer already succeeded in this step. Call finish to record it; a new action is a new request from the caller."
            }, None
        # In flight from here until settled, whichever response called it: a
        # handoff or a finish arriving while it runs waits for it rather than
        # moving the caller past an action nobody recorded.
        _visit = self._verify_customer_visit
        self._verify_customer_pending = True
        self._verify_customer_settled.clear()
        try:
            result = await _flow_tool_find_or_create_customer(args, flow_manager)
        except BaseException:
            self._verify_customer_settle()
            raise
        if (
            self._verify_customer_visit is not _visit
            or self._verify_customer_active_step != "verify_customer"
        ):
            # The flow moved while the tool ran. Nothing here may write state
            # a later step now owns, so the result goes back as it is.
            self._verify_customer_settle()
            return result, None
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
            self._verify_customer_settle()
            return result, None
        self._verify_customer_terminal = ("find_or_create_customer", result)
        _messages = self.context.get_messages()
        if _caller_turns(_messages) > self._verify_customer_snapshot_turns:
            # The turn this call answered arrived after the owner's snapshot, so
            # the owner never saw it. Counted rather than matched on its text,
            # because a caller who says the same word twice has said it twice.
            self._verify_customer_carried_turn = _newest_caller_message(_messages)
        try:
            _values = self.state.save_result("verify_customer", result)
        except StateRefused as refused:
            # The action happened and the save did not. The step stays open so
            # the model can repair it, and its mutations stay closed so the
            # repair cannot run the action a second time.
            self._verify_customer_settle()
            logger.warning(
                "terminal find_or_create_customer: not recorded: " + refused.message
            )
            return {
                **result,
                "refused": "Not recorded: "
                + refused.message
                + ". Call finish with the values this tool returned; recover a missing one with a read tool of this step, or finish with unserved_request saying the action happened and its details could not be saved.",
            }, None
        self._verify_customer_settle()
        logger.info("step verify_customer ended on find_or_create_customer")
        # Saved once, above. The advance is the finish handler's own tail,
        # shared rather than copied, and it saves nothing itself: a second save
        # wrote a second entry onto every list an assign appends to.
        return await self._verify_customer_advance_verify_customer(_values)

    async def _verify_customer_finish_verify_customer(
        self, args: Any, flow_manager: FlowManager
    ) -> tuple[dict[str, Any], Any]:
        """Record this step's result and move the flow on.

        Args:
            args: The result fields the model supplied.
            flow_manager: The flow this step runs in.

        Returns:
            The result the model sees, and the next node or None.
        """
        if self._verify_customer_active_step != "verify_customer":
            return {"status": "already handled"}, NO_RESPONSE
        if self._verify_customer_pending:
            # A tool that ends this step is still running, and its result
            # decides the step. Wait for it rather than ending the step under it
            # and leaving what it did unrecorded.
            try:
                await asyncio.wait_for(
                    self._verify_customer_settled.wait(),
                    timeout=TERMINAL_SETTLE_TIMEOUT_SECS,
                )
            except TimeoutError:
                self._verify_customer_settle()
            if self._verify_customer_active_step != "verify_customer":
                return {"status": "already handled"}, NO_RESPONSE
        args = dict(args)
        if self._verify_customer_terminal is not None:
            # Repairing a save the tool already made. The tool's own values win
            # where they validate: the model cannot manufacture a reference, and
            # a repair that retyped one would record an action nobody took.
            args = STEP_RESULTS["verify_customer"].retain(
                args, self._verify_customer_terminal[1]
            )
        # Validated before anything is recorded: a value that does not fit its
        # declared type never enters the state, the previous contents stand, and
        # the message goes back to the model so it can correct itself on the next
        # turn instead of the step recording something wrong. This framework
        # validates no tool argument itself, so without this the two targets
        # would behave differently on the same package.
        try:
            _values = self.state.save_result("verify_customer", dict(args))
        except StateRefused as refused:
            logger.warning("finish {}: {}", "verify_customer", refused.message)
            return {
                "refused": f"Not recorded: {refused.message}. Ask again, then call finish with a value that fits."
            }, None
        return await self._verify_customer_advance_verify_customer(_values)

    async def _verify_customer_advance_verify_customer(
        self, _values: dict[str, Any]
    ) -> tuple[dict[str, Any], Any]:
        """Move the flow on with this step's saved values: the next step, the owner, or the end.

        Args:
            _values: The values this step saved.

        Returns:
            The result the model sees, and the next node or None.
        """
        self._verify_customer_results["verify_customer"] = _values
        self._verify_customer_active_step = None
        # then: return — restore the owner's pre-flow context (messages and
        # tools); only a completed or unserved status crosses back.
        messages, tools = self._verify_customer_snapshot
        _settle_task_call(
            messages, "verify_customer", _group_status(self._verify_customer_results)
        )
        await self.queue_frame(
            LLMUpdateSettingsFrame(
                delta=LLMSettings(
                    system_instruction=_render(
                        COMPLAINT_SPECIALIST_PROMPT,
                        self.state,
                        site="agent:complaint_specialist",
                    )
                ),
            )
        )
        await self.flush_pipeline()
        self.context.set_messages(
            [
                *messages,
                *_carried_messages(self._verify_customer_carried_turn),
                {
                    "role": "developer",
                    "content": json.dumps(_group_status(self._verify_customer_results)),
                },
            ]
        )
        self.context.set_tools(tools)
        # Counted after the restore, so a turn a step consumed and carried back
        # is included: that turn is the one the owner is about to read, and it
        # is exactly the turn that must not be mistaken for a new request.
        self._ran_at_verify_customer = _caller_turns(self.context.get_messages())
        return {"status": "ok"}, None


def build_concierge_llm(state: CallState | None = None) -> LLMService[Any]:
    """Build the language model for this agent.

    Args:
        state: The call state, read by the router for its template variables.

    Returns:
        The configured LLM service.
    """
    return _GoogleVertexLLM(
        location="eu",
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GoogleLLMService.Settings(
            model="gemini-3.1-flash-lite",
            system_instruction=_render(CONCIERGE_PROMPT, state, site="agent:concierge"),
            thinking=GoogleLLMService.ThinkingConfig(**{"thinking_level": "MINIMAL"}),
        ),
    )


def build_concierge_tts(slng_session_id: str) -> TTSService:
    """Build the voice for this agent.

    Args:
        slng_session_id: This call's SLNG session id, sent with the agent id.

    Returns:
        The configured TTS service.
    """
    return SlngTTSService(
        api_key=os.environ["SLNG_API_KEY"],
        voice="62ae83ad-4f6a-430b-af41-a9bede9286ca",
        model="cartesia/sonic:3.5",
        language=Language("en"),
        warm_standby_enabled=True,
        world_part="eu-north",
        external_agent_id="salon-concierge",
        external_session_id=slng_session_id,
    )


class ConciergeAgent(TracedLLMWorker):
    """Agent: concierge."""

    def __init__(
        self,
        state: CallState | None = None,
        context: LLMContext | None = None,
        call_context: dict[str, Any] | None = None,
        dev_metrics: DevReporter | None = None,
        *,
        slng_session_id: str,
    ) -> None:
        """Build this agent's model and voice and join the call's shared context.

        Args:
            state: The call's typed variables, shared by every agent.
            context: The LLM context every agent on the call shares.
            call_context: The call's own facts and transport handles.
            dev_metrics: The dev-page reporter, or None outside `unmute dev`.
            slng_session_id: This call's SLNG session id, sent with the agent id.
        """
        # Always supplied when the package keeps state; None only in a test.
        self.state: CallState = cast(CallState, state)
        if context is not None:
            self.context = context
        self.call_context = call_context if call_context is not None else {}

        llm = build_concierge_llm(state)
        tts = build_concierge_tts(slng_session_id)
        self._dev = dev_metrics
        if self._dev:
            self._dev.observe_llm(llm)
            self._dev.observe_tts(tts)
        super().__init__(
            "concierge", llm=llm, pipeline=Pipeline([llm, tts]), bridged=()
        )
        # Every tool the model called in one response, recorded before any of
        # their handlers run. A step that ends on its own tool has to know
        # whether a handoff was called beside it: this framework runs both
        # handlers and whichever finishes first would otherwise close the step
        # under the other one.
        self._response_calls: tuple[str, ...] = ()

        @llm.event_handler("on_function_calls_started")
        async def _record_calls(service: Any, function_calls: list[Any]) -> None:
            """Remember the tools the model called in this response."""
            self._response_calls = tuple(call.function_name for call in function_calls)

    async def queue_frame(self, frame: Any, *args: Any, **kwargs: Any) -> None:
        """Stamp the frame for the dev page, then queue it as usual.

        Args:
            frame: The frame to queue.
            *args: Passed to the base worker.
            **kwargs: Passed to the base worker.
        """
        if self._dev:
            self._dev.stamp(frame)
        await super().queue_frame(frame, *args, **kwargs)

    async def on_activated(self, args: dict[str, Any] | None) -> None:
        """Set this agent's prompt and scope, then ask for its first reply when told to.

        Args:
            args: The activation arguments the previous agent or the entry passed.
        """
        token = self._dev.enter_activation(args) if self._dev else None
        try:
            await self.queue_frame(
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(
                        system_instruction=_render(
                            CONCIERGE_PROMPT, self.state, site="agent:concierge"
                        )
                    ),
                )
            )
            await super().on_activated(args)
            # Pipecat 1.8 and 1.9 only run on activation when messages are nonempty.
            # A handoff already shaped the shared context; request the reply
            # without adding a synthetic message or waiting for the caller.
            if args and args.get("run_llm") and not args.get("messages"):
                await self.queue_frame(LLMRunFrame())
        finally:
            if self._dev:
                self._dev.exit_activation(token)

    def _bind_state(
        self, handler: Callable[..., Awaitable[Any]]
    ) -> Callable[[Any, FlowManager], Awaitable[Any]]:
        """Give a module-level flows handler this worker's call state.

        A task tool can then read injected variables the same way an agent tool does.

        Args:
            handler: The module-level flows handler.

        Returns:
            A handler with the call state already bound.
        """

        async def _bound(args: Any, flow_manager: FlowManager) -> Any:
            """Call the handler with this worker's state."""
            return await handler(args, flow_manager, state=self.state)

        return _bound

    @_direct_tool(cancel_on_interruption=False)
    async def to_complaints(self, params: FunctionCallParams) -> None:
        """The caller has a complaint or service problem.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
        """
        # context.history on this handoff. One LLMContext is shared for the whole
        # call, so the receiver is given the shaped list rather than a copy.
        self.context.set_messages(_speech_only(self.context.get_messages()))
        await self.activate_worker(
            "complaint_specialist",
            args=LLMWorkerActivationArgs(
                metadata=self._dev.activation_metadata() if self._dev else None,
                messages=[],
                run_llm=True,
            ),
            deactivate_self=True,
            result_callback=params.result_callback,
        )

    @_direct_tool
    async def look_up_salon_info(self, params: FunctionCallParams, query: str) -> None:
        """Look up anything about the salon itself: services and prices, how long things take, opening hours, the stylists and what each one does, booking, deposits, cancellation and lateness terms, payment methods, parking, and accessibility. Use this whenever the caller asks a question about how the salon works, not only about price. Quote the document rather than estimating.
        Prefer this over asking the caller for anything. A question about the salon is not a booking request: look it up and answer it, and only start verification if the caller actually wants to make, move, or cancel an appointment.

        The results may not answer the question. They are ordered best first, and each carries a relevance score where a higher number means a closer match. If nothing returned actually answers what was asked, say you do not have that information rather than offering the closest result.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
            query (str): What to look up. Use the caller's own words.
        """  # noqa: D205
        await params.result_callback(await knowledge.look_up("services", query))

    @_direct_tool
    async def end_call(self, params: FunctionCallParams) -> None:
        """End the call once the caller's own words say the call is over: "no, that's all", "I'm good, thanks", "goodbye". A question is never one of those, however tired it sounds. On a live call on 2026-09-16 the caller asked "can we just recap when I have the next appointment?" and this ended the call on them. Answer what they asked and wait for them to say they are done.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
        """
        await params.llm.push_frame(
            LLMMessagesAppendFrame(
                [
                    {
                        "role": "developer",
                        "content": "Say one short goodbye in your own words, then end the call. Do not offer anything else first.",
                    }
                ],
                run_llm=False,
            )
        )
        await params.result_callback({"ended": True})
        await params.pipeline_worker.end()

    @_direct_tool(cancel_on_interruption=False)
    async def to_manager(self, params: FunctionCallParams) -> None:
        """The caller explicitly asks for a manager or is clearly and strongly frustrated.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
        """
        logger.info("human transfer fired: to_manager (cold)")
        transfer_result = self.call_context.get("_transfer_result")
        if transfer_result is not None:
            # Asked twice. Replay the first answer rather than dial again.
            await params.result_callback(transfer_result)
            return
        phone_call = self.call_context.get("_phone_call")
        if not phone_call or not phone_call.get("call_id"):
            # No phone call, so nothing to hand over: the browser and console
            # transports have no call for your carrier to redirect. Saying so beats
            # raising, and beats telling the model the caller was transferred.
            self.call_context["_transfer_result"] = {
                "failed": "this session is not a phone call, so it cannot be transferred"
            }
            await params.result_callback(self.call_context["_transfer_result"])
            return
        # Claimed before any transfer work awaits: a request arriving while
        # the first one is in flight must not slip past the guard.
        self.call_context["_transfer_result"] = {
            "in_progress": "A transfer is already under way; do not start another."
        }

        # The announcement is the first verb of the document below, spoken by your
        # carrier rather than by this agent. Applying the update replaces the call's
        # document, which tears this media stream down, so a line the agent started
        # speaking would be cut off by its own transfer.
        try:
            await _update_carrier_call(
                phone_call["call_id"],
                _transfer_twiml(os.environ["MANAGER_PHONE_NUMBER"], 30),
            )
        except Exception as exc:  # noqa: BLE001 - any failure leaves the call untouched
            # The request failed, so the call is untouched and the caller is still
            # here listening. Tell them, and never report a transfer that did not
            # happen.
            logger.warning("cold transfer failed: {}", exc)
            self.call_context["_transfer_result"] = {
                "failed": "The transfer could not be completed; the call is ending."
            }
            failure_error = None
            try:
                await params.llm.push_frame(
                    LLMMessagesAppendFrame(
                        [
                            {
                                "role": "developer",
                                "content": "Tell the caller nobody can take the call right now, apologize, and say goodbye.",
                            }
                        ],
                        run_llm=False,
                    )
                )
                await params.result_callback(self.call_context["_transfer_result"])
            except BaseException as exc:  # noqa: BLE001 - held, then re-raised once the call has ended
                failure_error = exc
            try:
                await params.pipeline_worker.end()
            except BaseException:
                if failure_error is None:
                    raise
                logger.exception("failed to end call after transfer failure")
            if failure_error is not None:
                raise failure_error  # noqa: B904 - the transfer error is logged above
            return
        # The stream ends as your carrier applies the document. Recorded first so a
        # request that arrives in that window replays this rather than re-firing.
        self.call_context["_transfer_result"] = {"transfer_started": True}
        await params.result_callback(self.call_context["_transfer_result"])

    @_direct_tool
    async def verify_customer(self, params: FunctionCallParams) -> None:
        """Run this when the caller explicitly corrects their phone number, and when a complaint is about to be written down. Never before a booking: the `book` flow runs this step itself when the saved number is not confirmed. A change of date, time or service is not a phone correction. The caller's words right above the status are what the step answered; it served the part it owns, so act only on what remains.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
        """
        # Already ran, and nobody has spoken since. Then this is not a second
        # request, it is the model reading its own completed result as though it
        # were one. On a live call on 2026-09-16 (trace 917975e9, LiveKit, same
        # package) a booking saved, the flow re-entered on the caller's own
        # "3 o'clock" still sitting above the result, and the last thing the
        # caller heard was an agent saying it was off to check the diary.
        #
        # Refused here rather than asked for in the prompt, which this package
        # did twice and which failed twice. Before the announcement below, so a
        # refused call speaks nothing at all.
        if _caller_turns(self.context.get_messages()) == getattr(
            self, "_ran_at_verify_customer", None
        ):
            await params.result_callback(
                {
                    "refused": "This already ran and finished, and the caller has "
                    "not spoken since, so its result is the answer to what they "
                    "asked. Reply to them from the values in your prompt. Run it "
                    "again only once they have asked for something new."
                }
            )
            return
        # Cover the entry: two model requests pass between the caller finishing and
        # the step's first word, and this puts speech in them. TTSSpeakFrame is a
        # DataFrame, so it is queued and speech starts while the flow is built.
        # Queued at the very start of the method, before anything else runs.
        await self.queue_frame(
            TTSSpeakFrame(
                _announce(
                    "delegate:verify_customer",
                    [
                        "Let me confirm who I'm speaking to.",
                        "One second, let me check who I've got.",
                        "Right, let me make sure I have you.",
                    ],
                )
            )
        )
        self._verify_customer_visit = object()
        self._verify_customer_results = {}
        # Per invocation: the tool that ended a step and what it returned, an
        # event a sibling handoff waits on, and the caller turn the step
        # consumed on its way out.
        self._verify_customer_terminal = None
        self._verify_customer_pending = False
        self._verify_customer_settled = asyncio.Event()
        self._verify_customer_carried_turn = None
        self._verify_customer_active_step = "verify_customer"
        flow = FlowManager(
            llm=self.llm,
            context_aggregator=LLMContextAggregatorPair(self.context),
            worker=self,
        )
        # Resolve this tool call so it never dangles (V3/B1), but with
        # run_llm=False: the flow's first node (respond_immediately) is the sole
        # responder, so the owner never runs a second completion and the caller
        # hears the task's opening line once, not twice (V7/B4).
        await params.result_callback(
            {"status": "running the verify_customer task"},
            properties=FunctionCallResultProperties(run_llm=False),
        )
        # V2/B2: drain the resolved owner call before snapshotting. Otherwise
        # restoration erases that call and the unchanged request delegates again.
        await self.flush_pipeline()
        self._verify_customer_snapshot = (
            copy.deepcopy(self.context.get_messages()),
            self.context.tools,
        )
        # How many caller turns the owner has seen, so a step that ends on its
        # tool can tell whether the turn it consumed is one the owner never saw.
        self._verify_customer_snapshot_turns = _caller_turns(
            self.context.get_messages()
        )
        # context.history on this task. Shaped after the snapshot above, so the
        # finish path restores the owner's own context whatever this step saw.
        self.context.set_messages(_speech_only(self.context.get_messages()))
        await flow.initialize(await self._verify_customer_node_verify_customer())

    def _verify_customer_settle(self) -> None:
        """Mark the running mutation over, one way or another."""
        self._verify_customer_pending = False
        self._verify_customer_settled.set()

    async def _verify_customer_node_verify_customer(self) -> NodeConfig:
        """Build the flow node for the verify_customer step.

        Returns:
            The node that runs this step's prompt and tools.
        """
        # This step's own ending, whatever the step before it did. Held on the
        # flow rather than the step, so the tool that ended the previous step
        # closed this one's mutations too: a verified caller was told their
        # first booking had already succeeded.
        self._verify_customer_terminal = None
        self._verify_customer_pending = False
        self._verify_customer_settled.clear()
        self._verify_customer_carried_turn = None
        # Entered, so whatever it confirms is being decided again. Withdrawn
        # before role_message renders, so this step never reads a confirmation
        # it is about to replace.
        self.state.withdraw_confirmation("verify_customer")
        self.context.set_messages(_speech_only(self.context.get_messages()))
        return NodeConfig(
            name="verify_customer",
            role_message=_render(
                VERIFY_CUSTOMER_TASK_PROMPT, self.state, site="task:verify_customer"
            ),
            task_messages=[
                {
                    "role": "developer",
                    "content": "Begin this step. Work from what the caller has already said.",
                }
            ],
            functions=[
                FlowsFunctionSchema(
                    name="find_or_create_customer",
                    description="Look up or create one salon customer from the exact confirmed phone number. Use only after the digit readback got a clear yes. Reuse the record that already owns the number, or create one for a number that is new. Never guess a number or pass one the caller has not confirmed. A caller giving a replacement number is not confirmation. Ask the readback question first; call this tool only after a later caller message agrees to it.",
                    properties={
                        "phone": {
                            "description": "The exact confirmed number in E.164: a plus sign, then 10 to 15 digits, with no spaces, brackets or dashes. The plus sign is part of the number and is never dropped; a text run sent the digits alone and the lookup could not tell which country it was. Never infer a country code the caller did not say.",
                            "type": "string",
                        }
                    },
                    required=["phone"],
                    handler=_flow_visit(
                        self,
                        "verify_customer",
                        self._trace_flow_tool(
                            "find_or_create_customer",
                            self._verify_customer_terminal_verify_customer_find_or_create_customer,
                        ),
                    ),
                ),
                FlowsFunctionSchema(
                    name="finish_verify_customer_verify_customer",
                    description="Record the result of this step and finish.",
                    properties=STEP_RESULTS["verify_customer"].tool_parameters()[
                        "properties"
                    ],
                    required=[],
                    handler=_flow_visit(
                        self,
                        "verify_customer",
                        self._trace_flow_tool(
                            "finish_verify_customer_verify_customer",
                            self._verify_customer_finish_verify_customer,
                        ),
                    ),
                ),
            ],
        )

    async def _verify_customer_terminal_verify_customer_find_or_create_customer(
        self, args: Any, flow_manager: FlowManager
    ) -> tuple[dict[str, Any], Any]:
        """Run find_or_create_customer and end this step when it says it did the work.

        The result never reaches the model on a success: the tool has already
        made the decision the model would have been asked to confirm. Anything
        else goes back as an ordinary result and the step stays open.

        Args:
            args: The arguments the model supplied.
            flow_manager: The flow this step runs in.

        Returns:
            The result the model sees, and whether a reply is generated.
        """
        if self._verify_customer_active_step != "verify_customer":
            return {"status": "already handled"}, NO_RESPONSE
        if self._verify_customer_terminal is not None:
            logger.info(
                "terminal tool refused: find_or_create_customer cannot run again in this step"
            )
            return {
                "refused": "find_or_create_customer already succeeded in this step. Call finish to record it; a new action is a new request from the caller."
            }, None
        # In flight from here until settled, whichever response called it: a
        # handoff or a finish arriving while it runs waits for it rather than
        # moving the caller past an action nobody recorded.
        _visit = self._verify_customer_visit
        self._verify_customer_pending = True
        self._verify_customer_settled.clear()
        try:
            result = await _flow_tool_find_or_create_customer(args, flow_manager)
        except BaseException:
            self._verify_customer_settle()
            raise
        if (
            self._verify_customer_visit is not _visit
            or self._verify_customer_active_step != "verify_customer"
        ):
            # The flow moved while the tool ran. Nothing here may write state
            # a later step now owns, so the result goes back as it is.
            self._verify_customer_settle()
            return result, None
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
            self._verify_customer_settle()
            return result, None
        self._verify_customer_terminal = ("find_or_create_customer", result)
        _messages = self.context.get_messages()
        if _caller_turns(_messages) > self._verify_customer_snapshot_turns:
            # The turn this call answered arrived after the owner's snapshot, so
            # the owner never saw it. Counted rather than matched on its text,
            # because a caller who says the same word twice has said it twice.
            self._verify_customer_carried_turn = _newest_caller_message(_messages)
        try:
            _values = self.state.save_result("verify_customer", result)
        except StateRefused as refused:
            # The action happened and the save did not. The step stays open so
            # the model can repair it, and its mutations stay closed so the
            # repair cannot run the action a second time.
            self._verify_customer_settle()
            logger.warning(
                "terminal find_or_create_customer: not recorded: " + refused.message
            )
            return {
                **result,
                "refused": "Not recorded: "
                + refused.message
                + ". Call finish with the values this tool returned; recover a missing one with a read tool of this step, or finish with unserved_request saying the action happened and its details could not be saved.",
            }, None
        self._verify_customer_settle()
        logger.info("step verify_customer ended on find_or_create_customer")
        # Saved once, above. The advance is the finish handler's own tail,
        # shared rather than copied, and it saves nothing itself: a second save
        # wrote a second entry onto every list an assign appends to.
        return await self._verify_customer_advance_verify_customer(_values)

    async def _verify_customer_finish_verify_customer(
        self, args: Any, flow_manager: FlowManager
    ) -> tuple[dict[str, Any], Any]:
        """Record this step's result and move the flow on.

        Args:
            args: The result fields the model supplied.
            flow_manager: The flow this step runs in.

        Returns:
            The result the model sees, and the next node or None.
        """
        if self._verify_customer_active_step != "verify_customer":
            return {"status": "already handled"}, NO_RESPONSE
        if self._verify_customer_pending:
            # A tool that ends this step is still running, and its result
            # decides the step. Wait for it rather than ending the step under it
            # and leaving what it did unrecorded.
            try:
                await asyncio.wait_for(
                    self._verify_customer_settled.wait(),
                    timeout=TERMINAL_SETTLE_TIMEOUT_SECS,
                )
            except TimeoutError:
                self._verify_customer_settle()
            if self._verify_customer_active_step != "verify_customer":
                return {"status": "already handled"}, NO_RESPONSE
        args = dict(args)
        if self._verify_customer_terminal is not None:
            # Repairing a save the tool already made. The tool's own values win
            # where they validate: the model cannot manufacture a reference, and
            # a repair that retyped one would record an action nobody took.
            args = STEP_RESULTS["verify_customer"].retain(
                args, self._verify_customer_terminal[1]
            )
        # Validated before anything is recorded: a value that does not fit its
        # declared type never enters the state, the previous contents stand, and
        # the message goes back to the model so it can correct itself on the next
        # turn instead of the step recording something wrong. This framework
        # validates no tool argument itself, so without this the two targets
        # would behave differently on the same package.
        try:
            _values = self.state.save_result("verify_customer", dict(args))
        except StateRefused as refused:
            logger.warning("finish {}: {}", "verify_customer", refused.message)
            return {
                "refused": f"Not recorded: {refused.message}. Ask again, then call finish with a value that fits."
            }, None
        return await self._verify_customer_advance_verify_customer(_values)

    async def _verify_customer_advance_verify_customer(
        self, _values: dict[str, Any]
    ) -> tuple[dict[str, Any], Any]:
        """Move the flow on with this step's saved values: the next step, the owner, or the end.

        Args:
            _values: The values this step saved.

        Returns:
            The result the model sees, and the next node or None.
        """
        self._verify_customer_results["verify_customer"] = _values
        self._verify_customer_active_step = None
        # then: return — restore the owner's pre-flow context (messages and
        # tools); only a completed or unserved status crosses back.
        messages, tools = self._verify_customer_snapshot
        _settle_task_call(
            messages, "verify_customer", _group_status(self._verify_customer_results)
        )
        await self.queue_frame(
            LLMUpdateSettingsFrame(
                delta=LLMSettings(
                    system_instruction=_render(
                        CONCIERGE_PROMPT, self.state, site="agent:concierge"
                    )
                ),
            )
        )
        await self.flush_pipeline()
        self.context.set_messages(
            [
                *messages,
                *_carried_messages(self._verify_customer_carried_turn),
                {
                    "role": "developer",
                    "content": json.dumps(_group_status(self._verify_customer_results)),
                },
            ]
        )
        self.context.set_tools(tools)
        # Counted after the restore, so a turn a step consumed and carried back
        # is included: that turn is the one the owner is about to read, and it
        # is exactly the turn that must not be mistaken for a new request.
        self._ran_at_verify_customer = _caller_turns(self.context.get_messages())
        return {"status": "ok"}, None

    @_direct_tool
    async def book(self, params: FunctionCallParams) -> None:
        """The caller wants to create, move or cancel a booking. Includes a change to an appointment just made. The caller's words right above the status are what the step answered; it served the part it owns, so act only on what remains.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
        """
        # Already ran, and nobody has spoken since. Then this is not a second
        # request, it is the model reading its own completed result as though it
        # were one. On a live call on 2026-09-16 (trace 917975e9, LiveKit, same
        # package) a booking saved, the flow re-entered on the caller's own
        # "3 o'clock" still sitting above the result, and the last thing the
        # caller heard was an agent saying it was off to check the diary.
        #
        # Refused here rather than asked for in the prompt, which this package
        # did twice and which failed twice. Before the announcement below, so a
        # refused call speaks nothing at all.
        if _caller_turns(self.context.get_messages()) == getattr(
            self, "_ran_at_book", None
        ):
            await params.result_callback(
                {
                    "refused": "This already ran and finished, and the caller has "
                    "not spoken since, so its result is the answer to what they "
                    "asked. Reply to them from the values in your prompt. Run it "
                    "again only once they have asked for something new."
                }
            )
            return
        self._book_visit = object()
        self._book_results = {}
        # Per invocation: the tool that ended a step and what it returned, an
        # event a sibling handoff waits on, and the caller turn the step
        # consumed on its way out.
        self._book_terminal = None
        self._book_pending = False
        self._book_settled = asyncio.Event()
        self._book_carried_turn = None
        # The steps this invocation will run, decided once as the flow starts.
        # A step whose confirmation already holds is not entered at all.
        self._book_plan = [
            name
            for name, confirmed in (
                ("verify_customer", "customer_phone"),
                ("manage_booking", ""),
            )
            if not (confirmed and self.state.is_confirmed(confirmed))
        ]

        self._book_active_step = self._book_plan[0]
        flow = FlowManager(
            llm=self.llm,
            context_aggregator=LLMContextAggregatorPair(self.context),
            worker=self,
        )
        # Resolve this tool call so it never dangles (V3/B1), but with
        # run_llm=False: the flow's first node (respond_immediately) is the sole
        # responder, so the owner never runs a second completion and the caller
        # hears the task's opening line once, not twice (V7/B4).
        await params.result_callback(
            {"status": "running the book flow"},
            properties=FunctionCallResultProperties(run_llm=False),
        )
        # V2/B2: drain the resolved owner call before snapshotting. Otherwise
        # restoration erases that call and the unchanged request delegates again.
        await self.flush_pipeline()
        self._book_snapshot = (
            copy.deepcopy(self.context.get_messages()),
            self.context.tools,
        )
        # How many caller turns the owner has seen, so a step that ends on its
        # tool can tell whether the turn it consumed is one the owner never saw.
        self._book_snapshot_turns = _caller_turns(self.context.get_messages())
        # context.history on this task. Shaped after the snapshot above, so the
        # finish path restores the owner's own context whatever this step saw.
        self.context.set_messages(_speech_only(self.context.get_messages()))
        await flow.initialize(await self._book_node(self._book_active_step))

    async def _book_node(self, name: str) -> NodeConfig:
        """Build one step's node by name, because the chain is decided at run time.

        Args:
            name: The step to enter.

        Returns:
            The step's flow node.
        """
        return await {
            "verify_customer": self._book_node_verify_customer,
            "manage_booking": self._book_node_manage_booking,
        }[name]()

    def _book_next(self, name: str) -> str | None:
        """Find the step after this one in this invocation's plan.

        Args:
            name: The step that just finished.

        Returns:
            The next step's name, or None when this was the last.
        """
        plan = self._book_plan
        position = plan.index(name) + 1
        return plan[position] if position < len(plan) else None

    def _book_settle(self) -> None:
        """Mark the running mutation over, one way or another."""
        self._book_pending = False
        self._book_settled.set()

    async def _book_node_verify_customer(self) -> NodeConfig:
        """Build the flow node for the verify_customer step.

        Returns:
            The node that runs this step's prompt and tools.
        """
        # This step's own ending, whatever the step before it did. Held on the
        # flow rather than the step, so the tool that ended the previous step
        # closed this one's mutations too: a verified caller was told their
        # first booking had already succeeded.
        self._book_terminal = None
        self._book_pending = False
        self._book_settled.clear()
        self._book_carried_turn = None
        # Entered, so whatever it confirms is being decided again. Withdrawn
        # before role_message renders, so this step never reads a confirmation
        # it is about to replace.
        self.state.withdraw_confirmation("verify_customer")
        self.context.set_messages(_speech_only(self.context.get_messages()))
        # Read once. Alternatives pick a line per call, so the seeded turn and
        # the spoken one have to come from the same read.
        _open = _announce(
            "task:verify_customer",
            [
                "Let me confirm who I'm speaking to.",
                "One second, let me check who I've got.",
                "Right, let me make sure I have you.",
            ],
        )
        # The step's own line, queued rather than handed to the framework as a
        # `tts_say` pre-action, and that is not a style choice. A pre-action
        # holds the node until the ActionFinishedFrame behind it reaches the
        # worker sink, and a frame this worker queues does not move until the
        # tool call that is building this flow returns. The first node of a
        # flow is built inside that call, so the two wait on each other: on a
        # live Pipecat call on 2026-09-16 a task group was entered, said
        # nothing, asked the model nothing, and held the caller for eleven
        # seconds. Queued, nothing waits, and the line is spoken in the same
        # place it would have been.
        #
        # append_to_context=False because the seeded assistant turn below is
        # how the line enters the context. Appending here as well would put it
        # in twice.
        await self.queue_frame(TTSSpeakFrame(_open, append_to_context=False))
        return NodeConfig(
            name="verify_customer",
            role_message=_render(
                BOOK_VERIFY_CUSTOMER_TASK_PROMPT,
                self.state,
                site="task:verify_customer",
            ),
            task_messages=[
                {"role": "assistant", "content": _open},
                {
                    "role": "developer",
                    "content": "Begin this step. Work from what the caller has already said.",
                },
            ],
            # The seeded turn above is the line the frame spoke, and it is what
            # stops the model opening by acknowledging a sentence it would
            # otherwise not be able to see: the frame carries no context write
            # of its own, and a live call on 2026-09-16 answered its own
            # announcement with "Got it.".
            #
            # The line goes first and the instruction last. The Gemini adapter
            # sends a developer message after the first as a user turn, so the
            # other order ended the request on the model's own sentence: on a
            # call on 2026-09-24 two step openings came back empty and the
            # caller sat through 15 and 10 seconds of silence, and a booking
            # step answered "Begin this step." by asking again for the service
            # and day the caller had already given.
            functions=[
                FlowsFunctionSchema(
                    name="find_or_create_customer",
                    description="Look up or create one salon customer from the exact confirmed phone number. Use only after the digit readback got a clear yes. Reuse the record that already owns the number, or create one for a number that is new. Never guess a number or pass one the caller has not confirmed. A caller giving a replacement number is not confirmation. Ask the readback question first; call this tool only after a later caller message agrees to it.",
                    properties={
                        "phone": {
                            "description": "The exact confirmed number in E.164: a plus sign, then 10 to 15 digits, with no spaces, brackets or dashes. The plus sign is part of the number and is never dropped; a text run sent the digits alone and the lookup could not tell which country it was. Never infer a country code the caller did not say.",
                            "type": "string",
                        }
                    },
                    required=["phone"],
                    handler=_flow_visit(
                        self,
                        "book",
                        self._trace_flow_tool(
                            "find_or_create_customer",
                            self._book_terminal_verify_customer_find_or_create_customer,
                        ),
                    ),
                ),
                FlowsFunctionSchema(
                    name="finish_book_verify_customer",
                    description="Record the result of this step and finish.",
                    properties=STEP_RESULTS["verify_customer"].tool_parameters()[
                        "properties"
                    ],
                    required=[],
                    handler=_flow_visit(
                        self,
                        "book",
                        self._trace_flow_tool(
                            "finish_book_verify_customer",
                            self._book_finish_verify_customer,
                        ),
                    ),
                ),
            ],
        )

    async def _book_terminal_verify_customer_find_or_create_customer(
        self, args: Any, flow_manager: FlowManager
    ) -> tuple[dict[str, Any], Any]:
        """Run find_or_create_customer and end this step when it says it did the work.

        The result never reaches the model on a success: the tool has already
        made the decision the model would have been asked to confirm. Anything
        else goes back as an ordinary result and the step stays open.

        Args:
            args: The arguments the model supplied.
            flow_manager: The flow this step runs in.

        Returns:
            The result the model sees, and whether a reply is generated.
        """
        if self._book_active_step != "verify_customer":
            return {"status": "already handled"}, NO_RESPONSE
        if self._book_terminal is not None:
            logger.info(
                "terminal tool refused: find_or_create_customer cannot run again in this step"
            )
            return {
                "refused": "find_or_create_customer already succeeded in this step. Call finish to record it; a new action is a new request from the caller."
            }, None
        # In flight from here until settled, whichever response called it: a
        # handoff or a finish arriving while it runs waits for it rather than
        # moving the caller past an action nobody recorded.
        _visit = self._book_visit
        self._book_pending = True
        self._book_settled.clear()
        try:
            result = await _flow_tool_find_or_create_customer(args, flow_manager)
        except BaseException:
            self._book_settle()
            raise
        if (
            self._book_visit is not _visit
            or self._book_active_step != "verify_customer"
        ):
            # The flow moved while the tool ran. Nothing here may write state
            # a later step now owns, so the result goes back as it is.
            self._book_settle()
            return result, None
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
            self._book_settle()
            return result, None
        self._book_terminal = ("find_or_create_customer", result)
        _messages = self.context.get_messages()
        if _caller_turns(_messages) > self._book_snapshot_turns:
            # The turn this call answered arrived after the owner's snapshot, so
            # the owner never saw it. Counted rather than matched on its text,
            # because a caller who says the same word twice has said it twice.
            self._book_carried_turn = _newest_caller_message(_messages)
        try:
            _values = self.state.save_result("verify_customer", result)
        except StateRefused as refused:
            # The action happened and the save did not. The step stays open so
            # the model can repair it, and its mutations stay closed so the
            # repair cannot run the action a second time.
            self._book_settle()
            logger.warning(
                "terminal find_or_create_customer: not recorded: " + refused.message
            )
            return {
                **result,
                "refused": "Not recorded: "
                + refused.message
                + ". Call finish with the values this tool returned; recover a missing one with a read tool of this step, or finish with unserved_request saying the action happened and its details could not be saved.",
            }, None
        self._book_settle()
        logger.info("step verify_customer ended on find_or_create_customer")
        # Saved once, above. The advance is the finish handler's own tail,
        # shared rather than copied, and it saves nothing itself: a second save
        # wrote a second entry onto every list an assign appends to.
        return await self._book_advance_verify_customer(_values)

    async def _book_finish_verify_customer(
        self, args: Any, flow_manager: FlowManager
    ) -> tuple[dict[str, Any], Any]:
        """Record this step's result and move the flow on.

        Args:
            args: The result fields the model supplied.
            flow_manager: The flow this step runs in.

        Returns:
            The result the model sees, and the next node or None.
        """
        if self._book_active_step != "verify_customer":
            return {"status": "already handled"}, NO_RESPONSE
        if self._book_pending:
            # A tool that ends this step is still running, and its result
            # decides the step. Wait for it rather than ending the step under it
            # and leaving what it did unrecorded.
            try:
                await asyncio.wait_for(
                    self._book_settled.wait(), timeout=TERMINAL_SETTLE_TIMEOUT_SECS
                )
            except TimeoutError:
                self._book_settle()
            if self._book_active_step != "verify_customer":
                return {"status": "already handled"}, NO_RESPONSE
        args = dict(args)
        if self._book_terminal is not None:
            # Repairing a save the tool already made. The tool's own values win
            # where they validate: the model cannot manufacture a reference, and
            # a repair that retyped one would record an action nobody took.
            args = STEP_RESULTS["verify_customer"].retain(args, self._book_terminal[1])
        # Validated before anything is recorded: a value that does not fit its
        # declared type never enters the state, the previous contents stand, and
        # the message goes back to the model so it can correct itself on the next
        # turn instead of the step recording something wrong. This framework
        # validates no tool argument itself, so without this the two targets
        # would behave differently on the same package.
        try:
            _values = self.state.save_result("verify_customer", dict(args))
        except StateRefused as refused:
            logger.warning("finish {}: {}", "verify_customer", refused.message)
            return {
                "refused": f"Not recorded: {refused.message}. Ask again, then call finish with a value that fits."
            }, None
        return await self._book_advance_verify_customer(_values)

    async def _book_advance_verify_customer(
        self, _values: dict[str, Any]
    ) -> tuple[dict[str, Any], Any]:
        """Move the flow on with this step's saved values: the next step, the owner, or the end.

        Args:
            _values: The values this step saved.

        Returns:
            The result the model sees, and the next node or None.
        """
        self._book_results["verify_customer"] = _values
        _next = self._book_next("verify_customer")
        if self._book_results["verify_customer"].get("unserved_request"):
            # The step could not serve what the caller asked. Running the next
            # one would answer a question nobody asked, so the flow stops here
            # and the owner is handed the unserved status.
            logger.info("group stopped: verify_customer ended unserved")
            _next = None
        self._book_active_step = _next
        if _next is not None:
            return _task_status(
                self._book_results["verify_customer"]
            ), await self._book_node(_next)
        flow_messages = None
        flow_tools = None
        completion_started = False
        try:
            flow_messages = copy.deepcopy(self.context.get_messages())
            flow_tools = self.context.tools
            completion_started = True
            return await self._book_complete_verify_customer()
        except BaseException as error:
            # The save already committed. Keep its terminal claim; a retry cannot save it twice.
            try:
                if flow_messages is not None:
                    self.context.set_messages(flow_messages)
                if flow_tools is not None:
                    self.context.set_tools(flow_tools)
                if completion_started:
                    await self.queue_frame(
                        LLMUpdateSettingsFrame(
                            delta=LLMSettings(
                                system_instruction=_render(
                                    BOOK_VERIFY_CUSTOMER_TASK_PROMPT,
                                    self.state,
                                    site="task:verify_customer",
                                )
                            ),
                        )
                    )
                    await self.flush_pipeline()
            except BaseException as rollback_error:
                raise error from rollback_error
            raise

    async def _book_complete_verify_customer(self) -> tuple[dict[str, Any], Any]:
        """Hand control back to the owner, or end the call, once the last step is done.

        Returns:
            The result the model sees, and the end node or None.
        """
        # then: return — restore the owner's pre-flow context (messages and
        # tools); only a completed or unserved status crosses back.
        messages, tools = self._book_snapshot
        _settle_task_call(messages, "book", _group_status(self._book_results))
        await self.queue_frame(
            LLMUpdateSettingsFrame(
                delta=LLMSettings(
                    system_instruction=_render(
                        CONCIERGE_PROMPT, self.state, site="agent:concierge"
                    )
                ),
            )
        )
        await self.flush_pipeline()
        self.context.set_messages(
            [
                *messages,
                *_carried_messages(self._book_carried_turn),
                {
                    "role": "developer",
                    "content": json.dumps(_group_status(self._book_results)),
                },
            ]
        )
        self.context.set_tools(tools)
        # Counted after the restore, so a turn a step consumed and carried back
        # is included: that turn is the one the owner is about to read, and it
        # is exactly the turn that must not be mistaken for a new request.
        self._ran_at_book = _caller_turns(self.context.get_messages())
        return {"status": "ok"}, None

    async def _book_node_manage_booking(self) -> NodeConfig:
        """Build the flow node for the manage_booking step.

        Returns:
            The node that runs this step's prompt and tools.
        """
        # This step's own ending, whatever the step before it did. Held on the
        # flow rather than the step, so the tool that ended the previous step
        # closed this one's mutations too: a verified caller was told their
        # first booking had already succeeded.
        self._book_terminal = None
        self._book_pending = False
        self._book_settled.clear()
        self._book_carried_turn = None
        self.context.set_messages(_speech_only(self.context.get_messages()))
        # Read once. Alternatives pick a line per call, so the seeded turn and
        # the spoken one have to come from the same read.
        _open = _announce(
            "task:manage_booking",
            ["One sec.", "Let me have a look.", "One moment, let me look."],
        )
        # The step's own line, queued rather than handed to the framework as a
        # `tts_say` pre-action, and that is not a style choice. A pre-action
        # holds the node until the ActionFinishedFrame behind it reaches the
        # worker sink, and a frame this worker queues does not move until the
        # tool call that is building this flow returns. The first node of a
        # flow is built inside that call, so the two wait on each other: on a
        # live Pipecat call on 2026-09-16 a task group was entered, said
        # nothing, asked the model nothing, and held the caller for eleven
        # seconds. Queued, nothing waits, and the line is spoken in the same
        # place it would have been.
        #
        # append_to_context=False because the seeded assistant turn below is
        # how the line enters the context. Appending here as well would put it
        # in twice.
        await self.queue_frame(TTSSpeakFrame(_open, append_to_context=False))
        return NodeConfig(
            name="manage_booking",
            role_message=_render(
                BOOK_MANAGE_BOOKING_TASK_PROMPT, self.state, site="task:manage_booking"
            ),
            task_messages=[
                {"role": "assistant", "content": _open},
                {
                    "role": "developer",
                    "content": "Begin this step. Work from what the caller has already said.",
                },
            ],
            # The seeded turn above is the line the frame spoke, and it is what
            # stops the model opening by acknowledging a sentence it would
            # otherwise not be able to see: the frame carries no context write
            # of its own, and a live call on 2026-09-16 answered its own
            # announcement with "Got it.".
            #
            # The line goes first and the instruction last. The Gemini adapter
            # sends a developer message after the first as a user turn, so the
            # other order ended the request on the model's own sentence: on a
            # call on 2026-09-24 two step openings came back empty and the
            # caller sat through 15 and 10 seconds of silence, and a booking
            # step answered "Begin this step." by asking again for the service
            # and day the caller had already given.
            functions=[
                FlowsFunctionSchema(
                    name="find_slots",
                    description="Read the diary: what the caller already has booked, and what times are free. Call this before offering any time and before changing any booking. It never writes anything, so call it whenever you are unsure what the caller holds.\nGive a date to see free times on that date. Leave the service out to see every service. The caller's own bookings come back every time, whatever you asked, so you never need a second call to find out what they hold.\nAsk the caller which day they want before you call this, and never guess one. A call for a day nobody named answers nothing, and the second call you then have to make speaks a second announcement the caller did not need to hear.",
                    properties={
                        "date": {
                            "description": 'The day to look at, YYYY-MM-DD. Always a real day, never an empty string: the caller\'s own bookings come back whatever day you ask for, so an empty date buys nothing and answers nothing.\nSend the day the caller named, and **never a day they did not name**. Today is not a default. A caller who asks for an appointment without saying when needs the question "which day?", not a guess: a guess reads the wrong day, comes back with nothing, and costs them the wait twice over. A live call on 2026-09-16 guessed today, got `full`, called again for tomorrow, and the caller heard two announcements in three seconds.',
                            "type": "string",
                        },
                        "service": {
                            "description": "Narrow the free times to one service, or `any` to see all three. Use `any` when the caller has not said which service they want.",
                            "enum": ["any", "haircut", "hair-color", "blowout"],
                            "type": "string",
                        },
                    },
                    required=["date", "service"],
                    handler=self._trace_flow_tool(
                        "find_slots", self._bind_state(_flow_tool_find_slots)
                    ),
                ),
                FlowsFunctionSchema(
                    name="save_booking",
                    description="Write the diary: book a new time, move an existing booking, or cancel one. This is the only tool that changes anything, so nothing is booked, moved or cancelled until it runs and says so.\nCall it only after the caller has heard the whole thing back in one sentence, the service, the day, the time and their phone number, and has said yes to that sentence. Choosing a time is not a yes. Use a slot_id and a booking_id exactly as find_slots returned them; never invent either.",
                    properties={
                        "action": {
                            "description": "book a new time, move an existing booking, or cancel one",
                            "enum": ["book", "move", "cancel"],
                            "type": "string",
                        },
                        "additional": {
                            "description": "True only when the caller asked for a second appointment on top of one they already hold. A caller who holds a booking is changing it, so this is false for an ordinary booking and the tool refuses with has_booking.",
                            "type": "boolean",
                        },
                        "booking_id": {
                            "description": "The exact booking_id from find_slots when moving or cancelling. An empty string when booking, which has no booking yet.",
                            "type": "string",
                        },
                        "confirmed": {
                            "description": "True only after the caller said yes to the sentence you read back.",
                            "type": "boolean",
                        },
                        "slot_id": {
                            "description": "The exact slot_id from find_slots when booking or moving. An empty string when cancelling, which needs no slot.",
                            "type": "string",
                        },
                    },
                    required=[
                        "action",
                        "slot_id",
                        "booking_id",
                        "confirmed",
                        "additional",
                    ],
                    handler=_flow_visit(
                        self,
                        "book",
                        self._trace_flow_tool(
                            "save_booking",
                            self._book_terminal_manage_booking_save_booking,
                        ),
                    ),
                ),
                FlowsFunctionSchema(
                    name="to_complaints",
                    description="The caller has a complaint or service problem.",
                    properties={},
                    required=[],
                    handler=_flow_visit(
                        self,
                        "book",
                        self._trace_flow_tool(
                            "to_complaints",
                            self._book_transfer_manage_booking_to_complaints,
                        ),
                    ),
                ),
                FlowsFunctionSchema(
                    name="finish_book_manage_booking",
                    description="Record the result of this step and finish.",
                    properties=STEP_RESULTS["manage_booking"].tool_parameters()[
                        "properties"
                    ],
                    required=[],
                    handler=_flow_visit(
                        self,
                        "book",
                        self._trace_flow_tool(
                            "finish_book_manage_booking",
                            self._book_finish_manage_booking,
                        ),
                    ),
                ),
            ],
        )

    async def _book_transfer_manage_booking_to_complaints(
        self, args: Any, flow_manager: FlowManager
    ) -> tuple[dict[str, Any], Any]:
        """Hand the caller to complaint_specialist from inside the manage_booking step.

        Args:
            args: The arguments the model supplied, none for a handoff.
            flow_manager: The flow this step runs in.

        Returns:
            The result the model sees, and whether a reply is generated.
        """
        if self._book_active_step != "manage_booking":
            return {"status": "already handled"}, NO_RESPONSE
        if self._book_pending or not {"save_booking"}.isdisjoint(self._response_calls):
            # A tool that ends this step is running, or was called in the same
            # response and has not started yet. The caller asked for two things
            # at once, and the step's own work has to be committed before the
            # caller is moved.
            try:
                await asyncio.wait_for(
                    self._book_settled.wait(), timeout=TERMINAL_SETTLE_TIMEOUT_SECS
                )
            except TimeoutError:
                self._book_settle()
            if self._book_active_step != "manage_booking":
                # The flow moved on while this handoff waited.
                return {"status": "already handled"}, NO_RESPONSE
            if "manage_booking" not in self._book_results:
                logger.info("handoff held: the step's own action did not complete")
                return {"status": "not transferred: the action did not complete"}, None
        self._book_active_step = None
        flow_messages = None
        flow_tools = None
        try:
            flow_messages = copy.deepcopy(self.context.get_messages())
            flow_tools = self.context.tools
            _, tools = self._book_snapshot
            self.context.set_messages(_speech_only(self.context.get_messages()))
            self.context.set_tools(tools)
            await self.activate_worker(
                "complaint_specialist",
                args=LLMWorkerActivationArgs(
                    metadata=self._dev.activation_metadata() if self._dev else None,
                    messages=[],
                    run_llm=True,
                ),
                deactivate_self=True,
            )
        except BaseException:
            self._book_active_step = "manage_booking"
            if flow_messages is not None:
                self.context.set_messages(flow_messages)
            if flow_tools is not None:
                self.context.set_tools(flow_tools)
            raise
        return {"transferred": True}, NO_RESPONSE

    async def _book_terminal_manage_booking_save_booking(
        self, args: Any, flow_manager: FlowManager
    ) -> tuple[dict[str, Any], Any]:
        """Run save_booking and end this step when it says it did the work.

        The result never reaches the model on a success: the tool has already
        made the decision the model would have been asked to confirm. Anything
        else goes back as an ordinary result and the step stays open.

        Args:
            args: The arguments the model supplied.
            flow_manager: The flow this step runs in.

        Returns:
            The result the model sees, and whether a reply is generated.
        """
        if self._book_active_step != "manage_booking":
            return {"status": "already handled"}, NO_RESPONSE
        if self._book_terminal is not None:
            logger.info(
                "terminal tool refused: save_booking cannot run again in this step"
            )
            return {
                "refused": "save_booking already succeeded in this step. Call finish to record it; a new action is a new request from the caller."
            }, None
        # In flight from here until settled, whichever response called it: a
        # handoff or a finish arriving while it runs waits for it rather than
        # moving the caller past an action nobody recorded.
        _visit = self._book_visit
        self._book_pending = True
        self._book_settled.clear()
        try:
            result = await _flow_tool_save_booking(args, flow_manager, state=self.state)
        except BaseException:
            self._book_settle()
            raise
        if self._book_visit is not _visit or self._book_active_step != "manage_booking":
            # The flow moved while the tool ran. Nothing here may write state
            # a later step now owns, so the result goes back as it is.
            self._book_settle()
            return result, None
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
            self._book_settle()
            return result, None
        self._book_terminal = ("save_booking", result)
        _messages = self.context.get_messages()
        if _caller_turns(_messages) > self._book_snapshot_turns:
            # The turn this call answered arrived after the owner's snapshot, so
            # the owner never saw it. Counted rather than matched on its text,
            # because a caller who says the same word twice has said it twice.
            self._book_carried_turn = _newest_caller_message(_messages)
        try:
            _values = self.state.save_result("manage_booking", result)
        except StateRefused as refused:
            # The action happened and the save did not. The step stays open so
            # the model can repair it, and its mutations stay closed so the
            # repair cannot run the action a second time.
            self._book_settle()
            logger.warning("terminal save_booking: not recorded: " + refused.message)
            return {
                **result,
                "refused": "Not recorded: "
                + refused.message
                + ". Call finish with the values this tool returned; recover a missing one with a read tool of this step, or finish with unserved_request saying the action happened and its details could not be saved.",
            }, None
        self._book_settle()
        logger.info("step manage_booking ended on save_booking")
        if not {"to_complaints"}.isdisjoint(self._response_calls):
            # A handoff was called in the same response. It owns the ending: the
            # save is committed, and advancing here would race its transition.
            # Recorded first, because the handoff reads this to know the step's
            # own work actually landed.
            self._book_results["manage_booking"] = _values
            return _task_status(_values), NO_RESPONSE
        # Saved once, above. The advance is the finish handler's own tail,
        # shared rather than copied, and it saves nothing itself: a second save
        # wrote a second entry onto every list an assign appends to.
        return await self._book_advance_manage_booking(_values)

    async def _book_finish_manage_booking(
        self, args: Any, flow_manager: FlowManager
    ) -> tuple[dict[str, Any], Any]:
        """Record this step's result and move the flow on.

        Args:
            args: The result fields the model supplied.
            flow_manager: The flow this step runs in.

        Returns:
            The result the model sees, and the next node or None.
        """
        if self._book_active_step != "manage_booking":
            return {"status": "already handled"}, NO_RESPONSE
        if self._book_pending:
            # A tool that ends this step is still running, and its result
            # decides the step. Wait for it rather than ending the step under it
            # and leaving what it did unrecorded.
            try:
                await asyncio.wait_for(
                    self._book_settled.wait(), timeout=TERMINAL_SETTLE_TIMEOUT_SECS
                )
            except TimeoutError:
                self._book_settle()
            if self._book_active_step != "manage_booking":
                return {"status": "already handled"}, NO_RESPONSE
        args = dict(args)
        if self._book_terminal is not None:
            # Repairing a save the tool already made. The tool's own values win
            # where they validate: the model cannot manufacture a reference, and
            # a repair that retyped one would record an action nobody took.
            args = STEP_RESULTS["manage_booking"].retain(args, self._book_terminal[1])
        # Validated before anything is recorded: a value that does not fit its
        # declared type never enters the state, the previous contents stand, and
        # the message goes back to the model so it can correct itself on the next
        # turn instead of the step recording something wrong. This framework
        # validates no tool argument itself, so without this the two targets
        # would behave differently on the same package.
        try:
            _values = self.state.save_result("manage_booking", dict(args))
        except StateRefused as refused:
            logger.warning("finish {}: {}", "manage_booking", refused.message)
            return {
                "refused": f"Not recorded: {refused.message}. Ask again, then call finish with a value that fits."
            }, None
        return await self._book_advance_manage_booking(_values)

    async def _book_advance_manage_booking(
        self, _values: dict[str, Any]
    ) -> tuple[dict[str, Any], Any]:
        """Move the flow on with this step's saved values: the next step, the owner, or the end.

        Args:
            _values: The values this step saved.

        Returns:
            The result the model sees, and the next node or None.
        """
        self._book_results["manage_booking"] = _values
        _next = self._book_next("manage_booking")
        if self._book_results["manage_booking"].get("unserved_request"):
            # The step could not serve what the caller asked. Running the next
            # one would answer a question nobody asked, so the flow stops here
            # and the owner is handed the unserved status.
            logger.info("group stopped: manage_booking ended unserved")
            _next = None
        self._book_active_step = _next
        if _next is not None:
            return _task_status(
                self._book_results["manage_booking"]
            ), await self._book_node(_next)
        flow_messages = None
        flow_tools = None
        completion_started = False
        try:
            flow_messages = copy.deepcopy(self.context.get_messages())
            flow_tools = self.context.tools
            completion_started = True
            return await self._book_complete_manage_booking()
        except BaseException as error:
            # The save already committed. Keep its terminal claim; a retry cannot save it twice.
            try:
                if flow_messages is not None:
                    self.context.set_messages(flow_messages)
                if flow_tools is not None:
                    self.context.set_tools(flow_tools)
                if completion_started:
                    await self.queue_frame(
                        LLMUpdateSettingsFrame(
                            delta=LLMSettings(
                                system_instruction=_render(
                                    BOOK_MANAGE_BOOKING_TASK_PROMPT,
                                    self.state,
                                    site="task:manage_booking",
                                )
                            ),
                        )
                    )
                    await self.flush_pipeline()
            except BaseException as rollback_error:
                raise error from rollback_error
            raise

    async def _book_complete_manage_booking(self) -> tuple[dict[str, Any], Any]:
        """Hand control back to the owner, or end the call, once the last step is done.

        Returns:
            The result the model sees, and the end node or None.
        """
        # then: return — restore the owner's pre-flow context (messages and
        # tools); only a completed or unserved status crosses back.
        messages, tools = self._book_snapshot
        _settle_task_call(messages, "book", _group_status(self._book_results))
        await self.queue_frame(
            LLMUpdateSettingsFrame(
                delta=LLMSettings(
                    system_instruction=_render(
                        CONCIERGE_PROMPT, self.state, site="agent:concierge"
                    )
                ),
            )
        )
        await self.flush_pipeline()
        self.context.set_messages(
            [
                *messages,
                *_carried_messages(self._book_carried_turn),
                {
                    "role": "developer",
                    "content": json.dumps(_group_status(self._book_results)),
                },
            ]
        )
        self.context.set_tools(tools)
        # Counted after the restore, so a turn a step consumed and carried back
        # is included: that turn is the one the owner is about to read, and it
        # is exactly the turn that must not be mistaken for a new request.
        self._ran_at_book = _caller_turns(self.context.get_messages())
        return {"status": "ok"}, None


def _flow_visit(
    worker: Any, delegate: str, handler: Callable[..., Awaitable[Any]]
) -> Callable[[Any, FlowManager], Awaitable[Any]]:
    """Wrap a step handler so it does nothing once its flow has been left.

    Args:
        worker: The worker that owns the flow.
        delegate: The delegate tool that started this flow.
        handler: The step's handler.

    Returns:
        The guarded handler.
    """
    visit = getattr(worker, "_" + delegate + "_visit", None)

    async def invoke(args: Any, flow_manager: FlowManager) -> Any:
        """Run the handler only while this visit is still the current one."""
        if getattr(worker, "_" + delegate + "_visit", None) is not visit:
            return {"status": "already handled"}, NO_RESPONSE
        return await handler(args, flow_manager)

    return invoke


# --- task tools (flows handlers) ----------------------------------------------
# Tools available inside task steps; stable module-level handlers so a
# re-registered function name always resolves to the same callable.


async def _flow_tool_find_or_create_customer(
    args: Any, flow_manager: FlowManager
) -> Any:
    """Look up or create one salon customer from the exact confirmed phone number. Use only after the digit readback got a clear yes. Reuse the record that already owns the number, or create one for a number that is new. Never guess a number or pass one the caller has not confirmed. A caller giving a replacement number is not confirmation. Ask the readback question first; call this tool only after a later caller message agrees to it.

    Args:
        args (Any): The arguments the model supplied.
        flow_manager (FlowManager): The flow the step runs in.
    """
    result = tools.find_or_create_customer.find_or_create_customer(**dict(args))
    if inspect.isawaitable(result):
        result = await result
    return result


async def _flow_tool_find_slots(
    args: Any, flow_manager: FlowManager, state: CallState
) -> Any:
    """Read the diary: what the caller already has booked, and what times are free. Call this before offering any time and before changing any booking. It never writes anything, so call it whenever you are unsure what the caller holds.
    Give a date to see free times on that date. Leave the service out to see every service. The caller's own bookings come back every time, whatever you asked, so you never need a second call to find out what they hold.
    Ask the caller which day they want before you call this, and never guess one. A call for a day nobody named answers nothing, and the second call you then have to make speaks a second announcement the caller did not need to hear.

    Args:
        args (Any): The arguments the model supplied.
        flow_manager (FlowManager): The flow the step runs in.
        state (State): The call's typed variables.
    """  # noqa: D205
    refusal = _refusal(
        "find_slots", state, [("customer_phone", "run verify_customer first.")]
    )
    if refusal:
        return {"refused": refusal}
    result = tools.find_slots.find_slots(
        **dict(args), customer_phone=state.plain("customer_phone")
    )
    if inspect.isawaitable(result):
        result = await result
    return result


async def _flow_tool_save_booking(
    args: Any, flow_manager: FlowManager, state: CallState
) -> Any:
    """Write the diary: book a new time, move an existing booking, or cancel one. This is the only tool that changes anything, so nothing is booked, moved or cancelled until it runs and says so.
    Call it only after the caller has heard the whole thing back in one sentence, the service, the day, the time and their phone number, and has said yes to that sentence. Choosing a time is not a yes. Use a slot_id and a booking_id exactly as find_slots returned them; never invent either.

    Args:
        args (Any): The arguments the model supplied.
        flow_manager (FlowManager): The flow the step runs in.
        state (State): The call's typed variables.
    """  # noqa: D205
    refusal = _refusal(
        "save_booking", state, [("customer_phone", "run verify_customer first.")]
    )
    if refusal:
        return {"refused": refusal}
    result = tools.save_booking.save_booking(
        **dict(args), customer_phone=state.plain("customer_phone")
    )
    if inspect.isawaitable(result):
        result = await result
    return result
