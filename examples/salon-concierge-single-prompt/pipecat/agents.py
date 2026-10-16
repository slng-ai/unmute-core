"""The agents, their tools, and the task steps they run as flows."""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Callable
from typing import Any, cast

from loguru import logger
from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMUpdateSettingsFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.workers.llm import tool
from pipecat_slng import SlngTTSService

import tools.find_or_create_customer
import tools.find_slots
import tools.record_complaint
import tools.save_booking
from call_state import CallState
from prompts import CONCIERGE_PROMPT
from session import _render
from utils import knowledge
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
        external_agent_id="salon-concierge-single-prompt",
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
    async def find_or_create_customer(
        self, params: FunctionCallParams, phone: str
    ) -> None:
        """Look up or create one salon customer from the exact confirmed phone number. Use only after the digit readback got a clear yes. Reuse the record that already owns the number, or create one for a number that is new. Never guess a number or pass one the caller has not confirmed. A caller giving a replacement number is not confirmation. Ask the readback question first; call this tool only after a later caller message agrees to it.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
            phone (str): The exact confirmed number in E.164: a plus sign, then 10 to 15 digits, with no spaces, brackets or dashes. The plus sign is part of the number and is never dropped; a text run sent the digits alone and the lookup could not tell which country it was. Never infer a country code the caller did not say.
        """
        result = tools.find_or_create_customer.find_or_create_customer(phone=phone)
        if inspect.isawaitable(result):
            result = await result
        await params.result_callback(result)

    @_direct_tool
    async def find_slots(
        self, params: FunctionCallParams, customer_phone: str, date: str, service: str
    ) -> None:
        """Read the diary: what the caller already has booked, and what times are free. Call this before offering any time and before changing any booking. It never writes anything, so call it whenever you are unsure what the caller holds.
        Give a date to see free times on that date. Leave the service out to see every service. The caller's own bookings come back every time, whatever you asked, so you never need a second call to find out what they hold.
        Ask the caller which day they want before you call this, and never guess one. A call for a day nobody named answers nothing, and the second call you then have to make speaks a second announcement the caller did not need to hear.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
            customer_phone (str): The caller's number exactly as find_or_create_customer returned it, in E.164: a plus sign then digits, with no spaces. Copy it, never retype it.
            date (str): The day to look at, YYYY-MM-DD. Always a real day, never an empty string: the caller's own bookings come back whatever day you ask for, so an empty date buys nothing and answers nothing.
                Send the day the caller named, and **never a day they did not name**. Today is not a default. A caller who asks for an appointment without saying when needs the question "which day?", not a guess: a guess reads the wrong day, comes back with nothing, and costs them the wait twice over. A live call on 2026-09-16 guessed today, got `full`, called again for tomorrow, and the caller heard two announcements in three seconds.
            service (str): Narrow the free times to one service, or `any` to see all three. Use `any` when the caller has not said which service they want. One of: any, haircut, hair-color, blowout.
        """  # noqa: D205
        result = tools.find_slots.find_slots(
            customer_phone=customer_phone, date=date, service=service
        )
        if inspect.isawaitable(result):
            result = await result
        await params.result_callback(result)

    @_direct_tool
    async def save_booking(
        self,
        params: FunctionCallParams,
        action: str,
        additional: bool,
        booking_id: str,
        confirmed: bool,
        customer_phone: str,
        slot_id: str,
    ) -> None:
        """Write the diary: book a new time, move an existing booking, or cancel one. This is the only tool that changes anything, so nothing is booked, moved or cancelled until it runs and says so.
        Call it only after the caller has heard the whole thing back in one sentence, the service, the day, the time and their phone number, and has said yes to that sentence. Choosing a time is not a yes. Use a slot_id and a booking_id exactly as find_slots returned them; never invent either.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
            action (str): book a new time, move an existing booking, or cancel one One of: book, move, cancel.
            additional (bool): True only when the caller asked for a second appointment on top of one they already hold. A caller who holds a booking is changing it, so this is false for an ordinary booking and the tool refuses with has_booking.
            booking_id (str): The exact booking_id from find_slots when moving or cancelling. An empty string when booking, which has no booking yet.
            confirmed (bool): True only after the caller said yes to the sentence you read back.
            customer_phone (str): The caller's number exactly as find_or_create_customer returned it, in E.164: a plus sign then digits, with no spaces. Copy it, never retype it.
            slot_id (str): The exact slot_id from find_slots when booking or moving. An empty string when cancelling, which needs no slot.
        """  # noqa: D205
        result = tools.save_booking.save_booking(
            action=action,
            additional=additional,
            booking_id=booking_id,
            confirmed=confirmed,
            customer_phone=customer_phone,
            slot_id=slot_id,
        )
        if inspect.isawaitable(result):
            result = await result
        await params.result_callback(result)

    @_direct_tool(announce="Let me get that written down.")
    async def record_complaint(
        self,
        params: FunctionCallParams,
        customer_phone: str,
        requested_resolution: str,
        summary: str,
    ) -> None:
        """Record the verified customer's complaint after enough facts are known. This does not replace a requested manager transfer.

        Args:
            params (FunctionCallParams): The call handle; the result goes back through it.
            customer_phone (str): The caller's number exactly as find_or_create_customer returned it, in E.164: a plus sign then digits, with no spaces. Copy it, never retype it.
            requested_resolution (str): What the customer wants the salon to do, or an empty string
            summary (str): Short factual summary of the complaint
        """
        result = tools.record_complaint.record_complaint(
            customer_phone=customer_phone,
            requested_resolution=requested_resolution,
            summary=summary,
        )
        if inspect.isawaitable(result):
            result = await result
        await params.result_callback(result)

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
