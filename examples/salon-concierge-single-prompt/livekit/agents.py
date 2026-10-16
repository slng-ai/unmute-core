"""The agents and the task steps they delegate to, with their tools."""

from __future__ import annotations

import inspect
import os
import time
from collections.abc import AsyncIterable
from typing import Annotated, Literal

from livekit import api, rtc
from livekit.agents import (
    NOT_GIVEN,
    Agent,
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
from livekit.agents.llm import ChatChunk, Tool
from pydantic import Field

import tools.find_or_create_customer
import tools.find_slots
import tools.record_complaint
import tools.save_booking
from prompts import CONCIERGE_PROMPT
from session import _render
from settings import logger
from utils import dev_metrics, knowledge
from utils.dev_metrics import dev_llm_node, dev_say
from utils.telephony import _refer_uri

# --- agents ----------------------------------------------------------------


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

    async def on_enter(self) -> None:
        """Open this agent's turn as it takes over the call."""
        # Rendered once, on the way in: no step this agent runs writes state,
        # so nothing it names can change under it.
        await self.update_instructions(
            _render(CONCIERGE_PROMPT, self.session.userdata, site="agent:concierge")
        )
        if not self._initial:
            self.session.generate_reply()
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
    async def find_or_create_customer(
        self,
        ctx: RunContext,
        phone: Annotated[
            str,
            Field(
                description="The exact confirmed number in E.164: a plus sign, then 10 to 15 digits, with no spaces, brackets or dashes. The plus sign is part of the number and is never dropped; a text run sent the digits alone and the lookup could not tell which country it was. Never infer a country code the caller did not say."
            ),
        ],
    ) -> dict:
        """Look up or create one salon customer from the exact confirmed phone number. Use only after the digit readback got a clear yes. Reuse the record that already owns the number, or create one for a number that is new. Never guess a number or pass one the caller has not confirmed. A caller giving a replacement number is not confirmation. Ask the readback question first; call this tool only after a later caller message agrees to it."""
        result = tools.find_or_create_customer.find_or_create_customer(phone=phone)
        if inspect.isawaitable(result):
            result = await result
        return result

    @function_tool
    async def find_slots(
        self,
        ctx: RunContext,
        customer_phone: Annotated[
            str,
            Field(
                description="The caller's number exactly as find_or_create_customer returned it, in E.164: a plus sign then digits, with no spaces. Copy it, never retype it."
            ),
        ],
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
        result = tools.find_slots.find_slots(
            customer_phone=customer_phone, date=date, service=service
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
        customer_phone: Annotated[
            str,
            Field(
                description="The caller's number exactly as find_or_create_customer returned it, in E.164: a plus sign then digits, with no spaces. Copy it, never retype it."
            ),
        ],
        slot_id: Annotated[
            str,
            Field(
                description="The exact slot_id from find_slots when booking or moving. An empty string when cancelling, which needs no slot."
            ),
        ],
    ) -> dict:
        """Write the diary: book a new time, move an existing booking, or cancel one. This is the only tool that changes anything, so nothing is booked, moved or cancelled until it runs and says so.
        Call it only after the caller has heard the whole thing back in one sentence, the service, the day, the time and their phone number, and has said yes to that sentence. Choosing a time is not a yes. Use a slot_id and a booking_id exactly as find_slots returned them; never invent either.
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
        return result

    @function_tool
    async def record_complaint(
        self,
        ctx: RunContext,
        customer_phone: Annotated[
            str,
            Field(
                description="The caller's number exactly as find_or_create_customer returned it, in E.164: a plus sign then digits, with no spaces. Copy it, never retype it."
            ),
        ],
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
        # Not awaited, on purpose: see the same line in webhook_tool.
        dev_say(self.session, "Let me get that written down.")
        result = tools.record_complaint.record_complaint(
            customer_phone=customer_phone,
            requested_resolution=requested_resolution,
            summary=summary,
        )
        if inspect.isawaitable(result):
            result = await result
        return result

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
