"""The salon concierge, written once for LiveKit and Pipecat.

Two main agents share one call: ``concierge`` books, and ``complaint_specialist`` takes
complaints. ``agents`` describes both as ``AgentSpec``s, which ``UnmuteAgent`` runs on
LiveKit and ``UnmuteFlows`` on Pipecat. Everything here is framework-neutral and imports
only unmute-core. The framework projects build the pipeline, the handoff tools and
``end_call``.

Ported from Unmute's examples/salon-concierge (agent.yaml, tools/*.yaml, the prompts).
"""

from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any, NamedTuple

from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel
from unmute_core.agent import AgentSpec
from unmute_core.call import save_call_start
from unmute_core.greeting import Greeting
from unmute_core.idle import IdleNudges
from unmute_core.knowledge import KnowledgeBase
from unmute_core.prefetch import CallValue, Clock, Prefetch
from unmute_core.state import CallInfo, CallState
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.tools import ToolRegistry, local_tool

from salon.state import State
from salon.store import TIMEZONE, BookingAction, ComplaintRecord, Customer, Diary, Salon

PACKAGE = Path(str(files("salon")))
AGENT_ID = "salon-concierge"
"""The agent's id, the same on LiveKit and Pipecat."""
GREETING = Greeting(
    text="Hi, you've reached Sage and Stone. Robin speaking, what can I do for you?"
)
PACE = "balanced"
"""Unmute's turn pace: how long the caller may pause before their turn ends."""
IDLE = IdleNudges(
    first_nudge_delay_seconds=20,
    first_nudge_text="Are you still there?",
    second_nudge_delay_seconds=15,
    second_nudge_text="I'm still here whenever you're ready.",
    hangup_delay_seconds=10,
    final_hangup_text="I'll let you go for now. Call us back any time. Goodbye.",
)
MAX_DURATION_S = 900
END_CALL = (
    "End the call once the caller's own words say the call is over: \"no, that's all\", "
    '"I\'m good, thanks", "goodbye". A question is never one of those, however tired it '
    'sounds. On a live call on 2026-09-16 the caller asked "can we just recap when I have '
    'the next appointment?" and this ended the call on them. Answer what they asked and '
    "wait for them to say they are done."
)
SAY_GOODBYE_WITH_THE_CALL = (
    "In the same reply as this call, and before it, say one short goodbye in your own words. "
    "Nothing is said after the call."
)
GOODBYE_LINE = "Thanks for calling. Goodbye."
TO_COMPLAINTS = "The caller has a complaint or service problem."
TO_COMPLAINTS_LINE = "I'm sorry to hear that. Let me take this down properly."
TO_CONCIERGE = "The caller changes to a request owned by another specialist."
CONFIRM = {"customer_phone": "verify_customer"}
WHEN = "{{result.appointment.date}} at {{result.appointment.time}}"
"""A booking's day and time, as code says them: "tomorrow, October 9 at 9:00 AM"."""
RECEIPTS = {
    "booked": (
        "Lovely, your {{result.appointment.service}} is booked for "
        f"{WHEN}. Anything else I can do for you?",
        f"Great, you're all set for {WHEN}. Is there anything else?",
        f"Perfect, you're in the diary for {WHEN}. Anything else I can help with?",
    ),
    "moved": (
        f"Done, I've moved it to {WHEN}. Anything else?",
        f"Great, you're now booked for {WHEN}. Is there anything else I can do?",
        f"All sorted, it's moved to {WHEN}. Anything else I can help with?",
    ),
    "cancelled": (
        "That's cancelled for you. Anything else I can help with?",
        "Done, I've cancelled it. Is there anything else?",
        "All cancelled. Anything else I can do for you?",
    ),
}
"""The lines spoken after each kind of save, in turn. The concierge makes no model request.

A booking or a move names its service, day and time, from the diary, said by their type:
the service as words, the day from the salon's today, the time on the clock. The day can be
"today" or "tomorrow", so a line puts it after "for" or "to", never after "on".
"""
CONFIRMS = {
    "find_or_create_customer": (
        "Is the best number to reach you {{input.phone}}?",
        "Just to check, is that {{input.phone}}?",
    ),
    "book_appointment": (
        "Shall I book your {{input.service}} for {{input.date}} at {{input.time}}?",
        "Would you like your {{input.service}} booked for {{input.date}} at {{input.time}}?",
    ),
    "move_appointment": (
        "Shall I move it to {{input.date}} at {{input.time}}?",
        "Would you like it moved to {{input.date}} at {{input.time}}?",
    ),
    "cancel_appointment": (
        "Shall I cancel your {{input.service}} for {{input.date}} at {{input.time}}?",
        "Just to check, cancel your {{input.service}} for {{input.date}} at {{input.time}}?",
    ),
}
"""What code asks before each write runs, filled from the model's call.

The write runs when the model calls it again with the same input, once the caller heard
the question to its end. The model cannot skip it, and a cut-off question is asked again.
"""
SAVES = {
    "book_appointment": "booked",
    "move_appointment": "moved",
    "cancel_appointment": "cancelled",
}
"""Each write and the status that ends the booking step."""
LOCKED_PHONE = {"customer_phone": "{{customer_phone}}"}


def prompt(name: str) -> str:
    """Return one of the package's prompts, its ``{{value}}`` placeholders unfilled."""
    return (PACKAGE / "prompts" / f"{name}.md").read_text(encoding="utf-8")


CONCIERGE_PROMPT = prompt("concierge")
SPECIALIST_PROMPT = prompt("complaint_specialist")


def _embedder() -> OpenAIEmbedding:
    """Return the embedder both knowledge bases use."""
    return OpenAIEmbedding(model="text-embedding-3-small")


SERVICES = KnowledgeBase(
    "services",
    PACKAGE / "knowledge" / "services",
    chunk_size=220,
    chunk_overlap=40,
    embed=_embedder,
    credential_env="OPENAI_API_KEY",
)
REFUNDS = KnowledgeBase(
    "refunds", PACKAGE / "knowledge" / "refunds", embed=_embedder, credential_env="OPENAI_API_KEY"
)


def start_knowledge() -> None:
    """Load both knowledge bases. Call it once per process, before the first call."""
    SERVICES.start()
    REFUNDS.start()


class NoTransfer(BaseModel):
    """Why the manager transfer did not happen."""

    failed: str


def to_manager() -> NoTransfer:
    """Refuse the manager transfer: this example has no phone line to transfer on."""
    return NoTransfer(failed="this session is not a phone call, so it cannot be transferred")


def tools(salon: Salon) -> ToolRegistry:
    """Return every tool of both agents, on one diary.

    ``customer_phone`` is locked to the call state. A number the caller has not confirmed
    refuses the call before the diary sees it.
    """
    return ToolRegistry(
        [
            local_tool(
                salon.find_or_create_customer,
                output=Customer,
                description=(
                    "Look up or create the salon customer for the caller's phone number. Call "
                    "it as soon as you have the whole number, the saved one or the one the "
                    "caller just said. It first asks the caller to confirm the number, so never "
                    "ask that yourself. Call it again with the same number after their yes, and "
                    "it runs. A number that cannot be right comes back refused, saying what to "
                    "ask the caller next."
                ),
            ),
            local_tool(
                salon.find_slots,
                output=Diary,
                locked=LOCKED_PHONE,
                description=(
                    "Read the diary: what the caller already has booked, and what times are "
                    "free. Call this before offering any time and before changing any "
                    "booking. It never writes anything, so call it whenever you are unsure "
                    "what the caller holds.\n"
                    "Give a date to see free times on that date. Leave the service out to see "
                    "every service. The caller's own bookings come back every time, whatever "
                    "you asked, so you never need a second call to find out what they hold.\n"
                    "Ask the caller which day they want before you call this, and never guess "
                    "one. A call for a day nobody named answers nothing, and the second call "
                    "you then have to make speaks a second announcement the caller did not "
                    "need to hear."
                ),
            ),
            local_tool(
                salon.book_appointment,
                output=BookingAction,
                locked=LOCKED_PHONE,
                description=(
                    "Book a free time: the service, day and time exactly as find_slots "
                    "returned them. Call it as soon as the caller has picked a time. It first "
                    "asks the caller to confirm the booking, so never ask that yourself. Call "
                    "it again with the same values after their yes, and it books. Nothing is "
                    "booked until it runs and says so."
                ),
            ),
            local_tool(
                salon.move_appointment,
                output=BookingAction,
                locked=LOCKED_PHONE,
                description=(
                    "Move the caller's booking to a free time: its booking_id, and the new day "
                    "and time exactly as find_slots returned them. It first asks the caller to "
                    "confirm the move, so never ask that yourself. Call it again with the same "
                    "values after their yes, and it moves the booking."
                ),
            ),
            local_tool(
                salon.cancel_appointment,
                output=BookingAction,
                locked=LOCKED_PHONE,
                description=(
                    "Cancel the caller's booking: its booking_id, service, day and time exactly "
                    "as find_slots returned them. It first asks the caller to confirm the "
                    "cancellation, so never ask that yourself. Call it again with the same "
                    "values after their yes, and it cancels."
                ),
            ),
            local_tool(
                salon.record_complaint,
                output=ComplaintRecord,
                locked=LOCKED_PHONE,
                announce="Let me get that written down.",
                description=(
                    "Record the verified customer's complaint after enough facts are known. "
                    "This does not replace a requested manager transfer."
                ),
            ),
            SERVICES.tool(
                "look_up_salon_info",
                (
                    "Look up anything about the salon itself: services and prices, how long "
                    "things take, opening hours, the stylists and what each one does, "
                    "booking, deposits, cancellation and lateness terms, payment methods, "
                    "parking, and accessibility. Use this whenever the caller asks a question "
                    "about how the salon works, not only about price. Quote the document "
                    "rather than estimating.\n"
                    "Prefer this over asking the caller for anything. A question about the "
                    "salon is not a booking request: look it up and answer it, and only start "
                    "verification if the caller actually wants to make, move, or cancel an "
                    "appointment."
                ),
            ),
            REFUNDS.tool(
                "look_up_refund_policy",
                (
                    "Look up the salon's refund and complaints policy. Use this before you "
                    "state any refund, redo, timescale, or goodwill offer, so you quote the "
                    "policy instead of guessing it. It covers the redo window, the three "
                    "refund tiers, colour corrections, retail returns, and how a complaint is "
                    "handled."
                ),
            ),
            local_tool(
                to_manager,
                description=(
                    "The caller explicitly asks for a manager or is clearly and strongly "
                    "frustrated."
                ),
            ),
        ]
    )


VERIFY_CUSTOMER: dict[str, Any] = {
    "name": "verify_customer",
    "when": (
        "Run this when the caller explicitly corrects their phone number, and when a "
        "complaint is about to be written down. Never before a booking: the `book` flow runs "
        "this step itself when the saved number is not confirmed. A change of date, time or "
        "service is not a phone correction."
    ),
    "instructions": prompt("verify_customer"),
    "opening": "listen",
    "announce": [
        "Sure, what's the best number to reach you on?",
        "Of course. What number should I use?",
        "Sure. What's a good number for you?",
    ],
    "opens_with": {"tool": "find_or_create_customer", "arguments": {"phone": "customer_phone"}},
    "tools": ["find_or_create_customer"],
    "finish": [
        {
            "tool": "find_or_create_customer",
            "success": {"status": ["existing", "created"]},
            "confirm": list(CONFIRMS["find_or_create_customer"]),
        }
    ],
    "assign": [
        {"variable": "customer_phone", "path": ["customer_phone"]},
        {"variable": "customer_verified", "path": ["status"]},
    ],
    "history": "messages",
}

CONCIERGE_TASKS = TasksSpec.model_validate(
    {
        "mode": "record",
        "tasks": [
            VERIFY_CUSTOMER,
            {
                "name": "manage_booking",
                "instructions": prompt("manage_booking"),
                "announce": [
                    "Let me have a look.",
                    "One moment, let me check the diary.",
                    "Let me see what we have.",
                ],
                "tools": ["find_slots", *SAVES],
                "opens_with": {
                    "tool": "find_slots",
                    "arguments": {
                        "service": "requested_service",
                        "date": "requested_day",
                        "intent": "requested_intent",
                    },
                    "say": ["{{result.offer}}"],
                },
                "finish": [
                    {
                        "tool": tool,
                        "success": {"status": [status]},
                        "say": list(RECEIPTS[status]),
                        "confirm": list(CONFIRMS[tool]),
                    }
                    for tool, status in SAVES.items()
                ],
                "replies": [
                    {
                        "tool": "find_slots",
                        "when": {"offered": [True]},
                        "say": ["{{result.offer}}"],
                    }
                ],
                "assign": [{"variable": "appointment", "path": ["appointment"]}],
                "history": "messages",
                "handoffs": ["to_complaints"],
            },
        ],
        "groups": [
            {
                "name": "book",
                "when": (
                    "The caller wants to create, move or cancel a booking. Includes a change "
                    "to an appointment just made."
                ),
                "steps": ["verify_customer", "manage_booking"],
                "context_scope": "shared",
                "skip_when_confirmed": {"verify_customer": "customer_phone"},
                "arguments": ["requested_service", "requested_day", "requested_intent"],
            }
        ],
        "task_only_tools": ["find_or_create_customer", "find_slots", *SAVES],
    }
)
SPECIALIST_TASKS = TasksSpec.model_validate(
    {
        "mode": "record",
        "tasks": [VERIFY_CUSTOMER],
        "task_only_tools": ["find_or_create_customer"],
    }
)
CONCIERGE_TOOLS = (
    "look_up_salon_info",
    "to_manager",
    "find_or_create_customer",
    "find_slots",
    *SAVES,
)
SPECIALIST_TOOLS = (
    "record_complaint",
    "look_up_refund_policy",
    "to_manager",
    "find_or_create_customer",
)


class Agents(NamedTuple):
    """The call's two main agents, the same on LiveKit and Pipecat.

    Attributes:
        concierge: Books, moves and cancels appointments, and answers questions about the
            salon. The call starts with it.
        specialist: Takes a complaint down and quotes the refund policy.
    """

    concierge: AgentSpec
    specialist: AgentSpec


def agents(salon: Salon) -> Agents:
    """Return both agents, on one diary. Build them once per process.

    Each agent names itself, so a ``handoff`` event shows which one holds the call. Both
    answer on the framework's own LLM, so neither carries a model, and the specs hold
    nothing of one call: the state, the handoff tools and ``end_call`` go to the framework
    adapter.

    Args:
        salon: The diary every call books into.
    """
    registry = tools(salon)
    concierge = TaskPlan(
        CONCIERGE_TASKS, state=State, confirm=CONFIRM, tools=registry.only(CONCIERGE_TOOLS)
    )
    specialist = TaskPlan(
        SPECIALIST_TASKS, state=State, confirm=CONFIRM, tools=registry.only(SPECIALIST_TOOLS)
    )
    return Agents(
        concierge=AgentSpec(CONCIERGE_PROMPT, name="concierge", tasks=concierge, timezone=TIMEZONE),
        specialist=AgentSpec(
            SPECIALIST_PROMPT, name="complaint_specialist", tasks=specialist, timezone=TIMEZONE
        ),
    )


PREFETCH = Prefetch(
    [
        Clock(
            "today",
            "Europe/Madrid",
            {"booking_date": "date", "booking_weekday": "day_of_week", "salon_local_time": "time"},
        ),
        CallValue("caller", "calling_number", ["customer_phone"]),
    ]
)


GREETING.check(PREFETCH)


async def new_call(
    call: CallInfo | None = None, dispatched: Mapping[str, object] | None = None
) -> CallState[State]:
    """Return a fresh call state, with today's date and the caller's number read in.

    A browser call has no caller ID. ``UNMUTE_CALL_START`` can simulate one, like
    ``{"call": {"direction": "inbound", "calling_number": "+34612345678"}}``.

    Args:
        call: What the line carries. A web call's when None.
        dispatched: The dispatch's ``call_start`` object, when there is one.
    """
    state = CallState(State, confirm=CONFIRM, call=call or CallInfo(agent_id=AGENT_ID))
    save_call_start(state, dispatched)
    await PREFETCH.run(state)
    return state
