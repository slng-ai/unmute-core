"""Wasla's agent, written once for LiveKit and Pipecat.

Lina rings the holder of one overdue account. One main agent and three task groups:

1. ``account``: ``identify`` checks the date of birth, read back by code, then
   ``talk_account`` asks if now is a good time, says the amount in code's words, and
   records a promise to pay, or a concern, each read back for a yes.
2. ``callback``: a day and time to call back, read back, then the call ends.
3. ``wrong_number``: takes the number off the list on a yes, then the call ends.

Only the holder's name is read before the greeting. The account's details come from a
successful date-of-birth check, so nothing about the account is known, to the model or
the prompt, before the holder is verified.
``APP`` is the ``VoiceApp`` the ``livekit`` and ``pipecat`` projects run.
"""

from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from pydantic import BaseModel
from unmute_core.agent import AgentSpec
from unmute_core.call import CallStartError, save_call_start
from unmute_core.greeting import Greeting
from unmute_core.prefetch import Clock, Lookup, Prefetch
from unmute_core.state import CallInfo, CallStartRule, CallState
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.tools import ToolRegistry, local_tool
from voice_stack import VoiceApp

from wasla.ledger import SUPPORT, TIMEZONE, Ledger, payment_policy
from wasla.state import State

AGENT_ID = "wasla-reminders"
VOICE = "933563129e564b19a115bedd57b7406a"
"""Fish Audio's "Sarah": a warm, calm voice."""
GREETING = Greeting(
    text=(
        "Hello, this is Lina calling from Wasla. Am I speaking with {{first_name}} {{last_name}}?"
    )
)
CONFIRM = {"birth_date": "identify"}
"""Only the ``identify`` step confirms the date of birth."""
CALL_START: dict[str, CallStartRule] = {"account_id": "required"}
"""What a dispatch must carry: the account the call is about."""
ACCOUNT = ("amount_words", "merchant", "purchase_spoken", "due_spoken")
"""The account's details. Only a successful date-of-birth check returns them."""
LOCKED = {"account_id": "{{account_id}}", "call": "{{call.session_id}}"}
CONCERN_SAID = {
    "hardship": (
        "I've passed that on. Please contact support with any documents, and they'll look at "
        f"it with you. {SUPPORT} Is there anything else you'd like to add?"
    ),
    "legal-matter": (
        "I've passed that to our internal team. They'll look into it and contact you within "
        "twenty-four hours. Is there anything else you'd like to add?"
    ),
    "purchase-dispute": (
        "I've recorded your dispute. Our team will look into the purchase and contact you; "
        "nothing more is asked of you until then. Is there anything else you'd like to add?"
    ),
    "refusal-to-pay": (
        "I've noted your answer and your feedback. Thank you for your time today. Is there "
        "anything else you'd like to add?"
    ),
}
"""What code says once each kind of concern is noted."""


def prompt(name: str) -> str:
    """Return one of the package's prompts, its ``{{value}}`` placeholders unfilled."""
    return (files("wasla") / "prompts" / f"{name}.md").read_text(encoding="utf-8")


class Ended(BaseModel):
    """What ``end_call`` returns."""

    ended: bool


def end_call() -> Ended:
    """End the call once it is over. Say your goodbye in the same reply."""
    return Ended(ended=True)


def tools(ledger: Ledger) -> ToolRegistry:
    """Return every tool, on one ledger. The account and the call are always locked."""
    return ToolRegistry(
        [
            local_tool(ledger.verify_holder, locked={"account_id": "{{account_id}}"}),
            local_tool(
                ledger.good_time,
                locked={
                    "amount_words": "{{amount_words}}",
                    "merchant": "{{merchant}}",
                },
            ),
            local_tool(ledger.record_promise, locked=LOCKED, interruption="continue"),
            local_tool(ledger.schedule_callback, locked=LOCKED, interruption="continue"),
            local_tool(ledger.mark_wrong_number, locked=LOCKED, interruption="continue"),
            local_tool(ledger.note_concern, locked=LOCKED, interruption="continue"),
            local_tool(payment_policy),
            local_tool(end_call, effect="ends_conversation"),
        ],
        state=State,
    )


def concern_rule(kind: str) -> dict[str, Any]:
    """Return the finish rule for one kind of concern: the same question, its own receipt."""
    return {
        "tool": "note_concern",
        "success": {"kind": [kind]},
        "confirm": ["Shall I pass this to our team as a {{input.kind}}: {{input.note}}?"],
        "say": [CONCERN_SAID[kind]],
    }


CALLBACK_RULE: dict[str, Any] = {
    "tool": "schedule_callback",
    "success": {"status": ["CB"]},
    "confirm": ["Shall I note a call back for {{input.day}} at {{input.at}}?"],
    "say": ["That's noted: we'll call back {{result.when}}. Thank you for your time. Goodbye."],
}
"""The callback's read-back and receipt, the same wherever it is taken."""

TASKS = TasksSpec.model_validate(
    {
        "mode": "record",
        "tasks": [
            {
                "name": "identify",
                "when": (
                    "Only when the holder wants to correct the date of birth they gave. Never "
                    "before `account`: that flow checks the date of birth itself."
                ),
                "instructions": prompt("identify"),
                "opening": "listen",
                "announce": [
                    "Thank you. To make sure I'm speaking with the right person, could you "
                    "tell me your date of birth?"
                ],
                "tools": ["verify_holder", "end_call"],
                "finish": [
                    {
                        "tool": "verify_holder",
                        "success": {"status": ["verified"]},
                        "confirm": ["Is your date of birth {{input.birth_date}}?"],
                        "say": [
                            "Thank you, that matches. Is now a good time to talk about your "
                            "Wasla account?"
                        ],
                    }
                ],
                "replies": [
                    {
                        "tool": "verify_holder",
                        "when": {"status": ["mismatch"]},
                        "say": [
                            "I'm sorry, that doesn't match our records. Could you tell me your "
                            "date of birth again?"
                        ],
                    },
                    {
                        "tool": "verify_holder",
                        "when": {"status": ["locked"]},
                        "say": [
                            f"I'm sorry, I can't go on with this call today. {SUPPORT} Goodbye."
                        ],
                    },
                ],
                "assign": [
                    {"variable": "birth_date", "path": ["birth_date"]},
                    *({"variable": name, "path": [name]} for name in ACCOUNT),
                ],
            },
            {
                "name": "talk_account",
                "instructions": prompt("talk_account"),
                "opening": "listen",
                "tools": [
                    "good_time",
                    "record_promise",
                    "note_concern",
                    "schedule_callback",
                    "payment_policy",
                ],
                "replies": [
                    {
                        "tool": "good_time",
                        "when": {"good_time": [True]},
                        "say": ["{{result.said}}"],
                    }
                ],
                "finish": [
                    {
                        "tool": "good_time",
                        "success": {"good_time": [False]},
                        "say": [
                            "No problem at all. When would be a good day and time to call you back?"
                        ],
                    },
                    {
                        "tool": "record_promise",
                        "success": {"status": ["PTP"]},
                        "confirm": [
                            "Shall I record your promise to pay {{amount_words}} for {{input.day}}?"
                        ],
                        "say": [
                            "Thank you. Your promise to pay {{result.when}} is recorded. You "
                            "can pay in the Wasla app or on our website. Is there anything "
                            "else I can help with?"
                        ],
                    },
                    *(concern_rule(kind) for kind in CONCERN_SAID),
                    CALLBACK_RULE,
                ],
                "assign": [
                    {"variable": "outcome", "path": ["status"], "optional": True},
                    {"variable": "concern", "path": ["kind"], "optional": True},
                ],
            },
            {
                "name": "take_callback",
                "instructions": prompt("callback"),
                "tools": ["schedule_callback"],
                "opens_with": {
                    "tool": "schedule_callback",
                    "arguments": {"day": "callback_day", "at": "callback_time"},
                },
                "finish": [CALLBACK_RULE],
                "assign": [{"variable": "outcome", "path": ["status"]}],
            },
            {
                "name": "take_off_list",
                "instructions": prompt("wrong_number"),
                "tools": ["mark_wrong_number", "end_call"],
                "opens_with": {"tool": "mark_wrong_number"},
                "finish": [
                    {
                        "tool": "mark_wrong_number",
                        "success": {"status": ["WN"]},
                        "confirm": [
                            "I'm sorry to have bothered you. Shall I take this number off our "
                            "list, so we don't call it again?"
                        ],
                        "say": [
                            "Done. You won't get any more calls from us on this number. Sorry "
                            "for the trouble. Goodbye."
                        ],
                    }
                ],
                "assign": [{"variable": "outcome", "path": ["status"]}],
            },
        ],
        "groups": [
            {
                "name": "account",
                "when": "The person says they are the account holder, or asks what this is about.",
                "steps": ["identify", "talk_account"],
                "context_scope": "shared",
                "skip_when_confirmed": {"identify": "birth_date"},
            },
            {
                "name": "callback",
                "when": (
                    "Someone other than the holder answered, or the holder wants to talk "
                    "another time."
                ),
                "steps": ["take_callback"],
                "context_scope": "shared",
                "arguments": ["callback_day", "callback_time"],
                "then": "end",
            },
            {
                "name": "wrong_number",
                "when": "Nobody of the holder's name can be reached at this number.",
                "steps": ["take_off_list"],
                "context_scope": "shared",
                "then": "end",
            },
        ],
        "task_only_tools": [
            "verify_holder",
            "good_time",
            "record_promise",
            "schedule_callback",
            "mark_wrong_number",
            "note_concern",
        ],
    }
)


class Wasla:
    """The reminder line, built once per process and shared by every call.

    Attributes:
        ledger: The ledger every call reads and records into.
        plan: The checked tasks, on the ledger's tools.
        prefetch: Today on the Dubai clock, and whose account the call is about.
    """

    def __init__(self, ledger: Ledger | None = None) -> None:
        """Build the plan and the prefetch once. A broken row fails here, at import.

        Args:
            ledger: The ledger. A fresh demo ledger when None.
        """
        self.ledger = ledger or Ledger()
        registry = tools(self.ledger)
        self.plan = TaskPlan(TASKS, state=State, confirm=CONFIRM, tools=registry)
        opened = local_tool(self.ledger.open_account, locked={"account_id": "{{account_id}}"})
        self.prefetch = Prefetch(
            [
                Clock("today", TIMEZONE, {"today": "date", "weekday": "day_of_week"}),
                Lookup(
                    "account",
                    opened,
                    {name: name for name in ("first_name", "last_name", "deadline")},
                    inputs=["account_id"],
                ),
            ],
            state=State,
        )
        GREETING.check(self.prefetch)

    async def new_call(
        self, call: CallInfo | None = None, call_start: Mapping[str, object] | None = None
    ) -> CallState[State]:
        """Return a fresh call state, with today and the account read in.

        The dispatch's ``call_start`` names the account, like ``{"account_id": "WA-1002"}``.
        ``UNMUTE_CALL_START`` does the same on a console or browser call.

        Args:
            call: What the line carries. A web call's when None.
            call_start: The dispatch's ``call_start`` values, when there are some.

        Raises:
            CallStartError: When the dispatch names no account, a malformed one, or one the
                ledger does not hold. The call never starts, so nobody is rung about nothing.
        """
        state = CallState(
            State,
            confirm=CONFIRM,
            call_start=CALL_START,
            call=call or CallInfo(agent_id=AGENT_ID, direction="outbound"),
        )
        save_call_start(state, call_start)
        await self.prefetch.run(state)
        if state.plain("first_name") is None:
            # The lookup refused an id the ledger does not hold.
            raise CallStartError(f"call_start: account_id: no account {state.plain('account_id')}")
        return state

    def agent(self, state: CallState) -> AgentSpec:
        """Return the call's agent: Lina's prompt and the tasks, on the Dubai clock."""
        return AgentSpec(
            prompt("main"), name="lina", tasks=self.plan, language="en", timezone=TIMEZONE
        )


WASLA = Wasla()
APP = VoiceApp(
    agent_id=AGENT_ID,
    greeting=GREETING,
    new_call=WASLA.new_call,
    agent=WASLA.agent,
    voice=VOICE,
    language="en",
    pace="balanced",
    max_duration_s=600,
)
