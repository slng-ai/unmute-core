"""Kestrel's agent, written once for LiveKit and Pipecat.

One main agent, Sam, and one task group, ``claim``, that runs two steps:

1. ``find_policy``: the caller's policy number and date of birth, read back by code and
   looked up on their yes. The group's arguments carry what the caller already said, so
   when both are known the read-back plays with no model request.
2. ``take_report``: code asks each question of the report in a fixed order, the model only
   notes the answers (``note_facts``), and code reads the whole claim back before
   ``file_claim`` files it on the caller's yes. A change after the read-back changes the
   locked report, so the library reads the claim back again before anything is filed.

Every line the caller hears in the flow is code's, except the model's own short answers.
``APP`` is the ``VoiceApp`` the ``livekit`` and ``pipecat`` projects run.
"""

from collections.abc import Mapping
from importlib.resources import files

from pydantic import BaseModel
from unmute_core.agent import AgentSpec
from unmute_core.call import save_call_start
from unmute_core.greeting import Greeting
from unmute_core.idle import IdleNudges
from unmute_core.prefetch import Clock, Prefetch
from unmute_core.state import CallInfo, CallState
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.tools import Assign, ToolRegistry, ToolReply, local_tool
from voice_stack import VoiceApp

from kestrel.claims import TIMEZONE, ClaimsDesk, coverage_guide, not_a_policy_number
from kestrel.state import State

AGENT_ID = "kestrel-claims"
VOICE = "8d8aec63a56c460cb7def795bdf754fa"
"""Fish Audio's "Oliver": a calm English voice."""
GREETING = Greeting(
    text="Thanks for calling Kestrel Insurance claims, this is Sam. Can I take your policy "
    "number, please?"
)
GOODBYE = "Thanks for calling Kestrel. Take care, and drive safely."
IDLE = IdleNudges(
    first_nudge_delay_seconds=15,
    first_nudge_text="Are you still there? Take your time.",
    second_nudge_delay_seconds=20,
    second_nudge_text="I'm still here whenever you're ready.",
    hangup_delay_seconds=15,
    final_hangup_text="I'll let you go for now. Call us back any time. Goodbye.",
)
FILED = (
    "Your claim is filed. Your claim number is {{result.claim_number}}. A claims handler "
    "will call you within one working day. Is there anything else I can help with?"
)
NOTED = (
    ToolReply(
        when={"status": ["future_day"]},
        say=["That day hasn't happened yet. On which day did it happen?"],
    ),
    ToolReply(when={"complete": [False]}, say=["{{result.next_question}}"]),
)
"""What code says after ``note_facts``, wherever it runs: a future day asked again, else
the next question. The first that matches is said."""


def prompt(name: str) -> str:
    """Return one of the package's prompts, its ``{{value}}`` placeholders unfilled."""
    return (files("kestrel") / "prompts" / f"{name}.md").read_text(encoding="utf-8")


class Ended(BaseModel):
    """What ``end_call`` returns."""

    ended: bool


def end_call() -> Ended:
    """End the call once the caller says they are done. Say your goodbye in the same reply."""
    return Ended(ended=True)


def tools(desk: ClaimsDesk) -> ToolRegistry:
    """Return every tool, on one claims desk.

    Locked arguments come from the call, never from the model: the call's id, the
    report so far and the policy found. ``note_facts`` saves the report and its read-back
    as soon as it returns, so the next answer merges into it. The model never reads the
    report or the read-back: a read-back it could see, it would say in its own words.
    """
    return ToolRegistry(
        [
            local_tool(
                desk.lookup_policy,
                announce=["Let me find that.", "One moment."],
                precheck=not_a_policy_number,
                replies=[
                    ToolReply(
                        when={"status": ["not_a_number"]},
                        say=[
                            "A policy number is KP and six digits. Could you read me yours "
                            "again?"
                        ],
                    )
                ],
            ),
            local_tool(
                desk.note_facts,
                locked={"report": "{{report}}", "policy": "{{policy}}"},
                hidden=("report", "read_back"),
                replies=NOTED,
            ),
            local_tool(
                desk.file_claim,
                locked={
                    "report": "{{report}}",
                    "policy_number": "{{policy.policy_number}}",
                    "call": "{{call.session_id}}",
                },
                announce="Filing that now.",
                interruption="continue",
            ),
            local_tool(coverage_guide),
            local_tool(end_call, effect="ends_conversation"),
        ],
        state=State,
        assign={
            "note_facts": [
                Assign(variable="report", path=("report",)),
                Assign(variable="read_back", path=("read_back",)),
            ]
        },
    )


TASKS = TasksSpec.model_validate(
    {
        "mode": "record",
        "tasks": [
            {
                "name": "find_policy",
                "instructions": prompt("find_policy"),
                "tools": ["lookup_policy"],
                "opens_with": {
                    "tool": "lookup_policy",
                    "arguments": {"policy_number": "policy_number", "birth_date": "birth_date"},
                },
                "finish": [
                    {
                        "tool": "lookup_policy",
                        "success": {"found": [True]},
                        "confirm": [
                            "Is your policy number {{input.policy_number}}, with a date of "
                            "birth of {{input.birth_date}}?"
                        ],
                        "say": ["Thank you, {{result.first_name}}, I have your policy."],
                    }
                ],
                "replies": [
                    {
                        "tool": "lookup_policy",
                        "when": {"found": [False], "status": ["looked_up"]},
                        "say": [
                            "I'm sorry, I can't find a policy with that number and date of "
                            "birth. Could you check the number on your policy documents?"
                        ],
                    }
                ],
                "assign": [
                    {"variable": "policy", "path": ["policy"]},
                    {"variable": "policy_number", "path": ["policy", "policy_number"]},
                ],
            },
            {
                "name": "take_report",
                "instructions": prompt("take_report"),
                "tools": ["note_facts", "file_claim"],
                "opens_with": {"tool": "note_facts", "say": ["{{result.next_question}}"]},
                "finish": [
                    {
                        "tool": "file_claim",
                        "success": {"filed": [True]},
                        "confirm": ["{{read_back}}"],
                        "say": [FILED],
                    }
                ],
                "assign": [{"variable": "claim_number", "path": ["claim_number"]}],
            },
        ],
        "groups": [
            {
                "name": "claim",
                "when": (
                    "The caller gives a policy number or a date of birth, or wants to report "
                    "an accident or damage to their car."
                ),
                "steps": ["find_policy", "take_report"],
                "context_scope": "shared",
                "arguments": ["policy_number", "birth_date"],
            }
        ],
        "task_only_tools": ["lookup_policy", "note_facts", "file_claim"],
    }
)

PREFETCH = Prefetch(
    [Clock("today", TIMEZONE, {"today": "date", "weekday": "day_of_week"})], state=State
)
GREETING.check(PREFETCH)


class Kestrel:
    """The claims line, built once per process and shared by every call.

    Attributes:
        desk: The claims desk every call files into.
        plan: The checked tasks, on the desk's tools.
    """

    def __init__(self, desk: ClaimsDesk | None = None) -> None:
        """Build the plan once. A broken task, tool or placeholder fails here, at import.

        Args:
            desk: The claims desk. A fresh demo desk when None.
        """
        self.desk = desk or ClaimsDesk()
        self.plan = TaskPlan(TASKS, state=State, tools=tools(self.desk))

    async def new_call(
        self, call: CallInfo | None = None, call_start: Mapping[str, object] | None = None
    ) -> CallState[State]:
        """Return a fresh call state, with today's date read in.

        Args:
            call: What the line carries. A web call's when None.
            call_start: The dispatch's ``call_start`` values, when there are some.
        """
        state = CallState(State, call=call or CallInfo(agent_id=AGENT_ID))
        save_call_start(state, call_start)
        await PREFETCH.run(state)
        return state

    def agent(self, state: CallState) -> AgentSpec:
        """Return the call's agent: Sam's prompt and the claim tasks, on the London clock."""
        return AgentSpec(
            prompt("main"), name="sam", tasks=self.plan, language="en-GB", timezone=TIMEZONE
        )


KESTREL = Kestrel()
APP = VoiceApp(
    agent_id=AGENT_ID,
    greeting=GREETING,
    new_call=KESTREL.new_call,
    agent=KESTREL.agent,
    voice=VOICE,
    language="en",
    pace="patient",
    max_duration_s=900,
    idle=IDLE,
)
