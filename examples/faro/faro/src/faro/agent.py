"""Faro's agents, written once for LiveKit and Pipecat.

Faro explains in English or in Spanish, so it is two agents built from one template:
``faro_en`` and ``faro_es``. They share the prompt and the tools. What differs is the
language of the lines the library itself says (the number and profile read-backs and
the opening questions of a step) and of dates. A caller who asks to switch language is
handed to the other agent, which saves the new ``explanation``, so every line code says
from then on is in that language.

The lesson itself needs no task. ``continue_lesson``, ``grade_answer``, ``give_hint``,
``read_progress`` and ``remember`` are main-agent tools: the registry saves what each
result carries, and code says each result's ``said``. Two tasks hold what must be read
back before it happens: a number the caller says (``other_number``) and a new profile
(``sign_up``).

``APP`` is the ``VoiceApp`` the ``livekit`` and ``pipecat`` projects run.
"""

import os
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from pydantic import BaseModel
from unmute_core.agent import AgentSpec
from unmute_core.call import save_call_start
from unmute_core.greeting import Greeting
from unmute_core.prefetch import CallValue, Lookup, Prefetch
from unmute_core.state import CallInfo, CallState
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.tools import Assign, ToolRegistry, ToolReply, local_tool
from voice_stack import Handoff, VoiceApp

from faro.desk import Desk
from faro.state import Lang, State
from faro.store import Store

AGENT_ID = "faro-companion"
VOICE = "15e288e662ca4072b2276d5c7c962bb2"
"""Fish Audio's "Isabella": warm, and clear in both languages."""
TIMEZONE = "Europe/Madrid"
HELLO = "Hi, I'm Faro, your language companion. Si prefieres hablar en español, dímelo."
GREETING = Greeting(
    text=f"{HELLO} {{{{number_question}}}}",
    instruction=(
        f'Say exactly: "{HELLO} What is your full phone number, starting with plus thirty-four?"'
    ),
)
"""The greeting reads the caller's number back. With no caller ID, the model asks for it."""
LINES: dict[Lang, dict[str, str]] = {
    "EN": {
        "ask_number": "What is your full phone number, starting with plus thirty-four?",
        "number": "Let me check: is your number {{input.phone}}?",
        "ask_name": "Great. What would you like me to call you?",
        "profile": (
            "Shall I create your profile as {{input.name}}, learning {{input.learn}} for "
            "{{input.goal}}, at level {{input.level}}?"
        ),
        "switch": "The caller asks you to explain in Spanish, or keeps speaking Spanish to you.",
    },
    "ES": {
        "ask_number": (
            "¿Me dices tu número de teléfono completo, empezando por más treinta y cuatro?"
        ),
        "number": "A ver si lo tengo bien: ¿tu número es {{input.phone}}?",
        "ask_name": "Genial. ¿Cómo quieres que te llame?",
        "profile": "¿Creo tu perfil como {{input.name}}, con nivel {{input.level}}?",
        "switch": "The caller asks you to explain in English, or keeps speaking English to you.",
    },
}
"""The lines the library says itself, in each explanation language."""
SAID = ["{{result.said}}"]
LEARNER = [
    Assign(variable="learner_id", path=("learner_id",)),
    Assign(variable="learner_name", path=("learner_name",)),
    Assign(variable="notes", path=("notes",)),
]
LESSON = [
    Assign(variable="waiting", path=("waiting",)),
    Assign(variable="helped", path=("helped",)),
]
PROMPT = (files("faro") / "prompts" / "main.md").read_text(encoding="utf-8")


def prompt(name: str) -> str:
    """Return one of the package's step prompts."""
    return (files("faro") / "prompts" / f"{name}.md").read_text(encoding="utf-8")


class Ended(BaseModel):
    """What ``end_call`` returns."""

    ended: bool


def end_call() -> Ended:
    """End the call once the learner says they are done. Say your goodbye in the same reply."""
    return Ended(ended=True)


def tools(desk: Desk) -> ToolRegistry:
    """Return every tool, with what each result saves and the line code says for it.

    The learner, the language and the exercise waiting are locked: the model never names
    them. A lesson tool before the learner is known is refused, naming what to do first.
    """
    learner = {"learner_id": "{{learner_id}}", "explanation": "{{explanation}}"}
    return ToolRegistry(
        [
            local_tool(
                desk.use_this_number,
                locked={"phone": "{{phone}}", "explanation": "{{explanation}}"},
            ),
            local_tool(desk.check_number, locked={"explanation": "{{explanation}}"}),
            local_tool(
                desk.create_profile,
                locked={"phone": "{{phone}}", "explanation": "{{explanation}}"},
                interruption="continue",
            ),
            local_tool(desk.continue_lesson, locked=learner),
            local_tool(
                desk.grade_answer,
                locked={**learner, "waiting": "{{waiting}}", "helped": "{{helped}}"},
                interruption="continue",
            ),
            local_tool(desk.give_hint, locked={"waiting": "{{waiting}}"}),
            local_tool(desk.read_progress, locked=learner),
            local_tool(desk.remember, locked=learner),
            local_tool(end_call, effect="ends_conversation"),
        ],
        state=State,
        assign={
            "use_this_number": LEARNER,
            "continue_lesson": LESSON,
            "grade_answer": LESSON,
            "give_hint": [Assign(variable="helped", path=("helped",))],
            "remember": [Assign(variable="notes", path=("notes",))],
        },
        replies=[
            ToolReply(tool=name, say=SAID)
            for name in (
                "use_this_number",
                "give_hint",
                "continue_lesson",
                "grade_answer",
                "read_progress",
                "remember",
            )
        ],
    )


def plan(lang: Lang, registry: ToolRegistry) -> TaskPlan:
    """Return the two tasks that read back before they act, in one explanation language."""
    lines = LINES[lang]
    found: dict[str, Any] = {
        "tool": "check_number",
        "success": {"status": ["known", "new"]},
        "confirm": [lines["number"]],
        "say": SAID,
    }
    learner = [
        {"variable": "learner_id", "path": ["learner_id"], "optional": True},
        {"variable": "learner_name", "path": ["learner_name"], "optional": True},
        {"variable": "notes", "path": ["notes"], "optional": True},
    ]
    spec = TasksSpec.model_validate(
        {
            "mode": "record",
            "tasks": [
                {
                    "name": "other_number",
                    "when": (
                        "The caller says the number read back is wrong, or there was no number "
                        "to read back."
                    ),
                    "instructions": prompt("other_number"),
                    "opening": "listen",
                    "announce": [lines["ask_number"]],
                    "arguments": ["phone"],
                    "opens_with": {"tool": "check_number", "arguments": {"phone": "phone"}},
                    "tools": ["check_number"],
                    "finish": [found],
                    "assign": [{"variable": "phone", "path": ["phone"]}, *learner],
                },
                {
                    "name": "sign_up",
                    "when": "A caller with no profile agrees to create one.",
                    "instructions": prompt("sign_up"),
                    "opening": "listen",
                    "announce": [lines["ask_name"]],
                    "tools": ["create_profile"],
                    "finish": [
                        {
                            "tool": "create_profile",
                            "success": {"status": ["created"]},
                            "confirm": [lines["profile"]],
                            "say": SAID,
                        }
                    ],
                    "assign": learner,
                },
            ],
            "task_only_tools": ["check_number", "create_profile"],
        }
    )
    return TaskPlan(spec, state=State, tools=registry)


class Faro:
    """The companion, built once per process and shared by every call.

    Attributes:
        desk: The tools' store of learners.
        plans: The tasks in each explanation language.
        prefetch: The caller's number, and which language to greet a known one in.
    """

    def __init__(self, store: Store | None = None) -> None:
        """Build both agents' plans once. A wrong row fails here, at import.

        Args:
            store: The learners. ``FARO_DB``, or ``faro.sqlite3`` here, when None.
        """
        self.desk = Desk(store or Store(os.getenv("FARO_DB", "faro.sqlite3")))
        registry = tools(self.desk)
        self.plans = {lang: plan(lang, registry) for lang in LINES}
        language = local_tool(self.desk.language_of, locked={"phone": "{{phone}}"})
        self.prefetch = Prefetch(
            [
                CallValue("caller", "calling_number", ["phone"]),
                Lookup(
                    "language",
                    language,
                    {"explanation": "explanation", "phone_spaced": "phone_spaced"},
                    inputs=["phone"],
                ),
            ],
            state=State,
        )

    async def new_call(
        self, call: CallInfo | None = None, call_start: Mapping[str, object] | None = None
    ) -> CallState[State]:
        """Return a fresh call state: the caller's number, and the language to greet in.

        A browser call has no caller ID. ``UNMUTE_CALL_START`` can simulate one, like
        ``{"call": {"direction": "inbound", "calling_number": "+34600555666"}}``.
        """
        state = CallState(State, call=call or CallInfo(agent_id=AGENT_ID))
        save_call_start(state, call_start)
        await self.prefetch.run(state)
        if (spaced := state.plain("phone_spaced")) is not None:
            question = (
                "Let me check: is your number"
                if state.plain("explanation") == "EN"
                else ("A ver si lo tengo bien: ¿tu número es")
            )
            state.save({"number_question": f"{question} {spaced}?"})
        return state

    def agents(self, state: CallState) -> dict[str, AgentSpec]:
        """Return both agents of a call: ``faro_en`` and ``faro_es``."""
        return {
            f"faro_{lang.lower()}": AgentSpec(
                PROMPT,
                name=f"faro_{lang.lower()}",
                tasks=self.plans[lang],
                language=lang.lower(),
                timezone=TIMEZONE,
                handoffs=("to_spanish",) if lang == "EN" else ("to_english",),
            )
            for lang in LINES
        }

    def agent(self, state: CallState) -> AgentSpec:
        """Return the agent the call starts with: the caller's own language, else English."""
        return self.agents(state)[f"faro_{str(state.plain('explanation')).lower()}"]


FARO = Faro()
APP = VoiceApp(
    agent_id=AGENT_ID,
    greeting=GREETING,
    new_call=FARO.new_call,
    agent=FARO.agent,
    agents=FARO.agents,
    handoffs={
        "faro_en": (
            Handoff(
                tool="to_spanish",
                description=LINES["EN"]["switch"],
                to="faro_es",
                saves={"explanation": "ES"},
            ),
        ),
        "faro_es": (
            Handoff(
                tool="to_english",
                description=LINES["ES"]["switch"],
                to="faro_en",
                saves={"explanation": "EN"},
            ),
        ),
    },
    voice=VOICE,
    language="es",
    pace="patient",
    max_duration_s=1200,
)
