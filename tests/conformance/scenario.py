"""Conformance scenarios: one file of test data that every adapter runs the same way.

A scenario holds the agent (instructions, task spec, canned tool results), the caller's
turns with the model's scripted replies, and what the agent must do: the lines it says in
each turn, the tools it runs, the task events and the saved state. Every adapter runs on
the ``State`` and the tools of ``tests/tasks/job.py``. A scenario may hold more main
agents, which handoff tools give the call to.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Literal

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, ConfigDict

from tests.dev.contract import problems as record_problems
from tests.outputs import AnyResult
from tests.tasks.job import State, registry
from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks, EventTool, save_call_start
from unmute_core.dev import DevRecorder
from unmute_core.greeting import Greeting
from unmute_core.handoff import History
from unmute_core.hooks import Hooks, PromptSuffix, Refuse
from unmute_core.model import (
    SUMMARY_SITE,
    ModelEvent,
    ModelRequest,
    TextDelta,
    ToolCall,
    Usage,
    fill_messages,
)
from unmute_core.prefetch import Lookup, Prefetch
from unmute_core.state import CallState, StateSchema
from unmute_core.tasks import NOT_YET_TOOL_NAME, OpeningTool, TaskPlan, TasksSpec
from unmute_core.telemetry import STT_SPAN, TOOL_SPAN, TTS_SPAN, Telemetry
from unmute_core.telephony import (
    NoPhoneCaller,
    OnUnavailable,
    Transfer,
    TransferError,
    transfer_tool,
)
from unmute_core.tools import Assign, ToolEffect, ToolRegistry, ToolReply, ToolSpec, Transcript


class Customer(BaseModel):
    """A shape in the State of a ``shaped_state`` scenario."""

    name: str
    city: str


class ShapedState(State):
    """The test State with a ``customer`` shape, for a locked argument that names it."""

    customer: Customer | None = None
    amount: float | None = None


SCENARIOS = Path(__file__).parent / "scenarios"
STEP_ENDED = '{"status": "completed"}'
MAIN = "main"


class _Strict(BaseModel):
    """A scenario part: a key the model does not know is an error, not a silent skip."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Call(_Strict):
    """A tool call the model makes, or a tool run the agent must make.

    ``extra`` is the provider data the model sends with its call, like a Gemini thought
    signature.
    """

    name: str
    arguments: dict[str, Any] = {}
    extra: dict[str, Any] = {}


class Reply(_Strict):
    """One scripted model reply, and the request it answers.

    Attributes:
        site: The prompt site the request must come from.
        heard: When set, the caller lines the request must hold, in order.
        result: When set, the newest tool output the request must hold.
        result_holds: When set, text the newest tool output the request holds must contain.
            It leaves out how an adapter words an error around its message.
        prompt: When set, the system prompt the request must hold, unfilled.
        tools: When set, the names of the tools the request offers, in order.
        tool_choice: When set, the tool choice the request must carry, like ``required``
            for a finish check.
        order: When set, the role of each user, assistant and tool message with text, in
            order. It pins where a carried caller turn sits.
        last_system: When set, the text of the request's last system message, like the
            block of values ``save_values`` adds.
        text: The reply's text.
        calls: The reply's tool calls.
        fail: The model call fails instead of answering.
        fail_after_text: The model says ``text``, then the call fails: the stream ends in an
            error after output, the way a connection drops mid-reply.
        variables: When set, values the request must carry beside its prompt, like a value
            a background prefetch saved.
        prompt_ends: When set, the text the request's first system message ends with.
        params: When set, the extra body fields the request must carry.
        usage: When set, the input and output tokens the reply reports last. Its ``llm``
            span must show them.
        holds: When set, text one of the request's system messages must hold, like the
            values block. A cascade request sends the block as a message of its own, and
            a realtime model reads it at the end of its instructions.
        replayed: When set, tool name to the ``extra_content`` every earlier call of that
            name the request replays must carry.
        model: When set, the name of the model the request must go to: ``scripted`` for
            the call's own, or an agent's ``model``.
        filled: When set, text the request's system messages must start with, once their
            placeholders are filled from the request's variables. A cascade request carries
            variables and a realtime one has them filled already, so both show the same
            text. What follows the prompt, like a greeting cue, differs by adapter.
        not_last_role: When set, a role the request's last message must not have, like
            ``assistant`` for a request that must not end on the agent's own line.
    """

    site: str
    heard: list[str] | None = None
    result: str | None = None
    result_holds: str | None = None
    prompt: str | None = None
    tools: list[str] | None = None
    tool_choice: str | None = None
    order: list[str] | None = None
    last_system: str | None = None
    text: str = ""
    calls: list[Call] = []
    fail: bool = False
    fail_after_text: bool = False
    variables: dict[str, str] | None = None
    prompt_ends: str | None = None
    params: dict[str, Any] | None = None
    usage: tuple[int, int] | None = None
    holds: str | None = None
    replayed: dict[str, dict[str, Any]] | None = None
    model: str | None = None
    filled: str | None = None
    not_last_role: str | None = None


class Turn(_Strict):
    """One caller line, the replies the model gives in that turn, and the lines said aloud.

    ``realtime_spoken``, when set, is what a realtime model says instead: it says no tool
    announce line, for one.
    """

    caller: str
    model: list[Reply]
    spoken: list[str]
    realtime_spoken: list[str] | None = None


class Expected(_Strict):
    """What the agent must have done by the end of the call.

    Attributes:
        agent: The main agent that holds the call at the end: ``main`` or one of ``agents``.
        build_error: When set, building the agent must fail with an error whose text holds
            this. The run then ends: nothing else is checked.
        ended: The agent ended the call itself, before the caller hung up.
        opening: The lines said before the caller's first line.
        event_calls: The runs of tools the model never chose, prefetch and events, in order.
        line_said: The lines a phone line said before the call ended.
    """

    state: dict[str, Any]
    tool_calls: list[Call]
    events: list[str]
    agent: str = MAIN
    build_error: str | None = None
    ended: bool = False
    opening: list[str] = []
    event_calls: list[Call] = []
    line_said: list[str] = []


class TransferCase(_Strict):
    """A transfer tool of the main agent, on a fake phone line.

    Attributes:
        on_unavailable: What a transfer nobody takes does.
        fails: Why nobody takes it. None puts the caller through.
        no_caller: The call has no phone leg.
    """

    on_unavailable: OnUnavailable = "return_to_caller"
    fails: str | None = None
    no_caller: bool = False


class ToolOptions(_Strict):
    """How one of the job's tools behaves beyond its result: ``ToolSpec`` fields."""

    announce: list[str] = []
    effect: ToolEffect = "returns_data"
    announce_wait: bool = False
    show_result: bool = True
    result_instructions: str = ""
    locked: dict[str, Any] = {}
    parameters: dict[str, Any] | None = None
    output: dict[str, str] | None = None
    replies: list[ToolReply] = []

    def fields(self) -> dict[str, Any]:
        """Return the ``ToolSpec`` fields to replace.

        ``output`` becomes a model of its type text, and a locked ``$transcript`` the
        transcript as text.
        """
        fields = self.model_dump(exclude_unset=True)
        if self.output is not None:
            fields["output"] = output_model(self.output)
        if "locked" in fields:
            fields["locked"] = {
                key: Transcript(text=True) if value == "$transcript" else value
                for key, value in self.locked.items()
            }
        return fields


def output_model(fields: dict[str, str]) -> type[BaseModel]:
    """Return a tool's output model, each field typed in the State type grammar.

    A field is required unless its type ends in ``| None``.
    """
    rows = [{"name": name, "type": text} for name, text in fields.items()]
    return StateSchema([], [{"name": "Output", "fields": rows}]).shape("Output")


class EventOptions(_Strict):
    """A tool that runs on events: its events, its locked arguments, and what it gives back.

    A locked value ``$transcript`` is the transcript as text. ``delay_s`` makes the tool
    slow, so a request that does not wait for it misses its result.
    """

    on: list[str]
    locked: dict[str, Any] = {}
    show_result: bool = True
    delay_s: float = 0
    assign: dict[str, str] = {}
    note: bool = False
    result_instructions: str = ""


class PrefetchOptions(_Strict):
    """A prefetch lookup over a tool: what it saves, and whether the greeting waits for it."""

    assign: dict[str, str] = {}
    background: bool = False
    delay_s: float = 0


class Context(_Strict):
    """What the agent that takes the call hears: ``HandoffContext`` without a summarizer."""

    history: History = "messages"
    max_messages: int | None = None
    include_tool_calls: bool = True


class Handoff(_Strict):
    """A handoff tool: its agent, its line, what that agent hears, the tool it opens with."""

    to: str
    announce: str | None = None
    context: Context = Context()
    opens_with: OpeningTool | None = None


class Agent(_Strict):
    """A main agent of the call.

    Attributes:
        instructions: Its prompt.
        tasks: Its tasks and groups.
        tools: The test tools its plan holds. None holds all of them.
        handoffs: Handoff tool name to the agent it hands the call to, or to the handoff.
            The first reply after the agent takes the call leaves them out.
        model: The name of its own model, a view of the scripted model. None runs it on the
            call's model.
    """

    instructions: str
    tasks: TasksSpec
    tools: list[str] | None = None
    handoffs: dict[str, str | Handoff] = {}
    model: str | None = None

    def handoff(self, tool: str) -> Handoff:
        """Return one of its handoff tools in full."""
        value = self.handoffs[tool]
        return Handoff(to=value) if isinstance(value, str) else value


class HookOptions(_Strict):
    """The call's hooks.

    Attributes:
        prompt_suffix: A ``PromptSuffix`` on every prompt.
        refuse: Tool name to the message a ``before_tool`` hook refuses it with.
        results: Tool name to the result an ``after_tool`` hook puts in its place.
        params: Extra body fields a ``before_model`` hook adds to every request.
    """

    prompt_suffix: str | None = None
    refuse: dict[str, str] = {}
    results: dict[str, Any] = {}
    params: dict[str, Any] = {}


class Scenario(_Strict):
    """One conversation that every adapter runs.

    The top-level ``instructions``, ``tasks`` and ``handoffs`` are the agent that answers
    first, ``main``. ``agents`` are the others a handoff can give the call to. ``at_end``
    scripts the requests the call's end makes, after the caller hangs up.

    ``call_start`` holds the values the call is dispatched with, each one optional. Its
    ``call`` object sets ``state.call``, as a dispatch does.
    ``prefetch`` holds lookups over tools, by name, run as the call opens.
    ``save_values`` gives the main agent the save tool.
    ``native_tools`` are framework tools of the main agent, name to canned result, that a
    task offers through its ``native_tools``.

    ``event_tools`` are tools on events, by name. Their results come from ``tool_results``.
    ``greeting`` is the fixed line the call opens with. ``worded_greeting`` has the model
    word it instead, with the ``opening`` replies. Neither waits for the caller.
    ``ready_fails`` makes the check before the greeting fail.
    ``hooks`` are the call's hooks.
    ``transfers`` are transfer tools of the main agent, by name, on a fake phone line.
    ``then_targets`` are the handoffs a group whose ``then`` is ``transfer`` makes, by its
    ``then_target``.
    ``shaped_state`` adds a ``customer`` shape (``name`` and ``city``) and a float
    ``amount`` to the State.
    ``caller_hangs_up`` makes the caller leave after the last turn, before the call's end.
    ``max_duration_s`` limits the call: the driver starts the adapter's max-duration limit
    with it once the call has opened, and the call ends when it runs out.
    ``language`` and ``timezone`` are the call's, for the lines code speaks. ``assign`` and
    ``replies`` are the main agent's registry rows: what a tool's result saves, and the
    line code speaks for it.
    ``realtime`` runs the scenario on a realtime model too, through a fake server. A
    realtime model reads one filled prompt with the values after it, so its requests show
    no ``variables`` and no values message of their own.
    """

    name: str
    about: str
    instructions: str
    tasks: TasksSpec
    handoffs: dict[str, str | Handoff] = {}
    agents: dict[str, Agent] = {}
    tool_results: dict[str, list[Any]] = {}
    tool_options: dict[str, ToolOptions] = {}
    call_start: dict[str, Any] = {}
    prefetch: dict[str, PrefetchOptions] = {}
    save_values: bool = False
    native_tools: dict[str, Any] = {}
    event_tools: dict[str, EventOptions] = {}
    greeting: str | None = None
    worded_greeting: bool = False
    opening: list[Reply] = []
    ready_fails: bool = False
    hooks: HookOptions = HookOptions()
    transfers: dict[str, TransferCase] = {}
    then_targets: dict[str, Handoff] = {}
    realtime: bool = False
    shaped_state: bool = False
    caller_hangs_up: bool = False
    max_duration_s: float | None = None
    language: str = "en"
    timezone: str | None = None
    assign: dict[str, list[Assign]] = {}
    replies: list[ToolReply] = []
    turns: list[Turn]
    at_end: list[Reply] = []
    expect: Expected

    @property
    def opens(self) -> Greeting:
        """How the call opens."""
        if self.greeting is not None:
            return Greeting(text=self.greeting)
        return Greeting() if self.worded_greeting else Greeting(speaks_first="user")

    @property
    def controls(self) -> set[str]:
        """The tools that hand the call over: every agent's handoff tools and the transfers."""
        return {
            *self.handoffs,
            *(tool for other in self.agents.values() for tool in other.handoffs),
            *self.transfers,
        }

    def agent(self, name: str) -> Agent:
        """Return one main agent of the call by name."""
        if name == MAIN:
            return Agent(instructions=self.instructions, tasks=self.tasks, handoffs=self.handoffs)
        return self.agents[name]


def agent_spec(scenario: "Scenario", run: "Run", name: str, *, own_model: bool) -> AgentSpec:
    """One main agent of the call, the same on every adapter.

    ``own_model`` gives the agent its view of the scripted model. A realtime driver runs
    every agent on the session's model instead.
    """
    part = scenario.agent(name)
    return AgentSpec(
        instructions=part.instructions,
        name=name,
        tasks=run.plans[name],
        model=run.model.named(part.model) if own_model else None,
        save_values=scenario.save_values and name == MAIN,
        language=scenario.language,
        timezone=scenario.timezone,
        handoffs=tuple(part.handoffs),
    )


def handoff_tool(name: str, to: str) -> dict[str, Any]:
    """The schema of a handoff tool, the same on every adapter."""
    return {
        "name": name,
        "description": f"Hand the call to the {to} agent.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def native_tool(name: str) -> dict[str, Any]:
    """The schema of a native tool, the same on every adapter."""
    return {
        "name": name,
        "description": f"The native {name} tool.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def load_scenarios() -> list[Scenario]:
    """Every scenario file, sorted by name."""
    return [
        Scenario.model_validate_json(path.read_text()) for path in sorted(SCENARIOS.glob("*.json"))
    ]


def event_line(stage: str, data: dict[str, Any]) -> str:
    """One task event as a short line. The run id is random, so it is left out."""
    parts = [stage]
    parts += [f"{key}={data[key]}" for key in ("group", "task", "step") if key in data]
    if "saved" in data:
        parts.append(f"saved={','.join(data['saved'])}")
    return " ".join(parts)


class ScriptedModel:
    """A ``ModelCall`` that answers each request with the current turn's next reply.

    A request at the wrong site, or past the turn's script, goes into ``problems``. It does
    not raise: an error inside a framework task can hang the run instead of failing it.
    """

    model = "scripted"

    def __init__(self) -> None:
        self.pending: list[Reply] = []
        self.requests: list[ModelRequest] = []
        self.answered: list[Reply | None] = []
        self.problems: list[str] = []

    def script(self, replies: list[Reply]) -> None:
        """Queue one turn's replies."""
        self.pending = list(replies)

    def named(self, name: str | None) -> "ScriptedModel | NamedModel":
        """Return this model, or a view of it under another name: an agent's own model."""
        return self if name is None else NamedModel(self, name)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Answer with the next reply's text, then its tool calls."""
        self.requests.append(request)
        number = len(self.requests)
        if not self.pending:
            self.answered.append(None)
            self.problems.append(f"request {number} at {request.site!r} has no scripted reply")
            return
        reply = self.pending.pop(0)
        self.answered.append(reply)
        if reply.fail:
            raise RuntimeError("the scripted model failed")
        if reply.site != request.site:
            self.problems.append(
                f"request {number} came from {request.site!r}, the script expects {reply.site!r}"
            )
        heard = [m["content"] for m in request.messages if m["role"] == "user"]
        if reply.heard is not None and heard != reply.heard:
            self.problems.append(
                f"request {number} heard {heard}, the script expects {reply.heard}"
            )
        outputs = [m["content"] for m in request.messages if m["role"] == "tool"]
        result = outputs[-1] if outputs else None
        if reply.result is not None and result != reply.result:
            self.problems.append(
                f"request {number} read {result!r}, the script expects {reply.result!r}"
            )
        if reply.result_holds is not None and reply.result_holds not in (result or ""):
            self.problems.append(
                f"request {number} read {result!r}, the script expects it to hold "
                f"{reply.result_holds!r}"
            )
        if (
            reply.variables is not None
            and (carried := {key: request.variables.get(key) for key in reply.variables})
            != reply.variables
        ):
            self.problems.append(
                f"request {number} carried {carried}, the script expects {reply.variables}"
            )
        prompt = next((m["content"] for m in request.messages if m["role"] == "system"), None)
        if reply.prompt_ends is not None and not _text(prompt).endswith(reply.prompt_ends):
            self.problems.append(
                f"request {number} had prompt {prompt!r}, the script expects it to end with "
                f"{reply.prompt_ends!r}"
            )
        if reply.params is not None and request.params != reply.params:
            self.problems.append(
                f"request {number} sent {request.params}, the script expects {reply.params}"
            )
        if reply.prompt is not None and prompt != reply.prompt:
            self.problems.append(
                f"request {number} had prompt {prompt!r}, the script expects {reply.prompt!r}"
            )
        tools = [tool["function"]["name"] for tool in request.tools]
        if reply.tools is not None and tools != reply.tools:
            self.problems.append(
                f"request {number} offered {tools}, the script expects {reply.tools}"
            )
        if reply.model is not None and request.model != reply.model:
            self.problems.append(
                f"request {number} went to {request.model!r}, the script expects {reply.model!r}"
            )
        if reply.filled is not None and not (filled := _filled_prompt(request)).startswith(
            reply.filled
        ):
            self.problems.append(
                f"request {number} had the filled prompt {filled!r}, the script expects it "
                f"to start with {reply.filled!r}"
            )
        if (
            reply.not_last_role is not None
            and request.messages
            and request.messages[-1]["role"] == reply.not_last_role
        ):
            self.problems.append(
                f"request {number} ends on a {reply.not_last_role} message, the script "
                "expects it not to"
            )
        if reply.tool_choice is not None and request.tool_choice != reply.tool_choice:
            self.problems.append(
                f"request {number} chose tools {request.tool_choice!r}, the script expects "
                f"{reply.tool_choice!r}"
            )
        systems = [_text(m.get("content")) for m in request.messages if m["role"] == "system"]
        last = systems[-1] if systems else None
        if reply.last_system is not None and last != reply.last_system:
            self.problems.append(
                f"request {number} ends its system messages with {last!r}, the script "
                f"expects {reply.last_system!r}"
            )
        if reply.holds is not None and not any(reply.holds in text for text in systems):
            self.problems.append(
                f"request {number} has no system message with {reply.holds!r}: {systems}"
            )
        order = [role for role, _ in _said(request.messages)]
        if reply.order is not None and order != reply.order:
            self.problems.append(
                f"request {number} has the messages {order}, the script expects {reply.order}"
            )
        for name, extra in (reply.replayed or {}).items():
            carried = [
                call.get("extra_content")
                for message in request.messages
                if message["role"] == "assistant"
                for call in message.get("tool_calls") or []
                if call["function"]["name"] == name
            ]
            if not carried or any(found != extra for found in carried):
                self.problems.append(
                    f"request {number} replayed {name} with {carried}, the script expects {extra}"
                )
        if reply.text:
            yield TextDelta(reply.text)
        if reply.fail_after_text:
            raise RuntimeError("the scripted model failed after its text")
        for index, call in enumerate(reply.calls):
            yield ToolCall(
                f"call_{number}_{index}", call.name, json.dumps(call.arguments), call.extra
            )
        if reply.usage is not None:
            yield Usage(*reply.usage)


class NamedModel:
    """The scripted model under another name, as an agent's own model.

    Its requests join the scripted model's, so one script runs the whole call.
    """

    def __init__(self, scripted: ScriptedModel, name: str) -> None:
        self.scripted = scripted
        self.model = name

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Answer through the scripted model."""
        return self.scripted.stream(request)


class RecordingTool:
    """A tool that answers with its next canned result and logs each run."""

    def __init__(
        self,
        name: str,
        results: list[Any],
        log: list[Call],
        returned: list[tuple[Call, Any]] | None = None,
    ) -> None:
        self.name = name
        self.results = list(results)
        self.log = log
        self.returned = returned if returned is not None else []

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        """Log the run and return the next result, or ``{"ok": true}`` when none is left."""
        call = Call(name=self.name, arguments=arguments)
        self.log.append(call)
        result = self.results.pop(0) if self.results else {"ok": True}
        self.returned.append((call, result))
        return result


class EventRecorder:
    """A tool on events: it logs each run and answers its next canned result."""

    def __init__(self, name: str, results: list[Any], log: list[Call], delay_s: float) -> None:
        self.name = name
        self.results = list(results)
        self.log = log
        self.delay_s = delay_s

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        """Wait, log the run and return the next result."""
        await asyncio.sleep(self.delay_s)
        self.log.append(Call(name=self.name, arguments=arguments))
        return self.results.pop(0) if self.results else {"ok": True}


def recorded_tool(
    name: str, locked: dict[str, Any], results: list[Any], log: list[Call], delay_s: float
) -> ToolSpec:
    """A tool the model never sees, every argument locked, logging each run."""
    parameters = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in locked},
        "required": list(locked),
    }
    executor = EventRecorder(name, results, log, delay_s)
    return ToolSpec(name, f"The {name} tool.", parameters, AnyResult, executor, locked=locked)


def event_tool(name: str, options: EventOptions, results: list[Any], log: list[Call]) -> EventTool:
    """Build one tool on events, every argument a locked text."""
    locked = {
        key: Transcript(text=True) if value == "$transcript" else value
        for key, value in options.locked.items()
    }
    spec = replace(
        recorded_tool(name, locked, results, log, options.delay_s),
        show_result=options.show_result,
        result_instructions=options.result_instructions,
    )
    return EventTool(spec, options.on, assign=options.assign, note=options.note)


class FakeLine:
    """A phone line that logs each transfer as a tool run, and fails when told to."""

    warm: ClassVar[bool] = False

    def __init__(self, case: TransferCase, name: str, log: list[Call], said: list[str]) -> None:
        self.case = case
        self.name = name
        self.log = log
        self.said = said

    async def transfer(self, transfer: Transfer) -> None:
        """Log the run, then fail as the case says."""
        self.log.append(Call(name=self.name))
        if self.case.no_caller:
            raise NoPhoneCaller
        if self.case.fails is not None:
            raise TransferError(self.case.fails)

    async def say(self, line: str) -> None:
        """Keep the line."""
        self.said.append(line)


@dataclass
class Run:
    """The parts one scenario run needs, built the same way for every adapter.

    Attributes:
        plans: Each main agent's plan, on one set of tools and one ``State``.
        telemetry: The call's telemetry. Its spans stay in ``spans``.
        agent: The main agent that holds the call. The driver sets it at the end.
        ending: The requests made before the call's end, once it is scripted.
        ended: The agent ended the call before the driver did. The driver sets it.
    """

    plans: dict[str, TaskPlan]
    state: CallState
    model: ScriptedModel
    spans: InMemorySpanExporter = field(default_factory=InMemorySpanExporter)
    tool_calls: list[Call] = field(default_factory=list)
    returned: list[tuple[Call, Any]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    spoken: list[list[str]] = field(default_factory=list)
    callers: list[str] = field(default_factory=list)
    agent: str = MAIN
    ending: int | None = None
    ended: bool = False
    call: CallHooks = field(default_factory=CallHooks)
    event_calls: list[Call] = field(default_factory=list)
    opening: list[str] = field(default_factory=list)
    line_said: list[str] = field(default_factory=list)
    realtime: bool = False
    dev: list[dict[str, Any]] = field(default_factory=list)
    controls: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Build the telemetry that records into ``spans``, and watch the task events."""
        self.call.hooks.on("task.*", "group.*")(self.on_event)
        self.telemetry = Telemetry(
            agent="conformance",
            call_id="call-1",
            processors=[SimpleSpanProcessor(self.spans)],
        )

    def trace_problems(self) -> list[str]:
        """Check the finished trace against what the scripted model really got and said.

        Each request has one ``llm`` span. It shows the messages the model got, its prompt
        filled, the tools offered and the calls the reply chose. The event spans are the
        task events, in order. Each handoff names a main agent, and a step's handoff
        records ``handoff`` before ``task.handed_off``.
        """
        spans = sorted(self.spans.get_finished_spans(), key=lambda span: span.start_time or 0)
        names = {span.context.span_id: span.name for span in spans}
        problems = [
            f"an llm span sits in {names.get(span.parent.span_id) if span.parent else None}"
            for span in spans
            if span.name == "llm"
            and (span.parent is None or names.get(span.parent.span_id) != "turn")
        ]
        problems += self._reply_problems(spans) if self.realtime else self._request_problems(spans)
        return problems + self._event_problems(spans)

    def _reply_problems(self, spans: list[Any]) -> list[str]:
        """Check a realtime model's ``llm`` spans: one per reply, with its token counts.

        A realtime model's span is its reply, reported once it is over. A fixed line it was
        told to say is a reply too, though the scripted model never saw it.
        """
        replies = [dict(span.attributes or {}) for span in spans if span.name == "llm"]
        if len(replies) < len(self.model.requests):
            return [
                f"the trace has {len(replies)} llm spans for {len(self.model.requests)} requests"
            ]
        tokens = iter(
            (reply.get("gen_ai.usage.input_tokens"), reply.get("gen_ai.usage.output_tokens"))
            for reply in replies
        )
        return [
            f"no llm span shows the tokens {reply.usage}"
            for reply in self.model.answered
            if reply is not None and reply.usage is not None
            if not any(counted == reply.usage for counted in tokens)
        ]

    def _request_problems(self, spans: list[Any]) -> list[str]:
        """Check one ``llm`` span per request: what the model got, offered and chose."""
        requests = [dict(span.attributes or {}) for span in spans if span.name == "llm"]
        if len(requests) != len(self.model.requests):
            return [
                f"the trace has {len(requests)} llm spans for {len(self.model.requests)} requests"
            ]
        problems = []
        pairs = zip(requests, self.model.requests, self.model.answered, strict=True)
        for number, (span, request, reply) in enumerate(pairs, 1):
            sent = json.loads(str(span.get("input.value", "[]")))
            if _said(sent) != _said(request.messages):
                problems.append(
                    f"llm span {number} shows {_said(sent)}, not {_said(request.messages)}"
                )
            if any("{{" in str(m.get("content")) for m in sent if m["role"] == "system"):
                problems.append(f"llm span {number} shows an unfilled prompt")
            offered = json.loads(str(span.get("gen_ai.tool.definitions", "[]")))
            if _names(offered) != _names(request.tools):
                problems.append(f"llm span {number} offers {_names(offered)}")
            if span.get("gen_ai.request.model") != request.model:
                problems.append(
                    f"llm span {number} names {span.get('gen_ai.request.model')!r}, the "
                    f"request went to {request.model!r}"
                )
            output = str(span.get("output.value", ""))
            chose = _names(json.loads(output)["tool_calls"]) if output.startswith("{") else []
            if reply is not None and chose != [call.name for call in reply.calls]:
                problems.append(f"llm span {number} chose {chose}")
            tokens = (span.get("gen_ai.usage.input_tokens"), span.get("gen_ai.usage.output_tokens"))
            if reply is not None and reply.usage is not None and tokens != reply.usage:
                problems.append(f"llm span {number} shows tokens {tokens}, not {reply.usage}")
        return problems

    def _event_problems(self, spans: list[Any]) -> list[str]:
        """Check the event spans, the handoffs, the tool spans and the turns."""
        problems = []
        events = [
            (span.name, json.loads(str((span.attributes or {})["input.value"])))
            for span in spans
            if (span.attributes or {}).get("langfuse.observation.type") == "event"
        ]
        lines = [event_line(name, data) for name, data in events if name != "handoff"]
        if lines != self.events:
            problems.append(f"the trace has the events {lines}")
        handed = [name for name, _ in events if name in ("handoff", "task.handed_off")]
        problems += [
            "task.handed_off comes without a handoff right before it"
            for index, name in enumerate(handed)
            if name == "task.handed_off" and (index == 0 or handed[index - 1] != "handoff")
        ]
        names = {MAIN, *self.plans}
        problems += [
            f"a handoff names {data}"
            for name, data in events
            if name == "handoff"
            if not {data["from"], data["to"]} <= names
        ]
        return problems + self._tool_problems(spans) + self._turn_problems(spans)

    def _turn_problems(self, spans: list[Any]) -> list[str]:
        """Check one turn per caller line, with that line in and the lines said out.

        Every span of a turn's work sits in a turn. The call's input is the transcript, one
        entry per line, and its output is how the call ended and the final state.
        """
        turns = [dict(span.attributes or {}) for span in spans if span.name == "turn"]
        problems = []
        callers: list[str | None] = [*([None] if self.opening else []), *self.callers]
        spoken = [*([self.opening] if self.opening else []), *self.spoken]
        if len(turns) != len(callers):
            problems.append(f"the trace has {len(turns)} turns for {len(callers)} callers")
        for number, (turn, caller, said) in enumerate(zip(turns, callers, spoken, strict=False), 1):
            if turn.get("input.value") != caller:
                problems.append(f"turn {number} shows the caller {turn.get('input.value')!r}")
            if turn.get("output.value") != (" ".join(said) or None):
                problems.append(f"turn {number} shows the agent {turn.get('output.value')!r}")
        names = {span.context.span_id: span.name for span in spans}
        problems += [
            f"{span.name} sits in {names.get(span.parent.span_id) if span.parent else None}"
            for span in spans
            if span.name in (STT_SPAN, TTS_SPAN, TOOL_SPAN)
            or (span.attributes or {}).get("langfuse.observation.type") == "event"
            if span.parent is None or names.get(span.parent.span_id) != "turn"
        ]
        [call] = [dict(span.attributes or {}) for span in spans if span.name == "conversation"]
        transcript = [
            line
            for caller, said in zip(callers, spoken, strict=False)
            for line in [
                *([{"role": "user", "content": caller}] if caller is not None else []),
                *({"role": "assistant", "content": text} for text in said),
            ]
        ]
        if json.loads(str(call.get("input.value", "[]"))) != transcript:
            problems.append(f"the call shows the transcript {call.get('input.value')}")
        output = str(call.get("output.value", ""))
        ended = json.loads(output) if output.startswith("{") else output
        if not isinstance(ended, dict) or set(ended) != {"end_reason", "state", "confirmed"}:
            problems.append(f"the call shows the output {ended}")
        elif ended["state"] != self.state.dump():
            problems.append(f"the call shows the state {ended['state']}")
        return problems

    def _tool_problems(self, spans: list[Any]) -> list[str]:
        """Check that each call the model chose has one tool span, with its input and output.

        A finish check's ``not_yet`` is an answer, never run, so it has no span, and so is
        every call of a decision the call's end asks for. The input is the call's
        arguments. Every span has a result, or an error: a task the call ended in has
        none. When a later request
        holds the call's result, the span shows that same text. Otherwise each tool run
        shows what the tool returned, also when the model reads only how a step ended.
        A later step that opens on ``full`` history reads that ended answer, not the result.
        """
        tools: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            if span.name == TOOL_SPAN:
                attributes = dict(span.attributes or {})
                tools.setdefault(str(attributes["gen_ai.tool.call.id"]), []).append(attributes)
        read = {
            m["tool_call_id"]: _text(m.get("content"))
            for request in self.model.requests
            for m in request.messages
            if m["role"] == "tool"
        }
        problems = []
        answered = self.model.answered[: self.ending]
        for number, reply in enumerate(answered, 1):
            for index, call in enumerate(reply.calls if reply else ()):
                if call.name == NOT_YET_TOOL_NAME:
                    continue
                call_id = f"call_{number}_{index}"
                found = tools.get(call_id, [])
                if len(found) != 1:
                    problems.append(f"{call.name} ({call_id}) has {len(found)} tool spans")
                    continue
                if _arguments(found[0]) != call.arguments:
                    problems.append(
                        f"{call.name} ({call_id}) shows the input {_arguments(found[0])}"
                    )
                result = str(found[0].get("gen_ai.tool.call.result") or "")
                if not result and not found[0].get("tool.error"):
                    problems.append(f"{call.name} ({call_id}) has no output")
                elif call_id in read and result != read[call_id] and read[call_id] != STEP_ENDED:
                    problems.append(
                        f"{call.name} ({call_id}) shows {result!r}, the model read {read[call_id]!r}"
                    )
        by_call = [
            (Call(name=str(a["gen_ai.tool.name"]), arguments=_arguments(a)), a)
            for found in tools.values()
            for a in found
        ]
        for call, returned in self.returned:
            spans_of = [
                a
                for made, a in by_call
                if made.name == call.name
                and all(call.arguments.get(key) == value for key, value in made.arguments.items())
            ]
            if any(a.get("tool.error") for a in spans_of):
                continue
            shown = [json.loads(str(a.get("gen_ai.tool.call.result") or "null")) for a in spans_of]
            read_ids = {str(a["gen_ai.tool.call.id"]) for a in spans_of} & read.keys()
            if not any(_holds(item, returned) for item in shown) and not read_ids:
                problems.append(f"{call.name} returned {returned}, its spans show {shown}")
        return problems

    def recorder(self, target: Literal["pipecat", "livekit"]) -> DevRecorder:
        """Build the call's dev recorder, which keeps its records in ``dev``."""
        return DevRecorder("call-1", self.dev.append, target=target)

    def dev_problems(self) -> list[str]:
        """Check the dev records against what the call did, the same on every adapter.

        Every record passes the contract and the call ends. Each caller line is one input
        with that text, final, and an input the agent spoke after has a response linked to
        it. Each line the agent said is one of its text rows. Each tool run the trace shows has
        one ``tool`` operation, but a tool that starts a task or a group, which the run's
        ``handoff`` operation draws, and a step's opening tool, which code runs with no
        framework tool call to see. A handoff tool and a transfer tool have one ``handoff``
        operation per run and no ``tool_duration``.
        """
        if not self.dev:
            return ["the call sent no dev records"]
        problems = [
            f"dev record {record['kind']} {record['id']}: {problem}"
            for record in self.dev
            for problem in record_problems(json.dumps(record))
        ]
        latest = {(r["kind"], r["id"]): r[r["kind"]] for r in self.dev}
        if latest["call", "call"]["state"] == "open":
            problems.append("the dev records never end the call")
        exchanges = [(key[1], p) for key, p in latest.items() if key[0] == "exchange"]
        texts = [p for key, p in latest.items() if key[0] == "text" and p["speaker"] == "user"]
        inputs = [identity for identity, p in exchanges if p["role"] == "input"]
        heard = [
            "".join(
                p["separator_before"] + p["text"]
                for p in texts
                if p["exchange_id"] == identity and p["state"] == "final"
            )
            for identity in inputs
        ]
        if heard != self.callers:
            problems.append(f"the dev records hear the caller say {heard}")
        linked = {p.get("input_id") for _, p in exchanges if p["role"] == "response"}
        problems += [
            f"no response is linked to the input of caller line {number}"
            for number, (identity, said) in enumerate(zip(inputs, self.spoken, strict=False), 1)
            if said and identity not in linked
        ]
        replies = {
            p["text"].strip()
            for key, p in latest.items()
            if key[0] == "text" and p["speaker"] == "assistant"
        }
        problems += [
            f"no dev text row holds the line {line!r}"
            for line in [*self.opening, *(line for said in self.spoken for line in said)]
            if line not in replies
        ]
        delegates = {d.name for plan in self.plans.values() for d in plan.delegates()}
        runs = [
            dict(span.attributes or {})
            for span in self.spans.get_finished_spans()
            if span.name == TOOL_SPAN
        ]
        ran = sorted(
            str(run["gen_ai.tool.name"])
            for run in runs
            if run["gen_ai.tool.name"] not in delegates | self.controls
            if not str(run["gen_ai.tool.call.id"]).startswith("opening_")
        )
        tools = sorted(p["name"] for key, p in latest.items() if p.get("type") == "tool")
        if tools != ran:
            problems.append(f"the dev records show the tools {tools}, not {ran}")
        problems += self._dev_control_problems(latest, runs)
        return problems + self._dev_model_problems(latest)

    def _dev_control_problems(
        self, latest: dict[tuple[str, str], dict[str, Any]], runs: list[dict[str, Any]]
    ) -> list[str]:
        """Check that each run of a handoff or transfer tool has one ``handoff`` operation.

        None of them has a ``tool_duration``, because the tool hands the call over and
        returns no result.
        """
        went = sorted(
            str(run["gen_ai.tool.name"])
            for run in runs
            if run["gen_ai.tool.name"] in self.controls
            if not str(run["gen_ai.tool.call.id"]).startswith("opening_")
        )
        drawn = {
            key[1]: p["name"]
            for key, p in latest.items()
            if key[0] == "operation" and p["type"] == "handoff" and p["name"] in self.controls
        }
        found = sorted(drawn.values())
        problems = (
            [] if found == went else [f"the dev records show the handoffs {found}, not {went}"]
        )
        timed = [
            p["operation_id"]
            for key, p in latest.items()
            if key[0] == "measurement"
            and p["metric"] == "tool_duration"
            and p.get("operation_id") in drawn
        ]
        return problems + [f"the handoff {identity} has a tool_duration" for identity in timed]

    def _dev_model_problems(self, latest: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
        """Check one ``llm`` operation per request, naming the model the request went to.

        A side request, a handoff's summary or a step's finish check, has no operation on
        any adapter. A realtime model's operations are its replies, which the trace checks
        instead. The check stops at a failed request: a plain LiveKit agent learns of its
        requests from LiveKit's metrics, which a failed one never sends.
        """
        if self.realtime:
            return []
        named = [
            p.get("model")
            for key, p in latest.items()
            if key[0] == "operation" and p["type"] == "llm"
        ]
        went: list[tuple[str, bool]] = []
        for request, reply in zip(self.model.requests, self.model.answered, strict=True):
            if request.site == SUMMARY_SITE or NOT_YET_TOOL_NAME in _names(request.tools):
                continue
            if reply is not None and reply.fail:
                named = named[: len(went)]
                break
            went.append((request.model, reply is not None and reply.fail_after_text))
        if not _matches(named, went):
            return [f"the dev records show the llm operations {named} for the requests {went}"]
        return []

    @property
    def plan(self) -> TaskPlan:
        """The plan of the agent that answers first."""
        return self.plans[MAIN]

    def native(self, name: str, results: dict[str, Any], arguments: dict[str, Any]) -> object:
        """Run a native tool: log the call like a job tool's, and return its canned result."""
        call = Call(name=name, arguments=arguments)
        self.tool_calls.append(call)
        self.returned.append((call, results[name]))
        return results[name]

    def on_event(self, stage: str, data: dict[str, Any]) -> None:
        """The task events' observer: keep each event as a short line."""
        self.events.append(event_line(stage, data))

    def end_call(self, replies: list[Reply]) -> None:
        """Script the requests the call's end makes. Their calls are answers, never run."""
        self.ending = len(self.model.requests)
        self.model.script(replies)

    def unused(self) -> list[str]:
        """Name each scripted reply of the turn, or of the call's end, that no request took."""
        return [
            f"no request took the scripted {reply.site!r} reply" for reply in self.model.pending
        ]

    def outcome(self) -> dict[str, Any]:
        """What the run did, in the shape of ``expected(scenario)``."""
        return {
            "spoken": self.spoken,
            "state": {name: self.state.plain(name) for name in self.state.names},
            "tool_calls": self.tool_calls,
            "events": self.events,
            "agent": self.agent,
            "ended": self.ended,
            "opening": self.opening,
            "event_calls": self.event_calls,
            "line_said": self.line_said,
            "problems": self.model.problems
            + self.unused()
            + self.trace_problems()
            + self.dev_problems(),
        }


def _matches(named: list[Any], went: list[tuple[str, bool]]) -> bool:
    """Check the operations' models against the requests', in order.

    A request that failed after its text may have an operation or none: a failed request
    has no metrics on every adapter, but its text did go out.
    """
    if not went:
        return not named
    (model, optional), rest = went[0], went[1:]
    if named and named[0] == model and _matches(named[1:], rest):
        return True
    return optional and _matches(named, rest)


def _holds(shown: object, returned: object) -> bool:
    """Return True when a span shows what a tool returned, maybe with its output's defaults."""
    if isinstance(shown, dict) and isinstance(returned, dict):
        return all(key in shown and _holds(shown[key], value) for key, value in returned.items())
    return shown == returned


def _arguments(attributes: dict[str, Any]) -> dict[str, Any]:
    """Return the arguments a tool span shows, parsed."""
    return json.loads(str(attributes.get("gen_ai.tool.call.arguments") or "{}"))


def _text(content: object) -> str:
    """Return a message's text, from a string or a list of text parts."""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def _filled_prompt(request: ModelRequest) -> str:
    """Return the request's system messages, placeholders filled, joined by a blank line."""
    filled = fill_messages(request.messages, request.variables)
    return "\n\n".join(_text(m["content"]) for m in filled if m["role"] == "system")


def _said(messages: list[Any]) -> list[tuple[str, str]]:
    """Return each user, assistant and tool message as role and text, skipping empty ones."""
    return [
        (m["role"], _text(m.get("content")))
        for m in messages
        if m["role"] in ("user", "assistant", "tool") and _text(m.get("content"))
    ]


def _names(tools: list[Any]) -> list[str]:
    """Return the function names of OpenAI tools or tool calls."""
    return [tool["function"]["name"] for tool in tools]


def new_run(scenario: Scenario, *, realtime: bool = False) -> Run:
    """Build the plan, state, scripted model and recording tools for one run."""
    log: list[Call] = []
    returned: list[tuple[Call, Any]] = []
    recorded = registry(
        **{
            name: RecordingTool(name, scenario.tool_results.get(name, []), log, returned)
            for name in ("lookup_order", "search_web", "normalize")
        }
    )
    line_said: list[str] = []
    transfers = [
        transfer_tool(
            Transfer(name, "+34910000000", on_unavailable=case.on_unavailable),
            FakeLine(case, name, log, line_said),
        )
        for name, case in scenario.transfers.items()
    ]
    state_model = ShapedState if scenario.shaped_state else State
    tools = ToolRegistry(
        [
            *(
                replace(spec, **options.fields())
                if (options := scenario.tool_options.get(name))
                else spec
                for name in recorded.names
                if (spec := recorded.get(name))
            ),
            *transfers,
        ],
        state=state_model,
        assign=scenario.assign,
        replies=scenario.replies,
    )
    plans = {
        name: TaskPlan(
            agent.tasks,
            state=state_model,
            tools=tools if agent.tools is None else tools.only(agent.tools),
        )
        for name, agent in [(MAIN, scenario.agent(MAIN)), *scenario.agents.items()]
    }
    state = CallState(
        state_model,
        confirm=plans[MAIN].confirm,
        call_start=dict.fromkeys(set(scenario.call_start) - {"call"}, "optional"),
    )
    save_call_start(state, scenario.call_start)
    event_calls: list[Call] = []
    lookups = [
        Lookup(
            name,
            recorded_tool(name, {}, scenario.tool_results.get(name, []), event_calls, opt.delay_s),
            opt.assign,
            background=opt.background,
        )
        for name, opt in scenario.prefetch.items()
    ]

    hooks = Hooks(
        *(
            event_tool(name, options, scenario.tool_results.get(name, []), event_calls)
            for name, options in scenario.event_tools.items()
        )
    )
    add_hooks(hooks, scenario.hooks)
    if scenario.ready_fails:

        @hooks.before_call
        async def not_ready() -> None:
            raise RuntimeError("the TTS did not answer")

    call = CallHooks(
        state, hooks=hooks, prefetch=Prefetch(lookups, state=state_model) if lookups else None
    )
    return Run(
        plans=plans,
        state=state,
        model=ScriptedModel(),
        tool_calls=log,
        callers=[turn.caller for turn in scenario.turns],
        returned=returned,
        call=call,
        event_calls=event_calls,
        line_said=line_said,
        realtime=realtime,
        controls=scenario.controls,
    )


def add_hooks(hooks: Hooks, options: HookOptions) -> None:
    """Add the scenario's hooks."""
    if options.prompt_suffix is not None:
        hooks.add(PromptSuffix(options.prompt_suffix))
    if options.refuse:
        hooks.before_tool(
            lambda run: Refuse(options.refuse[run.name]) if run.name in options.refuse else None
        )
    if options.results:
        hooks.after_tool(lambda run, result: options.results.get(run.name))
    if options.params:
        hooks.before_model(lambda request: replace(request, params=request.params | options.params))


def expected(scenario: Scenario, *, realtime: bool = False) -> dict[str, Any]:
    """What every adapter's ``Run.outcome`` must equal, on a cascade or a realtime model."""
    return {
        "spoken": [
            turn.realtime_spoken if realtime and turn.realtime_spoken is not None else turn.spoken
            for turn in scenario.turns
        ],
        "state": scenario.expect.state,
        "tool_calls": scenario.expect.tool_calls,
        "events": scenario.expect.events,
        "agent": scenario.expect.agent,
        "ended": scenario.expect.ended,
        "opening": scenario.expect.opening,
        "event_calls": scenario.expect.event_calls,
        "line_said": scenario.expect.line_said,
        "problems": [],
    }
