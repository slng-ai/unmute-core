"""An agent file: one agent the simulator calls, written as JSON, with mocked tools.

The file holds what an ``AgentSpec`` needs as data: the instructions, the greeting, the
State's values and shapes, the tools, what the call reads as it starts, and the tasks in
the ``TasksSpec`` JSON form. Each tool is a deterministic mock (``_mocks``) or one of the
library's built-in tools, so the checks know what every tool returned.

``build`` turns a file into a fresh call: its state, hooks and ``AgentSpec``. Every call
gets its own, so calls can run side by side.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks, EventTool, save_call_start
from unmute_core.greeting import Greeting
from unmute_core.hooks import Hooks
from unmute_core.model import ModelCall
from unmute_core.prefetch import (
    CallValue,
    CallValueSource,
    Clock,
    ClockField,
    Lookup,
    Prefetch,
)
from unmute_core.state import CallInfo, CallState, StateSchema
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.testing.simulate._mocks import MockLog, MockResult, MockTool
from unmute_core.tools import ToolRegistry, ToolSpec, current_datetime_tool, user_phone_number_tool

AGENTS = Path(__file__).parent / "agents"
"""The agent files that ship with the simulator."""

NO_ARGUMENTS: Mapping[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

Mock = Literal["availability", "orders", "appointments", "customer_records", "sms", "hook", "fixed"]
Builtin = Literal["current_datetime", "user_phone_number", "end_call"]


class _File(BaseModel):
    """A part of an agent file. An unknown key is a typo, so it is refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolFile(_File):
    """One tool: a mock, or a built-in tool of the library.

    Attributes:
        name: The tool's name, as the model and the tasks call it.
        description: What the model reads. A built-in tool has its own.
        parameters: The JSON Schema of the arguments.
        mock: The mock that answers it.
        result: The result a ``fixed`` mock returns.
        builtin: ``current_datetime``, ``user_phone_number`` or ``end_call``.
        locked: Argument name to its value, like ``"{{caller_phone}}"``. The model never
            sees it.
        announce: A line said as the tool starts.
        result_instructions: How the model should use the result.
        goodbye: For ``end_call``: the goodbye the model says in the reply that ends the call.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: dict(NO_ARGUMENTS))
    mock: Mock | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    builtin: Builtin | None = None
    locked: dict[str, Any] = Field(default_factory=dict)
    announce: str = ""
    result_instructions: str = ""
    goodbye: str = ""

    @model_validator(mode="after")
    def _one_kind(self) -> Self:
        """Refuse a tool that is both a mock and a built-in tool, or neither."""
        if (self.mock is None) == (self.builtin is None):
            raise ValueError(f"tool {self.name!r} needs exactly one of mock and builtin")
        return self


class PrefetchFile(_File):
    """One thing the call reads as it starts, before the greeting.

    Attributes:
        name: The entry's name. A task's ``tool`` input reads a ``lookup`` by this name.
        clock: State value name to the clock field it gets, like ``{"today": "date"}``.
        call_value: The ``state.call`` field to copy, like ``calling_number``.
        assign: The State values a ``call_value`` fills.
        lookup: The tool a lookup runs: one of the agent's tools, or a built-in tool.
    """

    name: str
    clock: dict[str, ClockField] | None = None
    call_value: CallValueSource | None = None
    assign: list[str] = Field(default_factory=list)
    lookup: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> Self:
        """Refuse an entry that is not exactly one of a clock, a call value or a lookup."""
        if sum(kind is not None for kind in (self.clock, self.call_value, self.lookup)) != 1:
            raise ValueError(f"prefetch {self.name!r} needs one of clock, call_value and lookup")
        return self


class EventFile(_File):
    """A tool that runs on a call event rather than when the model calls it.

    Attributes:
        tool: One of the agent's tools.
        on: Its events, like ``call_end`` or ``first_user_message``.
        wait: Hold the next model request until it ran.
    """

    tool: str
    on: list[str]
    wait: bool = True


class AgentFile(_File):
    """One agent the simulator calls.

    Attributes:
        name: The file's name, which scenarios use.
        business: The made-up business it answers for.
        about: One sentence on what it does and what it is for.
        language: The call's language.
        timezone: The business's IANA zone. Today is today there.
        instructions: The main agent's prompt. ``{{value}}`` reads a State value.
        greeting: What it says first.
        variables: The State's values, as ``StateSchema`` reads them.
        shapes: The State's shapes.
        prefetch: What the call reads as it starts.
        tools: Every tool, task-only ones included.
        events: Tools that run on call events.
        tasks: The tasks and groups, as ``TasksSpec`` JSON. None for an agent with none.
    """

    name: str
    business: str
    about: str
    language: str = "en"
    timezone: str = "UTC"
    instructions: str
    greeting: str
    variables: list[dict[str, Any]] = Field(default_factory=list)
    shapes: list[dict[str, Any]] = Field(default_factory=list)
    prefetch: list[PrefetchFile] = Field(default_factory=list)
    tools: list[ToolFile] = Field(default_factory=list)
    events: list[EventFile] = Field(default_factory=list)
    tasks: dict[str, Any] | None = None


def load_agent(name: str) -> AgentFile:
    """Read an agent file: a bundled agent by name, or a path to a JSON file.

    Raises:
        FileNotFoundError: If there is no such agent.
    """
    path = Path(name)
    if path.suffix != ".json":
        path = AGENTS / f"{name}.json"
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in AGENTS.glob("*.json")))
        raise FileNotFoundError(f"no agent {name!r}: give a JSON file, or one of {known}")
    return AgentFile.model_validate_json(path.read_text())


@dataclass
class AgentCall:
    """One call to an agent file: what a framework driver builds its agent from.

    Attributes:
        agent: The agent file.
        spec: The agent, ready for ``UnmuteAgent`` or ``UnmuteFlows``.
        state: The call's state.
        call: The call's hooks.
        hooks: The hooks the recorder watches through.
        greeting: What the agent says first.
        mocks: Every mock call, with its locked arguments.
        plan: The agent's tasks, or None.
        tool_names: The agent's own tools.
    """

    agent: AgentFile
    spec: AgentSpec
    state: CallState
    call: CallHooks
    hooks: Hooks
    greeting: Greeting
    mocks: MockLog
    plan: TaskPlan | None = None
    tool_names: frozenset[str] = field(default_factory=frozenset)

    def delegates(self) -> set[str]:
        """Return the names of the tools that start a task or a group."""
        return {delegate.name for delegate in self.plan.delegates()} if self.plan else set()

    def providers(self) -> dict[str, str]:
        """Return each task to the one task that saves all it requires."""
        return dict(self.plan.providers) if self.plan else {}

    def via_tools(self) -> dict[str, str]:
        """Return each tool whose success enters a task, to that task."""
        return dict(self.plan.via_tasks) if self.plan else {}

    def task_values(self) -> set[str]:
        """Return the State values some task saves."""
        if self.plan is None:
            return set()
        return {row.variable for task in self.plan.tasks.values() for row in task.assign}


class _Ended(BaseModel):
    """What ``end_call`` returns."""

    ended: bool = True


class _EndCall:
    """Ends the call. The library hangs up once the tool has succeeded."""

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        """Say the call ended."""
        return {"ended": True}


def _tool(tool: ToolFile, timezone: str, mocks: MockLog) -> ToolSpec:
    """Build one tool of the file."""
    if tool.builtin == "current_datetime":
        return current_datetime_tool(timezone, name=tool.name)
    if tool.builtin == "user_phone_number":
        return user_phone_number_tool(name=tool.name)
    if tool.builtin == "end_call":
        goodbye = f' Say this goodbye in that same reply: "{tool.goodbye}"' if tool.goodbye else ""
        description = (tool.description or "End the call.") + goodbye
        return ToolSpec(
            tool.name,
            description,
            dict(NO_ARGUMENTS),
            _Ended,
            _EndCall(),
            effect="ends_conversation",
        )
    assert tool.mock is not None  # _one_kind
    return ToolSpec(
        tool.name,
        tool.description,
        tool.parameters,
        MockResult,
        MockTool(tool.mock, tool.name, tool.result, mocks),
        tool.locked,
        tool.announce or (),
        result_instructions=tool.result_instructions,
    )


def _prefetch(
    entries: list[PrefetchFile],
    tools: Mapping[str, ToolSpec],
    timezone: str,
    state: type[BaseModel],
) -> Prefetch | None:
    """Build what the call reads as it starts, or None when it reads nothing."""
    built: list[Clock | CallValue | Lookup] = []
    for entry in entries:
        if entry.clock is not None:
            built.append(Clock(entry.name, timezone, entry.clock))
        elif entry.call_value is not None:
            built.append(CallValue(entry.name, entry.call_value, entry.assign))
        else:
            assert entry.lookup is not None  # _one_kind
            run = tools.get(entry.lookup)
            if run is None and entry.lookup == "current_datetime":
                run = current_datetime_tool(timezone)
            if run is None:
                raise ValueError(f"prefetch {entry.name!r} runs {entry.lookup!r}, no such tool")
            built.append(Lookup(entry.name, run=run, assign={}))
    return Prefetch(built, state=state) if built else None


def caller_line(facts: Mapping[str, str]) -> dict[str, Any]:
    """Return the ``call_start`` object of a call from a caller's number, or an empty one."""
    number = facts.get("calling_number")
    if not number:
        return {}
    return {
        "call": {
            "direction": facts.get("direction", "inbound"),
            "calling_number": number,
            "remote_number": number,
        }
    }


def build(
    agent: AgentFile,
    model: ModelCall | None,
    *,
    line: Mapping[str, str] | None = None,
    hooks: Hooks | None = None,
) -> AgentCall:
    """Build a fresh call to the agent.

    Args:
        agent: The agent file.
        model: The model the agent runs on. None builds a call only to read its plan.
        line: What the line carries, like the caller's number: ``calling_number`` and
            ``direction``. None for a call with no caller ID.
        hooks: The call's hooks, with the recorder's observers. A fresh set when None.

    Returns:
        The call, ready for a framework driver.

    Raises:
        ValueError: If the file does not fit together, naming the first problem.
    """
    schema = StateSchema(agent.variables, agent.shapes)
    mocks = MockLog(agent.timezone)
    specs = {tool.name: _tool(tool, agent.timezone, mocks) for tool in agent.tools}
    events = {event.tool for event in agent.events}
    lookups = {entry.lookup for entry in agent.prefetch if entry.lookup}
    callable_specs = [spec for name, spec in specs.items() if name not in events | lookups]
    registry = ToolRegistry(callable_specs, state=schema.model)
    plan = None
    if agent.tasks is not None:
        plan = TaskPlan(
            TasksSpec.model_validate(agent.tasks),
            state=schema.model,
            confirm=schema.confirm,
            tools=registry,
        )
        plan.check_lookups(entry.name for entry in agent.prefetch if entry.lookup)
    hooks = hooks or Hooks()
    for event in agent.events:
        hooks.add(EventTool(specs[event.tool], on=event.on, wait=event.wait))
    state = CallState(
        schema.model, confirm=schema.confirm, call=CallInfo(agent_id=f"simulate-{agent.name}")
    )
    save_call_start(state, caller_line(line or {}), env=False)
    call = CallHooks(
        state, hooks=hooks, prefetch=_prefetch(agent.prefetch, specs, agent.timezone, schema.model)
    )
    spec = AgentSpec(
        instructions=agent.instructions,
        name="main",
        tasks=plan,
        tools=None if plan is not None else registry,
        model=model,
        language=agent.language,
        timezone=agent.timezone,
    )
    return AgentCall(
        agent=agent,
        spec=spec,
        state=state,
        call=call,
        hooks=hooks,
        greeting=Greeting(text=agent.greeting),
        mocks=mocks,
        plan=plan,
        tool_names=frozenset(specs),
    )


def describe(agent: AgentFile) -> str:
    """Return the agent as one line: its name, business and what it is for."""
    return f"{agent.name} ({agent.business}): {agent.about}"
