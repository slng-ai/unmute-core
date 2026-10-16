"""Parcel Desk's agent, built from its spec: the State, tools, tasks and greeting.

This is what a worker does with a spec it receives. ``build`` turns the config
into library objects once. ``Desk.new_call`` starts one call, and ``Desk.agent`` describes
the call's agent as an ``AgentSpec``, its model requests sent through the SLNG Context
Router. Nothing here imports LiveKit or Pipecat: ``livekit/agent.py`` runs the spec with
``UnmuteAgent`` and ``pipecat/bot.py`` with ``UnmuteFlows``, each with its own pipeline.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from unmute_core.agent import AgentSpec
from unmute_core.call import save_call_start
from unmute_core.greeting import Greeting
from unmute_core.prefetch import CallValue, Clock, Prefetch
from unmute_core.slng import RouterModelCall, SlngSettings
from unmute_core.state import CallInfo, CallState, StateSchema
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.tools import ToolRegistry, local_tool
from unmute_core.turn import TurnWindow, turn_window

from desk.spec import SPEC
from desk.tools import Depot, end_call


class RouterSettings(SlngSettings):
    """The router settings, with this desk's defaults. ``SLNG_*`` variables override them.

    The model stays the library's ``slng/auto``: the router config picks it.

    Attributes:
        router_url: The SLNG Context Router in the EU.
    """

    router_url: str = "https://eu-north.context-router.slng.ai/v1"


def prefetch(rows: list[dict[str, Any]], state: type[BaseModel]) -> Prefetch:
    """Return the spec's prefetch: a clock reading or a copied call value per row."""
    entries = [
        Clock(row["clock"], row["timezone"], row["assign"])
        if "clock" in row
        else CallValue(row["call_value"], row["source"], row["assign"])
        for row in rows
    ]
    return Prefetch(entries, state=state)


def tools(
    depot: Depot, options: Mapping[str, Mapping[str, Any]], state: type[BaseModel]
) -> ToolRegistry:
    """Return the spec's tools: each Python function with the options the spec gives it.

    Built with the State, so a locked argument that names a value the State lacks, or one
    of the wrong type, fails here, before the first call.
    """
    functions: dict[str, Callable[..., Any]] = {
        "find_parcel": depot.find_parcel,
        "reschedule_delivery": depot.reschedule_delivery,
        "save_callback_note": depot.save_callback_note,
        "end_call": end_call,
    }
    return ToolRegistry(
        [
            local_tool(function, name=name, **options.get(name, {}))
            for name, function in functions.items()
        ],
        state=state,
    )


@dataclass(frozen=True)
class Desk:
    """The agent, built once per process and shared by every call.

    Attributes:
        agent_id: The agent's id, which the router groups requests by.
        instructions: The main prompt, its placeholders unfilled.
        schema: The State, built from the spec's type text.
        plan: The tasks and groups, on the spec's tools.
        prefetch: What is read before the greeting.
        greeting: How a call opens.
        max_duration_s: The longest a call may run.
        pace: The turn pace: snappy, balanced or patient.
    """

    agent_id: str
    instructions: str
    schema: StateSchema
    plan: TaskPlan
    prefetch: Prefetch
    greeting: Greeting
    max_duration_s: float
    pace: str

    def window(self, framework: str) -> TurnWindow:
        """Return the spec's turn pace as one framework's window."""
        return turn_window(self.pace, framework)  # ty: ignore[invalid-argument-type]

    async def new_call(
        self, call: CallInfo | None = None, dispatched: Mapping[str, object] | None = None
    ) -> CallState:
        """Return a fresh call state, with the prefetch read in.

        A browser call has no caller ID. ``UNMUTE_CALL_START`` can simulate one, like
        ``{"call": {"direction": "inbound", "calling_number": "+447911123456"}}``.

        Args:
            call: What the line carries. A web call's when None.
            dispatched: The dispatch's ``call_start`` object, when there is one.
        """
        call = call or CallInfo(agent_id=self.agent_id)
        state = CallState(self.schema.model, confirm=self.schema.confirm, call=call)
        save_call_start(state, dispatched)
        await self.prefetch.run(state)
        return state

    def model_call(self, call: CallInfo) -> RouterModelCall:
        """Return the call's model call: the SLNG Context Router, one session per call.

        The router reads the call's own ids, the same on LiveKit and Pipecat. Needs
        ``SLNG_API_KEY``.

        Args:
            call: What the call carries, ``state.call``.
        """
        router = RouterSettings()
        return RouterModelCall(
            call,
            model=router.model,
            base_url=router.router_url,
            api_key=router.api_key.get_secret_value(),
        )

    def agent(self, state: CallState) -> AgentSpec:
        """Return the call's agent: the prompt and tasks, on the router for this call.

        The same spec runs on LiveKit and Pipecat. It takes no name: the call's
        ``agent_id`` names it. Needs ``SLNG_API_KEY``.

        Args:
            state: The call's state, from ``new_call``. The router reads ``state.call``.
        """
        return AgentSpec(self.instructions, tasks=self.plan, model=self.model_call(state.call))


def build(spec: Mapping[str, Any] = SPEC, depot: Depot | None = None) -> Desk:
    """Turn a spec into the agent, or raise what refuses it, naming the row.

    Args:
        spec: The spec.
        depot: The tools' data. A fresh demo depot when None.
    """
    schema = StateSchema(spec["variables"], spec["shapes"])
    registry = tools(depot or Depot(), spec["tools"], schema.model)
    plan = TaskPlan(
        TasksSpec.model_validate(spec["tasks"]),
        state=schema.model,
        confirm=schema.confirm,
        tools=registry,
    )
    reads = prefetch(spec["prefetch"], schema.model)
    greeting = Greeting(**spec["greeting"])
    greeting.check(reads)
    return Desk(
        agent_id=spec["agent_id"],
        instructions=spec["instructions"],
        schema=schema,
        plan=plan,
        prefetch=reads,
        greeting=greeting,
        max_duration_s=spec["max_duration_s"],
        pace=spec["turn_pace"],
    )
