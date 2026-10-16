"""What both adapters do with an ``AgentSpec`` before the agent takes the call."""

from collections.abc import Iterable

from unmute_core.agent._spec import AgentSpec
from unmute_core.call import CallHooks
from unmute_core.model import variable_key_max
from unmute_core.model._prompt import check_keys
from unmute_core.prefetch import Lookup
from unmute_core.state import CallState
from unmute_core.tasks import CallTasks, TaskPlan, TasksSpec
from unmute_core.tools import ToolRegistry


def agent_state(state: CallState | None, call: CallHooks | None) -> CallState | None:
    """Return the call state an agent runs on: the one given, else the call's, else None."""
    if state is not None:
        return state
    return call.state if call is not None else None


def agent_name(spec: AgentSpec, state: CallState | None) -> str | None:
    """Return the agent's name: its own, else the call's ``agent_id``, else None."""
    if spec.name is not None:
        return spec.name
    return state.call.agent_id if state is not None else None


def agent_tasks(
    spec: AgentSpec, state: CallState | None, call: CallHooks | None, *, always: bool
) -> CallTasks | None:
    """Return the agent's task runs on this call.

    Args:
        spec: The agent.
        state: The call state, from ``agent_state``.
        call: The call's hooks. Steps run them and read the call-start results.
        always: Build runs for an agent without tasks too, holding its tools. Pipecat runs
            every agent on them. LiveKit leaves such an agent without runs.

    Returns:
        The runs, or None for an agent without tasks when ``always`` is False.
    """
    if spec.tasks is not None and state is not None:
        plan = spec.tasks
    elif not always:
        return None
    else:
        state = state if state is not None else CallState()
        plan = TaskPlan(TasksSpec(), state=state.model, tools=spec.tools or ToolRegistry([]))
    return CallTasks(
        plan,
        state,
        hooks=call,
        call_start=call.results if call is not None else None,
        language=spec.language,
        timezone=spec.timezone,
    )


def check_agent(
    spec: AgentSpec,
    state: CallState | None,
    *,
    call: CallHooks | None,
    then_targets: Iterable[str],
    models: Iterable[object],
) -> None:
    """Refuse an agent the call cannot run, the same way on every framework.

    Args:
        spec: The agent.
        state: The call state, from ``agent_state``.
        call: The call's hooks. With a prefetch, task inputs are checked against it.
        then_targets: The names of the agents a group may transfer to.
        models: The model calls the agent's requests go through. One that fills the prompt
            on its own side, like the router, limits how long a placeholder key may be.

    Raises:
        ValueError: If tasks or ``save_values`` come without a state, a group transfers to
            an agent ``then_targets`` lacks, a task input reads a tool no prefetch lookup
            runs, or a placeholder key is longer than a model call takes.
    """
    if spec.tasks is not None and state is None:
        raise ValueError("An agent with tasks needs state=CallState(...) or call=CallHooks(...).")
    if spec.save_values and state is None:
        raise ValueError(
            "An agent with save_values needs state=CallState(...) or call=CallHooks(...)."
        )
    if (limit := variable_key_max(*models)) is not None:
        check_keys(spec.instructions, "agent instructions", limit)
    plan = spec.tasks
    if plan is None:
        return
    plan.check_keys(limit)
    plan.check_then_targets(then_targets)
    if call is not None:
        plan.check_lookups(
            entry.name
            for entry in (call.prefetch.entries if call.prefetch is not None else ())
            if isinstance(entry, Lookup)
        )
