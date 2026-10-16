"""A tool that saves State values no task saves, and a prompt block with their current text.

The model calls the tool to record what it learns outside a task. The block lists every
value and its text, so the model knows what is set. A value a task saves stays that task's.
Unmute has no such tool: there, only a task's ``assign`` fills a value.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from unmute_core.state import CallState, StateOrigin, StateRefused
from unmute_core.tools._spec import ToolError, ToolSpec

SAVE_VALUES_TOOL = "save_values"
VALUES_HEADER = "Values of this call."
"""The block's first words, which mark its message in a chat."""


def is_values(role: object, text: object) -> bool:
    """Return True for the system message that lists the call's values."""
    return role == "system" and str(text or "").startswith(VALUES_HEADER)


class _SaveResult(BaseModel):
    """What the save tool answers: it dumps to ``{"ok": true, "saved": {...}}``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool = True
    saved: dict[str, Any]


class _SaveValues:
    """Saves the values the model sends. A null puts a value back to its default."""

    checks_extra_arguments = True

    def __init__(self, name: str, owners: Mapping[str, str]) -> None:
        """Keep the tool's name and the values tasks own."""
        self.name = name
        self.owners = dict(owners)

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        """Save the values as one batch, or refuse all of them.

        Raises:
            ToolError: If a value belongs to a task, is unknown, or does not fit.
        """
        if state is None:
            raise ToolError(f"{self.name} needs the call state")
        for name in arguments:
            if (task := self.owners.get(name)) is not None:
                raise ToolError(f"{name} is saved by the task {task}. Start that task to change it")
            if name in state.call_start:
                raise ToolError(f"{name} is set when the call starts. It cannot change")
            if name not in state.names:
                raise ToolError(
                    f"{name} is not a value of this call. The values are: "
                    + ", ".join(free_values(state, self.owners))
                )
        fields = state.model.model_fields
        by_alias = {info.alias or field: field for field, info in fields.items()}
        values = {
            name: fields[by_alias[name]].get_default(call_default_factory=True)
            if value is None
            else value
            for name, value in arguments.items()
        }
        try:
            state.save(values, origin=StateOrigin("tool", self.name))
        except StateRefused as refused:
            raise ToolError(f"Not saved: {refused.message}") from None
        return _SaveResult(saved={name: state.plain(name) for name in values}).model_dump()


def free_values(state: CallState, owners: Mapping[str, str] | None = None) -> list[str]:
    """Return the State values the main agent may save, in order.

    A value is free when no task saves it and the call did not start with it.

    Args:
        state: The call state.
        owners: Value name to the task that saves it, like ``TasksSpec.owners()``.
    """
    owners = owners or {}
    return [name for name in state.names if name not in owners and name not in state.call_start]


def values_tool(
    state: CallState,
    *,
    owners: Mapping[str, str] | None = None,
    name: str = SAVE_VALUES_TOOL,
) -> ToolSpec:
    """Build the tool the model calls to save State values that no task saves.

    Each value is an optional argument. A null puts it back to its default. A value a
    task saves is left out, and a call that names it is refused, naming the task. So is a
    value the call starts with, ``CallState(call_start=...)``.

    Args:
        state: The call state.
        owners: Value name to the task that saves it, like ``TasksSpec.owners()``.
        name: The tool name.

    Raises:
        ValueError: If every value belongs to a task or to the call's start, so there is
            nothing to save.
    """
    owners = dict(owners or {})
    free = free_values(state, owners)
    if not free:
        raise ValueError(
            "every State value is saved by a task or set when the call starts: there is "
            "nothing to save"
        )
    schema = state.model.model_json_schema(by_alias=True)
    properties = schema.get("properties", {})
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            value: _nullable(properties[value])
            | ({"description": text} if (text := properties[value].get("description")) else {})
            for value in free
        },
    }
    if "$defs" in schema:
        parameters["$defs"] = schema["$defs"]
    description = (
        "Save what you learn about the caller as soon as you learn it, before a tool that "
        "needs it. Send only the values you are saving. null clears one."
    )
    return ToolSpec(name, description, parameters, _SaveResult, _SaveValues(name, owners))


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a field's schema that also takes null, without its title, default and description."""
    kept = {k: v for k, v in schema.items() if k not in ("title", "default", "description")}
    if {"type": "null"} in kept.get("anyOf", []):
        return kept
    return {"anyOf": [kept, {"type": "null"}]}


def values_block(
    state: CallState,
    *,
    owners: Mapping[str, str] | None = None,
    tool: str = SAVE_VALUES_TOOL,
) -> str:
    """Return the prompt block that lists the call's values and their current text.

    Send it as the last system message of each main-agent request, built anew each time.
    ``AgentSpec(save_values=True)`` does on both frameworks. A value's
    braces are spaced out, so its text is never read as a placeholder.

    Args:
        state: The call state.
        owners: Value name to the task that saves it, like ``TasksSpec.owners()``.
        tool: The save tool's name.

    Returns:
        The block, one line per value. When a task saves or the call starts with every
        value, the header names no save tool, since the agent has none.
    """
    owners = owners or {}
    fields = {info.alias or field: info for field, info in state.model.model_fields.items()}

    def line(value: str) -> str:
        """Return one value's line: its label, and its text with braces spaced out."""
        about = fields[value].description
        task = owners.get(value)
        label = value + (f" ({about})" if about else "")
        if task is not None:
            label += f", saved by the task {task}"
        elif value in state.call_start:
            label += ", set when the call starts"
        text = state.render(value).replace("{{", "{ {").replace("}}", "} }")
        return f"- {label}: {text}"

    header = (
        f"{VALUES_HEADER} Save new ones with {tool}."
        if free_values(state, owners)
        else VALUES_HEADER
    )
    return "\n".join([header, *(line(value) for value in state.names)])
