"""What a ``CallState`` tells a listener after a save: what changed, and who changed it."""

from collections.abc import Callable
from typing import Literal

from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors

OriginKind = Literal[
    "step", "prefetch", "call", "call_start", "tool", "event", "withdraw", "start", "reset"
]
"""What can make a change to the state."""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class StateOrigin:
    """What made a change to the state.

    Attributes:
        kind: ``step`` (a task step), ``prefetch`` (a prefetch entry), ``call`` (the
            library, saving what the call carries), ``call_start`` (the values the call was dispatched with),
            ``tool`` (the save tool, or a tool result a registry saves), ``event`` (a tool
            on an event), ``withdraw`` (a step took its confirmation back), ``start`` (the values already
            there when a listener asked for them), or ``reset`` (a value went back to its
            default because something it was looked up from changed).
        name: The step or entry. Empty when there is none.
    """

    kind: OriginKind
    name: str = ""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class StateChange:
    """One value as it stands after a save.

    Attributes:
        field: The value's name.
        value: The value as JSON data.
        confirmed: Whether the caller has agreed to it. None for a value that never waits
            for the caller.
        by: What changed it.
    """

    field: str
    value: object
    confirmed: bool | None
    by: StateOrigin


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class StateEvent:
    """Everything one save changed, in the order the values changed.

    Attributes:
        origin: What made the save.
        changes: The values it changed, set or reset or flipped between confirmed and not.
    """

    origin: StateOrigin
    changes: tuple[StateChange, ...]


StateListener = Callable[[StateEvent], None]
"""Receives each ``StateEvent``. Called once per save that changed something."""
