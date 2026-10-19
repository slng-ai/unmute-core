"""A tool that runs on events, never chosen by the model."""

from collections.abc import Mapping, Sequence
from dataclasses import field
from typing import Annotated, Self

from pydantic import AfterValidator, ValidationInfo, model_validator
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.tools import ToolSpec

ROOT_EVENTS = ("first_user_message", "call_end")
"""Events that happen once a call."""
OUTCOMES = ("tool_succeeded", "tool_failed")
"""Events that follow one named tool: ``tool_succeeded:<tool>`` or ``tool_failed:<tool>``."""


def _events(events: Sequence[str], info: ValidationInfo) -> tuple[str, ...]:
    """Keep the events as a tuple, and refuse one that is no event.

    Raises:
        ValueError: If there is none, or one is unknown or a call-start event.
    """
    name = info.data["tool"].name if "tool" in info.data else "?"
    kept = tuple(events)
    if not kept:
        raise ValueError(f"event tool {name} needs at least one event in on")
    for event in kept:
        kind, _, source = event.partition(":")
        if kind == "call_start":
            raise ValueError(
                f"event tool {name}: work at call start is a prefetch. Run it as "
                "Lookup(..., background=True) in CallHooks(prefetch=...)"
            )
        if not (kind in ROOT_EVENTS and not source) and not (kind in OUTCOMES and source):
            raise ValueError(
                f"event tool {name}: {event!r} is no event. Write one of "
                f"{', '.join(ROOT_EVENTS)}, tool_succeeded:<tool> or tool_failed:<tool>"
            )
    return kept


@original_errors
@dataclass(frozen=True, config=VALUE)
class EventTool:
    """A tool that runs on events, never chosen by the model.

    Every argument the tool needs is locked: a constant, ``{{value}}``, ``{{call.<field>}}``
    or a ``Transcript``. ``{{call.first_user_message}}`` is what the caller said first,
    and at the end ``{{call.end_reason}}`` says why the call ended. A
    webhook's envelope also carries the event as ``trigger_event``. A tool at call start is
    a prefetch ``Lookup`` instead.

    Attributes:
        tool: The tool.
        on: Its events: ``first_user_message``, ``call_end``, ``tool_succeeded:<tool>`` or
            ``tool_failed:<tool>``. Kept as a tuple.
        assign: State value name to the top-level result key it gets, saved when the tool
            succeeds.
        note: True also shows the result to the model as a note. A failure is noted too.
        wait: True (the default) holds the next model request until the run ends, for up to
            ``first_request_timeout_s``. False starts the run on its event and never holds a
            request, so a tool that only tells another system something can be slow. The
            call's end still waits for it, for up to ``end_timeout_s``. The tools that follow
            it keep their own ``wait``. With ``note=True``, a note from a run nobody waited
            for can only reach a request that starts after the run ended.
    """

    tool: ToolSpec
    on: Annotated[Sequence[str], AfterValidator(_events)]
    assign: Mapping[str, str] = field(default_factory=dict)
    note: bool = False
    wait: bool = True

    @model_validator(mode="after")
    def _runs_alone(self) -> Self:
        """Refuse a tool someone would wait for, or one with an argument nothing fills.

        Raises:
            ValueError: If the tool has an announce line, or an argument with no locked value.
        """
        name = self.tool.name
        if self.tool.announce:
            raise ValueError(f"event tool {name}: nobody waits for it, so drop the announce line")
        if unset := self.tool.unlocked_required:
            raise ValueError(f"event tool {name}: argument {unset[0]} has no source. Lock it")
        return self
