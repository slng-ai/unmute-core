"""The ``tool`` operations of one LiveKit call, from a tool's start to its end."""

from collections.abc import Callable, Collection, Iterable
from types import MappingProxyType

from livekit.agents.voice.events import ToolCallEnded, ToolCallStarted
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.dev import DevRecorder

_TOOL_ENDS = MappingProxyType({"done": "returned", "error": "failed", "cancelled": "cancelled"})


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class _ToolRun:
    """A tool call that runs now: its operation, when it started and whether it hands over.

    Attributes:
        operation: The id of the call's operation.
        started: When LiveKit reported the call started, in seconds.
        control: The call is a human transfer, which gets no duration.
    """

    operation: str
    started: float
    control: bool


class ToolRuns:
    """Owns the tool calls that run now, and which tools draw as a handoff or a transfer."""

    def __init__(
        self, recorder: DevRecorder, *, handoffs: Collection[str], controls: Collection[str]
    ) -> None:
        """Keep the recorder and which tools draw as something else.

        Args:
            recorder: The call's recorder.
            handoffs: Tools that start a task or a group. They get no ``tool`` operation.
            controls: Tools that hand the call over without a result. They get no duration.
        """
        self._rec = recorder
        self._handoffs = set(handoffs)
        self._fixed_controls = frozenset(controls)
        self._controls = set(controls)
        self._delegates: set[str] = set()
        self._running: dict[str, _ToolRun] = {}
        self._chosen: dict[str, str] = {}

    @property
    def running(self) -> bool:
        """Whether a tool call is running."""
        return bool(self._running)

    @property
    def delegating(self) -> bool:
        """Whether a task or group handoff tool is running."""
        return bool(self._delegates)

    @property
    def controls(self) -> set[str]:
        """The tools that draw as a handoff now."""
        return self._controls

    def add_handoffs(self, names: Iterable[str]) -> None:
        """Draw these tools as a task or group handoff from now on."""
        self._handoffs |= set(names)

    def set_controls(self, agent_handoffs: Iterable[str]) -> None:
        """Draw the fixed controls and this agent's handoff tools as handoffs, and no others."""
        self._controls = set(self._fixed_controls) | set(agent_handoffs)

    def chose(self, operation: str, call_ids: Iterable[str]) -> None:
        """Note the tool calls a model request chose, so each tool names its request."""
        self._chosen.update(dict.fromkeys(call_ids, operation))

    def started(
        self,
        update: ToolCallStarted,
        created_at: float,
        *,
        target: Callable[[], str],
        last_request: str | None,
    ) -> None:
        """Open a ``tool`` operation as a tool starts.

        Args:
            update: The ``tool_call_started`` update.
            created_at: When LiveKit reported it, in seconds.
            target: Returns the response the call belongs to. It opens one when none is open,
                so it runs only for a call that gets an operation.
            last_request: The newest ``llm`` operation, the parent when no request named the call.
        """
        call = update.function_call
        if call.name in self._handoffs:
            self._delegates.add(call.call_id)
            return
        if call.call_id in self._running:
            return
        parent = self._chosen.pop(call.call_id, None) or last_request
        control = call.name in self._controls
        operation = self._rec.tool_started(
            call.name, exchange_id=target(), parent=parent, control=control
        )
        self._running[call.call_id] = _ToolRun(operation, created_at, control)

    def ended(
        self, update: ToolCallEnded, created_at: float, *, closing: Callable[[], bool]
    ) -> None:
        """End and time a tool's operation as the tool ends.

        Args:
            update: The ``tool_call_ended`` update.
            created_at: When LiveKit reported it, in seconds.
            closing: Says whether the session is closing, which turns a failure into a cancel.
        """
        self._delegates.discard(update.call_id)
        if (tool := self._running.pop(update.call_id, None)) is None:
            return
        state = _TOOL_ENDS[update.status]
        if state == "failed" and closing():
            state = "cancelled"
        self._rec.tool_ended(
            tool.operation,
            state,
            duration=created_at - tool.started,
            source="livekit-tool-events",
            control=tool.control,
            reason="The tool raised an error." if state == "failed" else None,
        )
