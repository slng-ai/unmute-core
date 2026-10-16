"""The ``tool`` operations of one Pipecat call, from a tool's start to its end."""

import re
from collections.abc import Collection

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.dev import DevRecorder

_NS = 1_000_000_000


def error_kind(error: object) -> str:
    """Return the exception type an error text starts with, and never the message after it.

    A provider's error text can quote a prompt or a tool's arguments. The type name is an
    identifier from the code, so it is the only part that goes in a record.
    """
    head = str(error or "").split(":", 1)[0].strip()
    return head if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", head) else "an error"


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class _ToolRun:
    """A running tool call: its operation, its start in frame time, and whether it hands over."""

    operation: str
    timestamp: int
    control: bool


class ToolRuns:
    """Owns the tool calls that run now and writes their ``tool`` operations."""

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
        self._handoffs = frozenset(handoffs)
        self._controls = frozenset(controls)
        self._running: dict[str, _ToolRun] = {}

    @property
    def running(self) -> bool:
        """Whether a tool call is running."""
        return bool(self._running)

    def start(
        self,
        frame: FunctionCallInProgressFrame,
        timestamp: int,
        *,
        exchange_id: str | None,
        parent: str | None,
    ) -> None:
        """Open a ``tool`` operation under the model request that asked for it.

        Args:
            frame: The frame that says the call is in progress.
            timestamp: When it was pushed, in frame time.
            exchange_id: The exchange the call belongs to.
            parent: The operation of the model request that chose the call.
        """
        if frame.function_name in self._handoffs or frame.tool_call_id in self._running:
            return
        control = frame.function_name in self._controls
        operation = self._rec.tool_started(
            frame.function_name, exchange_id=exchange_id, parent=parent, control=control
        )
        self._running[frame.tool_call_id] = _ToolRun(operation, timestamp, control)

    def result(self, frame: FunctionCallResultFrame, timestamp: int) -> None:
        """End a call with its result, unless the result only says a task is running."""
        if frame.properties is not None and not frame.properties.is_final:
            return
        if frame.error:
            reason = f"The handler raised {error_kind(frame.error)}."
            self.end(frame.tool_call_id, "failed", timestamp, reason)
        else:
            self.end(frame.tool_call_id, "returned", timestamp)

    def cancelled(self, frame: FunctionCallCancelFrame, timestamp: int) -> None:
        """End a call that was cancelled or ran past its deadline."""
        self.end(
            frame.tool_call_id,
            "timed_out" if frame.run_llm else "cancelled",
            timestamp,
            "It ran past its deadline and was cancelled." if frame.run_llm else None,
        )

    def cancel_all(self, timestamp: int) -> None:
        """Cancel every call that runs now."""
        for call_id in list(self._running):
            self.end(call_id, "cancelled", timestamp)

    def end(self, call_id: str, state: str, timestamp: int, reason: str | None = None) -> None:
        """End a call's operation and time it."""
        if (tool := self._running.pop(call_id, None)) is None:
            return
        self._rec.tool_ended(
            tool.operation,
            state,
            duration=(timestamp - tool.timestamp) / _NS,
            source="pipecat-function-call-frames",
            control=tool.control,
            reason=reason,
        )
