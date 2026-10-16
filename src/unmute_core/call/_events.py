"""Tools on events: run them in order, save their results, leave notes for the model."""

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType

from unmute_core.hooks import EventTool
from unmute_core.state import CallState, StateOrigin, StateRefused
from unmute_core.state._call_info import CALL
from unmute_core.tasks._texts import literal
from unmute_core.tools import ToolError, ToolRegistry, ToolSpec
from unmute_core.tools._spec import ToolHooks, tool_succeeded

logger = logging.getLogger(__name__)

NOTE_MAX_CHARS = 15_000
"""The longest note a tool on an event adds to the chat."""
NOTES_HEADER = (
    "Results of tools that ran on their own in this call. "
    "Use them, but never read them aloud or share the raw data:"
)
"""The first line of the system message that holds the notes."""

_WHEN = MappingProxyType(
    {
        "first_user_message": "the caller first spoke",
        "tool_succeeded": "{tool} succeeded",
        "tool_failed": "{tool} failed",
    }
)
_OUTCOME_OF_EVENT = MappingProxyType(
    {"tool.succeeded": "tool_succeeded", "tool.failed": "tool_failed"}
)


class EventToolRunner:
    """Run one call's tools on events, in the background, and keep what they leave behind.

    It owns the events already fired, the background runs a model request waits for, and
    the notes. It saves the caller's first message and why the call ended into
    ``state.call``.
    """

    def __init__(
        self,
        events: Sequence[EventTool],
        *,
        tools: ToolRegistry,
        state: CallState,
        transcript: Callable[[], list[dict[str, object]]],
        tool_hooks: ToolHooks,
    ) -> None:
        """Index the tools by the events they run on.

        Args:
            events: The tools on events.
            tools: A registry holding exactly those tools.
            state: The call's state.
            transcript: Returns the chat so far.
            tool_hooks: What each tool call passes as its ``hooks``: the call's ``CallHooks``.
        """
        self._on_event: dict[str, list[EventTool]] = defaultdict(list)
        for spec in events:
            for event in dict.fromkeys(spec.on):
                self._on_event[event].append(spec)
        self._tools = tools
        self._state = state
        self._transcript = transcript
        self._tool_hooks = tool_hooks
        self._fired: set[str] = set()
        self._notes: list[str] = []
        self._heard_caller = False
        self.pending: set[asyncio.Future[None]] = set()
        """The runs still going. A model request and the call's end wait for them."""

    @property
    def notes(self) -> tuple[str, ...]:
        """The notes the tools left so far, oldest first."""
        return tuple(self._notes)

    def notes_block(self) -> str | None:
        """The system message a main-agent request ends with, or None without notes."""
        if not self._notes:
            return None
        return "\n".join([NOTES_HEADER, *(f"- {note}" for note in self._notes)])

    def heard_caller(self, text: str) -> bool:
        """Save the caller's first message as ``call.first_user_message``.

        Returns:
            True the first time, False once the caller has spoken before.
        """
        if self._heard_caller:
            return False
        self._heard_caller = True
        call = self._state.call.model_copy(update={"first_user_message": text})
        self._state.save({CALL: call}, origin=StateOrigin("call", "first_user_message"))
        return True

    def began_end(self, reason: str) -> None:
        """Save why the call ends as ``call.end_reason``, for the tools that run at its end."""
        call = self._state.call.model_copy(update={"end_reason": reason})
        self._state.save({CALL: call}, origin=StateOrigin("call", "end"))

    def on_tool_event(self, event: str, data: object) -> None:
        """Start the tools that follow a tool, when a model-called tool succeeds or fails."""
        outcome = _OUTCOME_OF_EVENT.get(event)
        if outcome is not None and isinstance(data, Mapping) and data.get("source") == "model":
            self.spawn(self.follow(f"{outcome}:{data['name']}"))

    def spawn(self, work: Awaitable[None]) -> asyncio.Future[None]:
        """Run tools in the background and track them in ``pending``."""
        task = asyncio.ensure_future(work)
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)
        return task

    async def fire(self, event: str) -> None:
        """Run an event's tools once a call, in order."""
        if event in self._fired:
            return
        self._fired.add(event)
        await self.follow(event)

    async def follow(self, event: str) -> None:
        """Run the tools of one event, in order, and the tools that follow each."""
        for spec in self._on_event.get(event, ()):
            await self._run_tool(spec, event)

    async def _run_tool(self, spec: EventTool, event: str) -> None:
        """Run one tool on an event, save and note its result, and run the tools after it."""
        tool = spec.tool
        kind, _, source = event.partition(":")
        result = await self._call(spec, event, kind)
        succeeded = tool_succeeded(result)
        if succeeded and spec.assign and isinstance(result, Mapping):
            self._save(spec, result)
        shown = self._tools.for_model(tool.name, result)
        if spec.note and kind != "call_end" and (shown is result or not succeeded):
            self._note(tool, kind, source, result, succeeded=succeeded)
        if isinstance(result, dict) and result.get("refused"):
            return
        await self.follow(f"{'tool_succeeded' if succeeded else 'tool_failed'}:{tool.name}")

    async def _call(self, spec: EventTool, event: str, kind: str) -> object:
        """Call the tool with the event and the transcript. A failure becomes an ``error`` result."""
        name = spec.tool.name
        context = {"trigger_event": kind, "transcript": self._transcript()}
        try:
            return await self._tools.call(
                name, {}, self._state, context=context, hooks=self._tool_hooks, source="event"
            )
        except ToolError as error:
            return {"error": error.message}
        except Exception as error:
            logger.exception("tool %s on %s failed", name, event)
            return {"error": f"{name} failed: {error}"}

    def _note(
        self, tool: ToolSpec, kind: str, source: str, result: object, *, succeeded: bool
    ) -> None:
        """Add the note a tool with ``note=True`` leaves for the model.

        A tool that succeeded puts its ``result_instructions`` before the result, so the
        length cut never drops them.
        """
        when = _WHEN[kind].format(tool=source)
        text = json.dumps(result, ensure_ascii=False, default=str)
        use = tool.result_instructions.strip() if succeeded else ""
        how = f" Use the result like this: {use}" if use else ""
        note = literal(f"{tool.name} ran on its own when {when}.{how} Result: {text}")
        self._notes.append(note[:NOTE_MAX_CHARS])

    def _save(self, spec: EventTool, result: Mapping[str, object]) -> None:
        """Save the assigned fields of a result. A value that does not fit is only logged."""
        values = {name: result.get(key) for name, key in spec.assign.items()}
        try:
            self._state.save(values, origin=StateOrigin("event", spec.tool.name))
        except StateRefused as refused:
            logger.warning("event tool %s saved nothing: %s", spec.tool.name, refused.message)
