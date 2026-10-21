"""The timeline of one simulated call, built from the library's own hook points.

Nothing here knows the framework. The recorder watches the call through ``Hooks`` and the
``CallState``, the same on LiveKit and Pipecat:

* ``task.*`` and ``group.*`` events: names, run ids and saved names.
* every save, with what made it: a step (its ``writer``), the main agent's save tool or a
  tool result (origin ``tool``), the prefetch, an event tool.
* every request the agent's model answered (``wrap_model``): its site, its latency, what it
  said and which tools it called. A rate limit before the first chunk is retried here: it
  is our key's limit, not the agent's fault. A 5xx, a timeout or a lost link that remains
  is an ``environment`` error, which leaves the call out of the pass rate.
* every tool call: a registry tool through ``wrap_tool``, with its arguments and result. A
  tool the adapter owns, like a task's delegate tool or a step's ``finish``, from the
  model's call and the result a later request carries. A delegate's call is kept where its
  result is read, after the task events of its run; one never read has no output.

The call loop adds what each side said and who hung up.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from unmute_core.hooks import Hooks
from unmute_core.model import ModelEvent, ModelRequest, TextDelta, ToolCall
from unmute_core.state import CallState, StateChange, StateEvent
from unmute_core.tasks import FINISH_TOOL_NAME, NOT_YET_TOOL_NAME
from unmute_core.testing.simulate._endpoints import is_rate_limit, is_upstream
from unmute_core.tools import ToolRun

RUNTIME_TOOLS = (FINISH_TOOL_NAME, NOT_YET_TOOL_NAME)
"""Tools the runtime offers a step, which return nothing of the world."""
MODEL_RETRIES = 4
"""How often a request the agent's endpoint rate-limited before answering is sent again."""
ENVIRONMENT = "environment"
"""The error kinds that start with this are the endpoint's fault, not the agent's."""
HARNESS = "harness"
"""The error kinds that start with this are the simulator's own failures."""


def as_data(value: object) -> object:
    """Return a value as JSON data: a model's dump, JSON data as it is, anything else as text."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def parse_json(value: object) -> object:
    """Return JSON text parsed, or the value as it is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


class Recorder:
    """The events of one call, in order, with a time in seconds since the call began.

    Attributes:
        events: Each event: ``t``, ``kind`` and its data.
        errors: Each error: its ``kind`` and data. A kind that starts with ``environment``
            or ``harness`` is not the agent's.
        open_requests: Model requests sent and not yet answered.
        last_activity: When anything last happened, on the monotonic clock.
    """

    def __init__(self, registry_tools: frozenset[str]) -> None:
        """Start an empty timeline.

        Args:
            registry_tools: The tools ``wrap_tool`` records. Every other tool call is
                rebuilt from the model's requests, with the result the model read: a via
                tool's is its task's run record.
        """
        self._start = time.monotonic()
        self._registry = registry_tools
        self._awaiting: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.open_requests = 0
        self.last_activity = self._start

    def touch(self) -> None:
        """Note that something happened now."""
        self.last_activity = time.monotonic()

    def elapsed(self) -> float:
        """Return the seconds since the call began."""
        return round(time.monotonic() - self._start, 3)

    def add(self, kind: str, **data: Any) -> dict[str, Any]:
        """Add one event, and return it."""
        event = {"t": self.elapsed(), "kind": kind, **data}
        self.events.append(event)
        self.touch()
        return event

    def error(self, kind: str, **data: Any) -> None:
        """Add one error to the call and to the timeline."""
        self.errors.append({"kind": kind, **data})
        self.add("error", error_kind=kind, **data)

    def attach(self, hooks: Hooks) -> None:
        """Watch the call's task events, failures, model requests and tool calls."""
        hooks.on("task.*", "group.*")(self._task_event)
        hooks.on("call.failed")(self._failed)
        hooks.wrap_model(self._wrap_model)
        hooks.wrap_tool(self._wrap_tool)

    def watch(self, state: CallState) -> None:
        """Watch every save, and mark where the saves held for the call's end begin."""
        state.listen(self._saved)
        settle = state.settle_held

        async def settle_marked(**options: Any) -> list[str]:
            """Mark the call's end in the timeline once, then make the held saves."""
            if not any(event["kind"] == "call_end_saves" for event in self.events):
                self.add("call_end_saves")
            return await settle(**options)

        state.settle_held = settle_marked  # ty: ignore[invalid-assignment]

    def _task_event(self, stage: str, data: Any) -> None:
        """Keep one task or group event."""
        self.add("task_event", stage=stage, data=dict(data or {}))

    def _failed(self, event: str, data: Any) -> None:
        """Keep a failure of the call: the agent's, unless the endpoint already failed it."""
        data = dict(data or {})
        self.error(
            "call_failed", component=str(data.get("component")), error=str(data.get("error"))
        )

    def _saved(self, event: StateEvent) -> None:
        """Keep one save, split by what made each change."""
        by_origin: dict[tuple[str, str], list[StateChange]] = {}
        for change in event.changes:
            by_origin.setdefault((change.by.kind, change.by.name), []).append(change)
        for (kind, name), changes in by_origin.items():
            if kind == "start":
                continue
            if kind == "withdraw":
                self.add("state_withdraw", task=name, fields=[c.field for c in changes])
                continue
            self.add(
                "state_write",
                applied={c.field: c.value for c in changes},
                confirmed={c.field: c.confirmed for c in changes},
                writer=name if kind == "step" else None,
                origin=kind,
            )

    async def _wrap_tool(
        self, run: ToolRun, handler: Callable[[ToolRun], Awaitable[object]]
    ) -> object:
        """Keep one registry tool call: its arguments and its result, or its error."""
        if run.name not in self._registry:
            return await handler(run)
        try:
            result = await handler(run)
        except Exception as error:
            self.add("tool_call", name=run.name, arguments=dict(run.arguments), error=repr(error))
            raise
        self.add(
            "tool_call",
            name=run.name,
            arguments=dict(run.arguments),
            output=as_data(result),
            source=run.source,
        )
        return result

    def _results(self, request: ModelRequest) -> None:
        """Keep the result of each awaited tool call this request carries, where it is read.

        A step's ``finish`` or ``not_yet`` is in the timeline where the model called it;
        its result, a refusal, is filled in there. Any other tool goes in where its result
        is read, so a delegate's run record comes after the saves it reports.
        """
        for message in request.messages:
            call_id = message.get("tool_call_id")
            if message.get("role") != "tool" or call_id not in self._awaiting:
                continue
            event = self._awaiting.pop(call_id)
            event["output"] = parse_json(message.get("content"))
            if event["name"] not in RUNTIME_TOOLS:
                self.add("tool_call", **event)

    def _model_tool(self, call: ToolCall, site: str) -> None:
        """Keep a tool the adapter owns, and wait for its result in a later request."""
        if call.name in self._registry:
            return
        event = {
            "name": call.name,
            "call_id": call.id,
            "arguments": parse_json(call.arguments),
            "output": None,
            "site": site or "main",
        }
        if call.name in RUNTIME_TOOLS:
            event = self.add("tool_call", **event)
        self._awaiting[call.id] = event

    def unread(self) -> None:
        """Keep the tool calls whose result no model read, as the call ends, with no output."""
        for event in self._awaiting.values():
            if event["name"] not in RUNTIME_TOOLS:
                self.add("tool_call", **event)
        self._awaiting.clear()

    async def _wrap_model(
        self, request: ModelRequest, handler: Callable[[ModelRequest], AsyncIterator[ModelEvent]]
    ) -> AsyncIterator[ModelEvent]:
        """Keep one model request, and retry a rate limit that came before any answer."""
        self._results(request)
        self.open_requests += 1
        begin = time.monotonic()
        text: list[str] = []
        calls: list[str] = []
        try:
            for attempt in range(MODEL_RETRIES + 1):
                answered = False
                try:
                    async for event in handler(request):
                        answered = True
                        self.touch()
                        if isinstance(event, TextDelta):
                            text.append(event.text)
                        elif isinstance(event, ToolCall):
                            calls.append(event.name)
                            self._model_tool(event, request.site)
                        yield event
                    break
                except Exception as error:
                    if not answered and is_rate_limit(error) and attempt < MODEL_RETRIES:
                        self.add("llm_retry", site=request.site or "main", error=repr(error)[:200])
                        await asyncio.sleep(min(30, 2 * 2**attempt))
                        continue
                    kind = f"{ENVIRONMENT}_upstream_error" if is_upstream(error) else "model_error"
                    self.error(kind, site=request.site or "main", error=repr(error)[:300])
                    raise
        finally:
            self.open_requests -= 1
            self.add(
                "llm",
                site=request.site or "main",
                latency_ms=round((time.monotonic() - begin) * 1000),
                text="".join(text)[:300],
                tool_calls=calls,
                tool_choice=request.tool_choice,
            )

    def variables(self, state: CallState) -> dict[str, dict[str, Any]]:
        """Return each State value as the call left it: stored, confirmed, and its confirmer."""
        stored = state.dump()
        return {
            name: {
                "stored": value,
                "confirmed": state.is_confirmed(name) if name in state.confirm else None,
                "confirming_task": state.confirm.get(name),
            }
            for name, value in stored.items()
        }

    def transcript(self) -> list[dict[str, str]]:
        """Return what each side said, in order."""
        roles = {"caller_said": "caller", "agent_said": "agent"}
        return [
            {"role": roles[event["kind"]], "text": event["text"]}
            for event in self.events
            if event["kind"] in roles
        ]

    def not_agents(self) -> list[dict[str, Any]]:
        """Return the errors that are not the agent's: the endpoint's or the simulator's."""
        return [error for error in self.errors if error["kind"].startswith((ENVIRONMENT, HARNESS))]
