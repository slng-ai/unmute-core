"""The records of one call, in the dev format ``unmute dev`` reads.

The format is the contract of the Unmute CLI's ``internal/devmetrics/record.go``: every
record has ``version: 2``, the ``call_id``, an ``id``, a ``revision``, an ``order``, its
``kind``, and one payload named after the kind. A record is a snapshot of one thing, like an
exchange or an operation. Sending it again with a higher ``revision`` replaces it, and its
``order`` stays, so the page keeps it where it was. Behaviour follows the Unmute template
``pipecat_v1/dev_metrics.py.tmpl`` at unmute@a005bd2.

All times are seconds. A time that was not measured is absent, never zero.
"""

import contextlib
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BeforeValidator, Field, Strict, TypeAdapter, ValidationError
from pydantic_settings import SettingsConfigDict

from unmute_core.dev._records import (
    REQUEST_METRICS,
    Breakdown,
    Measurement,
    Operation,
    OperationKind,
    ProviderFirstOperation,
    Scope,
    TextRow,
    is_seconds,
)
from unmute_core.dev._store import DevSink, RecordStore
from unmute_core.settings import FLAG_WORDS, EnvSettings, SettingsError
from unmute_core.state import CallState, StateChange, StateEvent, StateOrigin

logger = logging.getLogger(__name__)

SENTINEL = "UNMUTE_METRIC"
"""The word that starts every line ``unmute dev`` reads."""
METRICS_ENV = "UNMUTE_DEV_METRICS"
"""The variable ``unmute dev`` sets for a bot it runs. Anything but ``""`` or ``"0"`` is on."""
CALL_ID_KEY = "unmute_dev_call_id"
"""The key the call's runner body carries the dev call id under."""
VALUE_MAX_BYTES = 4096
"""The longest state value a record carries. A longer one is cut, with ``truncated`` set."""

_CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_ENDED = frozenset({"ended", "returned"})
_RUN_ENDS = MappingProxyType(
    {
        "group.completed": "returned",
        "group.stopped": "cancelled",
        "task.completed": "returned",
        "task.unserved": "returned",
        "task.handed_off": "ended",
        "task.failed": "failed",
    }
)
"""How a task event ends the handoff of its run. The ones of a step in a group end nothing."""
_ORIGIN_KINDS = MappingProxyType({"tool": "step", "event": "prefetch", "call": "fact"})
"""State origins the reader has no word for, as the nearest one it has. The save tool names
itself like a step; a tool on an event is a lookup, like a prefetch; the library saving what
the call carries is the reader's ``fact``."""


def _valid_call_id(value: str) -> str:
    """Return the call id, or raise naming the rule the reader holds it to."""
    if not _CALL_ID.fullmatch(value):
        raise ValueError(
            f"dev call id {value!r} is not valid. Use 1 to 128 letters, digits, - or _"
        )
    return value


CallId = Annotated[str, Strict(), AfterValidator(_valid_call_id)]
"""A dev call id: 1 to 128 letters, digits, ``-`` or ``_``, which is what the reader takes."""
_CALL_ID_CHECK = TypeAdapter(CallId)


def _flag_text(value: object) -> object:
    """Strip the spaces around an on/off word. Spaces alone count as unset, so off."""
    return (value.strip() or False) if isinstance(value, str) else value


class DevSettings(EnvSettings):
    """What the dev loop sets in the environment of a bot it runs.

    Attributes:
        metrics: ``UNMUTE_DEV_METRICS``. True for 1, true, yes, y, on or t, false for 0,
            false, no, n, off or f, in any case.
    """

    model_config = SettingsConfigDict(env_prefix="UNMUTE_DEV_")

    metrics: Annotated[bool, BeforeValidator(_flag_text)] = Field(
        default=False, description="Send dev records on stdout. Set by unmute dev."
    )


def metrics_enabled() -> bool:
    """Return True when the dev loop asked for dev records through ``UNMUTE_DEV_METRICS``.

    A value that is not an on/off word logs a warning and counts as off, so a typo in this
    dev-only variable never breaks a call.
    """
    try:
        return DevSettings().metrics
    except SettingsError as error:
        refused = (
            error.__cause__.errors()[0]["input"]
            if isinstance(error.__cause__, ValidationError)
            else ""
        )
        logger.warning(
            "%s must be one of %s, not %r. Dev records stay off.", METRICS_ENV, FLAG_WORDS, refused
        )
        return False


def call_id_from(body: object) -> str | None:
    """Read the dev call id from a runner's body, or None when it has none that is valid.

    Args:
        body: The ``body`` of Pipecat's runner arguments, a dict when the dev loop made the call.

    Returns:
        The id, which has 1 to 128 letters, digits, ``-`` or ``_``.
    """
    value = body.get(CALL_ID_KEY) if isinstance(body, Mapping) else None
    try:
        return _CALL_ID_CHECK.validate_python(value)
    except ValidationError:
        return None


def stdout_sink(record: dict[str, Any]) -> None:
    """Print one record as a ``UNMUTE_METRIC`` line. A closed output is ignored.

    Args:
        record: The record to print.
    """
    with contextlib.suppress(OSError, ValueError):
        print(f"{SENTINEL} {json.dumps(record, separators=(',', ':'))}", flush=True)


def separator(previous: str, text: str, *, spacing: bool = False) -> str:
    """Return what goes between two chunks of text: nothing, or one space.

    Args:
        previous: The text before, empty when there is none.
        text: The chunk being added.
        spacing: True when the chunk carries its own spaces, as most LLM and transcript text
            does. Then nothing goes between.

    Returns:
        ``""`` or ``" "``.
    """
    if not previous or not text or spacing:
        return ""
    return "" if previous[-1].isspace() or text[0].isspace() else " "


def known_name(value: str | None) -> str | None:
    """Return a model or provider name, or None when the framework does not know it.

    Args:
        value: The name a framework reports. ``None``, ``""`` and ``"unknown"`` are not names.
    """
    return value if value and value != "unknown" else None


def _origin(origin: StateOrigin) -> dict[str, str]:
    """Return what made a state change, in the words the reader takes."""
    return {"kind": _ORIGIN_KINDS.get(origin.kind, origin.kind), "name": origin.name}


def _bounded(value: Any) -> tuple[Any, bool]:
    """Return a state value as it goes in a record, and whether it was cut.

    A value whose JSON is over 4 KiB becomes the first 4 KiB of that JSON, as a string.
    """
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    raw = text.encode()
    if len(raw) <= VALUE_MAX_BYTES:
        return value, False
    return raw[:VALUE_MAX_BYTES].decode(errors="ignore"), True


class DevRecorder:
    """Builds the dev records of one call and sends each to a sink.

    Make one per call. It sends the ``call`` record as it is made. Hook it up: the Pipecat
    or LiveKit ``DevObserver`` for the call, ``watch_state`` for the call's state and
    ``task_event`` for its tasks. Call ``end_call`` when the call is over, or let the
    observer do it.

    A sink that raises is logged and the record is lost. A record never fails the call.

    Attributes:
        call_id: The call's id.
        response: The response exchange that is open now, which a task's handoff belongs to.
            The observer keeps it.
    """

    def __init__(
        self,
        call_id: str,
        sink: DevSink = stdout_sink,
        *,
        target: Literal["pipecat", "livekit"] = "pipecat",
        model_calls: Literal["known", "partial"] = "known",
    ) -> None:
        """Start a call's records and send its ``call`` record, open.

        Args:
            call_id: The call's id: 1 to 128 letters, digits, ``-`` or ``_``.
            sink: Receives each record. The default prints ``UNMUTE_METRIC`` lines.
            target: The framework the call runs on.
            model_calls: ``partial`` when some model calls are not seen.

        Raises:
            ValueError: If ``call_id`` is not a valid id.
        """
        try:
            _CALL_ID_CHECK.validate_python(call_id)
        except ValidationError as error:
            raise (error.errors()[0].get("ctx", {}).get("error") or error) from None
        self.call_id = call_id
        self.response: str | None = None
        self._store = RecordStore(call_id, sink)
        self._runs: dict[str, str] = {}
        self._segments: dict[str, list[str]] = {}
        self._open_segment: dict[str, str] = {}
        self._messages: dict[str, str] = {}
        self._answers: dict[str, str] = {}
        self._error = False
        self.put(
            "call",
            "call",
            {
                "target": target,
                "state": "open",
                "input_boundaries": "partial",
                "model_calls": model_calls,
                "generated_text": "available",
            },
        )

    @property
    def closed(self) -> bool:
        """True once ``end_call`` ran. Nothing is sent after it."""
        return self._store.closed

    def new_id(self, prefix: str) -> str:
        """Return the next id for a kind of thing, like ``llm-3``. The count is the call's."""
        return self._store.new_id(prefix)

    def get(self, kind: str, identity: str) -> dict[str, Any] | None:
        """Return the payload last sent for a record, or None when there is none."""
        return self._store.get(kind, identity)

    def put(self, kind: str, identity: str, payload: dict[str, Any]) -> None:
        """Send a record's whole payload, unless it says nothing new.

        Args:
            kind: The record kind. It is also the payload's key.
            identity: The id of the thing the record describes.
            payload: Its whole state now.
        """
        self._store.put(kind, identity, payload)

    def update(self, kind: str, identity: str, **changes: Any) -> None:
        """Send a record again with some payload fields changed. An unknown record is ignored."""
        self._store.update(kind, identity, **changes)

    def close_response(self) -> None:
        """Say that no response exchange is open now, so a task's handoff belongs to none."""
        self.response = None

    def measure(
        self,
        identity: str,
        metric: str,
        value: float | None,
        source: str,
        *,
        scope: Scope = "unassigned",
        exchange_id: str | None = None,
        operation_id: str | None = None,
        pending: bool = False,
        reason: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        """Send one timing: measured, still pending, or unavailable with its reason.

        Args:
            identity: The measurement's id. Send it again to settle a pending one.
            metric: What was timed, like ``first_response`` or ``reply_latency``.
            value: Seconds. A value that is not a finite number of 0 or more is unavailable.
            source: Where the number came from.
            scope: ``call``, ``input``, ``response``, ``operation`` or ``unassigned``.
            exchange_id: The exchange it belongs to, for the ``input`` and ``response`` scopes.
            operation_id: The operation it belongs to, for the ``operation`` scope.
            pending: True when the number has not arrived yet.
            reason: Why it is unavailable, or only partly assigned.
            model: The model the service named.
            provider: The service that reported it.
        """
        state = "pending" if pending else "measured" if is_seconds(value) else "unavailable"
        self.put(
            "measurement",
            identity,
            Measurement(
                scope=scope,
                metric=metric,
                source=source,
                state=state,
                value=value if state == "measured" else None,
                exchange_id=exchange_id or None,
                operation_id=operation_id or None,
                reason=reason or None,
                model=model or None,
                provider=provider or None,
            ).dump(),
        )

    def measure_operation(
        self,
        operation: str,
        metric: str,
        value: float | None,
        source: str,
        *,
        identity: str | None = None,
        exchange_id: str | None = None,
        pending: bool = False,
        reason: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        """Send one timing of an operation. It is ``measure`` with the operation's scope.

        Args:
            operation: The operation it times.
            metric: What was timed, like ``first_response`` or ``tool_duration``.
            value: Seconds, or None when there is none.
            source: Where the number came from.
            identity: The measurement's id. The default is ``<operation>:<metric>``.
            exchange_id: The exchange the operation belongs to.
            pending: True when the number has not arrived yet.
            reason: Why it is unavailable, or only partly assigned.
            model: The model the service named.
            provider: The service that reported it.
        """
        self.measure(
            identity or f"{operation}:{metric}",
            metric,
            value,
            source,
            scope="operation",
            exchange_id=exchange_id,
            operation_id=operation,
            pending=pending,
            reason=reason,
            model=model,
            provider=provider,
        )

    def limited(self, **fields: Any) -> None:
        """Mark the call record as only partly observed.

        Args:
            **fields: The call fields that say what is limited, like ``model_calls="partial"``
                or ``generated_text="limited"``.
        """
        self.update("call", "call", **fields)

    def open_input(self) -> str:
        """Open an input exchange for the caller's turn and return its id."""
        identity = self.new_id("input")
        self.put("exchange", identity, {"role": "input", "state": "open", "link_status": "known"})
        return identity

    def open_response(self, input_id: str | None = None) -> str:
        """Open a response exchange, make it the open one and return its id.

        Args:
            input_id: The input it answers, or None when it follows no input.
        """
        identity = self.new_id("response")
        payload: dict[str, Any] = {
            "role": "response",
            "state": "open",
            "playback": "not_started",
            "link_status": "known" if input_id else "unavailable",
        }
        if input_id:
            payload["input_id"] = input_id
        self.put("exchange", identity, payload)
        self.response = identity
        return identity

    def direct_text(self, exchange_id: str, text: str, *, interrupted: bool = False) -> str:
        """Record a line the agent said outside the model's answer, as one text row.

        Args:
            exchange_id: The response it belongs to.
            text: What was said.
            interrupted: True when the caller cut the line off.

        Returns:
            The row's id.
        """
        identity = self.new_id("direct-text")
        self.put(
            "text",
            identity,
            TextRow(
                exchange_id=exchange_id,
                message_id=identity,
                speaker="assistant",
                text=text,
                state="incomplete" if interrupted else "final",
                origin="generated",
            ).dump(),
        )
        return identity

    def control_started(self, name: str, *, exchange_id: str | None, parent: str | None) -> str:
        """Open a ``handoff`` operation for a control tool, an agent handoff or a human transfer.

        The tool hands the call over and does not return a result, so its call time is the
        time of whatever it started. It gets a row and no ``tool_duration``.

        Args:
            name: The tool's name.
            exchange_id: The response it belongs to, if any.
            parent: The model request that asked for it, if known.

        Returns:
            The operation's id, for ``control_ended``.
        """
        operation = self.new_id("handoff")
        self.put(
            "operation",
            operation,
            Operation(
                type="handoff",
                name=name,
                state="running",
                exchange_id=exchange_id or None,
                parent_operation_id=parent or None,
            ).dump(),
        )
        return operation

    def control_ended(self, operation: str, state: str, reason: str | None = None) -> None:
        """End the operation of a control tool. It gets no ``tool_duration``.

        Args:
            operation: The id ``control_started`` returned.
            state: ``returned``, ``failed``, ``timed_out`` or ``cancelled``.
            reason: Why it did not return, for the states that take one.
        """
        self.update("operation", operation, state=state, **({"reason": reason} if reason else {}))

    def operation(
        self,
        kind: OperationKind,
        name: str,
        *,
        exchange_id: str | None = None,
        state: str = "running",
        model: str | None = None,
        provider: str | None = None,
        parent: str | None = None,
        request_id: str | None = None,
        provider_first: bool = False,
    ) -> str:
        """Send a new operation and return its id.

        Args:
            kind: ``llm``, ``stt``, ``tts``, ``tool`` or ``handoff``. It is also the id's prefix.
            name: What runs: a plugin's label, a service's name or a tool's name.
            exchange_id: The exchange it belongs to, if any.
            state: ``running`` for one that is open, or the state of one already over.
            model: The model, left out when empty.
            provider: Who serves it, left out when empty.
            parent: The model request that asked for it, for a tool.
            request_id: The provider's id for the request, left out when empty.
            provider_first: True to write ``provider`` and ``model`` before ``exchange_id``, as
                the Pipecat observer always has. Only the key order differs.

        Returns:
            The operation's id.
        """
        identity = self.new_id(kind)
        shape = ProviderFirstOperation if provider_first else Operation
        self.put(
            "operation",
            identity,
            shape(
                type=kind,
                name=name,
                state=state,
                exchange_id=exchange_id or None,
                model=model or None,
                provider=provider or None,
                parent_operation_id=parent or None,
                source_request_id=request_id or None,
            ).dump(),
        )
        return identity

    def pending_request(self, operation: str, exchange_id: str, source: str) -> None:
        """Send the two timings of a model request, transcript or speech as pending.

        Args:
            operation: The operation they time.
            exchange_id: The exchange it belongs to.
            source: Where the numbers will come from.
        """
        for metric in REQUEST_METRICS:
            self.measure_operation(
                operation, metric, None, source, exchange_id=exchange_id, pending=True
            )

    def settle_request(
        self,
        operation: str,
        state: str,
        *,
        timings: Mapping[str, float | None],
        source: str,
        unavailable: str,
        answer: str | None = None,
    ) -> None:
        """Settle a model request, its answer text and the timings it left pending.

        A pending timing is measured when the request ended and ``timings`` has its number.
        Any other is unavailable. A timing that is no longer pending stays as it is.

        Args:
            operation: The request's operation.
            state: ``ended``, ``cancelled`` or ``failed``.
            timings: Seconds by metric, ``first_response`` and ``request_duration``.
            source: Where the numbers came from.
            unavailable: The reason a timing that did not arrive gets.
            answer: The answer's text row, which ends final, or incomplete for any state but
                ``ended``.
        """
        self.update("operation", operation, state=state)
        if answer is not None:
            self.update("text", answer, state="final" if state == "ended" else "incomplete")
            self._answers.pop(answer, None)
        for metric in REQUEST_METRICS:
            identity = f"{operation}:{metric}"
            if (pending := self.get("measurement", identity)) is None:
                continue
            if pending["state"] != "pending":
                continue
            if state == "ended" and (value := timings.get(metric)) is not None:
                self.measure_operation(
                    operation, metric, value, source, exchange_id=pending.get("exchange_id")
                )
            else:
                self.update("measurement", identity, state="unavailable", reason=unavailable)

    def append_text(self, row: str, exchange_id: str, text: str, *, spacing: bool = True) -> None:
        """Add words to the answer text row of a model request and send the whole row.

        The row stays ``provisional`` until ``settle_request`` ends it. The text is kept here,
        so adding a chunk does not read it back from the last record.

        Args:
            row: The row's id.
            exchange_id: The response it belongs to.
            text: The chunk. It has to be words.
            spacing: True when the chunk carries its own spaces, as most LLM text does. Then
                nothing is put between chunks.
        """
        old = self._answers.get(row, "")
        full = self._answers[row] = old + separator(old, text, spacing=spacing) + text
        self.put(
            "text",
            row,
            TextRow(
                exchange_id=exchange_id,
                message_id=row,
                speaker="assistant",
                text=full,
                state="provisional",
                origin="generated",
            ).dump(),
        )

    def user_text(self, input_id: str, text: str, *, final: bool, spacing: bool = False) -> None:
        """Record what the caller said as a text row, provisional until final.

        The words of one segment go to one row. A final one closes the segment, and the next
        words open a new row of the same message.

        Args:
            input_id: The input exchange the caller's turn is.
            text: The words so far.
            final: True for the final transcript of this segment.
            spacing: True when the text carries its own spaces. Otherwise a space goes between
                this segment and the one before when neither has one.
        """
        segment = self._open_segment.get(input_id) or self.new_id("user-text")
        self._open_segment[input_id] = segment
        message = self._messages.setdefault(input_id, self.new_id("user-message"))  # burns an id
        earlier = self._segments.setdefault(input_id, [])
        previous = (self.get("text", earlier[-1]) or {}).get("text", "") if earlier else ""
        self.put(
            "text",
            segment,
            TextRow(
                exchange_id=input_id,
                message_id=message,
                speaker="user",
                text=text,
                state="final" if final else "provisional",
                origin="recognition",
                separator_before=separator(previous, text, spacing=spacing),
            ).dump(),
        )
        if final:
            earlier.append(segment)
            del self._open_segment[input_id]

    def typed_text(self, input_id: str, text: str) -> None:
        """Record words the caller typed as final text, when the input has no words yet.

        Args:
            input_id: The input exchange the caller's turn is.
            text: What the message said. Blank text is ignored.
        """
        if self._segments.get(input_id) or input_id in self._open_segment or not text.strip():
            return
        identity = self.new_id("user-text")
        self.put(
            "text",
            identity,
            TextRow(
                exchange_id=input_id,
                message_id=self._messages.setdefault(input_id, identity),
                speaker="user",
                text=text,
                state="final",
                origin="transcription",
            ).dump(),
        )
        self._segments.setdefault(input_id, []).append(identity)

    def tool_started(
        self, name: str, *, exchange_id: str | None, parent: str | None, control: bool
    ) -> str:
        """Open the operation of a tool call: a ``tool``, or a ``handoff`` for a control tool.

        Args:
            name: The tool's name.
            exchange_id: The response it belongs to, if any.
            parent: The model request that asked for it, if known.
            control: True for a tool that hands the call over and returns no result.

        Returns:
            The operation's id, for ``tool_ended``.
        """
        if control:
            return self.control_started(name, exchange_id=exchange_id, parent=parent)
        return self.operation("tool", name, exchange_id=exchange_id, parent=parent)

    def tool_ended(
        self,
        operation: str,
        state: str,
        *,
        duration: float,
        source: str,
        control: bool,
        reason: str | None = None,
    ) -> None:
        """End the operation of a tool call, and time it unless it is a control tool.

        Args:
            operation: The id ``tool_started`` returned.
            state: ``returned``, ``failed``, ``timed_out`` or ``cancelled``.
            duration: Seconds the call took.
            source: Where the duration came from.
            control: True for a control tool, which gets no ``tool_duration``.
            reason: Why it did not return, for the states that take one.
        """
        if control:
            self.control_ended(operation, state, reason)
            return
        self.update("operation", operation, state=state, **({"reason": reason} if reason else {}))
        self.measure_operation(
            operation,
            "tool_duration",
            duration,
            source,
            identity=f"{operation}:duration",
            exchange_id=(self.get("operation", operation) or {}).get("exchange_id"),
        )

    def settle_exchange(self, exchange_id: str, *, cut: bool, spoke: bool) -> None:
        """End a response exchange: ended, or interrupted when the caller cut it off.

        Args:
            exchange_id: The response.
            cut: True when the caller interrupted it.
            spoke: True when its speech started. Without it the playback is ``unknown``.
        """
        self.update(
            "exchange",
            exchange_id,
            state="interrupted" if cut else "ended",
            playback=("interrupted" if cut else "ended") if spoke else "unknown",
        )

    def breakdown(
        self, measured_from: object, total_secs: float, parts: Iterable[Mapping[str, Any]]
    ) -> None:
        """Send where the wait for a reply went, unless a part cannot sit in a record.

        Args:
            measured_from: What the wait counts from. None means there is no breakdown.
            total_secs: The whole wait, in seconds.
            parts: Each part's ``key``, ``label``, ``owner``, ``owner_kind``, ``start_time``
                and ``duration_secs``. One that is unnamed, unowned or untimed drops the whole
                breakdown.
        """
        if measured_from is None:
            return
        try:
            payload = Breakdown(
                measured_from=str(measured_from), total_secs=total_secs, parts=list(parts)
            ).dump()
        except ValidationError:
            return
        self.put("breakdown", self.new_id("breakdown"), payload)

    def state_changed(self, event: StateEvent) -> None:
        """Send one ``state`` record for a change to the call's state.

        A value over 4 KiB as JSON is cut, with ``truncated`` set. A value that is empty is sent
        as ``null``.

        Args:
            event: What ``CallState.listen`` told.
        """
        self.put(
            "state",
            self.new_id("state"),
            {
                "origin": _origin(event.origin),
                "changes": [self._state_change(change) for change in event.changes],
            },
        )

    @staticmethod
    def _state_change(change: StateChange) -> dict[str, Any]:
        """Return one change as the record words it."""
        value, cut = _bounded(change.value)
        entry: dict[str, Any] = {"field": change.field, "value": value}
        if cut:
            entry["truncated"] = True
        if change.confirmed is not None:
            entry["confirmed"] = change.confirmed
        return entry | {"by": _origin(change.by)}

    def watch_state(self, state: CallState) -> Callable[[], None]:
        """Send a ``state`` record for the values already set, then for each later change.

        Args:
            state: The call's state.

        Returns:
            A function that stops watching.
        """
        return state.listen(self.state_changed, replay=True)

    def task_event(self, name: str, data: dict[str, Any]) -> None:
        """Turn the task events of a delegate run into one ``handoff`` operation.

        The run opens a ``handoff`` operation, named for its group or its task, when it
        starts. The operation ends when the run ends: ``returned`` when it finished,
        ``failed`` when it failed, ``cancelled`` when its group stopped. A step of a group
        has no operation of its own. Pass this to ``CallTasks.listen``.

        Args:
            name: The event's stage, like ``task.started``.
            data: The event's data: ``task``, ``group``, ``task_run_id``.
        """
        run = data.get("task_run_id")
        title = data.get("group") or data.get("task")
        if run is None or title is None or self.closed:
            return
        if name in ("task.started", "group.started") and run not in self._runs:
            self._runs[run] = self.new_id("handoff")
            self.put(
                "operation",
                self._runs[run],
                Operation(
                    type="handoff", name=title, state="running", exchange_id=self.response or None
                ).dump(),
            )
        elif (state := _RUN_ENDS.get(name)) and run in self._runs:
            if name.startswith("task.") and data.get("group") and name != "task.handed_off":
                return
            reason = "The task failed." if state == "failed" else None
            self.update(
                "operation",
                self._runs.pop(run),
                state=state,
                **({"reason": reason} if reason else {}),
            )

    def mark_error(self, reason: str) -> None:
        """Note that the call will end in an error, and why. The first reason is kept."""
        if not self._error:
            self._error = True
            self.update("call", "call", reason=reason)

    def end_call(self, *, error: bool = False, reason: str | None = None) -> None:
        """Close everything still open, end the call and send nothing after.

        An exchange ends when its text and its speech finished. Anything else open ends
        ``incomplete``, and a measurement that never arrived is unavailable. Calling it twice
        does nothing.

        Args:
            error: True when the call ended in an error.
            reason: What ended it, in the words of whatever raised.
        """
        if self.closed:
            return
        if reason:
            self.update("call", "call", reason=reason)
        completed = self._completed()
        for kind, identity, payload in self._store.payloads():
            if (kind == "text" and payload["state"] == "provisional") or (
                kind == "operation" and payload["state"] == "running"
            ):
                self.update(kind, identity, state="incomplete")
            elif kind == "exchange" and payload["state"] == "open":
                self.update(
                    kind, identity, state="ended" if identity in completed else "incomplete"
                )
            elif kind == "measurement" and payload["state"] == "pending":
                self.update(
                    kind,
                    identity,
                    state="unavailable",
                    reason="The call ended before this measurement arrived.",
                )
        self.update("call", "call", state="error" if error or self._error else "ended")
        self._store.close()
        self._runs.clear()
        self._segments.clear()
        self._open_segment.clear()
        self._messages.clear()
        self._answers.clear()

    def _completed(self) -> set[str | None]:
        """Return the responses whose text is final, whose speech ended and whose work ended."""
        final: set[str | None] = set()
        spoken: set[str | None] = set()
        unfinished: set[str | None] = set()
        for kind, _, payload in self._store.payloads():
            response = payload.get("exchange_id")
            if kind == "text" and payload["speaker"] == "assistant":
                (final if payload["state"] == "final" else unfinished).add(response)
            elif kind == "operation":
                if payload["state"] not in _ENDED:
                    unfinished.add(response)
                elif payload["type"] == "tts":
                    spoken.add(response)
        return (final & spoken) - unfinished
