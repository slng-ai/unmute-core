"""The payloads of the dev records an operation, a text row, a measurement and a breakdown.

Each model dumps to the dict the reader's contract names: the keys in field order, and a
field that is ``None`` left out. ``DevSink`` still receives dicts.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

REQUEST_METRICS = ("first_response", "request_duration")
"""The two timings every model request, transcript and piece of speech has."""

Seconds = Annotated[int | float, Field(ge=0, allow_inf_nan=False)]
"""A time in seconds that can sit in a record: a finite number of 0 or more."""
_SECONDS = TypeAdapter(Seconds)


def is_seconds(value: object) -> bool:
    """Say whether a value is a finite number of 0 or more, so a record can carry it."""
    try:
        _SECONDS.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


OperationKind = Literal["llm", "stt", "tts", "tool", "handoff"]
"""The kinds of operation the reader knows."""

Scope = Literal["call", "input", "response", "operation", "unassigned"]
"""What a measurement is about."""


class _Payload(BaseModel):
    """A record payload: fixed fields, no extras, dumped without its empty fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def dump(self) -> dict[str, Any]:
        """Return the payload as the dict the record carries."""
        return self.model_dump(exclude_none=True)


class _OperationHead(_Payload):
    """What every operation starts with."""

    type: OperationKind
    name: str
    state: str


class Operation(_OperationHead):
    """One model request, transcript, piece of speech, tool call or handoff."""

    exchange_id: str | None = None
    model: str | None = None
    provider: str | None = None
    parent_operation_id: str | None = None
    source_request_id: str | None = None


class ProviderFirstOperation(_OperationHead):
    """An ``Operation`` whose provider and model come before its exchange, as Pipecat's do."""

    provider: str | None = None
    model: str | None = None
    exchange_id: str | None = None
    parent_operation_id: str | None = None
    source_request_id: str | None = None


class TextRow(_Payload):
    """One row of what the caller or the agent said."""

    exchange_id: str
    message_id: str
    speaker: Literal["user", "assistant"]
    text: str
    state: Literal["provisional", "final", "incomplete"]
    origin: Literal["recognition", "transcription", "generated"]
    separator_before: str = ""


class Measurement(_Payload):
    """One timing: measured, pending, or unavailable."""

    scope: Scope
    metric: str
    unit: Literal["seconds"] = "seconds"
    source: str
    state: Literal["pending", "measured", "unavailable"]
    value: int | float | None = None
    exchange_id: str | None = None
    operation_id: str | None = None
    reason: str | None = None
    model: str | None = None
    provider: str | None = None


class BreakdownPart(_Payload):
    """One part of the wait for a reply: named, owned and timed."""

    key: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    owner: Annotated[str, Field(min_length=1)]
    owner_kind: Annotated[str, Field(min_length=1)]
    start_time: Seconds
    duration_secs: Seconds


class Breakdown(_Payload):
    """Where the wait for a reply went, part by part."""

    measured_from: str
    total_secs: int | float
    parts: Annotated[list[BreakdownPart], Field(min_length=1)]
