"""The dev record contract, as a Python check.

A port of ``Extract`` and ``validate`` in unmute@a005bd2 ``internal/devmetrics/record.go``,
which is the source of truth. ``problems`` returns what the Go reader would refuse. The sample
lines in ``tests/data/devmetrics`` come from the same package's ``testdata``.
"""

import json
import math
import re
from typing import Any

MAX_RECORD_BYTES = 512 << 10
KINDS = ("call", "exchange", "text", "operation", "measurement", "breakdown", "state")
ENVELOPE = {"version", "kind", "call_id", "id", "revision", "order"}
FIELDS: dict[str, set[str]] = {
    "call": {"target", "state", "reason", "input_boundaries", "model_calls", "generated_text"},
    "exchange": {"role", "state", "input_id", "link_status", "playback"},
    "text": {
        "exchange_id",
        "message_id",
        "speaker",
        "text",
        "state",
        "origin",
        "separator_before",
    },
    "operation": {
        "exchange_id",
        "type",
        "name",
        "model",
        "provider",
        "source_request_id",
        "parent_operation_id",
        "state",
        "reason",
    },
    "measurement": {
        "exchange_id",
        "operation_id",
        "scope",
        "metric",
        "value",
        "unit",
        "state",
        "source",
        "reason",
        "model",
        "provider",
    },
    "breakdown": {"measured_from", "total_secs", "parts"},
    "state": {"origin", "changes"},
}
ORIGINS = ("start", "step", "call_start", "fact", "prefetch", "withdraw")


def _id(value: Any, optional: bool) -> bool:
    """Say whether a value is a valid id: printable ASCII without spaces, at most 128."""
    if value is None or value == "":
        return optional
    return isinstance(value, str) and len(value) <= 128 and all(33 <= ord(c) <= 126 for c in value)


def _finite(value: Any) -> bool:
    """Say whether a value is a finite number of 0 or more."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def problems(line: str) -> list[str]:
    """Return what makes a record line invalid, or an empty list when the reader takes it.

    Args:
        line: One record as JSON, without the sentinel.
    """
    if len(line) > MAX_RECORD_BYTES:
        return ["record too long"]
    record = json.loads(line)
    kind = record.get("kind")
    found = [f"unexpected field {key!r}" for key in record if key not in ENVELOPE | {kind}]
    if kind not in KINDS:
        return [*found, f"unknown kind {kind!r}"]
    found += [
        f"unexpected {kind} field {key!r}"
        for key in record.get(kind) or {}
        if key not in FIELDS[kind]
    ]
    if record.get("version") != 2:
        found.append("version is not 2")
    if (
        not _id(record.get("call_id"), False)
        or re.fullmatch(r"[A-Za-z0-9_-]+", record["call_id"]) is None
    ):
        found.append("bad call_id")
    if not _id(record.get("id"), False):
        found.append("bad id")
    for key in ("revision", "order"):
        if not isinstance(record.get(key), int) or record[key] <= 0:
            found.append(f"bad {key}")
    payloads = [k for k in KINDS if record.get(k) is not None]
    if payloads != [kind]:
        found.append(f"exactly one payload, named {kind}, expected. Found {payloads}")
        return found
    return found + _payload(kind, record[kind])


def _payload(kind: str, p: dict[str, Any]) -> list[str]:
    """Return the problems of one payload, as ``validate`` checks it."""
    one = lambda key, *allowed: p.get(key) in allowed  # noqa: E731
    bad: list[str] = []
    if kind == "call":
        ok = (
            one("target", "livekit", "pipecat")
            and one("state", "open", "ended", "error")
            and one("input_boundaries", "known", "partial")
            and one("model_calls", "known", "partial")
            and one("generated_text", "available", "limited")
        )
    elif kind == "exchange":
        ok = (
            one("role", "input", "response")
            and one("state", "open", "ended", "interrupted", "incomplete")
            and one("link_status", "known", "unavailable")
            and _id(p.get("input_id"), True)
            and one(
                "playback", None, "", "not_started", "speaking", "ended", "interrupted", "unknown"
            )
        )
        if p.get("input_id") and (p["role"] != "response" or p["link_status"] != "known"):
            ok = False
    elif kind == "text":
        ok = (
            _id(p.get("exchange_id"), True)
            and _id(p.get("message_id"), False)
            and one("speaker", "user", "assistant")
            and one("state", "provisional", "final", "incomplete")
            and one("origin", "recognition", "generated", "transcription")
            and one("separator_before", "", " ", "\n")
            and isinstance(p.get("text"), str)
        )
    elif kind == "operation":
        ok = (
            _id(p.get("exchange_id"), True)
            and _id(p.get("parent_operation_id"), True)
            and _id(p.get("source_request_id"), True)
            and bool(p.get("name"))
            and one("type", "llm", "stt", "tts", "tool", "handoff")
            and one(
                "state",
                "running",
                "ended",
                "returned",
                "failed",
                "timed_out",
                "cancelled",
                "incomplete",
            )
        )
        if p.get("reason") and p.get("state") not in ("failed", "timed_out", "cancelled"):
            ok = False
    elif kind == "measurement":
        ok = _measurement(p)
    elif kind == "breakdown":
        parts = p.get("parts") or []
        ok = (
            one("measured_from", "user_silence", "client_connected")
            and bool(parts)
            and _finite(p.get("total_secs"))
            and all(
                part.get("key")
                and part.get("label")
                and part.get("owner")
                and part.get("owner_kind") in ("service", "setting", "bot", "pipeline")
                and _finite(part.get("duration_secs"))
                and _finite(part.get("start_time"))
                for part in parts
            )
            and abs(sum(part["duration_secs"] for part in parts) - p["total_secs"]) <= 1e-6
        )
    else:
        origin = p.get("origin") or {}
        changes = p.get("changes") or []
        ok = (
            origin.get("kind") in ORIGINS
            and _id(origin.get("name"), True)
            and bool(changes)
            and all(
                _id(c.get("field"), False)
                and "value" in c
                and (c.get("by") or {}).get("kind") in (*ORIGINS, "reset")
                and _id((c.get("by") or {}).get("name"), True)
                and set(c) <= {"field", "value", "truncated", "confirmed", "by"}
                and isinstance(c.get("confirmed", True), bool)
                for c in changes
            )
        )
    if not ok:
        bad.append(f"invalid v2 {kind} record")
    return bad


def _measurement(p: dict[str, Any]) -> bool:
    """Say whether a measurement payload is valid, as ``validate`` checks it."""
    value = p.get("value")
    scope = p.get("scope")
    if not (
        _id(p.get("exchange_id"), True)
        and _id(p.get("operation_id"), True)
        and scope in ("call", "input", "response", "operation", "unassigned")
        and p.get("metric")
        in (
            "reply_latency",
            "first_response",
            "request_duration",
            "speech_duration",
            "tool_duration",
            "turn_detection",
            "transcription_delay",
            "text_aggregation",
            "first_speech",
            "playback_delay",
        )
        and p.get("unit") == "seconds"
        and p.get("source")
        and p.get("state") in ("pending", "measured", "unavailable")
    ):
        return False
    if (p["state"] == "measured") != (value is not None):
        return False
    if value is not None and not _finite(value):
        return False
    exchange, operation = p.get("exchange_id"), p.get("operation_id")
    if (scope == "operation" and not operation) or (
        scope in ("input", "response") and not exchange
    ):
        return False
    return not (
        (scope != "operation" and operation) or (scope in ("call", "unassigned") and exchange)
    )
