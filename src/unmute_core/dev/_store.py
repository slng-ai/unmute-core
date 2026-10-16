"""The versioned records of one call: revision, order, dedup and the sink."""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

DevSink = Callable[[dict[str, Any]], None]
"""Receives one record, as a dict. It may keep the dict."""


class RecordStore:
    """Keeps the last record of each thing and sends a new one to the sink.

    A record is sent when its payload changed. Its ``revision`` grows with each send and its
    ``order`` stays where it first was. A sink that raises is logged and the record is lost.
    """

    def __init__(self, call_id: str, sink: DevSink) -> None:
        """Start an empty store.

        Args:
            call_id: The id every record carries.
            sink: Receives each record.
        """
        self.call_id = call_id
        self._sink = sink
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        self._serial = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        """True once ``close`` ran. Nothing is sent after it."""
        return self._closed

    def new_id(self, prefix: str) -> str:
        """Return the next id for a kind of thing, like ``llm-3``. The count is the call's."""
        self._serial += 1
        return f"{prefix}-{self._serial}"

    def get(self, kind: str, identity: str) -> dict[str, Any] | None:
        """Return the payload last sent for a record, or None when there is none."""
        record = self._records.get((kind, identity))
        return record[kind] if record else None

    def payloads(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Return a snapshot of every record as its kind, id and payload, in first-sent order."""
        return [
            (kind, identity, record[kind]) for (kind, identity), record in self._records.items()
        ]

    def put(self, kind: str, identity: str, payload: dict[str, Any]) -> None:
        """Send a record's whole payload, unless it says nothing new.

        Args:
            kind: The record kind. It is also the payload's key.
            identity: The id of the thing the record describes.
            payload: Its whole state now.
        """
        if self._closed:
            return
        old = self._records.get((kind, identity))
        if old and old[kind] == payload:
            return
        record = {
            "version": 2,
            "kind": kind,
            "call_id": self.call_id,
            "id": identity,
            "revision": old["revision"] + 1 if old else 1,
            "order": old["order"] if old else len(self._records) + 1,
            kind: dict(payload),
        }
        self._records[kind, identity] = record
        try:
            self._sink(record)
        except Exception:
            logger.warning("a dev record could not be sent", exc_info=True)

    def update(self, kind: str, identity: str, **changes: Any) -> None:
        """Send a record again with some payload fields changed. An unknown record is ignored."""
        if (old := self.get(kind, identity)) is not None:
            self.put(kind, identity, {**old, **changes})

    def close(self) -> None:
        """Stop sending and forget every record."""
        self._closed = True
        self._records.clear()
