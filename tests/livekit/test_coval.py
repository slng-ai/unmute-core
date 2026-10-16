"""``correlate_coval`` finds the Coval simulation on a LiveKit job and the caller's origin."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest
from livekit import rtc
from livekit.agents import JobContext
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from unmute_core.livekit import correlate_coval
from unmute_core.telemetry import Coval, Sink, Telemetry

SIP = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
STANDARD = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD


@dataclass(kw_only=True)
class Memory(Sink):
    """Keeps the spans in memory."""

    spans: InMemorySpanExporter = field(default_factory=InMemorySpanExporter)

    def exporter(self) -> SpanExporter:
        return self.spans


class Room(rtc.EventEmitter[str]):
    """A room with the participants the test puts in it."""

    def __init__(self) -> None:
        super().__init__()
        self.remote_participants: dict[str, Any] = {}

    def join(self, identity: str, kind: Any = STANDARD, **attributes: str) -> Any:
        participant = SimpleNamespace(identity=identity, kind=kind, attributes=attributes)
        self.remote_participants[identity] = participant
        return participant


@dataclass
class Job:
    """The job context, a room and the sink, with the calls to ``activate`` kept."""

    room: Room = field(default_factory=Room)
    metadata: str = ""
    sink: Memory = field(default_factory=Memory)
    coval: Coval = field(default_factory=lambda: Coval(api_key="cv-key"))
    activations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.telemetry = Telemetry(self.sink, agent="salon", call_id="RM_1")
        cast(Any, self.coval).activate = lambda **kwargs: self.activations.append(kwargs)
        self.ctx = cast(
            JobContext, SimpleNamespace(room=self.room, job=SimpleNamespace(metadata=self.metadata))
        )

    def correlate(self) -> None:
        correlate_coval(self.ctx, self.coval, self.telemetry)

    async def settled(self, count: int) -> None:
        """Wait until the worker thread has made ``count`` activations."""
        for _ in range(200):
            if len(self.activations) >= count:
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)

    async def origin(self) -> object:
        """Return the origin on the call span, ending the call to read it."""
        self.telemetry.start_call()
        await self.telemetry.close()
        (call,) = self.sink.spans.get_finished_spans()
        return dict(call.attributes or {}).get("unmute.call.origin")


async def test_dispatch_metadata_comes_first() -> None:
    job = Job(metadata=json.dumps({"coval.simulation_id": "from-metadata"}))
    job.coval.simulation_id = "from-env"
    job.room.join("caller", SIP, **{"coval.simulation_id": "from-participant"})
    job.correlate()
    await job.settled(1)

    assert job.activations == [{"simulation_id": "from-metadata", "method": "dispatch_metadata"}]


async def test_the_environment_comes_before_participants_and_bad_metadata_is_ignored() -> None:
    job = Job(metadata="{not json")
    job.coval.simulation_id = "from-env"
    job.room.join("caller", SIP, **{"coval.simulation_id": "from-participant"})
    job.correlate()
    await job.settled(1)

    assert job.activations == [{"simulation_id": "from-env", "method": "environment"}]


async def test_a_participants_attributes_are_the_last_source() -> None:
    job = Job(metadata=json.dumps({"other": "value"}))
    job.room.join("caller", SIP, **{"sip.h.X-Coval-Simulation-Id": "from-sip"})
    job.correlate()
    await job.settled(1)

    assert job.activations == [{"simulation_id": "from-sip", "method": "sip_participant_attribute"}]


async def test_a_participant_who_joins_later_activates_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = Job()
    with caplog.at_level(logging.INFO):
        job.correlate()
        assert "No Coval simulation placed this call" in caplog.text
        assert job.activations == []

        joined = job.room.join("caller", SIP)
        job.room.emit("participant_connected", joined)
        joined.attributes = {"sip.h.x-coval-simulation-id": "late"}
        job.room.emit("participant_attributes_changed", {}, joined)
        await job.settled(1)
        joined.attributes = {"sip.h.x-coval-simulation-id": "other"}
        job.room.emit("participant_attributes_changed", {}, joined)
        job.room.emit("participant_connected", joined)
        await job.settled(2)

    assert job.activations == [{"simulation_id": "late", "method": "sip_participant_attribute"}]
    assert caplog.text.count("Coval tracing active for simulation") == 1
    assert "Coval tracing active for simulation late via sip_participant_attribute" in caplog.text


async def test_an_already_bound_call_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    job = Job(metadata=json.dumps({"coval.simulation_id": "sim"}))

    def refuse(**_: Any) -> None:
        raise RuntimeError("This call is already bound to a Coval ID.")

    cast(Any, job.coval).activate = refuse
    with caplog.at_level(logging.WARNING):
        job.correlate()
        await job.settled(1)

    assert "already bound" in caplog.text


@pytest.mark.parametrize(
    ("kind", "origin"),
    [(SIP, "phone"), (STANDARD, "browser"), (None, None)],
)
async def test_origin_is_on_the_call_span(kind: Any, origin: str | None) -> None:
    job = Job()
    if kind is not None:
        job.room.join("caller", kind)
    job.correlate()

    assert await job.origin() == origin


async def test_origin_follows_who_joins() -> None:
    job = Job()
    job.correlate()
    first = job.room.join("agent-bridge")
    job.room.emit("participant_connected", first)
    sip = job.room.join("caller", SIP)
    job.room.emit("participant_connected", sip)

    assert await job.origin() == "phone"
