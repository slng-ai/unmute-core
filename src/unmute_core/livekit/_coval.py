"""Find the Coval simulation that placed a LiveKit call, and how the caller arrived.

Ported from unmute@d70247b internal/generate/templates/livekit_v1/tracing_coval.py.tmpl:106-162
and :1006-1029. A SIP participant joins after the job starts and LiveKit delivers its
attributes later, so the check runs at the start, on each join and on each attribute change.
"""

import asyncio
import json
import logging

from livekit import rtc
from livekit.agents import JobContext

from unmute_core.telemetry import (
    CallOrigin,
    Coval,
    Telemetry,
    coval_simulation_id,
    log_activated,
    log_no_simulation,
)

logger = logging.getLogger(__name__)


def _dispatched_simulation(ctx: JobContext) -> str | None:
    """Return the simulation ID in the job's dispatch metadata. Bad JSON is ignored."""
    try:
        metadata = json.loads(ctx.job.metadata or "{}")
    except (TypeError, ValueError):
        return None
    return coval_simulation_id(metadata) if isinstance(metadata, dict) else None


def _simulation(ctx: JobContext, coval: Coval) -> tuple[str, str] | None:
    """Return the simulation ID and where it came from, or None while no source has one.

    The order is the dispatch metadata, then ``COVAL_SIMULATION_ID``, then each remote
    participant's attributes.
    """
    if found := _dispatched_simulation(ctx):
        return found, "dispatch_metadata"
    if coval.simulation_id:
        return coval.simulation_id, "environment"
    return next(
        (
            (found, "sip_participant_attribute")
            for participant in ctx.room.remote_participants.values()
            if (found := coval_simulation_id(participant.attributes))
        ),
        None,
    )


def _origin(ctx: JobContext) -> CallOrigin | None:
    """Return ``phone`` when a SIP participant is in the room, ``browser`` for any other.

    An empty room gives None, because a guess on the trace is worse than a gap.
    """
    participants = ctx.room.remote_participants.values()
    if any(p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP for p in participants):
        return "phone"
    return "browser" if participants else None


def correlate_coval(ctx: JobContext, coval: Coval, telemetry: Telemetry) -> None:
    """Bind the call to its Coval simulation as soon as one names it, and record the origin.

    Call it once from the job's entrypoint, with the ``Coval`` sink the call's ``Telemetry``
    was built with. It checks now, on ``participant_connected`` and on
    ``participant_attributes_changed``. Each check tells ``telemetry`` the call's origin. The
    first simulation ID activates the sink once, on a worker thread, so the event loop never
    waits for the held spans to export. A call no simulation names is filed as a Coval
    conversation when it ends.

    Args:
        ctx: The job context, for the dispatch metadata and the room.
        coval: The call's Coval sink.
        telemetry: The call's telemetry. It gets ``unmute.call.origin``.
    """
    activations: set[asyncio.Task[None]] = set()
    activated = False

    async def activate(simulation_id: str, method: str) -> None:
        """Send the held spans against the simulation, on a worker thread."""
        try:
            await asyncio.to_thread(coval.activate, simulation_id=simulation_id, method=method)
        except RuntimeError as error:
            logger.warning("Coval did not activate for simulation %s: %s", simulation_id, error)

    def check(*_: object) -> None:
        """Record the origin, and activate once when a simulation ID shows up."""
        nonlocal activated
        telemetry.describe_call(origin=_origin(ctx))
        if activated or (found := _simulation(ctx, coval)) is None:
            return
        activated = True
        simulation_id, method = found
        task = asyncio.get_running_loop().create_task(activate(simulation_id, method))
        activations.add(task)
        task.add_done_callback(activations.discard)
        log_activated(logger, simulation_id, method)

    ctx.room.on("participant_connected", check)
    ctx.room.on("participant_attributes_changed", check)
    check()
    if not activated:
        log_no_simulation(logger)
