"""Dev records: what one call did, as the lines ``unmute dev`` reads.

``DevRecorder`` builds the version 2 records of one call and sends each to a sink. The
default sink prints ``UNMUTE_METRIC <json>`` lines on stdout. It also takes the call's state
changes (``watch_state``) and its task events (``task_event``). On Pipecat,
``unmute_core.pipecat.DevObserver`` turns the pipeline's frames into the rest. On LiveKit,
``unmute_core.livekit.DevObserver`` turns the session's events into them.

The record shapes are the contract of the Unmute CLI's ``internal/devmetrics`` package.
"""

from unmute_core.dev._recorder import (
    METRICS_ENV,
    SENTINEL,
    CallId,
    DevRecorder,
    DevSettings,
    DevSink,
    call_id_from,
    known_name,
    metrics_enabled,
    separator,
    stdout_sink,
)

__all__ = [
    "METRICS_ENV",
    "SENTINEL",
    "CallId",
    "DevRecorder",
    "DevSettings",
    "DevSink",
    "call_id_from",
    "known_name",
    "metrics_enabled",
    "separator",
    "stdout_sink",
]
