"""Tracing for one call, sent to any set of providers over OTLP.

``Telemetry(Langfuse.from_env(), Coval.from_env(), agent=..., call_id=...)`` owns one
TracerProvider per call. Its ``start_*`` and ``end_*`` helpers emit the spans both adapters
share: ``conversation``, ``turn``, ``llm``, ``llm_tool_call``, ``task_step``, ``stt``,
``tts`` and one event span per decision. Any sink can
take ``filter=MetadataOnly()`` to send metadata only. docs/core/observe/telemetry.mdx lists the contract.
"""

from unmute_core.telemetry._coval import (
    Coval,
    coval_simulation_id,
    log_activated,
    log_no_simulation,
)
from unmute_core.telemetry._metadata import (
    DEFAULT_ATTRIBUTES,
    DEFAULT_RESOURCE,
    DEFAULT_SPAN_NAMES,
    MetadataOnly,
)
from unmute_core.telemetry._settings import (
    CovalSettings,
    LangfuseSettings,
    LogfireSettings,
    TelemetryConfigError,
)
from unmute_core.telemetry._sinks import (
    Langfuse,
    Logfire,
    Otlp,
    Sink,
)
from unmute_core.telemetry._telemetry import (
    CALL_SPAN,
    CLOSE_TIMEOUT_SECONDS,
    MODEL_SPAN,
    STT_SPAN,
    TASK_SPAN,
    TOOL_SPAN,
    TTS_SPAN,
    TURN_SPAN,
    CallOrigin,
    Telemetry,
)
from unmute_core.telemetry._turns import TurnEnd, TurnTracker

__all__ = [
    "CALL_SPAN",
    "CLOSE_TIMEOUT_SECONDS",
    "DEFAULT_ATTRIBUTES",
    "DEFAULT_RESOURCE",
    "DEFAULT_SPAN_NAMES",
    "MODEL_SPAN",
    "STT_SPAN",
    "TASK_SPAN",
    "TOOL_SPAN",
    "TTS_SPAN",
    "TURN_SPAN",
    "CallOrigin",
    "Coval",
    "CovalSettings",
    "Langfuse",
    "LangfuseSettings",
    "Logfire",
    "LogfireSettings",
    "MetadataOnly",
    "Otlp",
    "Sink",
    "Telemetry",
    "TelemetryConfigError",
    "TurnEnd",
    "TurnTracker",
    "coval_simulation_id",
    "log_activated",
    "log_no_simulation",
]
