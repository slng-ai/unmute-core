"""Unmute's tools, tasks, model call and telemetry on LiveKit Agents 1.8.

``UnmuteAgent`` is a LiveKit ``Agent``. Its tools come from a ``ToolRegistry``, its tasks
from a ``TaskPlan``, and every model request goes through its ``llm_node``: the session's
own LLM plugin, the agent's own (``LiveKitModel``), or a ``ModelCall`` like the SLNG Context
Router. Each agent of a call may have its own model and voice. ``yield_to_caller`` drops a
reply the caller went on speaking over, ``interruption`` sets what speech over a line
that plays does, and ``idle_nudges`` speaks to a quiet caller, also while a task runs.
``endpointing`` turns a turn pace into the session's endpointing options.
``sip_call_info`` reads what a SIP call carries, ``greet`` opens the call, and
``max_duration`` ends it after its longest allowed length. ``correlate_coval`` binds the
call to the Coval simulation that placed it. ``open_call`` passes the checks
before the greeting and holds the caller's audio until it plays, and ``attach_call`` lets a
``CallHooks`` fail and end the call. ``SipLine`` is a SIP call's phone leg: it transfers
the caller through ``transfer_tool`` and dials out.
For an agent of your own, ``function_tools`` gives a ``ToolRegistry`` as LiveKit tools,
``ModelCallLLM`` runs a ``ModelCall`` as the session's LLM, and ``trace_session`` traces the
session, with ``traced_tts`` for speech spans. docs/core/frameworks/livekit.mdx and docs/core/frameworks/standalone.mdx show
how. ``DevObserver`` turns the session's events into the records ``unmute dev`` shows.
Needs the ``[livekit]`` extra.
"""

try:
    import livekit.agents  # noqa: F401
except ImportError as error:
    raise ImportError(
        'unmute_core.livekit needs the livekit extra. Install it: uv add "unmute-core[livekit]"'
    ) from error

from unmute_core.livekit._agent import UnmuteAgent
from unmute_core.livekit._call import attach_call, greet, max_duration, open_call, sip_call_info
from unmute_core.livekit._coval import correlate_coval
from unmute_core.livekit._dev import DevObserver
from unmute_core.livekit._history import shape_chat_ctx
from unmute_core.livekit._idle import idle_nudges
from unmute_core.livekit._model import LiveKitModel, ModelCallLLM, filled_llm, traced_llm
from unmute_core.livekit._phone import BRIEFING_PERSONA, DialError, Dialled, SipLine
from unmute_core.livekit._tools import function_tools
from unmute_core.livekit._trace import SessionTrace, trace_session, traced_tts
from unmute_core.livekit._turn import endpointing
from unmute_core.livekit._yield import Backchannels, interruption, yield_to_caller

__all__ = [
    "BRIEFING_PERSONA",
    "Backchannels",
    "DevObserver",
    "DialError",
    "Dialled",
    "LiveKitModel",
    "ModelCallLLM",
    "SessionTrace",
    "SipLine",
    "UnmuteAgent",
    "attach_call",
    "correlate_coval",
    "endpointing",
    "filled_llm",
    "function_tools",
    "greet",
    "idle_nudges",
    "interruption",
    "max_duration",
    "open_call",
    "shape_chat_ctx",
    "sip_call_info",
    "trace_session",
    "traced_llm",
    "traced_tts",
    "yield_to_caller",
]
