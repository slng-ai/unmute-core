"""The Pipecat adapter. Needs the ``[pipecat]`` extra.

``UnmuteFlows`` runs an agent's tasks, with a Flows node for each step, and hands the call
between agents, each with its own model and, through a ``VoiceSwitcher``, its own voice.
``tools_schema`` gives the tools of an agent without tasks, with their handlers.
``ModelCallLLMService``
sends Pipecat's model requests through a ``ModelCall``. ``TelemetryObserver`` turns
pipeline frames into spans. ``DevObserver`` turns them into the records ``unmute dev`` shows. ``IdleNudgeObserver`` speaks to a quiet caller, then hangs up.
``end_call_tool`` ends the call with the goodbye in the same reply, after one model request.
``reply_text`` follows an LLM's replies for it, without ``UnmuteFlows``.
``interruption`` sets what speech over the agent does: cut it off, or wait for its line.
``vad_params`` and ``smart_turn_params`` turn a turn pace into the VAD and turn analyzer.
``telephony_call_info`` reads what a phone call carries, ``greeting_frames`` opens the call,
and ``max_duration`` ends it after its longest allowed length. ``user_mute_strategies``
turns interruption off, or protects the greeting and tool calls. ``open_call`` passes the
checks before the greeting and queues it, ``GreetingHold`` holds the caller's audio until
then, and ``attach_call`` lets a ``CallHooks`` fail and end the call. ``TwilioLine``,
``TelnyxLine`` and ``DailySipLine`` are a call's phone leg for ``transfer_tool``.
``forward_dial_in`` and ``dial_out`` connect a Daily room to a phone call.
``voicemail_tool`` gives the model a tool that ends an outbound call at a voicemail.
``correlate_coval`` binds the call to the Coval simulation that placed it.
"""

try:
    import pipecat  # noqa: F401
except ImportError as error:
    raise ImportError(
        'unmute_core.pipecat needs the pipecat extra. Install it: uv add "unmute-core[pipecat]"'
    ) from error

from unmute_core.pipecat._call import (
    GreetingHold,
    attach_call,
    fill_prompt,
    greeting_frames,
    max_duration,
    open_call,
    telephony_call_info,
)
from unmute_core.pipecat._coval import correlate_coval
from unmute_core.pipecat._dev import DevObserver
from unmute_core.pipecat._end import end_call_tool, reply_text
from unmute_core.pipecat._idle import IdleNudgeObserver
from unmute_core.pipecat._interruption import (
    Backchannels,
    Protected,
    interruption,
    user_mute_strategies,
)
from unmute_core.pipecat._mcp import mcp_tools
from unmute_core.pipecat._model import ModelCallLLMService
from unmute_core.pipecat._phone import (
    DailySipLine,
    TelnyxLine,
    TwilioLine,
    dial_out,
    disarm_hang_up,
    forward_dial_in,
    transfer_twiml,
)
from unmute_core.pipecat._realtime import attach_realtime
from unmute_core.pipecat._tasks import UnmuteFlows
from unmute_core.pipecat._telemetry import TelemetryObserver
from unmute_core.pipecat._tools import tools_schema
from unmute_core.pipecat._turn import smart_turn_params, vad_params
from unmute_core.pipecat._voicemail import voicemail_tool
from unmute_core.pipecat._voices import VoiceSwitcher

__all__ = [
    "Backchannels",
    "DailySipLine",
    "DevObserver",
    "GreetingHold",
    "IdleNudgeObserver",
    "ModelCallLLMService",
    "Protected",
    "TelemetryObserver",
    "TelnyxLine",
    "TwilioLine",
    "UnmuteFlows",
    "VoiceSwitcher",
    "attach_call",
    "attach_realtime",
    "correlate_coval",
    "dial_out",
    "disarm_hang_up",
    "end_call_tool",
    "fill_prompt",
    "forward_dial_in",
    "greeting_frames",
    "interruption",
    "max_duration",
    "mcp_tools",
    "open_call",
    "reply_text",
    "smart_turn_params",
    "telephony_call_info",
    "tools_schema",
    "transfer_twiml",
    "user_mute_strategies",
    "vad_params",
    "voicemail_tool",
]
