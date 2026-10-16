"""One agent of a call, described once for LiveKit and Pipecat.

``AgentSpec`` holds what makes an agent itself: its prompt, tools, tasks, models, language
and handoff tools. ``UnmuteAgent(spec)`` runs it on LiveKit and ``UnmuteFlows(spec, ...)``
on Pipecat. A handoff takes the next agent's spec on both. The call-wide objects (the state,
the hooks, the telemetry) and the framework's own objects go to the adapter.
"""

from unmute_core.agent._spec import AgentSpec

__all__ = ["AgentSpec"]
