"""``VoiceApp``: one agent, described once, that both frameworks run.

An example builds one ``VoiceApp`` in its own package, with no framework import. The
LiveKit and Pipecat modules of this package read it and build everything around it: the
session or the pipeline, speech, the model, the greeting, the turn pace and the call's
length. The app keeps what is the agent's own: how a call starts, which ``AgentSpec`` it
runs, and how it sounds.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from unmute_core.agent import AgentSpec
from unmute_core.greeting import Greeting
from unmute_core.idle import IdleNudges
from unmute_core.state import CallInfo, CallState
from unmute_core.turn import TurnPace

NewCall = Callable[[CallInfo | None, Mapping[str, object] | None], Awaitable[CallState]]
"""Starts one call: ``new_call(call, call_start)`` returns its state, prefetch read in."""


@dataclass(frozen=True)
class Handoff:
    """A tool that gives the call to another of the app's agents.

    It is the same on both frameworks: ``voice_stack`` turns it into a LiveKit function
    tool or a Pipecat ``FunctionSchema`` whose handler calls the adapter's ``hand_off``.
    The agent that offers it must name it in ``AgentSpec(handoffs=...)``, so the next
    agent's first reply cannot hand the call straight back.

    Attributes:
        tool: The tool's name, like ``to_spanish``.
        description: When the model calls it.
        to: The ``name`` of the agent that takes the call.
        announce: A line said first, which the caller cannot cut off.
        saves: State values saved as the call moves, like the language it now speaks.
    """

    tool: str
    description: str
    to: str
    announce: str | None = None
    saves: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VoiceApp:
    """One agent and how it sounds. Build it once per process; it holds nothing of a call.

    Attributes:
        agent_id: The agent's id. SLNG speech and the traces group calls by it.
        greeting: How a call opens.
        new_call: Starts one call and returns its state. It gets what the line carries
            (None on a browser or console call) and the dispatch's ``call_start`` values.
        agent: Returns the call's ``AgentSpec``, built after the state.
        voice: The Fish Audio voice id the agent speaks with.
        language: The language speech is recognised and spoken in, like ``en`` or ``es``.
            ``multi`` lets the speech-to-text model hear any language.
        tts_language: The language the voice speaks. Defaults to ``language``.
        pace: How long the caller may pause before their turn ends.
        max_duration_s: The longest a call may run.
        idle: What the agent says to a caller who goes quiet. None says nothing.
        agents: Every agent of the call by name, for an app with handoffs. ``agent``
            still picks the one the call starts with.
        handoffs: The handoff tools each agent offers, by the agent's name.
    """

    agent_id: str
    greeting: Greeting
    new_call: NewCall
    agent: Callable[[CallState], AgentSpec]
    voice: str
    language: str = "en"
    tts_language: str | None = None
    pace: TurnPace = "balanced"
    max_duration_s: float = 600
    idle: IdleNudges | None = None
    agents: Callable[[CallState], Mapping[str, AgentSpec]] | None = None
    handoffs: Mapping[str, tuple[Handoff, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def specs(self, state: CallState) -> tuple[AgentSpec, Mapping[str, AgentSpec]]:
        """Return the agent the call starts with, and every agent by name.

        Raises:
            ValueError: When a handoff names an agent the app does not build.
        """
        first = self.agent(state)
        every = dict(self.agents(state)) if self.agents is not None else {}
        if first.name is not None:
            every.setdefault(first.name, first)
        for offered in self.handoffs.values():
            for handoff in offered:
                if handoff.to not in every:
                    raise ValueError(
                        f"handoff {handoff.tool} goes to {handoff.to!r}, which the app does not "
                        f"build. Agents: {', '.join(sorted(every)) or 'none'}"
                    )
        return first, every

    @property
    def speaks(self) -> str:
        """The language the voice speaks."""
        return self.tts_language or self.language
