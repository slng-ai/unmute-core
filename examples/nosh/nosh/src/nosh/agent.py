"""Nosh's agent, written once for LiveKit and Pipecat.

Nosh has no tasks. It is one agent with three tools, and the tool registry does the rest:
each tool's checked result saves typed values (``Assign``) and code speaks the next line
(``ToolReply``), so a caller turn costs one model request. The greeting asks where; the
location is read back, with the craving asked in the same line when there is none; the
places are listed; a place's details are said; the goodbye is
the end tool's own line.

``APP`` is the ``VoiceApp`` the ``livekit`` and ``pipecat`` projects run.
"""

from collections.abc import Mapping
from importlib.resources import files

from pydantic import BaseModel
from unmute_core.agent import AgentSpec
from unmute_core.call import save_call_start
from unmute_core.greeting import Greeting
from unmute_core.state import CallInfo, CallStartRule, CallState
from unmute_core.tools import Assign, ToolRegistry, ToolReply, local_tool
from voice_stack import VoiceApp

from nosh.finder import TIMEZONE, Finder
from nosh.state import State

AGENT_ID = "nosh-finder"
VOICE = "ba875f2fb2ea4a4e9b175821014984b6"
"""Fish Audio's "Avery": bright and friendly."""
ASK_WHERE = "Whereabouts are you looking? An address or an area is fine."
GREETING = Greeting(
    text=f"Hey {{{{customer_name}}}}, it's Nosh! You wanted a hand finding somewhere good. {ASK_WHERE}"
)
"""Names the caller, from the dispatch."""
CALL_START: dict[str, CallStartRule] = {"customer_name": "required"}
"""What a dispatch must carry: who asked for the call."""
GREETING.check(None, CALL_START)
ASK_CRAVING = "What are you in the mood for? A cuisine, a bar, or no preference?"
GOODBYE = "Enjoy your meal! Bye for now."
PROMPT = (files("nosh") / "prompts" / "main.md").read_text(encoding="utf-8")


class Ended(BaseModel):
    """What ``end_call`` returns."""

    ended: bool


def end_call() -> Ended:
    """End the call once the caller is done. Say nothing: the application says goodbye."""
    return Ended(ended=True)


def tools(finder: Finder) -> ToolRegistry:
    """Return Nosh's tools, with what each result saves and the line code says for it."""
    return ToolRegistry(
        [
            local_tool(finder.find_area, announce=["Let me see.", "Hmm, let me look."]),
            local_tool(
                finder.find_places,
                locked={"spot": "{{spot}}"},
                announce=["Let me have a look.", "One sec, let me see what's around."],
                hidden=["listed"],
            ),
            local_tool(finder.place_details, locked={"offer": "{{offer}}"}, hidden=["spoken"]),
            local_tool(end_call, effect="ends_conversation", announce=GOODBYE, announce_wait=True),
        ],
        state=State,
        assign={
            "find_area": [
                Assign(variable="spot", path=("spot",)),
                Assign(variable="craving", path=("craving",)),
                Assign(variable="offer", path=("offer",)),
                Assign(variable="details", path=("details",)),
            ],
            "find_places": [
                Assign(variable="craving", path=("craving",)),
                Assign(variable="offer", path=("offer",)),
                Assign(variable="details", path=("details",)),
            ],
            "place_details": [Assign(variable="details", path=("details",))],
        },
        replies=[
            ToolReply(
                tool="find_area",
                when={"found": [True], "ask_craving": [False]},
                say=["So that's {{result.formatted_address}}, right?"],
            ),
            ToolReply(
                tool="find_area",
                when={"found": [True], "ask_craving": [True]},
                say=[f"So that's {{{{result.formatted_address}}}}. {ASK_CRAVING}"],
            ),
            ToolReply(
                tool="find_area",
                when={"found": [False]},
                say=[
                    "Hmm, I couldn't place that. Could you give me a nearby cross street, a "
                    "neighbourhood or the city?"
                ],
            ),
            ToolReply(
                tool="find_places",
                when={"ask_craving": [True]},
                say=[ASK_CRAVING],
            ),
            ToolReply(
                tool="find_places",
                when={"found": [True]},
                say=["I found {{result.listed}}. Do any of those sound good?"],
            ),
            ToolReply(
                tool="find_places",
                when={"found": [False]},
                say=[
                    "Hmm, I couldn't find anything like that right there. Want me to try a "
                    "bit wider, or a different kind of place?"
                ],
            ),
            ToolReply(tool="place_details", when={"found": [True]}, say=["{{result.spoken}}"]),
            ToolReply(
                tool="place_details",
                when={"found": [False]},
                say=[
                    "Sorry, I can't pull up {{result.name}}'s details just now. Want to hear "
                    "about another one?"
                ],
            ),
        ],
    )


class Nosh:
    """The finder, built once per process and shared by every call.

    Attributes:
        registry: The checked tools, on one place backend.
    """

    def __init__(self, finder: Finder | None = None) -> None:
        """Build the tools once. A wrong row, path or reply fails here, at import.

        Args:
            finder: The tools' backend. The place service, or the demo places, when None.
        """
        self.registry = tools(finder or Finder())

    async def new_call(
        self, call: CallInfo | None = None, call_start: Mapping[str, object] | None = None
    ) -> CallState[State]:
        """Return a fresh call state, with the caller's name from the dispatch.

        Raises:
            CallStartError: When the dispatch carries no name, or an empty one.
        """
        state = CallState(
            State,
            call_start=CALL_START,
            call=call or CallInfo(agent_id=AGENT_ID, direction="outbound"),
        )
        save_call_start(state, call_start)
        return state

    def agent(self, state: CallState) -> AgentSpec:
        """Return the call's agent: Nosh's prompt and tools, on the Madrid clock."""
        return AgentSpec(PROMPT, name="nosh", tools=self.registry, language="en", timezone=TIMEZONE)


NOSH = Nosh()
APP = VoiceApp(
    agent_id=AGENT_ID,
    greeting=GREETING,
    new_call=NOSH.new_call,
    agent=NOSH.agent,
    voice=VOICE,
    language="en",
    pace="snappy",
    max_duration_s=600,
)
