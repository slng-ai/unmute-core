"""Each agent of a call has its own model and voice, on a real text-only LiveKit session.

That the next model request goes to the next agent's model, and that its first reply offers
no handoff tool, is in tests/conformance (``handoff_own_model.json``). These cases cover the
LiveKit side: a plugin as the agent's model, its steps, its voice, and ``LiveKitModel``.
"""

import json
import re
from typing import Any, cast

import pytest
from livekit.agents import AgentSession, APIConnectOptions, function_tool, llm, tts

from tests.conformance.livekit_driver import Silence
from tests.livekit.fake import Reply, ScriptedLLM, Voice, call
from tests.livekit.test_trace import Memory
from tests.tasks.job import Canned, plan, registry, specialist_plan, state
from unmute_core.agent import AgentSpec
from unmute_core.handoff import HandoffContext
from unmute_core.livekit import LiveKitModel, UnmuteAgent
from unmute_core.livekit._step import StepTask
from unmute_core.model import FallbackModelCall, ModelRequest, TextDelta, ToolCall
from unmute_core.state import CallState
from unmute_core.telemetry import Telemetry

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


class Front(UnmuteAgent):
    """The orders desk, on the session's model and voice, with a handoff to the notes desk."""

    def __init__(
        self, call_state: CallState, notes: LiveKitModel, *, telemetry: Telemetry | None = None
    ) -> None:
        super().__init__(
            AgentSpec(
                "You help with orders.",
                tasks=plan(tools=registry(lookup_order=Canned(FOUND))),
                handoffs=("to_notes",),
            ),
            state=call_state,
            telemetry=telemetry,
        )
        self.notes_model = notes

    @function_tool
    async def to_notes(self) -> UnmuteAgent | str:
        """Hand the caller to the notes desk."""
        assert self.state is not None
        return await self.hand_off(Back(self.state, self.notes_model))


class Back(UnmuteAgent):
    """The notes desk, on its own model and its own voice, with a handoff back."""

    def __init__(self, call_state: CallState, model: LiveKitModel) -> None:
        super().__init__(
            AgentSpec(
                "You take notes.", tasks=specialist_plan(), model=model, handoffs=("to_orders",)
            ),
            state=call_state,
            tts=Voice("orion"),
        )

    @function_tool
    async def to_orders(self) -> UnmuteAgent | str:
        """Hand the caller back to orders."""
        return "Not now."


async def test_an_agent_and_its_steps_answer_on_its_own_plugin_and_voice() -> None:
    session_llm = ScriptedLLM()
    own = ScriptedLLM(call("find_order"), "Which order is it?", name="gpt-4o")
    voice = Voice("orion")
    tasks = plan(tools=registry(lookup_order=Canned(FOUND)))
    agent = UnmuteAgent(
        AgentSpec("You help with orders.", tasks=tasks, model=LiveKitModel(own)),
        state=state(tasks),
        tts=voice,
    )
    async with AgentSession(llm=session_llm) as session:
        await session.start(agent)
        await session.run(user_input="Where is my order?")
        step = session.current_agent

    assert not session_llm.requests
    main, step_request = own.requests
    assert "find_order" in main.tools
    assert "finish" in step_request.tools
    assert agent.tts is voice
    assert isinstance(step, StepTask)
    assert step.tts is voice
    assert step.llm is own


async def test_a_handoff_moves_the_call_to_the_next_agents_model_and_voice() -> None:
    sink = Memory()
    front_llm = ScriptedLLM(call("to_notes"), name="gpt-4o-mini")
    notes_llm = ScriptedLLM("Notes desk.", "Noted.", name="gpt-4o")
    first = Front(
        state(specialist_plan()),
        LiveKitModel(notes_llm),
        telemetry=Telemetry(sink, agent="desk", call_id="RM_1"),
    )
    async with AgentSession(llm=front_llm) as session:
        await session.start(first)
        await session.run(user_input="I need a note taken.")
        await session.run(user_input="Call me back on Monday.")
        desk = session.current_agent
    assert first._trace is not None and first._trace.closing is not None
    await first._trace.closing

    assert len(front_llm.requests) == 1
    assert "to_notes" in front_llm.requests[0].tools
    opening, after = notes_llm.requests
    assert "to_orders" not in opening.tools
    assert "to_orders" in after.tools
    assert isinstance(desk, Back)
    assert isinstance(desk.tts, Voice) and desk.tts.voice == "orion"
    models = [
        dict(span.attributes or {})["gen_ai.request.model"]
        for span in sink.spans.get_finished_spans()
        if span.name == "llm"
    ]
    assert models == ["gpt-4o-mini", "gpt-4o", "gpt-4o"]


async def test_a_plugin_writes_a_handoffs_summary_through_livekit_model() -> None:
    summarizer = ScriptedLLM("Notes wanted.", name="gpt-4o-mini")

    class Desk(UnmuteAgent):
        """Orders, handing off with a summary a plugin writes."""

        @function_tool
        async def to_notes(self) -> UnmuteAgent | str:
            """Hand the caller to the notes desk."""
            assert self.state is not None
            context = HandoffContext(history="summary", summarizer=LiveKitModel(summarizer))
            notes = UnmuteAgent(AgentSpec("You take notes."), state=self.state)
            return await self.hand_off(notes, context=context)

    model = ScriptedLLM(call("to_notes"), "Notes desk.")
    async with AgentSession(llm=model) as session:
        await session.start(Desk(AgentSpec("You help with orders."), state=state(plan())))
        await session.run(user_input="I need a note taken.")

    [request] = summarizer.requests
    assert request.tools == []
    assert request.texts == [("user", "user: I need a note taken.")]
    assert model.requests[1].texts == [
        ("assistant", "Summary of the conversation so far: Notes wanted.")
    ]


async def test_livekit_model_streams_a_request_as_any_model_call() -> None:
    plugin = ScriptedLLM(Reply("Hi Ana.", [("lookup_order", {"id": "A-12"})]), name="gpt-4o")
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look an order up.",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
        },
    }
    request = ModelRequest(
        messages=[
            {"role": "system", "content": "Help {{name}}."},
            {"role": "user", "content": "Hi."},
        ],
        tools=[tool],
        variables={"name": "Ana"},
    )
    model = FallbackModelCall([LiveKitModel(plugin)])

    events: list[Any] = [event async for event in model.stream(request)]

    assert events[0] == TextDelta("Hi Ana.")
    assert isinstance(events[1], ToolCall)
    assert (events[1].name, json.loads(events[1].arguments)) == ("lookup_order", {"id": "A-12"})
    [sent] = plugin.requests
    assert sent.prompt == "Help Ana."
    assert sent.tools == ["lookup_order"]
    assert (LiveKitModel(plugin).model, LiveKitModel(plugin).provider) == ("gpt-4o", "scripted")


SIGNED = {"google": {"thought_signature": "sig-1"}}


async def test_livekit_model_keeps_a_calls_signature_and_sends_it_back() -> None:
    plugin = ScriptedLLM(
        Reply(calls=[("lookup_order", {})], extra=SIGNED),
        Reply(calls=[("lookup_order", {})], extra=SIGNED),
        "Done.",
    )
    asked = ModelRequest(messages=[{"role": "user", "content": "Hi."}])

    [direct] = [event async for event in LiveKitModel(plugin).stream(asked)]
    [fallback] = [event async for event in FallbackModelCall([LiveKitModel(plugin)]).stream(asked)]
    assert isinstance(direct, ToolCall)
    assert isinstance(fallback, ToolCall)
    assert direct.extra == fallback.extra == SIGNED

    replayed = {
        "id": fallback.id,
        "type": "function",
        "function": {"name": fallback.name, "arguments": fallback.arguments},
        "extra_content": dict(fallback.extra),
    }
    again = ModelRequest(
        messages=[
            *asked.messages,
            {"role": "assistant", "tool_calls": [replayed]},
            {"role": "tool", "tool_call_id": fallback.id, "content": "{}"},
        ]
    )
    _ = [event async for event in FallbackModelCall([LiveKitModel(plugin)]).stream(again)]
    sent = plugin.requests[-1].chat_ctx.to_provider_format("openai")[0]
    assert sent[1]["tool_calls"][0]["extra_content"] == SIGNED


def test_a_handoff_the_agent_has_no_tool_for_is_refused() -> None:
    with pytest.raises(ValueError, match="handoffs names 'to_nowhere'"):
        UnmuteAgent(AgentSpec("Orders.", handoffs=("to_nowhere",)), state=state(plan()))


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({"model": ScriptedLLM()}, "model"),
        ({"summarizer": ScriptedLLM()}, "summarizer"),
        ({"task_models": {"fast": ScriptedLLM()}}, "task_models['fast']"),
    ],
)
def test_a_plugin_in_the_spec_is_refused_with_its_wrapper(fields: dict, field: str) -> None:
    with pytest.raises(TypeError, match=re.escape(f"AgentSpec {field} takes a ModelCall")):
        UnmuteAgent(AgentSpec("Orders.", **fields))


async def test_an_opening_cut_off_before_its_request_leaves_the_handoffs_in() -> None:
    desk = Back(state(specialist_plan()), LiveKitModel(ScriptedLLM()))
    heard = llm.ChatContext()
    heard.add_message(role="user", content="I need a note taken.")
    await desk.update_chat_ctx(heard)
    desk._gate.start_opening(desk.chat_ctx)
    later = heard.copy()
    later.add_message(role="user", content="Actually, call me back.")
    tools = cast(list[llm.Tool], list(desk.tools))

    assert "to_orders" not in [tool.id for tool in desk._gate.offered(heard.copy(), tools)]
    assert "to_orders" in [tool.id for tool in desk._gate.offered(later, tools)]


class _Sentence(tts.ChunkedStream):
    """One sentence of speech: a short sound."""

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="r", sample_rate=24000, num_channels=1, mime_type="audio/pcm"
        )
        output_emitter.push(b"\x00\x00" * 240)
        output_emitter.flush()


class Speaker(tts.TTS):
    """A voice that keeps each text it is asked to say."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
        )
        self.said: list[str] = []

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = ...,  # ty: ignore[invalid-parameter-default]
    ) -> tts.ChunkedStream:
        self.said.append(text)
        return _Sentence(tts=self, input_text=text, conn_options=conn_options)


SPANISH_LINES = ("Un momento.", "Un segundo.")


class English(UnmuteAgent):
    """The English desk, on the session's voice, with a handoff to the Spanish desk."""

    def __init__(self, call_state: CallState, spanish_voice: Speaker) -> None:
        super().__init__(AgentSpec("You speak English."), state=call_state)
        self.spanish_voice = spanish_voice

    @function_tool
    async def to_spanish(self) -> UnmuteAgent | str:
        """Hand the caller to the Spanish desk."""
        assert self.state is not None
        spanish = Spanish(self.state, self.spanish_voice)
        return await self.hand_off(spanish, announce=SPANISH_LINES)


class Spanish(UnmuteAgent):
    """The Spanish desk, on its own voice, with a handoff back."""

    def __init__(self, call_state: CallState, voice: Speaker) -> None:
        super().__init__(AgentSpec("Hablas español."), state=call_state, tts=voice)
        self.voice = voice

    @function_tool
    async def to_english(self) -> UnmuteAgent | str:
        """Hand the caller back to the English desk."""
        assert self.state is not None
        return await self.hand_off(English(self.state, self.voice))


async def test_the_handoff_line_is_said_in_the_next_agents_voice_and_rotates() -> None:
    english, spanish = Speaker(), Speaker()
    model = ScriptedLLM(
        call("to_spanish"), "Hola.", call("to_english"), "Hello.", call("to_spanish"), "Hola."
    )
    async with AgentSession(llm=model, tts=english) as session:
        session.output.audio = Silence()
        await session.start(English(CallState(), spanish))
        await session.run(user_input="Español, por favor.")
        await session.run(user_input="English, please.")
        await session.run(user_input="Español otra vez.")

    first, hola, second, hola_again = spanish.said
    assert {first, second} == set(SPANISH_LINES)
    assert (hola, hola_again) == ("Hola.", "Hola.")
    assert english.said == ["Hello."]
