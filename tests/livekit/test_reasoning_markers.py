"""``ModelCallLLM`` keeps a model's own reasoning markers out of the reply."""

from collections.abc import AsyncIterator

import pytest
from livekit.agents import AgentSession, llm

from unmute_core.agent import AgentSpec
from unmute_core.livekit import ModelCallLLM, UnmuteAgent
from unmute_core.model import FallbackModelCall, ModelEvent, ModelRequest, TextDelta

GEMMA_MODEL = "google/gemma-4-31b-it"
TRACE = pytest.mark.parametrize(
    "trace_attempts", [True, False], ids=["each attempt", "the request"]
)


class Speaks:
    """A model call that sends the pieces of text it was given, under a model name."""

    def __init__(self, model: str, *pieces: str) -> None:
        self.model = model
        self.pieces = pieces

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        for piece in self.pieces:
            yield TextDelta(piece)


async def _said(model: ModelCallLLM) -> str:
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Is my order on its way?")
    async with model.chat(chat_ctx=ctx) as stream:
        return "".join(
            [chunk.delta.content async for chunk in stream if chunk.delta and chunk.delta.content]
        )


def _llm(
    trace_attempts: bool, call: Speaks, reasoning_markers: tuple[tuple[str, str], ...] = ()
) -> ModelCallLLM:
    return ModelCallLLM(
        FallbackModelCall([call]),
        trace_attempts=trace_attempts,
        reasoning_markers=reasoning_markers,
    )


@TRACE
async def test_gemma_markers_are_stripped_for_gemma(trace_attempts: bool) -> None:
    call = Speaks(GEMMA_MODEL, "<|channel>thought\nCheck the order.<channel|>It is on its way.")
    assert await _said(_llm(trace_attempts, call)) == "It is on its way."


@TRACE
async def test_gemma_markers_are_stripped_when_the_model_name_has_no_prefix(
    trace_attempts: bool,
) -> None:
    call = Speaks("gemma-4-31b-it", "<|channel>thought\n<channel|>Yes.")
    assert await _said(_llm(trace_attempts, call)) == "Yes."


@TRACE
async def test_gemma_markers_are_left_alone_for_another_model(trace_attempts: bool) -> None:
    text = "<|channel>thought\nCheck the order.<channel|>It is on its way."
    assert await _said(_llm(trace_attempts, Speaks("gpt-4o", text))) == text


@TRACE
async def test_gemma_markers_cut_between_chunks_are_stripped(trace_attempts: bool) -> None:
    call = Speaks(
        GEMMA_MODEL,
        "<|chan",
        "nel>thou",
        "ght\nweigh",
        "ing<chan",
        "nel|",
        ">It is ",
        "on its way.",
    )
    assert await _said(_llm(trace_attempts, call)) == "It is on its way."


@TRACE
async def test_a_block_left_open_and_a_held_back_piece_are_handled_at_the_end(
    trace_attempts: bool,
) -> None:
    assert (
        await _said(_llm(trace_attempts, Speaks(GEMMA_MODEL, "Yes.<|channel>thought no"))) == "Yes."
    )
    assert await _said(_llm(trace_attempts, Speaks(GEMMA_MODEL, "Yes <|chan"))) == "Yes <|chan"


@TRACE
async def test_the_standard_tags_are_stripped_for_every_model(trace_attempts: bool) -> None:
    text = "</think>Hello <think>weighing it</think>there."
    for model in (GEMMA_MODEL, "gpt-4o"):
        assert await _said(_llm(trace_attempts, Speaks(model, text))) == "Hello there."


@TRACE
async def test_extra_markers_are_stripped_besides_the_models_own(trace_attempts: bool) -> None:
    call = Speaks("gpt-4o", "[[plan]]Hi", " <|channel>thought stays<channel|>.")
    model = _llm(trace_attempts, call, reasoning_markers=(("[[", "]]"),))
    assert await _said(model) == "Hi <|channel>thought stays<channel|>."


async def test_an_agent_turn_strips_the_markers_of_its_model() -> None:
    async def said(model: str) -> str:
        agent = UnmuteAgent(
            AgentSpec("Orders.", model=Speaks(model, "<|channel>thought\nhm<channel|>Hello there."))
        )
        async with AgentSession() as session:
            await session.start(agent)
            await session.run(user_input="Hi")
            return [item.text_content for item in session.history.items if item.type == "message"][
                -1
            ] or ""

    assert await said(GEMMA_MODEL) == "Hello there."
    assert await said("gpt-4o") == "<|channel>thought\nhm<channel|>Hello there."
