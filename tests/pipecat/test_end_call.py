"""``end_call_tool``: the goodbye rides in the reply that ends the call."""

from typing import Any, cast

import pytest
from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame, TTSSpeakFrame

from tests.pipecat.harness import FakeLLM, Harness, runs_llm
from unmute_core.call import AGENT_ENDED, GOODBYE_LINE, CallHooks
from unmute_core.pipecat import end_call_tool, reply_text
from unmute_core.pipecat._end import END_CALL_DESCRIPTION

GOODBYE = "Thanks for calling. Goodbye."


def harness() -> Harness:
    """A call whose main agent can end it."""
    call: Any = None
    tool = end_call_tool("End the call.", GOODBYE, lambda: call.flows.reply_text)
    call = Harness(native_tools=[tool])
    return call


def spoken(call: Harness) -> list[str]:
    """Every line the tool pushed to TTS."""
    return [frame.text for frame in call.llm.pushed if isinstance(frame, TTSSpeakFrame)]


async def test_a_goodbye_in_the_reply_ends_the_call_after_one_request() -> None:
    call = harness()
    call.user("No, that's all, thanks.")

    result, properties = await call.call("end_call", said="Goodbye!")

    assert result == {"ended": True}
    assert not runs_llm(properties)
    assert call.runs == 0
    assert spoken(call) == []
    assert call.worker.ended


async def test_a_reply_with_only_the_call_speaks_the_goodbye_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = harness()
    call.user("No, that's all, thanks.")

    said_before_the_end: list[list[str]] = []
    end = call.worker.end

    async def end_after_speaking(*, reason: str | None = None) -> None:
        said_before_the_end.append(spoken(call))
        await end(reason=reason)

    monkeypatch.setattr(call.worker, "end", end_after_speaking)

    _, properties = await call.call("end_call")

    assert not runs_llm(properties)
    assert spoken(call) == [GOODBYE]
    assert said_before_the_end == [[GOODBYE]]
    assert call.worker.ended


async def test_the_end_call_tool_notes_the_end_as_the_agents() -> None:
    hooks = CallHooks()
    tool = end_call_tool("End the call.", GOODBYE, lambda: "Bye.", call=hooks)
    call = Harness(native_tools=[tool], call=hooks)
    call.user("That is all.")

    await call.call("end_call", said="Bye.")

    assert call.worker.ended
    assert (await hooks.end("ended")).reason == AGENT_ENDED


def test_the_model_reads_livekits_end_call_text_first() -> None:
    end_call = pytest.importorskip("livekit.agents.beta.tools.end_call")
    tool = end_call_tool("Say goodbye first.", GOODBYE, lambda: "")

    assert tool.description == f"{END_CALL_DESCRIPTION}\nSay goodbye first."
    assert END_CALL_DESCRIPTION == end_call.END_CALL_DESCRIPTION


async def test_reply_text_follows_the_newest_reply_without_flows() -> None:
    llm = FakeLLM()
    text = reply_text(cast(Any, llm))
    [follow] = llm.hooks["on_before_push_frame"]

    await follow(llm, LLMFullResponseStartFrame())
    await follow(llm, LLMTextFrame("Good"))
    await follow(llm, LLMTextFrame("bye! "))
    assert text() == "Goodbye!"
    await follow(llm, LLMFullResponseStartFrame())
    assert text() == ""


async def test_with_no_goodbye_the_tool_says_the_english_goodbye() -> None:
    tool = end_call_tool("End the call.", None, lambda: "")
    call = Harness(native_tools=[tool])
    call.user("That is all.")

    await call.call("end_call")

    assert spoken(call) == [GOODBYE_LINE]


async def test_an_explicit_goodbye_is_said() -> None:
    tool = end_call_tool("End the call.", GOODBYE, lambda: "")
    call = Harness(native_tools=[tool])
    call.user("Eso es todo.")

    await call.call("end_call")

    assert spoken(call) == [GOODBYE]
