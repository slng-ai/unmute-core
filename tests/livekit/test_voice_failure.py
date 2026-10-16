"""A voice that fails for good (LiveKit 1.8.6 ends its segment and marks the reply interrupted)."""

import asyncio

import pytest
from livekit.agents import AgentSession, APIConnectOptions, APIStatusError, tts
from livekit.agents.voice.agent_session import SessionConnectOptions

from tests.conformance.livekit_driver import Lines, Silence
from tests.hooks.helpers import on_end
from tests.livekit.fake import ScriptedLLM, call, replies
from tests.livekit.test_dev import Records
from tests.tasks.job import Canned, job, plan, registry, state
from unmute_core.agent import AgentSpec
from unmute_core.call import ENDED, ERROR, FAILURE_LINE, CallEnd, CallHooks
from unmute_core.livekit import DevObserver, UnmuteAgent
from unmute_core.livekit._call import transcript

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}
NO_RETRY = SessionConnectOptions(
    tts_conn_options=APIConnectOptions(max_retry=0, retry_interval=0, timeout=2)
)


class _Sentence(tts.ChunkedStream):
    """One sentence of speech: a short sound, or the failure."""

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        voice = self._tts
        assert isinstance(voice, FailingVoice)
        voice.sentences += 1
        output_emitter.initialize(
            request_id="r", sample_rate=24000, num_channels=1, mime_type="audio/pcm"
        )
        if voice.sentences > voice.speaks:
            raise APIStatusError("down", status_code=400, retryable=False)
        output_emitter.push(b"\x00\x00" * 2400)
        output_emitter.flush()


class FailingVoice(tts.TTS):
    """A voice that says its first ``speaks`` sentences, then fails every one."""

    def __init__(self, speaks: int = 0) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
        )
        self.speaks = speaks
        self.sentences = 0

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = ...,  # ty: ignore[invalid-parameter-default]
    ) -> tts.ChunkedStream:
        return _Sentence(tts=self, input_text=text, conn_options=conn_options)


async def test_a_voice_that_fails_leaves_the_call_open_and_answers_the_next_turn() -> None:
    voice = FailingVoice(speaks=1)
    ends: list[CallEnd] = []
    failed: list[object] = []
    hooks = on_end(ends.append)
    hooks.on("call.failed")(lambda *args: failed.append(args))
    model = ScriptedLLM("Your order is on its way. It arrives on Monday.", "Anything else?")
    agent = UnmuteAgent(AgentSpec("Help."), tts=voice, call=CallHooks(hooks=hooks))
    lines = Lines()
    closed = asyncio.Event()
    async with AgentSession(llm=model, tts=voice, conn_options=NO_RETRY) as session:
        session.output.audio = Silence()
        session.output.transcription = lines
        session.on("close", lambda _event: closed.set())
        await session.start(agent)
        await session.run(user_input="Where is my order?")
        await session.run(user_input="Hello?")
        await asyncio.sleep(0.1)
        assert not closed.is_set()
        assert session.agent_state == "listening"
        assert len(model.requests) == 2
        cut, *silent = replies(session)
        assert cut.text_content and cut.text_content.startswith("Your order is on its way.")
        assert cut.interrupted
        assert silent == []
        assert transcript(session)[0]["content"] == "Where is my order?"
        assert [m["role"] for m in transcript(session)] == ["user", "assistant", "user"]
        assert FAILURE_LINE not in lines.lines
        session.shutdown()
        await closed.wait()
        await asyncio.sleep(0.1)
    assert failed == []
    assert [end.reason for end in ends] == [ENDED]
    assert [m["role"] for m in ends[0].transcript] == ["user", "assistant", "user"]


async def test_the_fourth_voice_failure_in_a_row_closes_the_call_as_an_error() -> None:
    voice = FailingVoice()
    ends: list[CallEnd] = []
    model = ScriptedLLM(*(f"Answer {n}." for n in range(5)))
    agent = UnmuteAgent(AgentSpec("Help."), tts=voice, call=CallHooks(hooks=on_end(ends.append)))
    closed = asyncio.Event()
    async with AgentSession(llm=model, tts=voice, conn_options=NO_RETRY) as session:
        session.output.audio = Silence()
        session.on("close", lambda _event: closed.set())
        await session.start(agent)
        for turn in range(3):
            await session.run(user_input=f"Turn {turn}.")
            await asyncio.sleep(0.1)
            assert not closed.is_set()
        await session.run(user_input="Turn 3.")
        async with asyncio.timeout(3):
            await closed.wait()
        await asyncio.sleep(0.1)
    assert [end.reason for end in ends] == [ERROR]


async def test_the_dev_records_show_a_reply_the_voice_cut_short() -> None:
    voice = FailingVoice(speaks=1)
    records = Records()
    model = ScriptedLLM("Your order is on its way. It arrives on Monday.")
    agent = UnmuteAgent(AgentSpec("Help."), tts=voice)
    async with AgentSession(llm=model, tts=voice, conn_options=NO_RETRY) as session:
        session.output.audio = Silence()
        session.output.transcription = Lines()
        DevObserver(records.recorder).attach(session)
        await session.start(agent)
        await session.run(user_input="Where is my order?")
        await asyncio.sleep(0.1)
    latest = records.latest()
    [reply] = records.ids("exchange", role="response")
    assert (latest["exchange", reply]["state"], latest["exchange", reply]["playback"]) == (
        "interrupted",
        "unknown",
    )
    [answer] = [t for t in records.of("text") if t["speaker"] == "assistant"]
    assert (answer["text"], answer["state"]) == (
        "Your order is on its way. It arrives on Monday.",
        "incomplete",
    )


@pytest.mark.parametrize(
    ("voice_works", "asked", "looked_up"), [(True, 1, True), (False, 2, False)]
)
async def test_a_confirm_line_the_voice_failed_to_say_is_asked_again(
    voice_works: bool, asked: int, looked_up: bool
) -> None:
    payload = job(mode="status")
    payload["tasks"][0]["finish"][0]["confirm"] = ["Shall I close order {{input.order_id}}?"]
    lookup = Canned(FOUND)
    tasks = plan(payload, tools=registry(lookup_order=lookup))
    voice = FailingVoice(speaks=100 if voice_works else 1)
    model = ScriptedLLM(
        call("find_order"),
        call("lookup_order", order_id="A-12"),
        call("lookup_order", order_id="A-12"),
    )
    agent = UnmuteAgent(AgentSpec("Orders.", tasks=tasks), state=state(tasks), tts=voice)
    lines = Lines()
    async with AgentSession(llm=model, tts=voice, conn_options=NO_RETRY) as session:
        session.output.audio = Silence()
        session.output.transcription = lines
        await session.start(agent)
        await session.run(user_input="Please close order A-12.")
        await session.run(user_input="Yes.")
    assert lines.lines.count("Shall I close order A-12?") == asked
    assert bool(lookup.calls) is looked_up
