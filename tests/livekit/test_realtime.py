"""A LiveKit session on a realtime model: fixed lines, refusals and fixed instructions."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from livekit.agents import NOT_GIVEN, AgentSession, llm
from livekit.agents.llm._realtime.openai import RealtimeModel

from tests.conformance.scenario import ScriptedModel
from tests.realtime.fake_server import FakeRealtimeServer
from tests.tasks.job import plan, state
from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks
from unmute_core.hooks import Hooks
from unmute_core.livekit import UnmuteAgent
from unmute_core.livekit._realtime import fixed_instructions, speak, speaks_itself
from unmute_core.realtime import say_line


class Model(llm.RealtimeModel):
    """A realtime model that only states what it can do."""

    def __init__(self, *, audio: bool = True, say: bool = False, mutable: bool = True) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=audio,
                manual_function_calls=True,
                mutable_chat_context=mutable,
                mutable_instructions=mutable,
                mutable_tools=mutable,
                supports_say=say,
            )
        )

    def session(self, *, turn_detection_disabled: bool = False) -> llm.RealtimeSession:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class Session:
    """Records whether a line went to ``say`` or to the model."""

    def __init__(self, model: object, tts: object = None) -> None:
        self.llm = model
        self.tts = tts
        self.current_agent = SimpleNamespace(llm=NOT_GIVEN, tts=NOT_GIVEN)
        self.said: list[str] = []
        self.generated: list[str] = []

    def say(self, text: str, **_options: object) -> str:
        self.said.append(text)
        return "said"

    def generate_reply(self, *, instructions: str, **_options: object) -> str:
        self.generated.append(instructions)
        return "generated"


def test_a_realtime_model_with_no_tts_says_a_fixed_line_itself() -> None:
    session = Session(Model())

    assert speaks_itself(cast(Any, session))
    assert speak(cast(Any, session), "One moment.") == "generated"
    assert session.generated == [say_line("One moment.")]


@pytest.mark.parametrize(
    "session",
    [
        Session(Model(), tts=object()),
        Session(Model(audio=False), tts=object()),
        Session(Model(say=True)),
        Session(object()),
    ],
    ids=["half-cascade", "text-and-tts", "model-can-say", "cascade"],
)
def test_a_fixed_line_goes_to_say_when_something_can_say_it(session: Session) -> None:
    assert not speaks_itself(cast(Any, session))
    assert speak(cast(Any, session), "One moment.") == "said"
    assert session.said == ["One moment."]


def test_the_agents_own_model_and_tts_come_before_the_sessions() -> None:
    session = Session(object(), tts=object())
    session.current_agent = SimpleNamespace(llm=Model(), tts=None)

    assert speaks_itself(cast(Any, session))


def test_a_live_model_fixes_its_instructions() -> None:
    assert fixed_instructions(cast(Any, Session(Model(mutable=False))))
    assert not fixed_instructions(cast(Any, Session(Model())))
    assert not fixed_instructions(cast(Any, Session(object())))


def test_save_values_is_refused_on_a_model_that_fixes_its_instructions() -> None:
    task_plan = plan({"mode": "record", "tasks": []})
    agent = UnmuteAgent(AgentSpec("Help.", save_values=True), state=state(task_plan))

    with pytest.raises(ValueError, match="save_values needs instructions that change"):
        agent._realtime.enter(Model(mutable=False))


def test_model_hooks_are_refused() -> None:
    hooks = Hooks()
    hooks.after_model(lambda *_args: None)
    task_plan = plan({"mode": "record", "tasks": []})
    call_state = state(task_plan)
    agent = UnmuteAgent(
        AgentSpec("Help."), state=call_state, call=CallHooks(call_state, hooks=hooks)
    )

    with pytest.raises(ValueError, match="the after_model hooks never run"):
        agent._realtime.enter(Model())


async def test_an_agent_with_tasks_closes_the_session_on_a_realtime_model() -> None:
    task_plan = plan()
    agent = UnmuteAgent(AgentSpec("Help.", tasks=task_plan), state=state(task_plan))
    async with FakeRealtimeServer(ScriptedModel()) as server:
        model = RealtimeModel(base_url=server.http_url, api_key="test", turn_detection=None)
        closed = asyncio.Event()
        try:
            async with AgentSession(llm=model) as session:
                session.on("close", lambda _event: closed.set())
                await session.start(agent)
                async with asyncio.timeout(5):
                    await closed.wait()
        finally:
            await model.aclose()


async def test_waiting_for_the_instructions_gives_up_on_a_resend_that_never_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unmute_core.livekit import _sync

    monkeypatch.setattr(_sync, "SETTLE_WAIT_SECONDS", 0.05)
    sync = object.__new__(_sync.RealtimeSync)
    never = asyncio.create_task(_forever())
    sync._instructing = {never}
    await asyncio.wait_for(sync.settled(), timeout=1)
    never.cancel()


async def _forever() -> None:
    await asyncio.Event().wait()
