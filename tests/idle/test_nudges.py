"""``IdleNudgeController`` says each line after its own quiet, then hangs up."""

import asyncio

import pytest
from pydantic import ValidationError

from unmute_core.idle import GENERATED_NUDGE_INSTRUCTIONS, IdleNudgeController, IdleNudges

CONFIG = IdleNudges(
    first_nudge_delay_seconds=15,
    second_nudge_delay_seconds=30,
    hangup_delay_seconds=15,
    first_nudge_text="First nudge",
    second_nudge_text="Second nudge",
    final_hangup_text="Final hangup",
)


async def _drain(iterations: int = 5) -> None:
    for _ in range(iterations):
        await asyncio.sleep(0)


class Call:
    """The lines said, the waits taken and the hang-ups of one controller."""

    def __init__(self, *, gated: bool = False, config: IdleNudges = CONFIG) -> None:
        self.sleeps: list[float] = []
        self.gates: list[asyncio.Event] = []
        self.spoken: list[str] = []
        self.events: list[tuple[str, dict | None]] = []
        self.hung_up = 0
        self.gated = gated
        self.controller = IdleNudgeController(
            config,
            speak=self.speak,
            hangup=self.hangup,
            on_event=lambda event, data: self.events.append((event, data)),
            sleep=self.sleep,
            generate=self.generate,
        )

    async def generate(self, instructions: str) -> bool:
        self.spoken.append(f"model: {instructions}")
        return False

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self.gated:
            gate = asyncio.Event()
            self.gates.append(gate)
            await gate.wait()

    async def speak(self, text: str) -> bool:
        self.spoken.append(text)
        return False

    async def hangup(self) -> None:
        self.hung_up += 1


async def test_the_three_lines_follow_their_waits_then_the_call_ends() -> None:
    call = Call()

    call.controller.on_agent_state_changed("listening", "thinking")
    await _drain()
    assert call.sleeps == []
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()

    assert call.sleeps == [15, 30, 15]
    assert call.spoken == ["First nudge", "Second nudge", "Final hangup"]
    assert call.hung_up == 1
    assert call.events[-1] == ("idle_nudges.hangup", None)


async def test_a_partial_transcript_resets_before_the_first_nudge() -> None:
    call = Call(gated=True)
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    call.controller.on_user_transcript("yeah")
    call.gates[0].set()
    await _drain()

    assert call.spoken == []
    call.controller.close()


async def test_suspend_cancels_the_wait_and_resume_starts_it_again() -> None:
    call = Call(gated=True)
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    assert call.sleeps == [15]

    call.controller.suspend("webhook")
    await _drain()
    call.gates[0].set()
    await _drain()
    assert call.spoken == []
    assert "idle_nudges.suspended" in [event for event, _ in call.events]

    call.controller.resume("webhook")
    await _drain()
    assert call.sleeps == [15, 15]
    call.gates[1].set()
    await _drain()
    assert call.spoken == ["First nudge"]
    call.controller.close()


async def test_nested_suspends_resume_only_after_the_last_release() -> None:
    call = Call(gated=True)
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    call.controller.suspend("tool_one")
    call.controller.suspend("tool_two")
    await _drain()
    call.gates[0].set()
    await _drain()

    call.controller.resume("tool_one")
    await _drain()
    assert call.sleeps == [15]
    call.controller.resume("tool_two")
    await _drain()
    assert call.sleeps == [15, 15]
    call.gates[1].set()
    await _drain()
    assert call.spoken == ["First nudge"]
    call.controller.close()


async def test_a_transcript_during_a_suspend_leaves_no_stale_nudge() -> None:
    call = Call(gated=True)
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    call.controller.suspend("webhook")
    call.controller.on_user_transcript("actually")
    call.gates[0].set()
    await _drain()
    assert call.spoken == []

    call.controller.resume("webhook")
    await _drain()
    assert call.sleeps == [15, 15]
    call.gates[1].set()
    await _drain()
    assert call.spoken == ["First nudge"]
    call.controller.close()


async def test_the_caller_cutting_into_the_final_line_cancels_the_hangup() -> None:
    call = Call()

    async def speak(text: str) -> bool:
        if text == "Final hangup":
            call.controller.on_user_transcript("give me a second")
            return True
        return False

    call.controller = IdleNudgeController(
        CONFIG, speak=speak, hangup=call.hangup, sleep=call.sleep, generate=call.generate
    )
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()

    assert call.sleeps == [15, 30, 15]
    assert call.hung_up == 0


async def test_a_disabled_setting_never_speaks() -> None:
    call = Call()
    call.controller = IdleNudgeController(
        CONFIG.model_copy(update={"enabled": False}), speak=call.speak, hangup=call.hangup
    )
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    assert call.spoken == []


async def test_a_nudge_with_no_text_tells_the_model_the_instruction_it_is_given() -> None:
    call = Call(
        config=IdleNudges(nudge_instruction="Pregunta si sigue ahí.", hangup_delay_seconds=None)
    )
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    assert call.spoken[0] == "model: Pregunta si sigue ahí."


async def test_a_nudge_with_no_text_is_worded_by_the_model() -> None:
    call = Call(config=IdleNudges(second_nudge_text="Still there?"))
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    assert call.spoken == [f"model: {GENERATED_NUDGE_INSTRUCTIONS}", "Still there?"]
    assert ("idle_nudges.first.fired", {"generated": True}) in call.events
    assert call.hung_up == 1


async def test_with_no_second_nudge_the_hangup_wait_follows_the_first() -> None:
    config = IdleNudges(
        first_nudge_text="Hello?", second_nudge_delay_seconds=None, hangup_delay_seconds=30
    )
    call = Call(config=config)
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    assert call.sleeps == [15, 30]
    assert call.spoken == ["Hello?"]
    assert "idle_nudges.final_message.fired" not in [event for event, _ in call.events]
    assert call.hung_up == 1


def test_a_second_text_needs_its_wait_and_a_nudge_with_no_text_needs_a_generator() -> None:
    with pytest.raises(ValidationError, match="a second nudge text needs a second nudge delay"):
        IdleNudges(second_nudge_delay_seconds=None, second_nudge_text="b")
    with pytest.raises(ValueError, match="pass generate"):
        IdleNudgeController(IdleNudges(), speak=Call().speak, hangup=Call().hangup)


async def test_nudge_only_never_hangs_up() -> None:
    config = IdleNudges(
        first_nudge_text="Hello?", second_nudge_text="Still there?", hangup_delay_seconds=None
    )
    call = Call(config=config)
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    assert call.sleeps == [15, 30]
    assert call.spoken == ["Hello?", "Still there?"]
    assert call.hung_up == 0


async def test_hang_up_only_ends_the_call_after_the_quiet_with_no_nudge() -> None:
    config = IdleNudges(
        first_nudge_delay_seconds=None, second_nudge_delay_seconds=None, hangup_delay_seconds=20
    )
    call = Call(config=config)
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    assert call.sleeps == [20]
    assert call.spoken == []
    assert call.hung_up == 1


async def test_hang_up_only_starts_over_when_the_caller_speaks() -> None:
    config = IdleNudges(
        first_nudge_delay_seconds=None, second_nudge_delay_seconds=None, hangup_delay_seconds=20
    )
    call = Call(gated=True, config=config)
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()
    call.controller.on_user_transcript("hi")
    call.gates[0].set()
    await _drain()
    assert call.hung_up == 0


def test_idle_modes_refuse_lines_with_no_wait() -> None:
    with pytest.raises(ValidationError, match="use enabled=False"):
        IdleNudges(first_nudge_delay_seconds=None, hangup_delay_seconds=None)
    with pytest.raises(ValidationError, match="no first nudge"):
        IdleNudges(first_nudge_delay_seconds=None)
    with pytest.raises(ValidationError, match="no first nudge"):
        IdleNudges(
            first_nudge_delay_seconds=None, second_nudge_delay_seconds=None, first_nudge_text="a"
        )
    with pytest.raises(ValidationError, match="final hang-up text needs a hang-up delay"):
        IdleNudges(hangup_delay_seconds=None, final_hangup_text="Bye")
    assert not IdleNudges(
        first_nudge_delay_seconds=None, second_nudge_delay_seconds=None
    ).worded_by_model


async def test_states_reported_during_a_generated_nudge_do_not_cancel_the_sequence() -> None:
    class Reporting(Call):
        async def generate(self, instructions: str) -> bool:
            self.spoken.append(f"model: {instructions}")
            self.controller.on_agent_state_changed("listening", "thinking")
            self.controller.on_agent_state_changed("thinking", "speaking")
            self.controller.on_agent_state_changed("speaking", "listening")
            return False

    call = Reporting(config=IdleNudges(second_nudge_text="Still there?"))
    call.controller.on_agent_state_changed("speaking", "listening")
    await _drain()

    assert call.spoken == [f"model: {GENERATED_NUDGE_INSTRUCTIONS}", "Still there?"]
    assert call.hung_up == 1
