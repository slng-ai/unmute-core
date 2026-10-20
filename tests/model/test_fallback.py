"""FallbackModelCall tries the next call only when that is safe.

The cases check behaviour only, with fake calls instead of LiveKit streams.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from unmute_core.model import (
    AnswerClock,
    AnswerStarted,
    FailOver,
    FallbackModelCall,
    MissingVariablesError,
    ModelEvent,
    ModelRequest,
    RequestRefusedError,
    Retry,
    TextDelta,
    ToolCall,
    Usage,
    answered_model,
    variable_key_max,
    with_answer_clock,
    with_attempts,
)
from unmute_core.model._fallback import SAME_MODEL_RETRIES_S

REQUEST = ModelRequest(messages=[{"role": "user", "content": "hi"}])


class StatusError(Exception):
    def __init__(
        self, status: int, headers: dict[str, str] | None = None, code: str | None = None
    ) -> None:
        super().__init__(f"status {status}")
        self.status_code = status
        self.code = code
        self.response = SimpleNamespace(headers=headers or {})


class FakeCall:
    """Plays one scripted outcome per attempt: an error, a delay, or a list of texts."""

    def __init__(
        self, *attempts: BaseException | float | list[str | BaseException | Usage]
    ) -> None:
        self.attempts = list(attempts)
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        outcome = self.attempts[min(self.calls, len(self.attempts) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, int | float):
            await asyncio.sleep(outcome)
            return
        for item in outcome:
            if isinstance(item, BaseException):
                raise item
            yield item if isinstance(item, Usage) else TextDelta(item)


async def collect(call: FallbackModelCall) -> list[str]:
    return [event.text async for event in call.stream(REQUEST) if isinstance(event, TextDelta)]


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry waits instead of waiting."""
    waited: list[float] = []

    async def fake_sleep(delay: float) -> None:
        waited.append(delay)

    monkeypatch.setattr("unmute_core.model._fallback.asyncio.sleep", fake_sleep)
    return waited


async def test_the_first_call_answers() -> None:
    first, second = FakeCall(["a", "b"]), FakeCall(["x"])
    assert await collect(FallbackModelCall([first, second])) == ["a", "b"]
    assert second.calls == 0


async def test_a_server_error_moves_to_the_next_call() -> None:
    first, second = FakeCall(StatusError(503)), FakeCall(["x"])
    assert await collect(FallbackModelCall([first, second])) == ["x"]
    assert first.calls == 1


@pytest.mark.parametrize("status", [400, 401, 402, 413, 422])
async def test_a_stop_status_never_reaches_the_next_call(status: int) -> None:
    second = FakeCall(["x"])
    with pytest.raises(StatusError):
        await collect(FallbackModelCall([FakeCall(StatusError(status)), second]))
    assert second.calls == 0


async def test_a_rate_limit_retries_the_same_call_after_retry_after(sleeps: list[float]) -> None:
    first = FakeCall(StatusError(429, {"Retry-After": "1"}), ["again"])
    second = FakeCall(["x"])
    assert await collect(FallbackModelCall([first, second])) == ["again"]
    assert (sleeps, first.calls, second.calls) == ([1.0], 2, 0)


async def test_a_rate_limit_retries_only_once(sleeps: list[float]) -> None:
    first = FakeCall(StatusError(429, {"Retry-After": "1"}))
    assert await collect(FallbackModelCall([first, FakeCall(["x"])])) == ["x"]
    assert (sleeps, first.calls) == ([1.0], 2)


NO_WAIT = [{"Retry-After": "0"}, {"Retry-After": "Thu, 01 Jan 1970 00:00:00 GMT"}]


@pytest.mark.parametrize("headers", NO_WAIT, ids=["zero", "past date"])
async def test_a_retry_after_for_no_wait_is_a_normal_retry(
    sleeps: list[float], headers: dict[str, str]
) -> None:
    events: list[Retry] = []
    busy = StatusError(429, headers)
    first = FakeCall(busy, busy, ["again"])
    chain = FallbackModelCall(
        [first, FakeCall(["x"])], same_model_retries=SAME_MODEL_RETRIES_S, on_retry=events.append
    )
    assert await collect(chain) == ["again"]
    assert (sleeps, first.calls, [(e.attempt, e.waited) for e in events]) == (
        [0.25, 0.75],
        3,
        [(1, False), (2, False)],
    )


@pytest.mark.parametrize("headers", NO_WAIT, ids=["zero", "past date"])
async def test_a_retry_after_for_no_wait_fails_over_when_there_are_no_retries(
    sleeps: list[float], headers: dict[str, str]
) -> None:
    first = FakeCall(StatusError(429, headers))
    chain = FallbackModelCall([first, FakeCall(["x"])], max_wait_s=0.0)
    assert await collect(chain) == ["x"]
    assert (sleeps, first.calls) == ([], 1)


@pytest.mark.parametrize("headers", [{"Retry-After": "30"}, {}])
async def test_a_long_or_missing_retry_after_fails_over(
    sleeps: list[float], headers: dict[str, str]
) -> None:
    first = FakeCall(StatusError(429, headers))
    assert await collect(FallbackModelCall([first, FakeCall(["x"])], max_wait_s=2.0)) == ["x"]
    assert (sleeps, first.calls) == ([], 1)


async def test_a_rate_limit_without_retry_after_retries_the_same_call_when_asked(
    sleeps: list[float],
) -> None:
    first = FakeCall(StatusError(429), StatusError(429), ["again"])
    chain = FallbackModelCall([first, FakeCall(["x"])], same_model_retries=(0, 0))
    assert await collect(chain) == ["again"]
    assert (sleeps, first.calls) == ([0, 0], 3)
    dead = FakeCall(StatusError(429))
    chain = FallbackModelCall([dead, FakeCall(["x"])], same_model_retries=(0, 0))
    assert await collect(chain) == ["x"]
    assert dead.calls == 3


@pytest.mark.parametrize(
    "error",
    [RequestRefusedError("bad option"), MissingVariablesError(["name"])],
    ids=["refused", "missing placeholder"],
)
async def test_a_refused_request_fails_over_at_once_with_no_retry(
    sleeps: list[float], error: RequestRefusedError
) -> None:
    moves: list[FailOver] = []
    retries: list[Retry] = []
    refused = FakeCall(error)
    chain = FallbackModelCall(
        [refused, FakeCall(["x"])],
        same_model_retries=SAME_MODEL_RETRIES_S,
        on_fail_over=moves.append,
        on_retry=retries.append,
    )
    assert await collect(chain) == ["x"]
    assert (refused.calls, sleeps, retries) == (1, [], [])
    assert [(move.status, move.error) for move in moves] == [(None, error)]


async def test_a_refused_request_on_the_last_call_is_raised(sleeps: list[float]) -> None:
    chain = FallbackModelCall(
        [FakeCall(RequestRefusedError("bad option"))], same_model_retries=SAME_MODEL_RETRIES_S
    )
    with pytest.raises(RequestRefusedError, match="bad option"):
        await collect(chain)
    assert sleeps == []


async def test_any_other_error_with_no_status_is_still_short_lived(sleeps: list[float]) -> None:
    first = FakeCall(ValueError("provider broke"), ["again"])
    chain = FallbackModelCall([first, FakeCall(["x"])], same_model_retries=SAME_MODEL_RETRIES_S)
    assert await collect(chain) == ["again"]
    assert (first.calls, sleeps) == (2, [0.25])


async def test_a_slow_first_event_fails_over() -> None:
    call = FallbackModelCall([FakeCall(5.0), FakeCall(["x"])], first_event_timeout_s=0.01)
    assert await collect(call) == ["x"]


async def test_an_empty_answer_fails_over() -> None:
    assert await collect(FallbackModelCall([FakeCall([]), FakeCall(["x"])])) == ["x"]


async def test_an_empty_answer_from_the_last_call_is_empty() -> None:
    assert await collect(FallbackModelCall([FakeCall([])])) == []


async def test_an_error_after_the_answer_started_is_raised() -> None:
    second = FakeCall(["x"])
    call = FallbackModelCall([FakeCall(["a", StatusError(503)]), second])
    seen: list[str] = []
    with pytest.raises(StatusError):
        async for event in call.stream(REQUEST):
            assert isinstance(event, TextDelta)
            seen.append(event.text)
    assert (seen, second.calls) == (["a"], 0)


async def test_the_last_error_is_raised_when_every_call_fails() -> None:
    with pytest.raises(StatusError, match="status 502"):
        await collect(FallbackModelCall([FakeCall(StatusError(503)), FakeCall(StatusError(502))]))


async def test_fallbacks_nest() -> None:
    inner = FallbackModelCall([FakeCall(StatusError(503)), FakeCall(StatusError(500))])
    assert await collect(FallbackModelCall([inner, FakeCall(["x"])])) == ["x"]


@pytest.mark.parametrize("error", [StatusError(503), StatusError(408), ConnectionError("reset")])
async def test_a_short_lived_failure_retries_the_same_call_when_asked(
    sleeps: list[float], error: BaseException
) -> None:
    first = FakeCall(error, error, ["same model"])
    second = FakeCall(["next model"])
    call = FallbackModelCall([first, second], same_model_retries=SAME_MODEL_RETRIES_S)
    assert await collect(call) == ["same model"]
    assert (first.calls, second.calls, sleeps) == (3, 0, [0.25, 0.75])


async def test_after_its_retries_the_call_fails_over(sleeps: list[float]) -> None:
    first = FakeCall(StatusError(503))
    second = FakeCall(["next model"])
    call = FallbackModelCall([first, second], same_model_retries=SAME_MODEL_RETRIES_S)
    assert await collect(call) == ["next model"]
    assert (first.calls, sleeps) == (3, [0.25, 0.75])


async def test_a_client_error_never_retries_the_same_call(sleeps: list[float]) -> None:
    first = FakeCall(StatusError(404))
    call = FallbackModelCall(
        [first, FakeCall(["next model"])], same_model_retries=SAME_MODEL_RETRIES_S
    )
    assert await collect(call) == ["next model"]
    assert (first.calls, sleeps) == (1, [])


async def test_without_retries_a_server_error_fails_over_at_once(sleeps: list[float]) -> None:
    first = FakeCall(StatusError(503), ["same model"])
    assert await collect(FallbackModelCall([first, FakeCall(["next model"])])) == ["next model"]
    assert (first.calls, sleeps) == (1, [])


async def test_a_retry_that_the_budget_cannot_hold_is_skipped() -> None:
    first = FakeCall(0.2, ["same model"])
    call = FallbackModelCall(
        [first, FakeCall(["next model"])],
        first_event_timeout_s=0.1,
        same_model_retries=SAME_MODEL_RETRIES_S,
    )
    assert await collect(call) == ["next model"]
    assert first.calls == 1


class CodeError(Exception):
    """An error with a status and a text code, like the OpenAI SDK's ``APIStatusError``."""

    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status
        self.code = code


class StoppingCall(FakeCall):
    stop_codes = frozenset({"project_mode_forbidden"})


async def test_a_stop_code_a_call_names_never_reaches_the_next_call() -> None:
    second = FakeCall(["x"])
    refusal = CodeError(403, "project_mode_forbidden")
    with pytest.raises(CodeError):
        await collect(FallbackModelCall([StoppingCall(refusal), second]))
    assert second.calls == 0
    assert await collect(FallbackModelCall([FakeCall(refusal), second])) == ["x"]


class KeyLimited(FakeCall):
    variable_key_max = 64


def test_the_chain_takes_the_tightest_key_limit_of_its_calls() -> None:
    assert variable_key_max(FallbackModelCall([KeyLimited(["x"]), FakeCall(["y"])])) == 64
    assert variable_key_max(FallbackModelCall([FakeCall(["y"])])) is None


async def test_usage_alone_is_an_empty_answer_and_usage_after_text_comes_through() -> None:
    usage = Usage(10, 2)
    second = FakeCall(["x", usage])
    events = [
        event async for event in FallbackModelCall([FakeCall([usage]), second]).stream(REQUEST)
    ]
    assert events == [TextDelta("x"), usage]
    last = [event async for event in FallbackModelCall([FakeCall([usage])]).stream(REQUEST)]
    assert last == [usage]


async def test_usage_before_the_answer_comes_after_it() -> None:
    usage = Usage(10, 2)
    events = [
        event async for event in FallbackModelCall([FakeCall([usage, "a", "b"])]).stream(REQUEST)
    ]
    assert events == [TextDelta("a"), TextDelta("b"), usage]
    second = FakeCall(["x"])
    failing = FakeCall([usage, StatusError(503)])
    assert await collect(FallbackModelCall([failing, second])) == ["x"]


class Named(FakeCall):
    def __init__(
        self, model: str, *attempts: BaseException | float | list[str | BaseException | Usage]
    ) -> None:
        super().__init__(*attempts)
        self.model = model


async def test_a_custom_stop_status_stops_the_chain() -> None:
    first, second = FakeCall(StatusError(404)), FakeCall(["b"])
    chain = FallbackModelCall([first, second], stop_statuses={404})
    with pytest.raises(StatusError):
        await collect(chain)
    assert second.calls == 0


async def test_a_custom_stop_code_stops_the_chain() -> None:
    error = StatusError(403, code="blocked")
    first, second = FakeCall(error), FakeCall(["b"])
    with pytest.raises(StatusError):
        await collect(FallbackModelCall([first, second], stop_codes={"blocked"}))
    assert second.calls == 0


async def test_on_fail_over_gets_one_event_per_move_in_order() -> None:
    boom = StatusError(503)
    calls = [Named("a", boom), Named("b", []), Named("c", ["ok"])]
    events: list[FailOver] = []
    assert await collect(FallbackModelCall(calls, on_fail_over=events.append)) == ["ok"]
    assert events == [
        FailOver(index=0, failed="a", next="b", status=503, error=boom),
        FailOver(index=1, failed="b", next="c", status=None, error=None),
    ]


async def test_on_retry_gets_one_event_per_pause_before_it(sleeps: list[float]) -> None:
    busy = StatusError(429, {"retry-after": "1"})
    down = StatusError(503)
    calls = [Named("a", busy, down, ["ok"])]
    events: list[Retry] = []
    chain = FallbackModelCall(
        calls, same_model_retries=SAME_MODEL_RETRIES_S, on_retry=events.append
    )
    assert await collect(chain) == ["ok"]
    assert events == [
        Retry(index=0, model="a", attempt=1, pause_s=1.0, waited=True, status=429, error=busy),
        Retry(index=0, model="a", attempt=2, pause_s=0.25, waited=False, status=503, error=down),
    ]
    assert sleeps == [1.0, 0.25]


async def test_a_retry_callback_that_raises_does_not_break_the_chain(sleeps: list[float]) -> None:
    def broken(event: Retry) -> None:
        raise RuntimeError("bug")

    chain = FallbackModelCall(
        [FakeCall(StatusError(503), ["ok"])], same_model_retries=(0.25,), on_retry=broken
    )
    assert await collect(chain) == ["ok"]


async def test_a_callback_that_raises_does_not_break_the_chain() -> None:
    def broken(event: FailOver) -> None:
        raise RuntimeError("bug")

    chain = FallbackModelCall([FakeCall(StatusError(503)), FakeCall(["ok"])], on_fail_over=broken)
    assert await collect(chain) == ["ok"]


async def test_sticky_starts_each_request_with_the_call_that_answered_last() -> None:
    dead = FakeCall(StatusError(503))
    chain = FallbackModelCall([dead, FakeCall(["x"])], sticky=True)
    for _ in range(3):
        assert await collect(chain) == ["x"]
    assert dead.calls == 1


async def test_without_sticky_a_dead_first_call_is_tried_every_time() -> None:
    dead = FakeCall(StatusError(503))
    chain = FallbackModelCall([dead, FakeCall(["x"])])
    for _ in range(3):
        assert await collect(chain) == ["x"]
    assert dead.calls == 3


async def test_sticky_tries_the_earlier_calls_after_the_active_one() -> None:
    calls = [
        Named("a", StatusError(503), ["a"]),
        Named("b", StatusError(503)),
        Named("c", ["c"], StatusError(503)),
    ]
    events: list[FailOver] = []
    chain = FallbackModelCall(calls, sticky=True, on_fail_over=events.append)
    assert await collect(chain) == ["c"]
    events.clear()
    assert await collect(chain) == ["a"]
    assert [(e.failed, e.next) for e in events] == [("c", "a")]
    assert chain.model == "a"


async def test_answered_names_the_call_that_answered() -> None:
    chain = FallbackModelCall([Named("a", StatusError(503)), Named("b", ["x"])])
    assert (chain.model, chain.answered) == ("a", "a")
    await collect(chain)
    assert (chain.answered, answered_model(chain)) == ("b", "b")


async def test_without_sticky_the_next_request_still_starts_with_the_first_call() -> None:
    a = Named("a", StatusError(503), ["x"])
    chain = FallbackModelCall([a, Named("b", ["y"])])
    await collect(chain)
    assert (chain.model, chain.answered) == ("a", "b")
    await collect(chain)
    assert (chain.model, chain.answered) == ("a", "a")


async def test_with_sticky_the_next_request_starts_with_the_call_that_answered() -> None:
    chain = FallbackModelCall([Named("a", StatusError(503)), Named("b", ["x"])], sticky=True)
    await collect(chain)
    assert (chain.model, chain.answered) == ("b", "b")


def test_answered_model_names_a_plain_call_by_its_model() -> None:
    assert answered_model(Named("a", ["x"])) == "a"


async def test_aclose_closes_each_call_that_can_close() -> None:
    closable = SimpleNamespace(closed=0, stream=FakeCall().stream)

    async def aclose() -> None:
        closable.closed += 1

    closable.aclose = aclose
    await FallbackModelCall([closable, FakeCall()]).aclose()  # ty: ignore[invalid-argument-type]
    assert closable.closed == 1


def test_a_chain_retries_itself() -> None:
    assert FallbackModelCall([FakeCall()]).retries_itself is True


class Tracked:
    """Streams texts, then waits forever, and records when its stream is closed."""

    def __init__(self, *texts: str) -> None:
        self.texts = texts
        self.closed = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        try:
            for text in self.texts:
                yield TextDelta(text)
            await asyncio.Event().wait()
        finally:
            self.closed.set()


async def test_a_reader_that_stops_early_closes_the_calls_stream() -> None:
    call = Tracked("Your table is booked")
    stream = FallbackModelCall([call]).stream(REQUEST)
    assert await anext(stream) == TextDelta("Your table is booked")
    await stream.aclose()
    assert call.closed.is_set()


async def test_an_empty_answer_can_end_the_request_instead_of_failing_over() -> None:
    moves: list[FailOver] = []
    usage = Usage(10, 0)
    first, second = FakeCall([usage]), FakeCall(["x"])
    chain = FallbackModelCall([first, second], empty_fails_over=False, on_fail_over=moves.append)
    assert [event async for event in chain.stream(REQUEST)] == [usage]
    assert (second.calls, moves) == (0, [])


@pytest.mark.parametrize(
    ("answer", "moves_on"),
    [([Usage(10, 0)], False), ([], True)],
    ids=["usage only", "no event"],
)
async def test_a_usage_can_count_as_an_empty_answer(answer: list, moves_on: bool) -> None:
    moves: list[FailOver] = []
    first, second = FakeCall(answer), FakeCall(["x"])
    chain = FallbackModelCall(
        [first, second], usage_counts_as_answer=True, on_fail_over=moves.append
    )
    expected = (["x"], 1, 1) if moves_on else ([], 0, 0)
    assert (await collect(chain), second.calls, len(moves)) == expected


class SlowToolCall:
    """Starts its answer at once, then streams one tool call for ``seconds``."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield AnswerStarted(response_id="r1")
        await asyncio.sleep(self.seconds)
        yield ToolCall("t1", "book", "{}")


@pytest.mark.parametrize(
    ("start_ends_budget", "first", "moves_on"),
    [(True, ToolCall("t1", "book", "{}"), False), (False, TextDelta("next"), True)],
    ids=["ends at the start", "ends at the answer"],
)
async def test_the_start_of_an_answer_can_end_the_first_event_budget(
    start_ends_budget: bool, first: ModelEvent, moves_on: bool
) -> None:
    moves: list[FailOver] = []
    chain = FallbackModelCall(
        [SlowToolCall(0.2), FakeCall(["next"])],
        first_event_timeout_s=0.05,
        start_ends_budget=start_ends_budget,
        on_fail_over=moves.append,
    )
    events = [event async for event in chain.stream(REQUEST)]
    assert (events, len(moves)) == ([first], int(moves_on))


async def test_a_call_that_starts_then_fails_still_fails_over() -> None:
    class StartsThenFails:
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            yield AnswerStarted()
            raise RuntimeError("dropped")

    chain = FallbackModelCall(
        [StartsThenFails(), FakeCall(["next"])], first_event_timeout_s=1.0, start_ends_budget=True
    )
    assert await collect(chain) == ["next"]


class StartsThen:
    """Starts its answer at once, then plays ``rest``: events, or an error to raise."""

    def __init__(self, *rest: ModelEvent | BaseException) -> None:
        self.rest = rest
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.calls += 1
        yield AnswerStarted(response_id="r1")
        for item in self.rest:
            if isinstance(item, BaseException):
                raise item
            yield item


def watched(*calls: Any, **options: Any) -> tuple[FallbackModelCall, list[object]]:
    """Return a chain with both callbacks and the list they fill."""
    seen: list[object] = []
    chain = FallbackModelCall(
        list(calls),
        on_fail_over=seen.append,
        on_retry=seen.append,
        **options,
    )
    return chain, seen


async def test_the_start_of_an_answer_ends_the_budget_when_it_is_the_answer() -> None:
    chain, seen = watched(
        SlowToolCall(0.2), FakeCall(["next"]), first_event_timeout_s=0.05, start_is_answer=True
    )
    events = [event async for event in chain.stream(REQUEST)]
    assert (events, seen) == ([AnswerStarted(response_id="r1"), ToolCall("t1", "book", "{}")], [])


async def test_the_start_of_an_answer_is_passed_on_once_when_it_is_the_answer() -> None:
    chain, _ = watched(StartsThen(TextDelta("Hi"), Usage(1, 2)), start_is_answer=True)
    assert [event async for event in chain.stream(REQUEST)] == [
        AnswerStarted(response_id="r1"),
        TextDelta("Hi"),
        Usage(1, 2),
    ]


async def test_the_start_of_an_answer_is_not_passed_on_by_default() -> None:
    chain, _ = watched(StartsThen(TextDelta("Hi")), start_ends_budget=True)
    assert [event async for event in chain.stream(REQUEST)] == [TextDelta("Hi")]


@pytest.mark.parametrize("retries", [(), (0.0,)], ids=["no retries", "one retry"])
async def test_an_error_after_the_start_is_raised_without_retry_or_fail_over(
    retries: tuple[float, ...], sleeps: list[float]
) -> None:
    first, second = StartsThen(StatusError(503)), FakeCall(["next"])
    chain, seen = watched(first, second, start_is_answer=True, same_model_retries=retries)
    stream = chain.stream(REQUEST)
    events = [await anext(stream)]
    with pytest.raises(StatusError):
        await anext(stream)
    assert (events, first.calls, second.calls, seen, sleeps) == (
        [AnswerStarted(response_id="r1")],
        1,
        0,
        [],
        [],
    )


async def test_an_error_before_the_start_still_fails_over_when_the_start_is_the_answer() -> None:
    first, second = FakeCall(StatusError(503)), StartsThen(TextDelta("next"))
    chain, seen = watched(first, second, start_is_answer=True)
    events = [event async for event in chain.stream(REQUEST)]
    assert events == [AnswerStarted(response_id="r1"), TextDelta("next")]
    assert [type(item) for item in seen] == [FailOver]


@pytest.mark.parametrize(
    "options",
    [{}, {"usage_counts_as_answer": True}, {"empty_fails_over": True}],
    ids=["default", "usage counts", "empty fails over"],
)
async def test_a_start_then_nothing_is_an_empty_answer_of_that_call(
    options: dict[str, bool],
) -> None:
    first, second = StartsThen(), FakeCall(["next"])
    chain, seen = watched(first, second, start_is_answer=True, **options)
    events = [event async for event in chain.stream(REQUEST)]
    assert (events, second.calls, seen, chain.answered) == (
        [AnswerStarted(response_id="r1")],
        0,
        [],
        "unknown",
    )


async def test_a_start_is_not_passed_on_by_a_chain_that_does_not_set_the_option() -> None:
    inner = FallbackModelCall([StartsThen(TextDelta("Hi"))], start_is_answer=True)
    outer = FallbackModelCall([inner, FakeCall(["next"])])
    assert [event async for event in outer.stream(REQUEST)] == [TextDelta("Hi")]


async def test_with_attempts_sees_each_attempt_when_the_start_is_the_answer() -> None:
    tried: list[object] = []

    async def stream(call, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        tried.append(call)
        async for event in call.stream(request):
            yield event

    first, second = FakeCall(RuntimeError("down")), StartsThen(TextDelta("ok"))
    chain = FallbackModelCall([first, second], start_is_answer=True)
    events = [event async for event in with_attempts(chain.stream(REQUEST), stream)]
    assert (events, tried) == ([AnswerStarted(response_id="r1"), TextDelta("ok")], [first, second])


async def test_with_attempts_runs_each_attempt_of_a_chain_inside_through_its_stream() -> None:
    tried: list[object] = []

    async def stream(call, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        tried.append(call)
        async for event in call.stream(request):
            yield event

    first, second = FakeCall(RuntimeError("down")), FakeCall(["ok"])
    chain = FallbackModelCall([first, second])
    events = [event async for event in with_attempts(chain.stream(REQUEST), stream)]
    assert (events, tried) == ([TextDelta("ok")], [first, second])
    assert [event async for event in FallbackModelCall([FakeCall(["ok"])]).stream(REQUEST)] == [
        TextDelta("ok")
    ]
    assert tried == [first, second]


class Timed:
    """Plays one outcome per attempt, and notes when each attempt began and ended."""

    def __init__(self, *outcomes: BaseException | str) -> None:
        self.outcomes = list(outcomes)
        self.began: list[float] = []
        self.ended: list[float] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        outcome = self.outcomes[min(len(self.began), len(self.outcomes) - 1)]
        self.began.append(time.perf_counter())
        try:
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome:
                yield TextDelta(outcome)
        finally:
            self.ended.append(time.perf_counter())


async def _answered(call: Any) -> AnswerClock:
    clock = AnswerClock()
    chain = call if isinstance(call, FallbackModelCall) else FallbackModelCall([call])
    [_ async for _ in with_answer_clock(chain.stream(REQUEST), clock)]
    return clock


async def test_the_answer_clock_starts_at_the_call_that_answered() -> None:
    first, second = Timed(StatusError(503)), Timed("ok")
    clock = await _answered(FallbackModelCall([first, second]))
    assert clock.started is not None
    assert first.ended[0] <= clock.started <= second.began[0]


async def test_the_answer_clock_starts_at_the_retry_that_answered(sleeps: list[float]) -> None:
    call = Timed(StatusError(503), "ok")
    clock = await _answered(FallbackModelCall([call], same_model_retries=(0.1,)))
    assert sleeps == [0.1]
    assert clock.started is not None
    assert call.ended[0] <= clock.started <= call.began[1]


async def test_the_answer_clock_starts_at_the_innermost_call_that_answered() -> None:
    first, second = Timed(StatusError(503)), Timed("ok")
    clock = await _answered(FallbackModelCall([FallbackModelCall([first, second])]))
    assert clock.started is not None
    assert first.ended[0] <= clock.started <= second.began[0]


async def test_the_answer_clock_stays_empty_when_no_call_answered() -> None:
    clock = await _answered(FallbackModelCall([Timed(StatusError(503)), Timed("")]))
    assert clock.started is None
    with pytest.raises(StatusError):
        await _answered(FallbackModelCall([Timed(StatusError(503))]))


async def test_a_chain_outside_an_answer_clock_runs_as_before() -> None:
    assert await collect(FallbackModelCall([Timed("ok")])) == ["ok"]
