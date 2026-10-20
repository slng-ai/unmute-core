"""Try several model calls in order, by the failover rules.

It holds no framework types. Its rules: a first-event budget per call, and no retry or
failover once the caller has part of the answer. A same-model retry of a short-lived failure is an
option, off by default: the SLNG Context Router already retries behind its own plan.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence, Set
from contextlib import asynccontextmanager
from enum import Enum, auto
from typing import Annotated, ClassVar

from pydantic import Field, SkipValidation, validate_call
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, named_errors, original_errors
from unmute_core.model._attempts import attempt_stream, note_answer
from unmute_core.model._call import (
    AnswerStarted,
    ModelCallArg,
    ModelEvent,
    ModelRequest,
    RequestRefusedError,
    Usage,
    close_model,
    model_name,
    variable_key_max,
)
from unmute_core.model._failover import (
    STOP_STATUSES,
    Next,
    error_code,
    http_status,
    next_step,
    retry_delay,
)

logger = logging.getLogger(__name__)

FIRST_EVENT_TIMEOUT_S = 45.0
"""A first-event budget that clears slow reasoning models, which think before they answer.
Thinking models took up to 33 s to start on a call simulator."""
SAME_MODEL_RETRIES_S = (0.25, 0.75)
"""Suggested pauses before each same-model retry. A proxy that still routes to a stopping
upstream fails again at once, so a short pause lets it pick a healthy one."""
MIN_RETRY_BUDGET_S = 0.5
"""A retry is skipped when less than this would be left of the first-event budget."""


def _short_lived(status: int | None) -> bool:
    """Return True for a failure the same model may not repeat: no status, 408, 429 or a 5xx."""
    return status is None or status in (408, 429) or status >= 500


class _Decision(Enum):
    """What follows a call that failed before it answered."""

    WAIT = auto()
    RETRY = auto()
    FAIL_OVER = auto()
    RAISE = auto()


@original_errors
@dataclass(frozen=True, config=VALUE)
class FailOver:
    """One move from a failed model call to the next.

    Attributes:
        index: The position of the call that failed.
        failed: The model name of the call that failed, or ``unknown``.
        next: The model name of the call tried next, or ``unknown``.
        status: The HTTP status of the failure, or None when there was none.
        error: The error the call raised, or None when it answered nothing.
    """

    index: int
    failed: str
    next: str
    status: int | None
    error: SkipValidation[BaseException | None]


@original_errors
@dataclass(frozen=True, config=VALUE)
class Retry:
    """One retry of the same model call after it failed before it answered.

    Attributes:
        index: The position of the call that failed and is tried again.
        model: The model name of that call, or ``unknown``.
        attempt: Which retry of this call this is, 1 for the first.
        pause_s: The seconds the chain waits before it sends the call again.
        waited: True when the pause is the error's ``Retry-After``, False when it is one of
            ``same_model_retries``.
        status: The HTTP status of the failure, or None when there was none.
        error: The error the call raised.
    """

    index: int
    model: str
    attempt: int
    pause_s: float
    waited: bool
    status: int | None
    error: SkipValidation[BaseException]


class FallbackModelCall:
    """A model call that moves to the next one when a call fails before it answers.

    It is optional. A framework's own fallback, like LiveKit's ``FallbackAdapter``, works
    without it. It is a ``ModelCall`` too, so it can nest. Build one per call: it remembers
    which call answered last.

    Attributes:
        retries_itself: True. The chain fails over, so an adapter must not retry the whole
            chain again.
    """

    retries_itself: ClassVar[bool] = True

    @named_errors("FallbackModelCall")
    @validate_call
    def __init__(
        self,
        calls: Annotated[Sequence[ModelCallArg], Field(min_length=1)],
        *,
        first_event_timeout_s: Annotated[float, Field(gt=0)] | None = None,
        max_wait_s: Annotated[float, Field(ge=0)] = 2.0,
        same_model_retries: Sequence[Annotated[float, Field(ge=0)]] = (),
        stop_statuses: Set[int] = STOP_STATUSES,
        stop_codes: Set[str] | None = None,
        on_fail_over: SkipValidation[Callable[[FailOver], None]] | None = None,
        on_retry: SkipValidation[Callable[[Retry], None]] | None = None,
        sticky: bool = False,
        empty_fails_over: bool = True,
        usage_counts_as_answer: bool = False,
        start_ends_budget: bool = False,
        start_is_answer: bool = False,
    ) -> None:
        """Keep the calls in the order they are tried.

        Args:
            calls: The model calls, first choice first.
            first_event_timeout_s: How long a call may take to start answering before the
                next one is tried, its same-model retries included. None waits as long as
                the call does. ``FIRST_EVENT_TIMEOUT_S`` clears slow reasoning models.
            max_wait_s: The longest ``Retry-After`` worth waiting for. A longer one fails over.
            same_model_retries: The pause before each retry of the same call after a
                short-lived failure (no status, 408, 429 or a 5xx). A
                ``RequestRefusedError`` is not one: it moves to the next call at once. A 429 whose
                ``Retry-After`` fits ``max_wait_s`` waits that long once instead.
                ``SAME_MODEL_RETRIES_S`` is the suggested setting. Empty moves to the next
                call at once.
            stop_statuses: The statuses that end the chain at once. ``STOP_STATUSES`` by default.
            stop_codes: The error codes that end the chain at once. None reads each call's
                own ``stop_codes``, like the router's ``project_mode_forbidden``.
            on_fail_over: Called at each move to the next call, with a ``FailOver``. An
                error it raises is logged and the chain goes on.
            on_retry: Called before each pause that sends the same call again, with a
                ``Retry``: a ``Retry-After`` wait or one of ``same_model_retries``. An error
                it raises is logged and the chain goes on.
            sticky: Start each request with the call that answered last, then the others in
                the given order, as LiveKit's ``FallbackAdapter(sticky=True)`` does. A dead
                first call is then tried once, not on every request.
            empty_fails_over: An answer with nothing in it moves to the next call, as a
                failure does. False ends the request there with the empty answer, for an app
                that handles an empty reply itself, like asking the same model again.
            usage_counts_as_answer: A call that sent its ``Usage`` and nothing else answered
                with an empty reply, which ends the request there. Only a call that sent no
                event at all moves on through ``empty_fails_over``. A provider that reports
                usage has finished its answer, while a stream with no event at all is often
                a broken response.
            start_ends_budget: End ``first_event_timeout_s`` at a call's ``AnswerStarted``,
                the provider's first chunk, instead of at its first text or tool call. A
                reply that is only a long tool call then counts as started at once.
                ``AnswerStarted`` is not passed on, unless ``start_is_answer`` is set.
            start_is_answer: Take a call's ``AnswerStarted`` as its answer. It implies
                ``start_ends_budget``. The chain passes that ``AnswerStarted`` on, once,
                for the call that answers, and stays with the call from then on. An error
                after it reaches the reader as it is: no retry, no failover, and no
                ``on_retry`` or ``on_fail_over`` call. A call that sends ``AnswerStarted``
                and nothing else ends the request with its own empty answer. So a long
                tool call that began at once is never sent again. A call that sends no
                ``AnswerStarted`` is judged as before, and ``empty_fails_over`` and
                ``usage_counts_as_answer`` apply to it only.

        Raises:
            ValueError: If ``calls`` is empty, or a wait is below zero.
        """
        self._calls = list(calls)
        self._first_event_timeout_s = first_event_timeout_s
        self._max_wait_s = max_wait_s
        self._same_model_retries = tuple(same_model_retries)
        self._stop_statuses = frozenset(stop_statuses)
        self._stop_codes = frozenset(
            stop_codes
            if stop_codes is not None
            else (code for call in calls for code in getattr(call, "stop_codes", ()))
        )
        self._on_fail_over = on_fail_over
        self._on_retry = on_retry
        self._sticky = sticky
        self._empty_fails_over = empty_fails_over
        self._usage_counts_as_answer = usage_counts_as_answer
        self._start_ends_budget = start_ends_budget
        self._start_is_answer = start_is_answer
        self._active = 0
        self._answered = 0

    @property
    def _next(self) -> object:
        """The call the next request starts with."""
        return self._calls[self._active if self._sticky else 0]

    @property
    def model(self) -> str:
        """The model name of the call the next request starts with."""
        return str(getattr(self._next, "model", "") or "")

    @property
    def provider(self) -> str:
        """Who serves the call the next request starts with."""
        return str(getattr(self._next, "provider", "") or "")

    @property
    def answered(self) -> str:
        """The model name of the call that answered last, the first before any answer."""
        return model_name(self._calls[self._answered])

    @property
    def variable_key_max(self) -> int | None:
        """The tightest placeholder key limit of the wrapped calls, or None for no limit."""
        return variable_key_max(*self._calls)

    async def aclose(self) -> None:
        """Close every call that has an ``aclose``."""
        await asyncio.gather(*(close_model(call) for call in self._calls))

    def _left(self, started: float) -> float | None:
        """Return what is left of a call's first-event budget, or None without one."""
        if self._first_event_timeout_s is None:
            return None
        return max(0.0, self._first_event_timeout_s - (asyncio.get_running_loop().time() - started))

    def _retry_pause(self, retries: int, started: float) -> float | None:
        """Return the pause before the next same-model retry, or None when none is left."""
        if retries >= len(self._same_model_retries):
            return None
        pause = self._same_model_retries[retries]
        left = self._left(started)
        return None if left is not None and left - pause < MIN_RETRY_BUDGET_S else pause

    def _fail_over(
        self, index: int, next_index: int, status: int | None, error: BaseException | None
    ) -> None:
        """Tell ``on_fail_over`` that call ``index`` gave way to call ``next_index``."""
        if self._on_fail_over is None:
            return
        event = FailOver(
            index=index,
            failed=model_name(self._calls[index]),
            next=model_name(self._calls[next_index]),
            status=status,
            error=error,
        )
        try:
            self._on_fail_over(event)
        except Exception:
            logger.exception("on_fail_over raised, the chain goes on")

    def _retry(
        self, index: int, attempt: int, pause_s: float, waited: bool, error: Exception
    ) -> None:
        """Tell ``on_retry`` that call ``index`` is sent again after ``pause_s``."""
        if self._on_retry is None:
            return
        event = Retry(
            index=index,
            model=model_name(self._calls[index]),
            attempt=attempt,
            pause_s=pause_s,
            waited=waited,
            status=http_status(error),
            error=error,
        )
        try:
            self._on_retry(event)
        except Exception:
            logger.exception("on_retry raised, the chain goes on")

    async def _first(
        self, events: AsyncIterator[ModelEvent], budget: float | None
    ) -> tuple[ModelEvent | None, Usage | None]:
        """Wait for the first answer event within the budget, and a ``Usage`` that came first.

        A ``Usage`` or an ``AnswerStarted`` is no answer. None when the stream holds nothing
        else. With ``start_ends_budget``, the budget covers the wait for ``AnswerStarted``
        only. With ``start_is_answer``, the ``AnswerStarted`` is the answer and is returned.
        """
        at_start = self._start_ends_budget or self._start_is_answer
        async with asyncio.timeout(budget):
            first, usage = await _answer(events, None, at_start=at_start)
        if isinstance(first, AnswerStarted) and not self._start_is_answer:
            # The provider began within the budget. Its answer may take longer.
            first, usage = await _answer(events, usage, at_start=False)
        return first, usage

    def _decide(
        self,
        error: Exception,
        index: int,
        last: bool,
        started: float,
        waited: bool,
        retries: int,
    ) -> tuple[_Decision, float]:
        """Choose what follows a call that failed before it answered.

        Args:
            error: What the call raised.
            index: The call's place in the chain.
            last: True for the last call of the chain.
            started: When this call's budget began, on the loop's clock.
            waited: True once a ``Retry-After`` wait happened for this call.
            retries: How many short retries this call had.

        Returns:
            The decision and the seconds to sleep first, 0 when it does not sleep. A
            ``RequestRefusedError`` moves to the next call at once: the call never sent the
            request, so sending it again cannot help.
        """
        if isinstance(error, RequestRefusedError):
            return (_Decision.RAISE if last else _Decision.FAIL_OVER), 0.0
        status = http_status(error)
        step = next_step(
            status,
            started=False,
            code=error_code(error),
            stop_statuses=self._stop_statuses,
            stop_codes=self._stop_codes,
        )
        if step is Next.RETRY and not waited:
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None) or {}
            if (delay := retry_delay(headers, budget_s=self._max_wait_s)) is not None:
                logger.warning("model call %d: status %s, retrying in %.1f s", index, status, delay)
                return _Decision.WAIT, delay
        if (
            step in (Next.FAIL_OVER, Next.RETRY)
            and _short_lived(status)
            and (pause := self._retry_pause(retries, started)) is not None
        ):
            logger.warning(
                "model call %d failed (status %s, %s), retrying it in %.2f s",
                index,
                status,
                type(error).__name__,
                pause,
            )
            return _Decision.RETRY, pause
        if step is Next.STOP or last:
            return _Decision.RAISE, 0.0
        return _Decision.FAIL_OVER, 0.0

    async def stream(self, request: ModelRequest) -> AsyncGenerator[ModelEvent]:
        """Stream from the first call that starts answering.

        Raises:
            Exception: The error of a call that stopped, failed after it started answering,
                or was the last one.
        """
        loop = asyncio.get_running_loop()
        order = (
            [self._active, *(i for i in range(len(self._calls)) if i != self._active)]
            if self._sticky
            else list(range(len(self._calls)))
        )
        for position, index in enumerate(order):
            call = self._calls[index]
            last = position == len(order) - 1
            next_index = order[position + 1] if not last else index
            waited = False
            retries = 0
            started = loop.time()
            while True:
                attempt_started = time.perf_counter()
                async with _closing(attempt_stream(call, request)) as events:
                    try:
                        first, usage = await self._first(events, self._left(started))
                    except Exception as error:
                        status = http_status(error)
                        decision, delay = self._decide(error, index, last, started, waited, retries)
                        if decision is _Decision.WAIT:
                            self._retry(index, int(waited) + retries + 1, delay, True, error)
                            await asyncio.sleep(delay)
                            waited = True
                        elif decision is _Decision.RETRY:
                            self._retry(index, int(waited) + retries + 1, delay, False, error)
                            await asyncio.sleep(delay)
                            retries += 1
                        elif decision is _Decision.RAISE:
                            raise
                        else:
                            logger.warning(
                                "model call %d failed (status %s, %s), trying the next one",
                                index,
                                status,
                                type(error).__name__,
                            )
                            self._fail_over(index, next_index, status, error)
                            break
                        continue
                    answered_empty = self._usage_counts_as_answer and usage is not None
                    if first is None and not last and self._empty_fails_over and not answered_empty:
                        logger.warning("model call %d answered nothing, trying the next one", index)
                        self._fail_over(index, next_index, None, None)
                        break
                    if first is None:
                        logger.warning("model call %d answered nothing: an empty reply", index)
                        if usage is not None:
                            yield usage
                    if first is not None:
                        self._active = self._answered = (
                            index  # ponytail: concurrent requests race, last wins
                        )
                        note_answer(attempt_started)
                        yield first
                        async for event in events:
                            yield event
                        if usage is not None:
                            yield usage
                    return


async def _answer(
    events: AsyncIterator[ModelEvent], usage: Usage | None, *, at_start: bool
) -> tuple[ModelEvent | None, Usage | None]:
    """Read a call's stream to its first answer event, keeping a ``Usage`` that came first.

    Returns:
        The first answer event, or the ``AnswerStarted`` when ``at_start`` stopped the read
        there. None when the stream holds no other. Then the last ``Usage`` before it.
    """
    while True:
        event = await anext(events, None)
        if isinstance(event, Usage):
            usage = event
        elif not isinstance(event, AnswerStarted) or at_start:
            return event, usage


@asynccontextmanager
async def _closing(stream: AsyncIterator[ModelEvent]) -> AsyncGenerator[AsyncIterator[ModelEvent]]:
    """Yield a call's stream, then close it once the chain is done with it.

    A stream is done with when it fails, times out, ends, or its reader stops early, like
    a caller who talks over the reply. Without the close, the call's request stays open
    until the stream is garbage collected.
    """
    events = aiter(stream)
    try:
        yield events
    finally:
        if (close := getattr(events, "aclose", None)) is not None:
            await close()
