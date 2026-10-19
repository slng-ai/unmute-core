"""What to do when a model request fails: retry it, try the next model, or stop.

The rules follow the SLNG Context Router: it already fails over between models and retries
behind its own plan, so a 5xx reaching the client means the next model, not the same one
again.
"""

from collections.abc import Collection, Iterator, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Literal

STOP_STATUSES = frozenset({400, 401, 402, 413, 422})
"""Bad request, bad key, no balance, payload too large, missing variables. No model fixes it."""
STOP_CODES: frozenset[str] = frozenset()
"""Error codes no other model fixes, whatever their status. None by default: a model call
names its own in ``stop_codes``, and ``FallbackModelCall`` reads them."""


class Next(StrEnum):
    """What to do after a failed model request."""

    RETRY = "retry"
    FAIL_OVER = "fail_over"
    STOP = "stop"


def _causes(error: BaseException) -> Iterator[BaseException]:
    """Yield an error, then its ``__cause__`` or ``__context__`` chain, each error once."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_status(value: object) -> bool:
    """Return True for an int from 100 to 599 that is not a bool."""
    return isinstance(value, int) and not isinstance(value, bool) and 100 <= value < 600


def _code_of(error: BaseException) -> object:
    """Read an error's ``code``, else the code in its ``body``, as the OpenAI SDK keeps it."""
    code = getattr(error, "code", None)
    if isinstance(code, str):
        return code
    body = getattr(error, "body", None)
    inner = body.get("error", body) if isinstance(body, dict) else None
    return inner.get("code") if isinstance(inner, dict) else None


def http_status(error: BaseException) -> int | None:
    """Return the HTTP status an error carries, looking through its causes.

    Reads ``status_code`` or ``status`` on the error, then on ``__cause__`` and
    ``__context__``. A bool or a number outside 100 to 599 is not a status.

    Args:
        error: The error a model request raised.

    Returns:
        The status, or None for a timeout, a lost connection or any other error.
    """
    return next(
        (
            value
            for cause in _causes(error)
            for attr in ("status_code", "status")
            if _is_status(value := getattr(cause, attr, None))
        ),
        None,
    )


_TIMEOUTS = frozenset(
    {("httpx", "TimeoutException"), ("httpx2", "TimeoutException"), ("openai", "APITimeoutError")}
)
"""Timeout classes by package and name: openai 2 runs on httpx, openai 3 on httpx2."""
_LOST_CONNECTIONS = frozenset(
    {("httpx", "TransportError"), ("httpx2", "TransportError"), ("openai", "APIConnectionError")}
)
"""Lost-connection classes by package and name. ``APITimeoutError`` is one too, so a
timeout is checked first."""


def _is_one_of(error: BaseException, classes: frozenset[tuple[str, str]]) -> bool:
    """Return whether the error's class, or a class it inherits, is one of ``classes``."""
    return any(
        (kind.__module__.split(".")[0], kind.__name__) in classes for kind in type(error).__mro__
    )


def transport_failure(error: BaseException) -> Literal["timeout", "connection"] | None:
    """Return whether a model request timed out or lost its connection, looking through causes.

    Classes are matched by package and name, so an error from ``httpx``, ``httpx2`` or the
    OpenAI SDK counts without importing any of them. Python's own ``TimeoutError`` and
    ``ConnectionError`` count too.

    Args:
        error: The error a model request raised.

    Returns:
        ``"timeout"``, ``"connection"``, or None for any other error.
    """
    for cause in _causes(error):
        if isinstance(cause, TimeoutError) or _is_one_of(cause, _TIMEOUTS):
            return "timeout"
        if isinstance(cause, ConnectionError) or _is_one_of(cause, _LOST_CONNECTIONS):
            return "connection"
    return None


def error_code(error: BaseException) -> str | None:
    """Return the text error code an error carries, like ``project_mode_forbidden``.

    Reads ``code`` on the error, as the OpenAI SDK's ``APIError`` keeps it, then the
    ``{"error": {"code": ...}}`` of its ``body``, then the same on its causes.

    Args:
        error: The error a model request raised.

    Returns:
        The code, or None when there is no text code.
    """
    return next(
        (code for cause in _causes(error) if isinstance(code := _code_of(cause), str) and code),
        None,
    )


def next_step(
    status: int | None,
    *,
    started: bool,
    code: str | None = None,
    stop_statuses: Collection[int] = STOP_STATUSES,
    stop_codes: Collection[str] = STOP_CODES,
) -> Next:
    """Decide what follows a failed request.

    Args:
        status: The HTTP status, or None when there was none.
        started: True once the caller got part of the answer. Then nothing is safe to repeat.
        code: The error's text code, from ``error_code``.
        stop_statuses: The statuses that stop. ``STOP_STATUSES`` by default.
        stop_codes: The error codes that stop. ``STOP_CODES`` by default.

    Returns:
        ``STOP`` after the answer started, for a status in ``stop_statuses`` (400, 401, 402,
        413 and 422 by default), and for a code in ``stop_codes``. ``RETRY`` for 429, after
        its ``Retry-After``. ``FAIL_OVER`` for everything else.
    """
    if started or status in stop_statuses or code in stop_codes:
        return Next.STOP
    if status == 429:
        return Next.RETRY
    return Next.FAIL_OVER


def retry_delay(
    headers: Mapping[str, str], *, budget_s: float, now: datetime | None = None
) -> float | None:
    """Read how long ``Retry-After`` asks to wait, if it asks for a wait that fits the budget.

    A ``Retry-After`` of 0, or an HTTP date that has passed, asks for no wait. It is read as
    if the header were absent.

    Args:
        headers: The response headers. The lookup ignores case.
        budget_s: The longest wait the call can afford.
        now: The current time, for an HTTP date. Defaults to now.

    Returns:
        Seconds to wait, or None when the header is absent, unreadable, asks for no wait or is
        over budget.
    """
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        delay = float(value)
    else:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        delay = max(0.0, (when - (now or datetime.now(UTC))).total_seconds())
    return delay if 0 < delay <= budget_s else None
