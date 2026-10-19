"""Which failed requests retry, fail over or stop."""

from datetime import UTC, datetime

import httpx
import openai
import pytest

from tests.openai_http import http
from unmute_core.model import Next, error_code, http_status, next_step, retry_delay


class StatusError(Exception):
    def __init__(self, status_code: object = None, status: object = None) -> None:
        super().__init__("upstream failed")
        self.status_code = status_code
        self.status = status


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, Next.STOP),
        (401, Next.STOP),
        (402, Next.STOP),
        (413, Next.STOP),
        (422, Next.STOP),
        (429, Next.RETRY),
        (403, Next.FAIL_OVER),
        (404, Next.FAIL_OVER),
        (408, Next.FAIL_OVER),
        (500, Next.FAIL_OVER),
        (502, Next.FAIL_OVER),
        (503, Next.FAIL_OVER),
        (504, Next.FAIL_OVER),
        (None, Next.FAIL_OVER),
    ],
)
def test_next_step_by_status(status: int | None, expected: Next) -> None:
    assert next_step(status, started=False) is expected


@pytest.mark.parametrize("status", [None, 429, 503])
def test_nothing_repeats_once_the_answer_started(status: int | None) -> None:
    assert next_step(status, started=True) is Next.STOP


def test_http_status_reads_either_attribute() -> None:
    assert http_status(StatusError(status_code=413)) == 413
    assert http_status(StatusError(status=429)) == 429


def test_http_status_looks_through_the_cause_chain() -> None:
    try:
        try:
            raise StatusError(status_code=503)
        except StatusError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as error:
        assert http_status(error) == 503


def test_http_status_looks_through_the_context_chain() -> None:
    try:
        try:
            raise StatusError(status=402)
        except StatusError:
            raise RuntimeError("during handling")  # noqa: B904
    except RuntimeError as error:
        assert http_status(error) == 402


@pytest.mark.parametrize("value", [True, 99, 600, "500", 413.0])
def test_http_status_ignores_what_is_not_a_status(value: object) -> None:
    assert http_status(StatusError(status_code=value)) is None


def test_http_status_survives_a_cycle() -> None:
    first, second = RuntimeError("a"), RuntimeError("b")
    first.__cause__, second.__cause__ = second, first
    assert http_status(first) is None


NOW = datetime(2026, 10, 6, 12, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": "1"}, 1.0),
        ({"retry-after": " 2 "}, 2.0),
        ({"Retry-After": "0"}, None),
        ({"Retry-After": "Tue, 06 Oct 2026 12:00:01 GMT"}, 1.0),
        ({"Retry-After": "Tue, 06 Oct 2026 11:59:00 GMT"}, None),
        ({"Retry-After": "Tue, 06 Oct 2026 12:00:00 GMT"}, None),
        ({"Retry-After": "3"}, None),
        ({"Retry-After": "soon"}, None),
        ({"Retry-After": "-1"}, None),
        ({}, None),
    ],
)
def test_retry_delay(headers: dict, expected: float | None) -> None:
    assert retry_delay(headers, budget_s=2.0, now=NOW) == expected


def test_error_code_reads_the_sdk_error_its_body_and_its_causes() -> None:
    response = http.Response(403, request=http.Request("POST", "https://router.test/v1"))
    body = {"error": {"message": "no", "code": "project_mode_forbidden"}}
    sdk = openai.APIStatusError("no", response=response, body=body)
    assert error_code(sdk) == "project_mode_forbidden"

    class BodyError(Exception):
        def __init__(self, code: object) -> None:
            super().__init__("refused")
            self.body = {"error": {"code": code}}

    try:
        try:
            raise BodyError("insufficient_balance")
        except BodyError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as wrapped:
        assert error_code(wrapped) == "insufficient_balance"
    assert error_code(BodyError(400)) is None
    assert error_code(ValueError("x")) is None


def test_a_stop_code_stops_whatever_its_status_and_none_is_set_by_default() -> None:
    assert next_step(403, started=False, code="blocked", stop_codes={"blocked"}) is Next.STOP
    assert next_step(403, started=False, code="project_mode_forbidden") is Next.FAIL_OVER


async def test_a_real_sdk_request_to_a_closed_port_is_a_lost_connection() -> None:
    from unmute_core.model import ModelRequest, OpenAIModelCall
    from unmute_core.model._failover import transport_failure

    call = OpenAIModelCall("gpt-test", base_url="http://127.0.0.1:9/v1", api_key="k")
    with pytest.raises(Exception) as caught:
        [
            event
            async for event in call.stream(
                ModelRequest(messages=[{"role": "user", "content": "hi"}])
            )
        ]
    assert transport_failure(caught.value) == "connection"
    await call.aclose()


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (TimeoutError(), "timeout"),
        (ConnectionError(), "connection"),
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("down"), "connection"),
        (ValueError("other"), None),
    ],
)
def test_transport_failure_names_a_timeout_or_a_lost_connection(error, kind) -> None:
    from unmute_core.model._failover import transport_failure

    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = error
    assert transport_failure(error) == kind
    assert transport_failure(wrapped) == kind
