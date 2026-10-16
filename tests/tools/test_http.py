"""Hardened HTTP: pinned addresses, same-origin redirects, one deadline, a bounded body.

The production client is tested in test_webhook.py, because the client lives in Webhook.
"""

import asyncio
from typing import Any

import httpx
import pytest

from unmute_core.tools._http import HardenedHttpError, pin_destination, send_pinned_request

PUBLIC = "93.184.216.34"


def _resolver(*addresses: str):
    async def resolve(_host: str, _port: int) -> tuple[str, ...]:
        return addresses

    return resolve


async def _send(client: httpx.AsyncClient, **overrides):
    options: dict[str, Any] = {
        "method": "GET",
        "url": "https://api.example.com/value",
        "headers": {},
        "content": b"",
        "allow_http": False,
        "timeout_seconds": 2,
        "max_redirects": 0,
        "max_response_bytes": 1024,
        "resolver": _resolver(PUBLIC),
    }
    return await send_pinned_request(client, **(options | overrides))


@pytest.mark.parametrize(
    "addresses",
    [
        ("127.0.0.1",),
        ("169.254.169.254",),
        ("10.0.0.1",),
        ("::1",),
        ("fe80::1",),
        (PUBLIC, "10.0.0.1"),
    ],
)
async def test_private_metadata_and_mixed_answers_are_refused(addresses):
    with pytest.raises(HardenedHttpError, match="globally routable"):
        await pin_destination(
            "https://api.example.com/path", allow_http=False, resolver=_resolver(*addresses)
        )


async def test_private_answers_pass_when_allowed():
    destination = await pin_destination(
        "http://localhost:8080/path",
        allow_http=True,
        allow_private=True,
        resolver=_resolver("127.0.0.1"),
    )
    assert destination.connection_url == "http://127.0.0.1:8080/path"
    assert destination.host_header == "localhost:8080"
    assert destination.sni_hostname is None


@pytest.mark.parametrize(
    "url",
    [
        "https://user@api.example.com/path",
        "https://api.example.com/path#fragment",
        "http://api.example.com/path",
        "ftp://api.example.com/path",
        "https://api.example.com:0/path",
        "https://api.example.com:abc/path",
        "https://api example.com/path",
    ],
)
async def test_unsafe_authority_is_refused(url):
    with pytest.raises(HardenedHttpError):
        await pin_destination(url, allow_http=False, resolver=_resolver(PUBLIC))


@pytest.mark.parametrize(
    ("url", "raw_path"),
    [
        ("https://api.example.com/path?", b"/path?"),
        ("https://api.example.com/path?a=1&&", b"/path?a=1&&"),
        ("https://api.example.com/%2F?x=+%20", b"/%2F?x=+%20"),
    ],
)
async def test_pinned_destination_keeps_the_exact_request_target(url, raw_path):
    destination = await pin_destination(url, allow_http=False, resolver=_resolver(PUBLIC))
    assert httpx.URL(destination.connection_url).raw_path == raw_path


async def test_ipv6_destination_brackets_both_authorities():
    destination = await pin_destination(
        "https://[2606:4700:4700::1111]:8443/path",
        allow_http=False,
        resolver=_resolver("2606:4700:4700::1111"),
    )
    assert destination.connection_url == "https://[2606:4700:4700::1111]:8443/path"
    assert destination.host_header == "[2606:4700:4700::1111]:8443"
    assert destination.sni_hostname == "2606:4700:4700::1111"


async def test_same_origin_redirect_keeps_host_and_sni():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(302, request=request, headers={"Location": "/next"})
        return httpx.Response(200, request=request, content=b'{"ok":true}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response, body = await _send(
            client,
            method="POST",
            url="https://api.example.com/start",
            headers={"Authorization": "Bearer SENTINEL"},
            content=b"payload",
            max_redirects=2,
        )
    assert response.status_code == 200
    assert body == b'{"ok":true}'
    assert [request.url.path for request in requests] == ["/start", "/next"]
    for request in requests:
        assert request.url.host == PUBLIC
        assert request.headers["Host"] == "api.example.com"
        assert request.extensions["sni_hostname"] == "api.example.com"
        assert request.headers["Authorization"] == "Bearer SENTINEL"


async def test_cross_origin_redirect_is_refused_before_credentials_are_sent():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            302, request=request, headers={"Location": "https://evil.example.net/steal"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HardenedHttpError, match="cross-origin"):
            await _send(client, headers={"Authorization": "Bearer SENTINEL"}, max_redirects=2)
    assert len(requests) == 1


async def test_too_many_redirects_are_refused():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, request=request, headers={"Location": "/again"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HardenedHttpError, match="redirect limit"):
            await _send(client, max_redirects=2)


async def test_rebinding_answer_is_refused_before_the_redirect_follow_up():
    requests: list[httpx.Request] = []
    answers = iter([(PUBLIC,), ("127.0.0.1",)])

    async def resolve(_host: str, _port: int) -> tuple[str, ...]:
        return next(answers)

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, request=request, headers={"Location": "/next"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HardenedHttpError, match="globally routable"):
            await _send(client, max_redirects=2, resolver=resolve)
    assert len(requests) == 1


async def test_response_is_bounded_while_streaming():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=b"x" * 20)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HardenedHttpError, match="size limit"):
            await _send(client, max_response_bytes=10)


async def test_cancelled_streaming_request_closes_the_response():
    started = asyncio.Event()
    closed = asyncio.Event()

    class BlockingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self) -> None:
            closed.set()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, stream=BlockingStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(_send(client))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()


async def test_dns_is_inside_the_total_timeout():
    async def resolve(_host: str, _port: int) -> tuple[str, ...]:
        await asyncio.sleep(1)
        return (PUBLIC,)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HardenedHttpError, match="timed out"):
            await _send(client, timeout_seconds=0.01, resolver=resolve)


async def test_dns_and_transport_share_one_timeout():
    async def resolve(_host: str, _port: int) -> tuple[str, ...]:
        await asyncio.sleep(0.03)
        return (PUBLIC,)

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.03)
        return httpx.Response(200, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HardenedHttpError, match="timed out"):
            await _send(client, timeout_seconds=0.05, resolver=resolve)
