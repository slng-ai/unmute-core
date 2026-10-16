"""HTTP requests that cannot be turned against the agent's own network.

DNS is resolved once, every address must be public, and the connection is pinned to the
checked address with the original Host header and TLS server name, so a second DNS answer
cannot point it elsewhere. Redirects stay on the same origin and are checked again. One
deadline covers DNS, every hop and the body. ``allow_private`` turns the address check off
for an author's own internal service.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Mapping
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors

type Resolver = Callable[[str, int], Awaitable[tuple[str, ...]]]
type IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class HardenedHttpError(RuntimeError):
    """A request was refused or failed. The message is safe to show the model."""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class PinnedDestination:
    """Where one request connects, and what it tells the server it asked for.

    Attributes:
        connection_url: The URL with the checked IP address in place of the host.
        host_header: The original authority, for the ``Host`` header.
        sni_hostname: The original host for TLS, or None for plain http.
        origin: Scheme, host and port, to keep redirects on the same origin.
    """

    connection_url: str
    host_header: str
    sni_hostname: str | None
    origin: tuple[str, str, int]


async def resolve_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve a host to its addresses, sorted, without blocking the event loop."""
    try:
        rows = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise HardenedHttpError("destination cannot be resolved") from error
    if addresses := tuple(sorted({str(row[4][0]) for row in rows})):
        return addresses
    raise HardenedHttpError("destination cannot be resolved")


def _parse_addresses(addresses: tuple[str, ...], *, allow_private: bool) -> list[IPAddress]:
    """Parse the resolver's answer. Every address must be public unless private is allowed."""
    parsed = []
    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError as error:
            raise HardenedHttpError("resolver returned an invalid address") from error
        if not allow_private and not address.is_global:
            raise HardenedHttpError("destination is not globally routable")
        parsed.append(address)
    return parsed


def check_destination(url: str, *, allow_http: bool) -> str | None:
    """Check the scheme and the authority of a URL, without a network call.

    Args:
        url: The URL to call.
        allow_http: Accept plain ``http`` as well as ``https``.

    Returns:
        Why the URL is refused, or None when it is fine.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in ({"https", "http"} if allow_http else {"https"}):
        return "destination scheme is not allowed"
    if not parsed.hostname or any(char.isspace() for char in parsed.netloc):
        return "destination authority is invalid"
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        return "destination authority is invalid"
    try:
        port = parsed.port
    except ValueError:
        return "destination port is invalid"
    if port is not None and not 1 <= port <= 65535:
        return "destination port is invalid"
    return None


async def pin_destination(
    url: str,
    *,
    allow_http: bool,
    allow_private: bool = False,
    resolver: Resolver = resolve_addresses,
) -> PinnedDestination:
    """Check a URL and pin it to one checked address.

    Args:
        url: The URL to call.
        allow_http: Accept plain ``http`` as well as ``https``.
        allow_private: Accept private, loopback and link-local addresses.
        resolver: Resolves a host and port to addresses. Tests pass their own.

    Returns:
        The pinned destination.

    Raises:
        HardenedHttpError: If the scheme, authority or any resolved address is refused.
    """
    if reason := check_destination(url, allow_http=allow_http):
        raise HardenedHttpError(reason)
    parsed = urlsplit(url)
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port
    assert parsed.hostname is not None
    addresses = _parse_addresses(await resolver(parsed.hostname, port), allow_private=allow_private)
    selected = min(addresses, key=lambda address: (address.version, address.packed))
    port_suffix = f":{port}" if port != default_port else ""
    connection_host = f"[{selected}]" if selected.version == 6 else str(selected)
    public_host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    connection_url = urlunsplit(
        (parsed.scheme, connection_host + port_suffix, parsed.path, parsed.query, "")
    )
    if "?" in url and not parsed.query:
        connection_url += "?"
    return PinnedDestination(
        connection_url=connection_url,
        host_header=public_host + port_suffix,
        sni_hostname=parsed.hostname if parsed.scheme == "https" else None,
        origin=(parsed.scheme, parsed.hostname.casefold(), port),
    )


async def send_pinned_request(
    client: httpx.AsyncClient,
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    content: bytes,
    allow_http: bool,
    timeout_seconds: float,
    max_redirects: int,
    max_response_bytes: int,
    allow_private: bool = False,
    resolver: Resolver = resolve_addresses,
) -> tuple[httpx.Response, bytes]:
    """Send one request through pinned connections and read a bounded body.

    Args:
        client: A client made with ``follow_redirects=False``.
        method: The HTTP method.
        url: The URL to call.
        headers: Request headers. ``Host`` is set here.
        content: The request body, or ``b""`` for none.
        allow_http: Accept plain ``http``.
        timeout_seconds: One budget for DNS, every redirect hop and the body.
        max_redirects: Same-origin redirects to follow.
        max_response_bytes: The largest body to read.
        allow_private: Accept private, loopback and link-local addresses.
        resolver: Resolves a host and port to addresses.

    Returns:
        The response, already closed, and its body.

    Raises:
        HardenedHttpError: If the request is refused, times out, fails in transport,
            redirects too often or off its origin, or answers with too large a body.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    current = url
    first_origin: tuple[str, str, int] | None = None
    for hop in range(max_redirects + 1):
        try:
            async with asyncio.timeout_at(deadline):
                destination = await pin_destination(
                    current, allow_http=allow_http, allow_private=allow_private, resolver=resolver
                )
        except TimeoutError as error:
            raise HardenedHttpError("request timed out") from error
        if first_origin is None:
            first_origin = destination.origin
        elif destination.origin != first_origin:
            raise HardenedHttpError("cross-origin redirects are not allowed")
        extensions = {}
        if destination.sni_hostname is not None:
            extensions["sni_hostname"] = destination.sni_hostname
        request = client.build_request(
            method,
            destination.connection_url,
            headers={**headers, "Host": destination.host_header},
            content=content or None,
            extensions=extensions,
        )
        try:
            async with asyncio.timeout_at(deadline):
                response = await client.send(request, stream=True)
                try:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location or hop == max_redirects:
                            raise HardenedHttpError("redirect limit exceeded")
                        current = urljoin(current, location)
                        continue
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) > max_response_bytes:
                            raise HardenedHttpError("response exceeded the size limit")
                    return response, bytes(body)
                finally:
                    await response.aclose()
        except TimeoutError as error:
            raise HardenedHttpError("request timed out") from error
        except httpx.TransportError as error:
            raise HardenedHttpError("request transport failed") from error
    raise HardenedHttpError("redirect limit exceeded")
