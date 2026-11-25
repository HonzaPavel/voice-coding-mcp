"""Small Starlette/ASGI helpers shared by the auth routes and the HTTP app.

`client_ip` trusts only loopback proxies: the server sits behind a reverse proxy on loopback, so
X-Forwarded-For is honored only through loopback hops. Run uvicorn with
`proxy_headers=False` so the peer address in the ASGI scope is the real socket peer.
"""

from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterable, Mapping
from typing import Any

from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse

from .json_file import loads_json
from .urls import is_loopback_ip

NO_STORE = {"Cache-Control": "no-store"}


def client_ip_from(peer: str | None, forwarded_for: str | None) -> str:
    """Trust only loopback hops: walk from the socket peer through
    X-Forwarded-For (right to left) while the hop is loopback; return the first untrusted
    address, or the leftmost one when every hop is trusted."""
    addresses = [peer or "unknown"]
    if forwarded_for:
        addresses += [part.strip() for part in reversed(forwarded_for.split(",")) if part.strip()]
    for address in addresses[:-1]:
        if not is_loopback_ip(address):
            return address
    return addresses[-1]


def client_key(ip: str) -> str:
    """The key per-IP limits count by: an IPv4 address (also when IPv4-mapped in IPv6) as itself, an
    IPv6 address as its /64 network (one subscriber usually holds a whole /64, so per-address limits
    would be trivial to dodge). Anything that is not an IP address is returned unchanged."""
    text = ip.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    text = text.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return ip
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.IPv6Network((address, 64), strict=False))
    return str(address)


def client_ip(conn: HTTPConnection) -> str:
    """The client address of a request behind a loopback proxy (see `client_ip_from`)."""
    peer = conn.client.host if conn.client else None
    return client_ip_from(peer, conn.headers.get("x-forwarded-for"))


def json_response(
    status: int,
    body: Any,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Compact JSON response with `Cache-Control: no-store` (plus `headers`)."""
    return JSONResponse(body, status_code=status, headers={**NO_STORE, **(headers or {})})


class BodyError(Exception):
    """A request body that cannot be accepted; `status` is 400, 413 or 415."""

    def __init__(self, status: int) -> None:
        super().__init__(f"body error {status}")
        self.status = status


def content_type(headers: Mapping[str, str]) -> tuple[str, str | None]:
    """Lowercase media type and charset parameter (unquoted, lowercase; None when absent) of the
    Content-Type header."""
    media, _, params = headers.get("content-type", "").partition(";")
    charset = None
    for param in params.split(";"):
        name, _, value = param.partition("=")
        if name.strip().lower() == "charset":
            charset = value.strip().strip('"').lower()
    return media.strip().lower(), charset


async def read_capped(headers: Mapping[str, str], chunks: AsyncIterable[bytes], limit: int) -> bytes:
    """Reads a whole body, raising BodyError(413) when Content-Length declares more than `limit`
    bytes or more than `limit` bytes arrive."""
    declared = headers.get("content-length")
    if declared is not None and declared.strip().isdigit() and int(declared) > limit:
        raise BodyError(413)
    body = bytearray()
    async for chunk in chunks:
        body.extend(chunk)
        if len(body) > limit:
            raise BodyError(413)
    return bytes(body)


async def read_json_body(headers: Mapping[str, str], chunks: AsyncIterable[bytes], limit: int) -> tuple[bytes, Any]:
    """A UTF-8 JSON body (object or array) of at most `limit` bytes, as (raw bytes, parsed value); an
    empty body counts as `{}` (raw `b"{}"`).

    BodyError(415) for a content encoding other than identity or a charset other than utf-8,
    413 above `limit`, 400 for invalid UTF-8 or invalid JSON (NaN and Infinity included)."""
    if headers.get("content-encoding", "identity").strip().lower() != "identity":
        raise BodyError(415)
    _, charset = content_type(headers)
    if charset is not None and charset != "utf-8":
        raise BodyError(415)
    raw = await read_capped(headers, chunks, limit)
    if not raw:
        return b"{}", {}
    try:
        return raw, loads_json(raw.decode("utf-8"), containers_only=True)
    except ValueError as err:  # UnicodeDecodeError is a ValueError
        raise BodyError(400) from err
