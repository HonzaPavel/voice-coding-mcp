"""Tests of Client ID Metadata Document resolution. Fetching goes through an httpx2 MockTransport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import pytest

from voice_coding_mcp.auth import CimdClient, CimdError, cache_ttl_ms, create_cimd_resolver

URL_ID = "https://claude.ai/oauth/mcp-oauth-client-metadata"
VALID: dict[str, Any] = {
    "client_id": URL_ID,
    "client_name": "Claude",
    "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
    "token_endpoint_auth_method": "none",
}


class Stub:
    def __init__(self, make: Callable[[], httpx2.Response]) -> None:
        self.calls: list[httpx2.Request] = []
        self._make = make
        self.client = httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.calls.append(request)
        return self._make()


def json_response(body: Any, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(
        200, content=json.dumps(body).encode(), headers={"content-type": "application/json", **(headers or {})}
    )


async def test_fetches_and_validates_an_allowlisted_document() -> None:
    stub = Stub(lambda: json_response(VALID))
    resolver = create_cimd_resolver([URL_ID], client=stub.client)
    client = await resolver.resolve(URL_ID)
    assert client == CimdClient(
        client_id=URL_ID,
        client_name="Claude",
        redirect_uris=tuple(VALID["redirect_uris"]),
        token_endpoint_auth_method="none",
    )
    assert len(stub.calls) == 1
    assert str(stub.calls[0].url) == URL_ID
    assert stub.calls[0].headers["accept"] == "application/json"


async def test_normalizes_the_client_name_like_dcr() -> None:
    stub = Stub(lambda: json_response({**VALID, "client_name": " \x1b]0;x\x07Cl\u202eaude " + "e" * 300}))
    client = await create_cimd_resolver([URL_ID], client=stub.client).resolve(URL_ID)
    assert client.client_name == ("]0;xClaude " + "e" * 300)[:200]


async def test_never_fetches_a_url_that_is_not_on_the_allowlist() -> None:
    stub = Stub(lambda: json_response(VALID))
    resolver = create_cimd_resolver([URL_ID], client=stub.client)
    with pytest.raises(CimdError):
        await resolver.resolve("https://evil.example/meta")
    with pytest.raises(CimdError):
        await resolver.resolve(f"{URL_ID}?x=1")
    assert stub.calls == []


async def _oversize_stream() -> AsyncIterator[bytes]:
    for _ in range(10):
        yield bytes(10_000)


INVALID: list[tuple[str, Callable[[], httpx2.Response], str]] = [
    ("client_id mismatch", lambda: json_response({**VALID, "client_id": "https://other.example/meta"}), "client_id"),
    (
        "missing client_name",
        lambda: json_response({k: v for k, v in VALID.items() if k != "client_name"}),
        "client_name",
    ),
    ("missing redirect_uris", lambda: json_response({**VALID, "redirect_uris": []}), "redirect_uris"),
    (
        "auth method not none",
        lambda: json_response({**VALID, "token_endpoint_auth_method": "client_secret_basic"}),
        "none",
    ),
    ("not JSON", lambda: httpx2.Response(200, content=b"<html>"), "JSON"),
    ("HTTP error", lambda: httpx2.Response(404, content=b"nope"), "404"),
    ("redirect response", lambda: httpx2.Response(302, headers={"location": "https://evil.example/"}), "redirect"),
    ("oversize by content-length", lambda: httpx2.Response(200, content=b"x" * 70_000), "too large"),
    ("oversize stream without content-length", lambda: httpx2.Response(200, content=_oversize_stream()), "too large"),
]


@pytest.mark.parametrize(("make", "pattern"), [(m, p) for _, m, p in INVALID], ids=[n for n, _, _ in INVALID])
async def test_rejects(make: Callable[[], httpx2.Response], pattern: str) -> None:
    stub = Stub(make)
    resolver = create_cimd_resolver([URL_ID], client=stub.client)
    with pytest.raises(CimdError, match=pattern):
        await resolver.resolve(URL_ID)
    assert len(stub.calls) == 1  # a redirect is never followed


async def test_rejects_when_fetch_fails_or_times_out() -> None:
    async def hang(_request: httpx2.Request) -> httpx2.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    resolver = create_cimd_resolver(
        [URL_ID], client=httpx2.AsyncClient(transport=httpx2.MockTransport(hang)), timeout_ms=20
    )
    with pytest.raises(CimdError, match="failed"):
        await resolver.resolve(URL_ID)

    def broken(_request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    resolver = create_cimd_resolver([URL_ID], client=httpx2.AsyncClient(transport=httpx2.MockTransport(broken)))
    with pytest.raises(CimdError, match="fetching the metadata document failed: ConnectError"):
        await resolver.resolve(URL_ID)


async def test_caches_per_max_age_capped_at_one_hour_5_minutes_by_default() -> None:
    t = 0
    headers = {"cache-control": "max-age=60"}
    stub = Stub(lambda: json_response(VALID, headers))
    resolver = create_cimd_resolver([URL_ID], client=stub.client, now=lambda: t)
    await resolver.resolve(URL_ID)
    t = 59_000
    await resolver.resolve(URL_ID)
    assert len(stub.calls) == 1
    t = 60_001
    headers = {}
    await resolver.resolve(URL_ID)
    assert len(stub.calls) == 2
    t += 299_000
    await resolver.resolve(URL_ID)
    assert len(stub.calls) == 2
    t += 2_000
    await resolver.resolve(URL_ID)
    assert len(stub.calls) == 3


def test_computes_cache_lifetimes() -> None:
    assert cache_ttl_ms(None) == 300_000
    assert cache_ttl_ms("public, max-age=120") == 120_000
    assert cache_ttl_ms("max-age=999999") == 3_600_000
    assert cache_ttl_ms("no-store") == 0
    assert cache_ttl_ms("public") == 300_000
