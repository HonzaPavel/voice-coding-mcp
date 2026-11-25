"""Client ID Metadata Documents.

A client_id that is an https URL names a JSON document describing the client. Only URLs on the
configured allowlist are ever fetched (no SSRF surface).

Fetching uses an injected `httpx2.AsyncClient` (tests pass one with `httpx2.MockTransport`):
GET with `accept: application/json`, `follow_redirects=False`, a 5 s timeout covering the whole
fetch including the body, and a 64 KB cap enforced while streaming (and up front from
Content-Length).
"""

from __future__ import annotations

import asyncio
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Literal

import httpx2

from ..clock import HOUR_MS, MINUTE_MS, SECOND_MS, Clock, now_ms
from ..json_file import loads_json
from ..urls import can_parse
from ..web import BodyError, read_capped
from .text import normalize_client_name

CIMD_TIMEOUT_MS = 5 * SECOND_MS
CIMD_MAX_BYTES = 64 * 1024
DEFAULT_TTL_MS = 5 * MINUTE_MS
MAX_TTL_MS = HOUR_MS

_MAX_AGE = re.compile(r'max-age\s*=\s*"?(\d+)"?', re.ASCII)


@dataclass(frozen=True)
class CimdClient:
    client_id: str
    client_name: str
    """Normalized with `text.normalize_client_name`."""
    redirect_uris: tuple[str, ...]
    token_endpoint_auth_method: Literal["none"] = "none"


class CimdError(Exception):
    """The document is not allowed, cannot be fetched or is invalid. The message is safe to show."""


def is_cimd_client_id(client_id: str) -> bool:
    """A client_id that looks like a metadata document URL (starts with `https://`)."""
    return client_id.startswith("https://")


def cache_ttl_ms(cache_control: str | None) -> int:
    """Cache lifetime from Cache-Control: max-age capped at 1 h, 5 min when absent, 0 for no-store/no-cache."""
    if not cache_control:
        return DEFAULT_TTL_MS
    directives = [d.strip(" \t") for d in cache_control.lower().split(",")]
    if any(d in ("no-store", "no-cache") for d in directives):
        return 0
    for d in directives:
        match = _MAX_AGE.fullmatch(d)
        if match:
            return min(int(match.group(1)) * SECOND_MS, MAX_TTL_MS)
    return DEFAULT_TTL_MS


def _validate(client_id: str, data: object) -> CimdClient:
    if not isinstance(data, dict):
        raise CimdError("metadata document is not a JSON object")
    if data.get("client_id") != client_id:
        raise CimdError("client_id in the metadata document does not match its URL")
    name = data.get("client_name")
    if not isinstance(name, str) or not name.strip():
        raise CimdError("client_name missing")
    uris = data.get("redirect_uris")
    if not isinstance(uris, list) or len(uris) == 0 or not all(isinstance(u, str) and can_parse(u) for u in uris):
        raise CimdError("redirect_uris missing or invalid")
    if data.get("token_endpoint_auth_method") != "none":
        raise CimdError("token_endpoint_auth_method must be none")
    return CimdClient(
        client_id=client_id,
        client_name=normalize_client_name(name),
        redirect_uris=tuple(uris),
    )


@dataclass(frozen=True)
class _CacheEntry:
    client: CimdClient
    expires_at: int


class CimdResolver:
    def __init__(
        self,
        allowlist: list[str] | tuple[str, ...],
        *,
        client: httpx2.AsyncClient | None = None,
        now: Clock | None = None,
        timeout_ms: int = CIMD_TIMEOUT_MS,
        max_bytes: int = CIMD_MAX_BYTES,
    ) -> None:
        """`allowlist`: exact client_id URLs that may be fetched (`oauth.cimd_client_ids`)."""
        self._allowlist = frozenset(allowlist)
        self._client = client
        self._now: Clock = now or now_ms
        self._timeout_ms = timeout_ms
        self._max_bytes = max_bytes
        self._cache: dict[str, _CacheEntry] = {}

    def is_allowed(self, client_id: str) -> bool:
        return client_id in self._allowlist

    async def _load(self, client_id: str) -> tuple[CimdClient, int]:
        if self._client is not None:
            return await self._load_timed(self._client, client_id)
        # No proxy from the environment.
        async with httpx2.AsyncClient(trust_env=False, follow_redirects=False) as client:
            return await self._load_timed(client, client_id)

    async def _load_timed(self, client: httpx2.AsyncClient, client_id: str) -> tuple[CimdClient, int]:
        # One timeout for the whole fetch including the body.
        try:
            async with asyncio.timeout(self._timeout_ms / SECOND_MS):
                return await self._load_with(client, client_id)
        except TimeoutError as err:
            raise CimdError("fetching the metadata document failed: TimeoutError") from err

    async def _load_with(self, client: httpx2.AsyncClient, client_id: str) -> tuple[CimdClient, int]:
        async with AsyncExitStack() as stack:
            try:
                response = await stack.enter_async_context(
                    client.stream(
                        "GET",
                        client_id,
                        headers={"accept": "application/json"},
                        follow_redirects=False,
                        timeout=self._timeout_ms / SECOND_MS,
                    )
                )
            except Exception as err:
                raise CimdError(f"fetching the metadata document failed: {type(err).__name__}") from err
            if 300 <= response.status_code < 400:
                raise CimdError("metadata document URL redirects, redirects are not followed")
            if response.status_code != 200:
                raise CimdError(f"metadata document returned HTTP {response.status_code}")
            try:
                raw = await read_capped(response.headers, response.aiter_bytes(), self._max_bytes)
            except BodyError as err:
                raise CimdError("metadata document too large") from err
            except Exception as err:
                raise CimdError(f"reading the metadata document failed: {type(err).__name__}") from err
        try:
            # utf-8-sig: a leading BOM is dropped.
            data = loads_json(raw.decode("utf-8-sig"))
        except ValueError as err:  # UnicodeDecodeError is a ValueError
            raise CimdError("metadata document is not valid JSON") from err
        return _validate(client_id, data), cache_ttl_ms(response.headers.get("cache-control"))

    async def resolve(self, client_id: str) -> CimdClient:
        """Fetches (or returns the cached) document. Raises CimdError when it is not allowed or invalid:
        not on the allowlist, fetch failure (error class name only), a redirect (3xx), status != 200,
        too large, not UTF-8 JSON, not an object, `client_id` != URL, missing `client_name`,
        missing/invalid `redirect_uris`, `token_endpoint_auth_method` != `none`."""
        if client_id not in self._allowlist:
            raise CimdError("client_id is not an allowed metadata document URL")
        cached = self._cache.get(client_id)
        if cached is not None and cached.expires_at > self._now():
            return cached.client
        client, ttl = await self._load(client_id)
        if ttl > 0:
            self._cache[client_id] = _CacheEntry(client=client, expires_at=self._now() + ttl)
        else:
            self._cache.pop(client_id, None)
        return client


def create_cimd_resolver(
    allowlist: list[str] | tuple[str, ...],
    *,
    client: httpx2.AsyncClient | None = None,
    now: Clock | None = None,
    timeout_ms: int = CIMD_TIMEOUT_MS,
    max_bytes: int = CIMD_MAX_BYTES,
) -> CimdResolver:
    return CimdResolver(allowlist, client=client, now=now, timeout_ms=timeout_ms, max_bytes=max_bytes)
