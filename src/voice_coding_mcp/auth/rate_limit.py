"""Fixed-window, in-memory rate limit per client IP: an object the route handlers call first."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from starlette.responses import Response

from ..clock import Clock, now_ms
from ..web import client_key, json_response


@dataclass(frozen=True)
class RateLimitOptions:
    max: int
    """Requests allowed per window and key."""
    window_ms: int
    now: Clock | None = None
    on_limited: Callable[[str], None] | None = None
    """Called with the IP of a rejected request, for logging."""


@dataclass
class _Entry:
    count: int
    reset_at: int


class RateLimiter:
    """Counts requests per IP (`web.client_ip`, so the loopback-proxy rule decides which address
    counts), keyed by `web.client_key` (IPv6 clients by their /64). Expired entries are swept when
    more than 1000 keys are held."""

    def __init__(self, options: RateLimitOptions) -> None:
        self._options = options
        self._now: Clock = options.now or now_ms
        self._hits: dict[str, _Entry] = {}

    def check(self, ip: str) -> Response | None:
        """Counts one request. None when allowed; otherwise a 429 JSON response
        `{"error": "temporarily_unavailable", "error_description": "Too many requests, try again later."}`
        with `Retry-After` (seconds, at least 1) and `Cache-Control: no-store`."""
        t = self._now()
        if len(self._hits) > 1000:
            for key in [k for k, e in self._hits.items() if e.reset_at <= t]:
                del self._hits[key]
        key = client_key(ip)
        entry = self._hits.get(key)
        if entry is None or entry.reset_at <= t:
            entry = _Entry(count=0, reset_at=t + self._options.window_ms)
            self._hits[key] = entry
        entry.count += 1
        if entry.count <= self._options.max:
            return None
        if self._options.on_limited is not None:
            self._options.on_limited(ip)
        return json_response(
            429,
            {"error": "temporarily_unavailable", "error_description": "Too many requests, try again later."},
            {"Retry-After": str(max(1, math.ceil((entry.reset_at - t) / 1000)))},
        )
