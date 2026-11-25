"""Time helpers shared by every module.

All clocks and timestamps in state files are integer milliseconds since the epoch. ISO strings are
ISO 8601 in UTC with millisecond precision and a `Z` suffix (`2026-01-02T03:04:05.000Z`); existing
state files use this format, so it must not change.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

from .text import plural

Clock = Callable[[], int]
"""Returns the current time in integer milliseconds since the epoch."""

SECOND_MS = 1_000
MINUTE_MS = 60 * SECOND_MS
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS


def elapsed_minutes(ms: int | float) -> int:
    """A duration in milliseconds as whole minutes, rounded, never negative."""
    return max(0, round(ms / MINUTE_MS))


def ago(ms: int | float) -> str:
    """An elapsed duration in milliseconds as "just now", "5 minutes ago", "3 hours ago" or
    "2 days ago", for spoken answers and the CLI alike."""
    minutes = elapsed_minutes(ms)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{plural(minutes, 'minute')} ago"
    hours = round(minutes / 60)
    if hours < 24:
        return f"{plural(hours, 'hour')} ago"
    return f"{plural(round(hours / 24), 'day')} ago"


def now_ms() -> int:
    """Current time in integer milliseconds since the epoch."""
    return time.time_ns() // 1_000_000


def to_iso(ms: int | float) -> str:
    """Formats epoch milliseconds as ISO 8601 UTC with milliseconds (`...T03:04:05.000Z`)."""
    whole = int(ms // 1000)
    millis = int(ms - whole * 1000)
    dt = datetime.fromtimestamp(whole, tz=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{millis:03d}Z"


def datetime_to_iso(dt: datetime) -> str:
    """Formats a datetime like `to_iso` (naive datetimes are taken as UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_iso(value: str | None) -> int | None:
    """Epoch milliseconds of an ISO 8601 string, or None when it cannot be parsed."""
    if not value:
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)
