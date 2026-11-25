"""JSON-lines logger to stdout and helpers that keep secrets out of logs.

Each line is one JSON object: `ts` (ISO, UTC, milliseconds), `level`, `msg`, then the fields of
the logger (from `child()`) and of the call. Fields named `ts`, `level` or `msg` and fields whose
value is None are dropped (None means "not set"; use a sentinel string if a null must be logged).
"""

from __future__ import annotations

import errno
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from .clock import datetime_to_iso

LogLevel = Literal["debug", "info", "warn", "error"]
LogFields = Mapping[str, Any]

LOG_LEVELS: tuple[LogLevel, ...] = ("debug", "info", "warn", "error")

_RESERVED = frozenset({"ts", "level", "msg"})


class Logger(Protocol):
    def debug(self, msg: str, fields: LogFields | None = None) -> None: ...
    def info(self, msg: str, fields: LogFields | None = None) -> None: ...
    def warn(self, msg: str, fields: LogFields | None = None) -> None: ...
    def error(self, msg: str, fields: LogFields | None = None) -> None: ...
    def child(self, fields: LogFields) -> Logger:
        """Returns a logger that adds `fields` to every line."""
        ...


def _rank(level: LogLevel) -> int:
    return LOG_LEVELS.index(level)


def _error_fields(err: BaseException) -> dict[str, Any]:
    out: dict[str, Any] = {"name": type(err).__name__, "message": str(err)}
    code = getattr(err, "code", None)
    if code is None and isinstance(err, OSError) and err.errno is not None:
        code = errno.errorcode.get(err.errno, err.errno)
    if code is not None:
        out["code"] = code
    return out


def _default(value: object) -> object:
    if isinstance(value, BaseException):
        return _error_fields(value)
    raise TypeError(f"not serializable: {type(value).__name__}")


def _stdout_write(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class JsonLogger:
    """The logger returned by `create_logger`."""

    def __init__(
        self,
        min_rank: int,
        write: Callable[[str], None],
        now: Callable[[], datetime],
        base: Mapping[str, Any],
    ) -> None:
        self._min = min_rank
        self._write = write
        self._now = now
        self._base = dict(base)

    def _emit(self, level: LogLevel, msg: str, fields: LogFields | None) -> None:
        if _rank(level) < self._min:
            return
        ts = datetime_to_iso(self._now())
        line: dict[str, Any] = {"ts": ts, "level": level, "msg": msg}
        for source in (self._base, fields):
            if not source:
                continue
            for key, value in source.items():
                if key not in _RESERVED and value is not None:
                    line[key] = value
        try:
            text = json.dumps(line, default=_default, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError, RecursionError):
            text = json.dumps(
                {"ts": ts, "level": level, "msg": msg, "logError": "unserializable fields"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        self._write(text)

    def debug(self, msg: str, fields: LogFields | None = None) -> None:
        self._emit("debug", msg, fields)

    def info(self, msg: str, fields: LogFields | None = None) -> None:
        self._emit("info", msg, fields)

    def warn(self, msg: str, fields: LogFields | None = None) -> None:
        self._emit("warn", msg, fields)

    def error(self, msg: str, fields: LogFields | None = None) -> None:
        self._emit("error", msg, fields)

    def child(self, fields: LogFields) -> Logger:
        return JsonLogger(self._min, self._write, self._now, {**self._base, **fields})


def create_logger(
    level: LogLevel = "info",
    write: Callable[[str], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> Logger:
    """A JSON-lines logger. `write` receives one line without the trailing newline (default: stdout);
    `now` is the clock for the `ts` field, for tests."""
    return JsonLogger(
        _rank(level),
        write or _stdout_write,
        now or (lambda: datetime.now(UTC)),
        {},
    )


class _SilentLogger:
    def debug(self, msg: str, fields: LogFields | None = None) -> None:
        pass

    def info(self, msg: str, fields: LogFields | None = None) -> None:
        pass

    def warn(self, msg: str, fields: LogFields | None = None) -> None:
        pass

    def error(self, msg: str, fields: LogFields | None = None) -> None:
        pass

    def child(self, fields: LogFields) -> Logger:
        return self


silent_logger: Logger = _SilentLogger()
"""A logger that drops everything, for tests."""


def fingerprint(secret: str) -> str:
    """Identifies a secret in logs without revealing it: `len=43 sha256=1a2b3c4d`.

    The length counts code points.
    """
    digest = hashlib.sha256(secret.encode("utf-8", "surrogatepass")).hexdigest()
    return f"len={len(secret)} sha256={digest[:8]}"


_QUERY_RE = re.compile(r"[?#]")


def strip_query(url: str) -> str:
    """Drops the query string and fragment from a URL or path, for logs."""
    match = _QUERY_RE.search(url)
    return url if match is None else url[: match.start()]


def as_log_level(value: str | None, default: LogLevel = "info") -> LogLevel:
    """`value` when it is a log level, else `default` (for `VCM_LOG_LEVEL`)."""
    for level in LOG_LEVELS:
        if value == level:
            return level
    return default
