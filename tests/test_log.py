"""Tests of the JSON logger."""

from __future__ import annotations

import errno
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from voice_coding_mcp.log import LogLevel, create_logger, fingerprint, silent_logger, strip_query


def capture(level: LogLevel | None = None) -> tuple[Any, list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []
    logger = create_logger(
        **({"level": level} if level else {}),
        write=lambda line: lines.append(json.loads(line)),
        now=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    return logger, lines


class TestCreateLogger:
    def test_writes_json_lines_with_ts_level_msg_fields(self) -> None:
        logger, lines = capture()
        logger.info("hello", {"a": 1, "b": "x"})
        assert lines == [{"ts": "2026-01-02T03:04:05.000Z", "level": "info", "msg": "hello", "a": 1, "b": "x"}]
        assert list(lines[0])[:3] == ["ts", "level", "msg"]

    def test_compact_line(self) -> None:
        raw: list[str] = []
        create_logger(write=raw.append, now=lambda: datetime(2026, 1, 2, tzinfo=UTC)).info("é", {"n": 1})
        assert raw == ['{"ts":"2026-01-02T00:00:00.000Z","level":"info","msg":"é","n":1}']

    def test_filters_below_level(self) -> None:
        logger, lines = capture("warn")
        logger.debug("d")
        logger.info("i")
        logger.warn("w")
        logger.error("e")
        assert [line["level"] for line in lines] == ["warn", "error"]

    def test_defaults_to_info(self) -> None:
        logger, lines = capture()
        logger.debug("d")
        logger.info("i")
        assert len(lines) == 1

    def test_child_loggers_merge_and_nest(self) -> None:
        logger, lines = capture()
        logger.child({"req": 1}).child({"task": 7}).info("x", {"extra": True})
        assert lines[0] == {
            "ts": "2026-01-02T03:04:05.000Z",
            "level": "info",
            "msg": "x",
            "req": 1,
            "task": 7,
            "extra": True,
        }

    def test_reserved_keys_not_overridden(self) -> None:
        logger, lines = capture()
        logger.info("real", {"msg": "fake", "level": "error", "ts": "never"})
        assert lines[0]["msg"] == "real"
        assert lines[0]["level"] == "info"
        assert lines[0]["ts"] == "2026-01-02T03:04:05.000Z"

    def test_none_fields_dropped(self) -> None:
        logger, lines = capture()
        logger.info("x", {"a": None, "b": 0})
        assert lines[0] == {"ts": "2026-01-02T03:04:05.000Z", "level": "info", "msg": "x", "b": 0}

    def test_serializes_errors_and_big_ints(self) -> None:
        logger, lines = capture()
        err = ValueError("boom")
        err.code = "EBOOM"  # type: ignore[attr-defined]
        logger.error("failed", {"err": err, "n": 10**20})
        assert lines[0]["err"] == {"name": "ValueError", "message": "boom", "code": "EBOOM"}
        assert lines[0]["n"] == 10**20

    def test_os_error_code_name(self) -> None:
        logger, lines = capture()
        logger.error("failed", {"err": FileNotFoundError(errno.ENOENT, "No such file")})
        assert lines[0]["err"]["code"] == "ENOENT"
        assert lines[0]["err"]["name"] == "FileNotFoundError"

    def test_survives_unserializable_fields(self) -> None:
        logger, lines = capture()
        cyclic: dict[str, Any] = {}
        cyclic["self"] = cyclic
        logger.info("cyclic", {"cyclic": cyclic})
        logger.info("object", {"thing": object()})
        assert lines[0]["msg"] == "cyclic"
        assert lines[0]["logError"] == "unserializable fields"
        assert lines[1]["logError"] == "unserializable fields"


def test_silent_logger() -> None:
    silent_logger.info("nothing")
    assert silent_logger.child({"a": 1}) is silent_logger


class TestFingerprint:
    def test_length_and_hash_prefix(self) -> None:
        secret = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
        digest = hashlib.sha256(secret.encode()).hexdigest()
        assert fingerprint(secret) == f"len=43 sha256={digest[:8]}"
        assert "abcdef" not in fingerprint(secret)

    def test_length_counts_code_points(self) -> None:
        assert fingerprint("😀").startswith("len=1 ")
        assert fingerprint("é").startswith("len=1 ")


def test_strip_query() -> None:
    assert strip_query("/authorize?code=secret&state=x") == "/authorize"
    assert strip_query("/a#frag") == "/a"
    assert strip_query("https://h.example/p?q=1#f") == "https://h.example/p"
    assert strip_query("/plain") == "/plain"
