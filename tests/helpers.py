"""Test helpers shared by the test modules (fake claude, task config, log capture).

- `FAKE_CLAUDE`: path of the fake `claude` executable (tests/fixtures/fake_claude.py).
- `claude_config()` / `tasks_config()`: config sections pointing at the fake.
- `read_records()`: what the fake recorded (argv, env, stdin, pid, cwd, childPid) per run.
- `until()`: polls a condition; `is_alive()`: process check.
- `LogCapture`: a debug logger that keeps its JSON lines.
- `FakeClock`: a settable clock in epoch ms; `config_data()`: the smallest valid config file data.
- `trace_lock_io()`: records flock, read and write calls of a module that edits a locked state file.
"""

from __future__ import annotations

import asyncio
import fcntl
import inspect
import json
import os
import stat
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

import pytest

from voice_coding_mcp.auth.store import IssuedTokens, NewGrant, OAuthStore
from voice_coding_mcp.config import ClaudeConfig, TasksConfig
from voice_coding_mcp.log import Logger, create_logger

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_CLAUDE = str(FIXTURES / "fake_claude.py")

# The fake is started directly (argv[0], no shell), so it must stay executable.
_mode = os.stat(FAKE_CLAUDE).st_mode
if not _mode & stat.S_IXUSR:
    os.chmod(FAKE_CLAUDE, _mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def claude_config(record_path: str, **overrides: Any) -> ClaudeConfig:
    """ClaudeConfig running the fake claude, which records each run to `record_path`.

    Overrides use snake_case names, e.g. `claude_config(rec, max_turns=3)`.
    """
    base: dict[str, Any] = {
        "bin": FAKE_CLAUDE,
        "model": None,
        "effort": None,
        "max_turns": None,
        "permission_mode": "auto",
        "setting_sources": ["user", "local"],
        "config_dir": None,
        "mcp_servers": {},
        "extra_args": [],
        "env": {"FAKE_CLAUDE_RECORD": record_path},
    }
    return ClaudeConfig(**{**base, **overrides})


def tasks_config(**overrides: Any) -> TasksConfig:
    """TasksConfig with the defaults, e.g. `tasks_config(max_concurrent=1)`."""
    base: dict[str, Any] = {
        "sync_wait_seconds": 25,
        "timeout_minutes": 15,
        "max_concurrent": 2,
        "max_queued": 10,
        "resume_within_hours": 12,
        "keep_days": 14,
    }
    return TasksConfig(**{**base, **overrides})


class FakeRecord(TypedDict):
    argv: list[str]
    env: dict[str, str]
    stdin: str
    pid: int
    cwd: str
    childPid: int | None


def read_records(path: str) -> list[FakeRecord]:
    """One record per fake claude run, in start order; empty when nothing ran."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    records: list[FakeRecord] = [json.loads(line) for line in text.split("\n") if line]
    return records


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def until(check: Callable[[], bool | Awaitable[bool]], timeout_ms: int = 5000) -> None:
    """Polls `check` every 20 ms until it is true; raises AssertionError after `timeout_ms`."""
    end = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < end:
        result = check()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met in time")


class LogCapture:
    """A debug-level JSON logger that keeps every line."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.logger: Logger = create_logger(level="debug", write=self.lines.append)

    @property
    def records(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.lines]

    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class FakeClock:
    """A `Clock` (epoch ms) that only moves when a test sets or advances `t`. Pass the instance or
    its bound `now`."""

    t: int

    def __call__(self) -> int:
        return self.t

    def now(self) -> int:
        return self.t


async def create_grant(store: OAuthStore, new: NewGrant) -> IssuedTokens:
    """A grant with fresh tokens, without the OAuth flow. Built on the store's internals (the
    endpoints only create grants through `exchange_authorization_code`)."""

    def op() -> IssuedTokens:
        tokens = store._add_grant(new, None)
        store._save()
        return tokens

    return await store._run(op)


def config_data(public_url: str, state_dir: str, project_root: str, **sections: Any) -> dict[str, Any]:
    """Config file data (camelCase keys) with the required keys plus `sections`, e.g. `oauth={...}`."""
    return {"publicUrl": public_url, "stateDir": state_dir, "projectRoots": [project_root], **sections}


def _traced(real: Callable[..., Any], event: str, events: list[str]) -> Callable[..., Any]:
    def traced(*args: Any) -> Any:
        events.append(event)
        return real(*args)

    return traced


def trace_lock_io(monkeypatch: pytest.MonkeyPatch, module: ModuleType, lock_path: str) -> list[str]:
    """Records, in order, `lock` and `unlock` for each flock (which must be on `lock_path`), and `read`
    and `write` for each call of `module.read_json` and `module.write_json_atomic(_identity)`."""
    events: list[str] = []
    real_flock = fcntl.flock

    def flock(fd: int, op: int) -> None:
        assert os.fstat(fd).st_ino == os.stat(lock_path).st_ino
        events.append({fcntl.LOCK_EX: "lock", fcntl.LOCK_UN: "unlock"}.get(op, str(op)))
        real_flock(fd, op)

    monkeypatch.setattr(fcntl, "flock", flock)
    for name, event in (
        ("read_json", "read"),
        ("write_json_atomic", "write"),
        ("write_json_atomic_identity", "write"),
    ):
        real = getattr(module, name, None)
        if real is not None:
            monkeypatch.setattr(module, name, _traced(real, event, events))
    return events
