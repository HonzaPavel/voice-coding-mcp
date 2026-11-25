"""The idle close of the server's Remote Control sessions, with a fake clock and a stub remote."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from tests.helpers import FakeClock, LogCapture, claude_config, tasks_config
from voice_coding_mcp.clock import MINUTE_MS
from voice_coding_mcp.idle_close import IdleCloser
from voice_coding_mcp.remote_control import SKIPPED
from voice_coding_mcp.sessions import LiveSession, rc_disconnected
from voice_coding_mcp.tasks.manager import Started, Task, TaskManager, TaskManagerOptions

SESSION = "88888888-0000-4000-8000-000000000001"
REOPENED = "88888888-0000-4000-8000-000000000002"
T0 = 1_800_000_000_000


@dataclass
class StubRemote:
    clock: FakeClock
    live: dict[str, LiveSession] = field(default_factory=dict)
    stopped: list[tuple[int, str]] = field(default_factory=list)
    closed: list[tuple[int, str]] = field(default_factory=list)
    fail: bool = False
    before_lock: Callable[[], None] | None = None
    """Runs where the real RemoteControl waits for its lock: the state may change meanwhile."""

    def read_live(self) -> dict[str, LiveSession]:
        return dict(self.live)

    def disconnected(self, live: LiveSession) -> bool:
        return rc_disconnected(live, self.clock())

    def _rejected(self, sid: str, only_if: Callable[[str, LiveSession], bool] | None) -> bool:
        if self.before_lock is not None:
            self.before_lock()
        entry = self.live.get(sid)
        return only_if is not None and (entry is None or not only_if(sid, entry))

    async def close(
        self, task: Task, *, reason: str = "cancel", only_if: Callable[[str, LiveSession], bool] | None = None
    ) -> str:
        """Records a session task's stop in `stopped`, a reopened task's in `closed`."""
        sid = task.remote_session_id or task.session_id or ""
        if self._rejected(sid, only_if):
            return SKIPPED
        if self.fail:
            return "could not stop"
        (self.stopped if task.is_session_task else self.closed).append((task.id, reason))
        self.live.pop(sid, None)
        return "closed"


def bg(status: str, *, updated: int | None, bridge: bool = True, started: int = T0) -> LiveSession:
    return LiveSession(
        pid=1,
        socket_path=None,
        status=status,
        name="x",
        kind="bg",
        bridge_session_id="session_01X" if bridge else None,
        started_at_ms=started,
        status_updated_at_ms=updated,
    )


@dataclass
class Env:
    manager: TaskManager
    remote: StubRemote
    clock: FakeClock
    closer: IdleCloser
    log: LogCapture


@pytest.fixture
async def env(real_tmp: Path, log_capture: LogCapture) -> AsyncIterator[Env]:
    root = real_tmp / "projects"
    (root / "alpha").mkdir(parents=True)
    clock = FakeClock(T0)
    manager = await TaskManager.create(
        TaskManagerOptions(
            state_dir=str(real_tmp / "state"),
            project_roots=[str(root)],
            claude=claude_config(str(real_tmp / "record.jsonl")),
            tasks=tasks_config(),
            logger=log_capture.logger,
            now=clock,
            live_session_ids=set,
        )
    )
    remote = StubRemote(clock)
    closer = IdleCloser(manager, remote, idle_minutes=60, logger=log_capture.logger, now=clock)
    yield Env(manager, remote, clock, closer, log_capture)
    await manager.shutdown("test end")


def session_task(env: Env) -> Task:
    started = env.manager.start_remote("alpha", "Oprav testy", title="Opravit testy")
    assert isinstance(started, Started)
    env.manager.remote_launched(started.task.id, session_id=SESSION, bg_id=SESSION[:8])
    return started.task


async def test_stops_a_session_task_idle_for_the_configured_time(env: Env) -> None:
    task = session_task(env)
    env.remote.live[SESSION] = bg("idle", updated=T0)
    env.clock.t = T0 + 59 * MINUTE_MS
    assert await env.closer.check() == []
    env.clock.t = T0 + 60 * MINUTE_MS
    assert await env.closer.check() == [task.id]
    assert env.remote.stopped == [(task.id, "idle")]
    after = env.manager.get(task.id)
    assert after is not None
    assert (after.status, after.error) == ("done", None)
    [entry] = [r for r in env.log.records if r["msg"] == "idle session closed"]
    assert (entry["taskId"], entry["state"], entry["idleMinutes"]) == (task.id, "idle", 60)
    # Logs never carry the title (prompt-derived text).
    assert "Opravit" not in env.log.text()


async def test_never_closes_sessions_that_work_or_wait(env: Env) -> None:
    session_task(env)
    for status in ("busy", "waiting"):
        env.remote.live[SESSION] = bg(status, updated=T0)
        env.clock.t = T0 + 5 * 60 * MINUTE_MS
        assert await env.closer.check() == []
    assert env.remote.stopped == []


async def test_without_status_updated_at_the_idle_time_starts_when_first_seen(env: Env) -> None:
    task = session_task(env)
    env.remote.live[SESSION] = bg("idle", updated=None)
    env.clock.t = T0 + 10 * MINUTE_MS
    assert await env.closer.check() == []
    env.clock.t = T0 + 69 * MINUTE_MS
    assert await env.closer.check() == []
    # Busy in between resets it.
    env.remote.live[SESSION] = bg("busy", updated=None)
    assert await env.closer.check() == []
    env.remote.live[SESSION] = bg("idle", updated=None)
    env.clock.t = T0 + 70 * MINUTE_MS
    assert await env.closer.check() == []
    env.clock.t = T0 + 130 * MINUTE_MS
    assert await env.closer.check() == [task.id]


async def test_closes_a_disconnected_session_after_the_time_seen_disconnected(env: Env) -> None:
    session_task(env)
    # Busy but without bridgeSessionId long after start: disconnected from the app.
    env.remote.live[SESSION] = bg("busy", updated=T0, bridge=False)
    env.clock.t = T0 + 5 * MINUTE_MS
    assert await env.closer.check() == []
    env.clock.t = T0 + 65 * MINUTE_MS
    assert len(await env.closer.check()) == 1
    assert env.remote.stopped[0][1] == "idle"


async def test_closes_the_reopened_session_of_a_finished_task(env: Env) -> None:
    task = session_task(env)
    env.manager.session_closed(task.id, "cancel")
    env.manager.set_remote_session(task.id, REOPENED, bg_id=REOPENED[:8])
    env.remote.live[REOPENED] = bg("idle", updated=T0)
    env.clock.t = T0 + 61 * MINUTE_MS
    assert await env.closer.check() == [task.id]
    assert env.remote.closed == [(task.id, "idle")]
    assert env.remote.stopped == []
    after = env.manager.get(task.id)
    assert after is not None
    assert after.status == "cancelled"


async def test_off_with_zero_and_failures_are_logged(env: Env) -> None:
    session_task(env)
    env.remote.live[SESSION] = bg("idle", updated=T0)
    env.clock.t = T0 + 120 * MINUTE_MS
    off = IdleCloser(env.manager, env.remote, idle_minutes=0, logger=env.log.logger, now=env.clock)
    assert not off.enabled
    assert await off.check() == []
    env.remote.fail = True
    assert await env.closer.check() == []
    assert any(r["msg"] == "idle session close failed" for r in env.log.records)
    running = env.manager.get(1)
    assert running is not None
    assert running.status == "running"


async def test_ignores_sessions_of_other_owners(env: Env) -> None:
    env.remote.live["99999999-0000-4000-8000-000000000001"] = replace(bg("idle", updated=T0), kind="interactive")
    env.clock.t = T0 + 600 * MINUTE_MS
    assert await env.closer.check() == []


async def test_rechecks_under_the_lock_and_skips_a_session_that_changed(env: Env) -> None:
    task = session_task(env)
    env.remote.live[SESSION] = bg("idle", updated=T0)
    env.clock.t = T0 + 61 * MINUTE_MS

    # The owner started a new turn while the stop waited for the lock: not stopped.
    def owner_types() -> None:
        env.remote.live[SESSION] = bg("busy", updated=env.clock.t)

    env.remote.before_lock = owner_types
    assert await env.closer.check() == []
    assert env.remote.stopped == []
    after = env.manager.get(task.id)
    assert after is not None
    assert after.status == "running"
    assert any(r["msg"] == "idle close skipped: session changed" for r in env.log.records)

    # Idle again, but only just (statusUpdatedAt moved): not due yet under the lock either.
    def fresh_turn_done() -> None:
        env.remote.live[SESSION] = bg("idle", updated=env.clock.t - MINUTE_MS)

    env.remote.live[SESSION] = bg("idle", updated=T0)
    env.remote.before_lock = fresh_turn_done
    assert await env.closer.check() == []
    assert env.remote.stopped == []
