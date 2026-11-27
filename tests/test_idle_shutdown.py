"""The idle shutdown: the check with a fake clock, what counts as activity on /mcp, what keeps the
server busy in the task manager, and `serve` stopping itself (with the CLI override)."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from tests.app_helpers import AppCtx, start_app
from tests.helpers import FakeClock, LogCapture, claude_config, create_grant, tasks_config, until
from tests.session_fixtures import assistant_text, write_transcript
from tests.test_cli import free_port, run_cli, write_config
from voice_coding_mcp import cli
from voice_coding_mcp.auth.store import NewClient, NewGrant
from voice_coding_mcp.clock import MINUTE_MS, now_ms
from voice_coding_mcp.config import SessionsConfig
from voice_coding_mcp.idle_shutdown import Activity, IdleShutdown
from voice_coding_mcp.log import silent_logger
from voice_coding_mcp.sessions import LiveSession, ReplyWatchOptions, ReplyWatchResult, SessionInfo
from voice_coding_mcp.tasks.manager import Started, TaskManager, TaskManagerOptions

T0 = 1_800_000_000_000


class Env:
    def __init__(self, *, idle_minutes: int = 10) -> None:
        self.clock = FakeClock(T0)
        self.activity = Activity(self.clock)
        self.busy = False
        self.fired = 0
        self.log = LogCapture()
        self.idle = IdleShutdown(
            self.activity,
            busy=lambda: self.busy,
            idle_minutes=idle_minutes,
            on_idle=self.on_idle,
            logger=self.log.logger,
            now=self.clock,
        )

    def on_idle(self) -> None:
        self.fired += 1

    def advance(self, minutes: float) -> None:
        self.clock.t += int(minutes * MINUTE_MS)


# --- the check ------------------------------------------------------------------------------------


def test_fires_once_after_the_idle_time_counted_from_the_start() -> None:
    env = Env()
    env.advance(9)
    assert not env.idle.check()
    env.advance(1)
    assert env.idle.check()
    assert env.fired == 1
    assert not env.idle.check()
    assert env.fired == 1
    [record] = [r for r in env.log.records if r["msg"] == "idle shutdown"]
    assert record["level"] == "info"
    assert record["idleMinutes"] == 10


def test_zero_turns_it_off() -> None:
    env = Env(idle_minutes=0)
    assert not env.idle.enabled
    env.advance(100_000)
    assert not env.idle.check()
    assert env.fired == 0


def test_a_tool_call_resets_the_idle_time() -> None:
    env = Env()
    env.advance(9)
    with env.activity.call():
        pass
    env.advance(9)
    assert not env.idle.check()
    env.advance(1)
    assert env.idle.check()


def test_busy_work_blocks_it_and_the_idle_time_counts_from_when_it_ended() -> None:
    env = Env()
    env.advance(30)
    env.busy = True
    assert not env.idle.check()
    env.advance(60)
    assert not env.idle.check()
    env.busy = False
    env.advance(9)
    assert not env.idle.check(), "idle since the last check that saw work, 9 minutes ago"
    env.advance(1)
    assert env.idle.check()


def test_a_tool_call_still_being_handled_blocks_it() -> None:
    env = Env()
    with env.activity.call():
        env.advance(60)
        assert not env.idle.check()
    env.advance(9)
    assert not env.idle.check()
    env.advance(1)
    assert env.idle.check()


async def test_the_loop_checks_periodically_and_stops() -> None:
    env = Env(idle_minutes=1)
    fast = IdleShutdown(
        env.activity,
        busy=lambda: False,
        idle_minutes=1,
        on_idle=env.on_idle,
        logger=silent_logger,
        now=env.clock,
        interval_ms=5,
    )
    fast.start()
    await asyncio.sleep(0.05)
    assert env.fired == 0
    env.advance(1)
    await until(lambda: env.fired == 1)
    await fast.stop()
    await fast.stop()


# --- activity on /mcp -----------------------------------------------------------------------------


@pytest.fixture
async def app(real_tmp: Path) -> AsyncIterator[tuple[AppCtx, Activity, FakeClock]]:
    clock = FakeClock(T0)
    activity = Activity(clock)
    async with start_app(real_tmp, activity=activity) as ctx:
        yield ctx, activity, clock


async def _token(ctx: AppCtx) -> str:
    client = await ctx.store.register_client(
        NewClient(
            client_name="Test",
            redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
            token_endpoint_auth_method="none",
        )
    )
    tokens = await create_grant(
        ctx.store,
        NewGrant(client_id=client.client.client_id, client_name="Test", scope="mcp", resource=ctx.config.resource_url),
    )
    return tokens.access_token


def _rpc(method: str, params: dict[str, Any] | None = None) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})


INIT_PARAMS = {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}


async def test_only_an_authenticated_tool_call_is_activity(app: tuple[AppCtx, Activity, FakeClock]) -> None:
    ctx, activity, clock = app
    headers = {
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
        "mcp-protocol-version": "2025-11-25",
    }
    auth = {**headers, "authorization": f"Bearer {await _token(ctx)}"}
    clock.t = T0 + MINUTE_MS
    call = _rpc("tools/call", {"name": "list_projects"})

    assert (await ctx.http.post("/mcp", headers=headers, content=call)).status_code == 401
    bad = {**headers, "authorization": "Bearer not-a-token"}
    assert (await ctx.http.post("/mcp", headers=bad, content=call)).status_code == 401
    assert (await ctx.http.post("/mcp", headers=auth, content=_rpc("initialize", INIT_PARAMS))).status_code == 200
    assert (await ctx.http.post("/mcp", headers=auth, content=_rpc("tools/list"))).status_code == 200
    assert (await ctx.http.get("/.well-known/oauth-authorization-server")).status_code == 200
    await ctx.http.get("/authorize")
    assert (await ctx.http.get("/wp-login.php")).status_code == 404
    assert activity.last_ms == T0

    res = await ctx.http.post("/mcp", headers=auth, content=call)
    assert res.status_code == 200
    assert activity.last_ms == T0 + MINUTE_MS
    assert activity.in_flight == 0


# --- busy work in the task manager ----------------------------------------------------------------


@pytest.fixture
async def projects(real_tmp: Path) -> Path:
    root = real_tmp / "projects"
    for name in ("alpha", "assistant"):
        (root / name).mkdir(parents=True)
    return root


async def _manager(real_tmp: Path, root: Path, **overrides: Any) -> TaskManager:
    return await TaskManager.create(
        TaskManagerOptions(
            state_dir=str(real_tmp / "state"),
            project_roots=[str(root)],
            claude=claude_config(str(real_tmp / "record.jsonl")),
            tasks=tasks_config(),
            sessions=SessionsConfig(days=7, reply_wait_minutes=1),
            logger=silent_logger,
            kill_grace_ms=500,
            reply_poll_ms=20,
            **overrides,
        )
    )


async def test_a_headless_task_keeps_the_server_busy(real_tmp: Path, projects: Path) -> None:
    manager = await _manager(real_tmp, projects)
    try:
        assert not manager.has_work()
        started = await manager.start("alpha", "SLEEP 60000")
        assert isinstance(started, Started)
        assert manager.has_work()
        await manager.cancel(started.task.id)
        await until(lambda: not manager.has_work())
    finally:
        await manager.shutdown("test end")


async def test_a_relay_waiting_for_its_reply_keeps_the_server_busy(real_tmp: Path, projects: Path) -> None:
    reply = asyncio.Event()
    watching = asyncio.Event()

    async def watch(_options: ReplyWatchOptions) -> ReplyWatchResult:
        watching.set()
        await reply.wait()
        return ReplyWatchResult(arrived=True, finished=True, reply="OK.", aborted=False)

    manager = await _manager(real_tmp, projects, watch_reply=watch)
    try:
        cwd = str(projects / "assistant")
        session_id = "22222222-0000-4000-8000-000000000001"
        transcript = write_transcript(str(real_tmp / "home"), session_id, [assistant_text(cwd, "earlier work")])
        target = SessionInfo(
            id=session_id,
            name="billing-fix",
            titled=True,
            project="assistant",
            cwd=cwd,
            last_activity_ms=now_ms(),
            transcript_path=transcript,
            live=LiveSession(pid=1, socket_path="/tmp/cc-socks/1.sock", status="idle", name="billing-fix"),
        )
        started = await manager.start_session("send", target, "Check the tests.\nRESULT SENT")
        assert isinstance(started, Started)
        await asyncio.wait_for(watching.wait(), 10)
        assert manager.has_work(), "no run slot while waiting, but still in progress"
        reply.set()
        await until(lambda: not manager.has_work())
    finally:
        await manager.shutdown("test end")


async def test_a_remote_control_session_task_does_not_keep_the_server_busy(real_tmp: Path, projects: Path) -> None:
    manager = await _manager(real_tmp, projects)
    try:
        started = manager.start_remote("alpha", "Fix the build.", title="Fix build")
        assert isinstance(started, Started)
        assert started.task.status == "running"
        assert not manager.has_work()
    finally:
        await manager.shutdown("test end")


# --- serve ----------------------------------------------------------------------------------------


@pytest.fixture
def seen_minutes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replaces the server's IdleShutdown with one that checks every 10 ms on a clock that moves a
    minute per reading; records the idle minutes it got. With 0 it stops the server at once by
    itself, so the test ends."""
    seen: list[int] = []

    class Fast(IdleShutdown):
        def __init__(self, activity: Activity, **kwargs: Any) -> None:
            seen.append(kwargs["idle_minutes"])
            clock = FakeClock(now_ms())

            def tick() -> int:
                clock.t += MINUTE_MS
                return clock.t

            super().__init__(activity, **{**kwargs, "now": tick, "interval_ms": 10})
            if kwargs["idle_minutes"] == 0:
                asyncio.get_running_loop().call_later(0.1, kwargs["on_idle"])

    monkeypatch.setattr(cli, "IdleShutdown", Fast)
    return seen


@pytest.mark.parametrize(
    "case",
    [(0, ["--idle-shutdown-minutes", "1"], 1), (2, [], 2), (5, ["--idle-shutdown-minutes", "0"], 0)],
)
async def test_serve_stops_itself_when_idle_and_the_flag_overrides_the_config(
    real_tmp: Path,
    seen_minutes: list[int],
    capsys: pytest.CaptureFixture[str],
    case: tuple[int, list[str], int],
) -> None:
    config_minutes, flag, expected = case
    path = write_config(real_tmp, listen={"host": "127.0.0.1", "port": free_port()}, idleShutdownMinutes=config_minutes)
    code, _io = await asyncio.wait_for(run_cli("serve", "--config", path, *flag, env=dict(os.environ)), 20)
    assert code == cli.EXIT_OK
    assert seen_minutes == [expected]
    msgs = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    stopping = next(m for m in msgs if m["msg"] == "server stopping")
    assert stopping["reason"] == "idle shutdown"
    assert msgs[-1]["msg"] == "server stopped"
    idle = [m for m in msgs if m["msg"] == "idle shutdown"]
    if expected:
        assert len(idle) == 1
        assert idle[0]["idleMinutes"] >= expected
    else:
        assert idle == []
