"""resume_task_in_remote_control: the RemoteControl module with the fake claude's `--bg` mode."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import LogCapture, claude_config, read_records
from tests.session_fixtures import claude_home, stop_background_sessions, write_live_session
from voice_coding_mcp.remote_control import (
    SKIPPED,
    RcFailed,
    RcOpened,
    RemoteControl,
    RemoteControlOptions,
    build_resume_args,
    build_start_args,
    desktop_resume_url,
    format_opened,
    stop_hook_settings,
)
from voice_coding_mcp.sessions import read_live_sessions
from voice_coding_mcp.tasks import manager as manager_module
from voice_coding_mcp.tasks.access import RunAccess, isolated_access
from voice_coding_mcp.tasks.manager import TITLE_MAX_CHARS, Task, format_session_name, sanitize_title, session_name
from voice_coding_mcp.tasks.prompt import REMOTE_SESSION_PROMPT

SESSION = "44444444-0000-4000-8000-000000000001"


@dataclass
class Env:
    root: Path
    home: str
    project: str
    record: str
    log: LogCapture

    def rc(
        self,
        *,
        bg: str = "",
        max_sessions: int = 3,
        link_wait_ms: int = 3000,
        extra_env: dict[str, str] | None = None,
        public_url: str | None = None,
        options: dict[str, Any] | None = None,
        **claude: Any,
    ) -> RemoteControl:
        env = {"FAKE_CLAUDE_RECORD": self.record, "FAKE_CLAUDE_BG": bg, **(extra_env or {})}
        return RemoteControl(
            RemoteControlOptions(
                claude=claude_config(self.record, config_dir=self.home, env=env, **claude),
                claude_home=self.home,
                mcp_config_path=str(self.root / "mcp-config.json"),
                max_sessions=max_sessions,
                logger=self.log.logger,
                link_wait_ms=link_wait_ms,
                poll_ms=50,
                public_url=public_url,
                **(options or {}),
            )
        )


@pytest.fixture
def env(real_tmp: Path, log_capture: LogCapture) -> Iterator[Env]:
    project = real_tmp / "projects" / "Alpha App"
    project.mkdir(parents=True)
    home = claude_home(str(real_tmp))
    try:
        yield Env(real_tmp, home, str(project), str(real_tmp / "record.jsonl"), log_capture)
    finally:
        stop_background_sessions(home)


def make_task(env: Env, **overrides: Any) -> Task:
    base: dict[str, Any] = {
        "id": 7,
        "kind": "task",
        "project": "Alpha App",
        "project_path": env.project,
        "prompt": "Oprav testy",
        "status": "done",
        "new_session": False,
        "created_at": "2026-11-24T10:00:00.000Z",
        "started_at": "2026-11-24T10:00:00.000Z",
        "finished_at": "2026-11-24T10:01:00.000Z",
        "resumed_session_id": None,
        "session_id": SESSION,
        "result": "Hotovo.",
        "error": None,
        "progress": [],
        "cost_usd": None,
        "duration_ms": None,
        "num_turns": None,
        "permission_denials": 0,
        "exit_code": 0,
    }
    return Task(**{**base, **overrides})


def test_session_name_uses_the_title_else_the_project_never_the_prompt(env: Env) -> None:
    assert session_name(make_task(env)) == "Alpha App (voice 7)"
    assert session_name(make_task(env, project="Školní web", id=12)) == "Školní web (voice 12)"
    assert session_name(make_task(env, name="Oprava tlačítka (voice 7)")) == "Oprava tlačítka (voice 7)"
    assert "Oprav testy" not in session_name(make_task(env))
    # Task files from before titles keep the old name of their Remote Control session.
    assert session_name(make_task(env, remote_session_id=SESSION)) == "vcm-alpha-app-7"
    assert session_name(make_task(env, project="___", id=3, remote_control=True)) == "vcm-project-3"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fix the login button", "Fix the login button"),
        ("  Oprava\n\tpřihlášení \x00 ", "Oprava přihlášení"),
        ("--dangerously-skip-permissions", "dangerously-skip-permissions"),
        ("- - -", None),
        ("a\u202eb\u2028c", "a b c"),
        ("", None),
        (None, None),
        ("word " * 20, ("word " * 12).strip()),
        ("x" * 80, "x" * 60),
    ],
)
def test_sanitize_title(raw: str | None, expected: str | None) -> None:
    assert sanitize_title(raw) == expected
    if expected is not None:
        assert len(expected) <= TITLE_MAX_CHARS


def test_format_session_name_sanitizes_the_label() -> None:
    assert format_session_name("-web\nshop", 4) == "web shop (voice 4)"
    assert format_session_name("---", 5) == "project (voice 5)"


def test_resume_args_reuse_the_task_flags_but_no_print_mode_flags(env: Env) -> None:
    claude = claude_config(env.record, model="opus", effort="high", max_turns=5, extra_args=["--verbose"])
    args = build_resume_args(claude, session_id=SESSION, name="a (voice 1)", mcp_config_path="/state/mcp-config.json")
    assert args == [
        "--bg",
        "--resume",
        SESSION,
        "--remote-control",
        "a (voice 1)",
        "--name",
        "a (voice 1)",
        "--permission-mode",
        "auto",
        "--strict-mcp-config",
        "--mcp-config",
        "/state/mcp-config.json",
        "--setting-sources",
        "user,local",
        "--model",
        "opus",
        "--effort",
        "high",
    ]


def trust_project(env: Env) -> None:
    """claude.configDir is the test claude home, so `.claude.json` is read from there."""
    trust = {"projects": {env.project: {"hasTrustDialogAccepted": True}}}
    Path(env.home, ".claude.json").write_text(json.dumps(trust), encoding="utf-8")


async def test_desktop_access_drops_the_strict_config_and_denies_the_voice_server(env: Env) -> None:
    trust_project(env)
    rc = env.rc(public_url="https://vcm.example.test", access="desktop")
    assert isinstance(await rc.open(make_task(env), session_busy=False, opened={}), RcOpened)
    rec = read_records(env.record)[0]
    argv = rec["argv"]
    assert "--strict-mcp-config" not in argv
    assert "--mcp-config" not in argv
    assert argv[argv.index("--setting-sources") + 1] == "user,project,local"
    settings = json.loads(argv[argv.index("--settings") + 1])
    assert settings == {"deniedMcpServers": [{"serverUrl": "https://vcm.example.test/*"}]}
    assert "ENABLE_CLAUDEAI_MCP_SERVERS" not in rec["env"]
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in rec["env"]


async def test_isolated_access_keeps_the_strict_config(env: Env) -> None:
    assert isinstance(await env.rc().open(make_task(env), session_busy=False, opened={}), RcOpened)
    rec = read_records(env.record)[0]
    assert "--strict-mcp-config" in rec["argv"]
    assert "--settings" not in rec["argv"]
    assert rec["env"]["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"


async def test_opens_the_session_and_returns_name_and_link(env: Env) -> None:
    result = await env.rc().open(make_task(env), session_busy=False, opened={})
    assert result == RcOpened(
        name="Alpha App (voice 7)",
        project="Alpha App",
        link=f"https://claude.ai/code/session_01FAKE{SESSION[:8]}",
        already_open=False,
        session_id=SESSION,
        bg_id=SESSION[:8],
    )
    rec = read_records(env.record)[0]
    assert rec["cwd"] == env.project
    assert rec["stdin"] == ""
    assert rec["argv"][:7] == [
        "--bg",
        "--resume",
        SESSION,
        "--remote-control",
        "Alpha App (voice 7)",
        "--name",
        "Alpha App (voice 7)",
    ]
    assert "Oprav testy" not in " ".join(rec["argv"])
    assert rec["env"]["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    assert rec["env"]["CLAUDE_CONFIG_DIR"] == env.home
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in rec["env"]
    text = format_opened(7, result)
    assert text.startswith("Task 7 is now open as session Alpha App (voice 7) in Alpha App.")
    assert text.endswith(f"Link (do not read it aloud): https://claude.ai/code/session_01FAKE{SESSION[:8]}")


async def test_a_second_call_returns_the_open_session_instead_of_starting_another(env: Env) -> None:
    rc = env.rc()
    first = await rc.open(make_task(env), session_busy=False, opened={})
    second = await rc.open(make_task(env), session_busy=False, opened={})
    assert isinstance(first, RcOpened)
    assert isinstance(second, RcOpened)
    assert second.already_open
    assert (second.name, second.link) == (first.name, first.link)
    assert len(read_records(env.record)) == 1
    assert format_opened(7, second).startswith("Task 7 is already open as session Alpha App (voice 7)")
    assert set(rc.live_sessions({SESSION: "Alpha App (voice 7)"})) == {SESSION}
    assert rc.live_sessions({}) == {}


async def test_drops_the_variable_that_keeps_remote_control_from_connecting(env: Env) -> None:
    rc = env.rc(extra_env={"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
    result = await rc.open(make_task(env), session_busy=False, opened={})
    assert isinstance(result, RcOpened)
    assert result.link is not None
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in read_records(env.record)[0]["env"]


async def test_follows_a_copy_the_cli_started_under_a_new_id(env: Env) -> None:
    result = await env.rc(bg="copy").open(make_task(env), session_busy=False, opened={})
    assert isinstance(result, RcOpened)
    assert result.session_id is not None
    assert result.session_id != SESSION
    assert result.link == f"https://claude.ai/code/session_01FAKE{result.session_id[:8]}"
    assert result.name == "Alpha App (voice 7)"


async def test_prefers_the_earlier_remote_control_session(env: Env) -> None:
    earlier = "55555555-0000-4000-8000-000000000002"
    task = make_task(env, remote_session_id=earlier)
    rc = env.rc()
    opened = await rc.open(task, session_busy=False, opened={earlier: task})
    assert isinstance(opened, RcOpened)
    assert opened.session_id == earlier
    assert read_records(env.record)[0]["argv"][2] == earlier
    # Live under the earlier id: returned, nothing started.
    again = await rc.open(task, session_busy=False, opened={earlier: task})
    assert isinstance(again, RcOpened)
    assert again.already_open
    assert len(read_records(env.record)) == 1


async def test_falls_back_to_the_task_session_when_the_earlier_one_is_gone(env: Env) -> None:
    task = make_task(env, remote_session_id="66666666-0000-4000-8000-000000000003")
    result = await env.rc(bg="missing").open(task, session_busy=False, opened={})
    assert isinstance(result, RcFailed)  # the fake fails every resume with "missing"
    argv = [r["argv"][2] for r in read_records(env.record)]
    assert argv == ["66666666-0000-4000-8000-000000000003", SESSION]


@pytest.mark.parametrize("status", ["queued", "running"])
async def test_refuses_a_task_that_has_not_finished(env: Env, status: str) -> None:
    result = await env.rc().open(make_task(env, status=status), session_busy=False, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "not_finished"
    assert "Wait for its result" in result.message
    assert read_records(env.record) == []


async def test_refuses_while_another_task_uses_the_session(env: Env) -> None:
    result = await env.rc().open(make_task(env), session_busy=True, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "session_busy"
    assert read_records(env.record) == []


async def test_refuses_a_relay_task_and_points_to_the_target_session(env: Env) -> None:
    task = make_task(env, kind="send", target_session_id="x", target_session_name="billing-fix")
    result = await env.rc().open(task, session_busy=False, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "relay_task"
    assert "Continue in billing-fix itself" in result.message
    assert read_records(env.record) == []


async def test_allows_an_ask_task_and_refuses_a_task_without_session(env: Env) -> None:
    ask = make_task(env, kind="ask", target_session_id="y", target_session_name="Přihlášení")
    assert isinstance(await env.rc().open(ask, session_busy=False, opened={}), RcOpened)
    failed = await env.rc().open(make_task(env, id=8, session_id=None, status="failed"), session_busy=False, opened={})
    assert isinstance(failed, RcFailed)
    assert failed.reason == "no_session"


async def test_explains_an_untrusted_folder(env: Env) -> None:
    result = await env.rc(bg="untrusted").open(make_task(env), session_busy=False, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "untrusted"
    assert "Open Claude Code once in that project on the Mac and confirm the trust question" in result.message
    assert read_live_sessions(env.home) == {}


async def test_reports_a_failed_launch(env: Env) -> None:
    result = await env.rc(bg="fail").open(make_task(env), session_busy=False, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "launch_failed"
    assert "something broke" in result.message


async def test_returns_the_name_without_link_after_the_timeout(env: Env) -> None:
    start = time.monotonic()
    result = await env.rc(bg="nolink", link_wait_ms=800).open(make_task(env), session_busy=False, opened={})
    assert 0.7 < time.monotonic() - start < 5
    assert result == RcOpened(
        name="Alpha App (voice 7)",
        project="Alpha App",
        link=None,
        already_open=False,
        session_id=SESSION,
        bg_id=SESSION[:8],
    )
    assert "The link is not ready yet" in format_opened(7, result)


def _now_ms() -> int:
    return int(time.time() * 1000)


async def test_limits_the_number_of_open_sessions(env: Env) -> None:
    other = os.getpid()
    s1, s2, gone, stale, starting = (f"55555555-0000-4000-8000-00000000000{i}" for i in (1, 2, 3, 4, 5))
    base = {"kind": "bg", "cwd": env.project, "startedAt": _now_ms() - 7_200_000}
    write_live_session(env.home, other, {**base, "sessionId": s1, "name": "s-1", "bridgeSessionId": "session_01A"})
    # Not counted: sessions the server did not open, and a session that lost Remote Control (the
    # owner does not see it in the app).
    write_live_session(env.home, 1, {**base, "sessionId": s2, "name": "s-2", "bridgeSessionId": "session_01B"})
    # (Other registry files, same live pid.)
    write_live_session(env.home, 900001, {**base, "pid": other, "sessionId": stale, "name": "s-4"})
    # Not counted: the owner continues a handed-over session in the desktop app (same id, interactive).
    handed = "55555555-0000-4000-8000-000000000006"
    write_live_session(
        env.home,
        900003,
        {**base, "pid": other, "kind": "interactive", "sessionId": handed, "bridgeSessionId": "session_01C"},
    )
    # Counted: a session still starting up (no link yet).
    write_live_session(
        env.home, 900002, {**base, "pid": other, "sessionId": starting, "startedAt": _now_ms(), "name": "s-5"}
    )
    beta = make_task(env, id=3, project="Beta")
    opened = {
        s1: beta,
        gone: make_task(env, id=4, project="Beta"),
        stale: make_task(env, id=5, project="Beta"),
        starting: make_task(env, id=6, project="Beta"),
        handed: make_task(env, id=7, project="Beta"),
    }
    result = await env.rc(max_sessions=2).open(make_task(env), session_busy=False, opened=opened)
    assert isinstance(result, RcFailed)
    assert result.reason == "limit"
    assert result.message == (
        "There are already 2 sessions open in the Claude app: Beta (voice 3) (task 3), Beta (voice 6) (task 6)."
        " To close one, ask to cancel its task, for example cancel task 3."
    )
    assert read_records(env.record) == []
    one = {s1: beta}
    failed = await env.rc(max_sessions=1).open(make_task(env), session_busy=False, opened=one)
    assert isinstance(failed, RcFailed)
    assert failed.message.startswith("There is already 1 session open in the Claude app: Beta (voice 3) (task 3).")
    assert isinstance(await env.rc(max_sessions=3).open(make_task(env), session_busy=False, opened=opened), RcOpened)


async def test_reopens_a_session_that_lost_remote_control(env: Env) -> None:
    task = make_task(env)
    first = await env.rc(bg="stale", link_wait_ms=300).open(task, session_busy=False, opened={})
    assert isinstance(first, RcOpened)
    assert first.link is None
    rc = env.rc()
    entry = rc.read_live()[SESSION]
    assert rc.disconnected(entry)
    # The stale session of this task does not count toward the limit, it is stopped and relaunched.
    again = await env.rc(max_sessions=1).open(task, session_busy=False, opened={SESSION: task})
    assert isinstance(again, RcOpened)
    assert not again.already_open
    assert again.replaced_stale
    assert again.link == f"https://claude.ai/code/session_01FAKE{SESSION[:8]}"
    argvs = [r["argv"] for r in read_records(env.record)]
    assert argvs[1] == ["stop", SESSION[:8]]
    assert argvs[2][:3] == ["--bg", "--resume", SESSION]
    assert format_opened(7, again).startswith(
        "The earlier session of task 7 had lost its connection to the Claude app, so it was closed."
        " Task 7 is now open again as session Alpha App (voice 7) in Alpha App."
    )
    live = rc.read_live()
    assert live[SESSION].pid != entry.pid
    assert not rc.disconnected(live[SESSION])


async def test_reopens_a_running_session_task_that_lost_remote_control(env: Env) -> None:
    task = make_task(env, status="done")
    await env.rc(bg="stale", link_wait_ms=300).open(task, session_busy=False, opened={})
    session_task = make_task(env, status="running", remote_control=True, remote_session_id=SESSION, bg_id=SESSION[:8])
    result = await env.rc().open(session_task, session_busy=False, opened={SESSION: session_task})
    assert isinstance(result, RcOpened)
    assert result.replaced_stale
    assert result.link is not None
    # Connected now: returned as it is.
    connected = await env.rc().open(session_task, session_busy=False, opened={SESSION: session_task})
    assert isinstance(connected, RcOpened)
    assert connected.already_open
    assert [r["argv"][0] for r in read_records(env.record)] == ["--bg", "stop", "--bg"]


async def test_reports_a_stale_session_that_cannot_be_stopped(env: Env) -> None:
    # Owned by the test process, so the fake's `claude stop` (which finds only its own "bg"
    # sleepers by session id) is pointed at a session it does not know.
    write_live_session(
        env.home,
        os.getpid(),
        {"sessionId": SESSION, "kind": "bg-other", "cwd": env.project, "startedAt": _now_ms() - 7_200_000},
    )
    rc = env.rc()
    rc._read_live = lambda home: {  # a stale bg entry the fake cannot stop
        sid: replace(e, kind="bg") for sid, e in read_live_sessions(home).items()
    }
    result = await rc.open(make_task(env), session_busy=False, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "stop_failed"
    assert result.message.startswith(
        "Session Alpha App (voice 7) of task 7 lost its connection to the Claude app, and it could not be closed"
    )
    assert [r["argv"] for r in read_records(env.record)] == [["stop", SESSION[:8]]]


async def test_close_stops_the_live_session_of_a_finished_task(env: Env) -> None:
    rc = env.rc()
    task = make_task(env)
    assert await rc.close(task) == "nothing"
    opened = await rc.open(task, session_busy=False, opened={})
    assert isinstance(opened, RcOpened)
    reopened = make_task(env, remote_session_id=opened.session_id)
    assert rc.live_session_of(reopened) == SESSION
    assert await rc.close(reopened) == "closed"
    assert SESSION not in rc.read_live()
    assert await rc.close(reopened) == "nothing"
    # Found by its id8 when the session id was never stored.
    await rc.open(task, session_busy=False, opened={})
    assert await rc.close(make_task(env, bg_id=SESSION[:8])) == "closed"


class Opener:
    """Records the URLs a handoff opens and whether the session was still live at that moment."""

    def __init__(self, rc: list[RemoteControl], failure: str | None = None) -> None:
        self.rc = rc
        self.failure = failure
        self.calls: list[tuple[str, bool]] = []

    async def __call__(self, url: str) -> str | None:
        self.calls.append((url, SESSION in self.rc[0].read_live()))
        return self.failure


def handoff_rc(env: Env, opener: Opener, **options: Any) -> RemoteControl:
    rc = env.rc(options={"handoff_to_desktop": True, "open_url": opener, "handoff_wait_ms": 3000, **options})
    opener.rc.append(rc)
    return rc


async def test_hands_a_closed_session_over_to_the_desktop_app_after_it_is_gone(env: Env) -> None:
    opener = Opener([])
    rc = handoff_rc(env, opener)
    task = make_task(env)
    opened = await rc.open(task, session_busy=False, opened={})
    assert isinstance(opened, RcOpened)
    assert await rc.close(make_task(env, remote_session_id=opened.session_id)) == "closed"
    await rc.wait_handoffs()
    assert opener.calls == [(f"claude://resume?session={SESSION}", False)]
    handoffs = [r for r in env.log.records if r["msg"] == "desktop handoff"]
    assert [(r["taskId"], r["ok"]) for r in handoffs] == [(7, True)]


async def test_hands_a_stopped_session_task_over_and_logs_a_failed_open(env: Env) -> None:
    opener = Opener([], failure="open exited with code 1")
    rc = handoff_rc(env, opener)
    opened = await rc.open(make_task(env), session_busy=False, opened={})
    assert isinstance(opened, RcOpened)
    running = make_task(env, status="running", remote_control=True, bg_id=SESSION[:8], remote_session_id=SESSION)
    assert await rc.close(running) == "closed"
    await rc.wait_handoffs()
    assert opener.calls == [(f"claude://resume?session={SESSION}", False)]
    [entry] = [r for r in env.log.records if r["msg"] == "desktop handoff"]
    assert entry["level"] == "warn"
    assert entry["ok"] is False


async def test_no_handoff_when_off_or_while_the_session_still_runs(env: Env) -> None:
    opener = Opener([])
    rc = env.rc(options={"open_url": opener})
    opener.rc.append(rc)
    opened = await rc.open(make_task(env), session_busy=False, opened={})
    assert isinstance(opened, RcOpened)
    assert await rc.close(make_task(env, remote_session_id=SESSION)) == "closed"
    await rc.wait_handoffs()
    assert opener.calls == []

    # The registry still lists the session (another process owns the transcript): never opened.
    still = read_live_sessions(env.home)
    stuck = handoff_rc(env, opener, read_live=lambda _home: still, handoff_wait_ms=200)
    await env.rc().open(make_task(env), session_busy=False, opened={})
    still = read_live_sessions(env.home)
    assert await stuck.close(make_task(env, remote_session_id=SESSION)) == "closed"
    await stuck.wait_handoffs()
    assert opener.calls == []
    assert any(r["msg"] == "desktop handoff skipped: session still running" for r in env.log.records)


async def test_records_a_successful_handoff_on_the_task(env: Env) -> None:
    handed: list[tuple[int, str]] = []
    rc = handoff_rc(env, Opener([]), on_handoff=lambda task_id, sid: handed.append((task_id, sid)))
    opened = await rc.open(make_task(env), session_busy=False, opened={})
    assert isinstance(opened, RcOpened)
    assert await rc.close(make_task(env, remote_session_id=opened.session_id)) == "closed"
    await rc.wait_handoffs()
    assert handed == [(7, SESSION)]
    # A failed open records nothing.
    failing = handoff_rc(env, Opener([], failure="open exited with code 1"), on_handoff=lambda *a: handed.append(a))
    assert isinstance(await failing.open(make_task(env), session_busy=False, opened={}), RcOpened)
    assert await failing.close(make_task(env, remote_session_id=SESSION)) == "closed"
    await failing.wait_handoffs()
    assert handed == [(7, SESSION)]


HANDED_OVER = {
    "handed_over_at": "2026-11-25T10:00:00.000Z",
    "desktop_session_id": SESSION,
    "remote_session_id": SESSION,
    "name": "Alpha App (voice 7)",
}


def desktop_entry(env: Env, *, bridge: bool, kind: str = "interactive") -> None:
    """The owner continues the session in the desktop app (or a terminal): not the server's bg session."""
    data: dict[str, Any] = {"sessionId": SESSION, "kind": kind, "status": "idle", "name": "Alpha App (voice 7)"}
    if bridge:
        data["bridgeSessionId"] = "session_01DESKTOP"
    write_live_session(env.home, os.getpid(), data)


async def test_a_session_open_in_the_desktop_app_with_remote_control_is_returned_not_resumed(env: Env) -> None:
    desktop_entry(env, bridge=True)
    task = make_task(env, **HANDED_OVER)
    result = await env.rc().open(task, session_busy=False, opened={SESSION: task})
    assert result == RcOpened(
        name="Alpha App (voice 7)",
        project="Alpha App",
        link="https://claude.ai/code/session_01DESKTOP",
        already_open=True,
        session_id=SESSION,
    )
    assert read_records(env.record) == []
    assert format_opened(7, result).startswith("Task 7 is already open as session Alpha App (voice 7) in Alpha App.")


async def test_a_session_open_in_the_desktop_app_without_remote_control_is_refused(env: Env) -> None:
    desktop_entry(env, bridge=False)
    result = await env.rc().open(make_task(env, **HANDED_OVER), session_busy=False, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "open_elsewhere"
    assert result.message.startswith(
        "Task 7's session Alpha App (voice 7) is open in the Claude desktop app on the Mac, so it is not opened a second time."
    )
    assert "switch on Remote Control for that session there" in result.message
    assert read_records(env.record) == []


async def test_any_live_session_the_server_did_not_start_is_never_resumed(env: Env) -> None:
    # Not handed over: the task's own session is open in a terminal, say.
    desktop_entry(env, bridge=False)
    result = await env.rc().open(make_task(env), session_busy=False, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "open_elsewhere"
    assert "in Claude Code on the Mac (the desktop app or a terminal)" in result.message
    # Only the desktop id is live (the copy the app took over): refused too.
    copy = "55555555-0000-4000-8000-000000000009"
    task = make_task(env, desktop_session_id=copy, handed_over_at="2026-11-25T10:00:00.000Z")
    os.remove(os.path.join(env.home, "sessions", f"{os.getpid()}.json"))
    write_live_session(env.home, os.getpid(), {"sessionId": copy, "kind": "interactive", "status": "busy"})
    refused = await env.rc().open(task, session_busy=False, opened={})
    assert isinstance(refused, RcFailed)
    assert "in the Claude desktop app on the Mac" in refused.message
    assert read_records(env.record) == []


async def test_a_handed_over_session_that_is_not_live_is_resumed_with_its_name(env: Env) -> None:
    result = await env.rc().open(make_task(env, **HANDED_OVER), session_busy=False, opened={})
    assert isinstance(result, RcOpened)
    assert not result.already_open
    argv = read_records(env.record)[0]["argv"]
    assert argv[:3] == ["--bg", "--resume", SESSION]
    assert argv[argv.index("--name") + 1] == "Alpha App (voice 7)"


def list_background(env: Env, session_id: str = SESSION, kind: str = "background") -> str:
    """The CLI's background list remembers a stopped session (the fake then copies it on resume)."""
    path = os.path.join(env.home, "fake-agents.json")
    entry = {"id": session_id[:8], "sessionId": session_id, "kind": kind, "state": "stopped", "name": "x", "cwd": "/"}
    Path(path).write_text(json.dumps([entry]), encoding="utf-8")
    return path


def read_agents(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


async def test_removes_a_stopped_background_session_first_so_the_resume_keeps_its_id(env: Env) -> None:
    path = list_background(env)
    result = await env.rc().open(make_task(env, **HANDED_OVER), session_busy=False, opened={})
    assert isinstance(result, RcOpened)
    assert result.session_id == SESSION
    argvs = [r["argv"] for r in read_records(env.record)]
    assert argvs[0] == ["rm", SESSION[:8]]
    assert argvs[1][:3] == ["--bg", "--resume", SESSION]
    assert read_agents(path) == []
    [entry] = [r for r in env.log.records if r["msg"] == "background session removed before resume"]
    assert entry["ok"] is True


async def test_a_refused_rm_still_resumes_as_a_copy(env: Env) -> None:
    list_background(env)
    result = await env.rc(bg="rmfail").open(make_task(env), session_busy=False, opened={})
    assert isinstance(result, RcOpened)
    assert result.session_id is not None
    assert result.session_id != SESSION
    assert [r["argv"][0] for r in read_records(env.record)] == ["rm", "--bg"]


async def test_only_background_entries_of_the_same_session_are_removed(env: Env) -> None:
    list_background(env, kind="interactive")
    assert isinstance(await env.rc().open(make_task(env), session_busy=False, opened={}), RcOpened)
    list_background(env, session_id="55555555-0000-4000-8000-00000000000a")
    assert isinstance(await env.rc().open(make_task(env, id=8), session_busy=False, opened={}), RcOpened)
    assert all(r["argv"][0] != "rm" for r in read_records(env.record))


def test_desktop_resume_url_takes_only_session_ids() -> None:
    assert desktop_resume_url(SESSION) == f"claude://resume?session={SESSION}"
    assert desktop_resume_url("x&y=1") is None
    assert desktop_resume_url("") is None


async def test_max_sessions_zero_turns_it_off(env: Env) -> None:
    result = await env.rc(max_sessions=0).open(make_task(env), session_busy=False, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "disabled"


async def test_ignores_session_ids_that_are_not_uuids(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    # The Task model drops ids that are not UUID-shaped, also for tasks built in code.
    monkeypatch.setattr(manager_module, "_CODE_LOGGER", env.log.logger)
    bad = "--help"
    no_session = make_task(env, session_id=bad)
    assert no_session.session_id is None
    result = await env.rc().open(no_session, session_busy=False, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "no_session"
    bad_remote = make_task(env, remote_session_id="not-a-uuid")
    assert bad_remote.remote_session_id is None
    opened = await env.rc().open(bad_remote, session_busy=False, opened={})
    assert isinstance(opened, RcOpened)
    assert [r["argv"][2] for r in read_records(env.record)] == [SESSION]
    warnings = [r for r in env.log.records if r["msg"] == "invalid session id ignored"]
    assert [(w["source"], w["taskId"]) for w in warnings] == [("task session_id", 7), ("task remote_session_id", 7)]
    assert bad not in env.log.text()
    assert "not-a-uuid" not in env.log.text()


def test_start_args_have_no_prompt_and_one_merged_settings_json(env: Env) -> None:
    claude = claude_config(env.record, model="opus", effort="high", max_turns=5, extra_args=["--verbose"])
    desktop = RunAccess(
        isolated=False,
        mcp_config_path=None,
        setting_sources=("user", "project", "local"),
        settings={"deniedMcpServers": [{"serverUrl": "https://vcm.example.test/*"}]},
    )
    args = build_start_args(claude, name="a (voice 1)", access=desktop, answer_path="/state/session-answers/1.json")
    assert args[:7] == ["--bg", "--remote-control", "a (voice 1)", "--name", "a (voice 1)", "--permission-mode", "auto"]
    assert args.count("--settings") == 1
    settings = json.loads(args[args.index("--settings") + 1])
    assert settings["deniedMcpServers"] == [{"serverUrl": "https://vcm.example.test/*"}]
    assert settings["hooks"] == stop_hook_settings("/state/session-answers/1.json")["hooks"]
    assert args[args.index("--append-system-prompt") + 1] == REMOTE_SESSION_PROMPT
    assert args[-4:] == ["--model", "opus", "--effort", "high"]
    for flag in ("-p", "--max-turns", "--verbose"):
        assert flag not in args

    isolated = build_start_args(claude, name="a (voice 1)", access=isolated_access(claude, "/state/mcp.json"))
    assert "--strict-mcp-config" in isolated
    assert "--settings" not in isolated


def test_the_stop_hook_command_survives_odd_paths(env: Env) -> None:
    target = env.root / "state dir; $(echo x)" / "1.json"
    target.parent.mkdir()
    command = stop_hook_settings(str(target))["hooks"]["Stop"][0]["hooks"][0]["command"]
    subprocess.run(["sh", "-c", command], input=b'{"last_assistant_message": "OK"}', check=True)
    assert json.loads(target.read_text(encoding="utf-8")) == {"last_assistant_message": "OK"}
    assert target.stat().st_mode & 0o077 == 0
    assert not (target.parent / "1.json.tmp").exists()


def test_resume_args_merge_the_stop_hook_into_the_one_settings_json(env: Env) -> None:
    claude = claude_config(env.record)
    desktop = RunAccess(
        isolated=False,
        mcp_config_path=None,
        setting_sources=("user", "project", "local"),
        settings={"deniedMcpServers": [{"serverUrl": "https://vcm.example.test/*"}]},
    )
    answer = "/state/session-answers/7.json"
    args = build_resume_args(
        claude, session_id=SESSION, name="a (voice 7)", mcp_config_path="/m.json", access=desktop, answer_path=answer
    )
    assert args.count("--settings") == 1
    settings = json.loads(args[args.index("--settings") + 1])
    assert settings["deniedMcpServers"] == [{"serverUrl": "https://vcm.example.test/*"}]
    assert settings["hooks"] == stop_hook_settings(answer)["hooks"]
    isolated = build_resume_args(claude, session_id=SESSION, name="a", mcp_config_path="/m.json", answer_path=answer)
    assert "--strict-mcp-config" in isolated
    assert json.loads(isolated[isolated.index("--settings") + 1]) == stop_hook_settings(answer)


async def test_a_reopened_session_writes_its_answers_with_the_stop_hook(env: Env) -> None:
    answers = env.root / "answers"
    answers.mkdir()
    rc = env.rc(bg="turn", options={"answer_path": lambda task_id: str(answers / f"{task_id}.json")})
    assert isinstance(await rc.open(make_task(env), session_busy=False, opened={}), RcOpened)
    saved = json.loads((answers / "7.json").read_text(encoding="utf-8"))
    assert (saved["session_id"], saved["last_assistant_message"]) == (SESSION, "OK")


async def test_close_waits_for_the_lock_and_reads_the_task_again(env: Env) -> None:
    states: dict[int, Task] = {}
    rc = env.rc(options={"get_task": states.get})
    assert isinstance(await rc.open(make_task(env), session_busy=False, opened={}), RcOpened)
    running = make_task(env, status="running", remote_control=True, remote_session_id=SESSION, bg_id=SESSION[:8])
    await rc._lock.acquire()
    job = asyncio.create_task(rc.close(running, reason="idle", only_if=lambda sid, entry: True))
    await asyncio.sleep(0.3)
    assert [r["argv"][0] for r in read_records(env.record)] == ["--bg"]
    # The task changed while the close waited (its session ids forgotten, say): with only_if and no
    # live session of the task read again, nothing is stopped.
    states[7] = make_task(env, status="done", remote_control=True)
    rc._lock.release()
    assert await job == SKIPPED
    assert [r["argv"][0] for r in read_records(env.record)] == ["--bg"]


async def test_only_if_is_checked_under_the_lock_with_the_live_session(env: Env) -> None:
    rc = env.rc()
    assert isinstance(await rc.open(make_task(env), session_busy=False, opened={}), RcOpened)
    running = make_task(env, status="running", remote_control=True, remote_session_id=SESSION, bg_id=SESSION[:8])
    seen: list[tuple[str, str | None]] = []

    def busy_now(sid: str, entry: Any) -> bool:
        seen.append((sid, entry.status))
        return False

    assert await rc.close(running, reason="idle", only_if=busy_now) == SKIPPED
    reopened = make_task(env, remote_session_id=SESSION)
    assert await rc.close(reopened, reason="idle", only_if=busy_now) == SKIPPED
    assert seen == [(SESSION, "idle"), (SESSION, "idle")]
    assert all(r["argv"][0] != "stop" for r in read_records(env.record))
    assert await rc.close(reopened, reason="idle", only_if=lambda sid, entry: True) == "closed"


async def test_a_stale_session_that_does_not_go_is_not_launched_a_second_time(env: Env) -> None:
    task = make_task(env)
    await env.rc(bg="stale", link_wait_ms=300).open(task, session_busy=False, opened={})
    list_background(env)
    frozen = read_live_sessions(env.home)
    # The stop works, but the registry keeps listing the session (it has not ended yet).
    rc = env.rc(options={"read_live": lambda _home: frozen, "stop_wait_ms": 300})
    result = await rc.open(task, session_busy=False, opened={})
    assert isinstance(result, RcFailed)
    assert result.reason == "still_stopping"
    assert result.message == (
        "Session Alpha App (voice 7) of task 7 lost its connection to the Claude app and is being closed,"
        " but it has not ended yet. Ask again in a moment to open it again."
    )
    # No `claude rm` and no second launch (it would only make a copy).
    assert [r["argv"][0] for r in read_records(env.record)] == ["--bg", "stop"]


async def test_shutdown_lets_a_pending_handoff_finish(env: Env) -> None:
    opener = Opener([])
    rc = handoff_rc(env, opener)
    opened = await rc.open(make_task(env), session_busy=False, opened={})
    assert isinstance(opened, RcOpened)
    assert await rc.close(make_task(env, remote_session_id=opened.session_id)) == "closed"
    await rc.aclose()
    assert opener.calls == [(f"claude://resume?session={SESSION}", False)]


async def test_shutdown_cancels_a_handoff_that_takes_too_long(env: Env) -> None:
    opener = Opener([])
    await env.rc().open(make_task(env), session_busy=False, opened={})
    still = read_live_sessions(env.home)
    rc = handoff_rc(env, opener, read_live=lambda _home: still, handoff_wait_ms=30_000, shutdown_wait_ms=200)
    assert await rc.close(make_task(env, remote_session_id=SESSION)) == "closed"
    start = time.monotonic()
    await rc.aclose()
    assert time.monotonic() - start < 2
    assert opener.calls == []
    assert any(r["msg"] == "desktop handoff cancelled at shutdown" for r in env.log.records)
