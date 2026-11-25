"""ask_session_readonly and send_message_to_session tasks against the fake claude."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from itertools import count
from pathlib import Path
from typing import Any, NoReturn

import pytest

from tests.helpers import claude_config, read_records, tasks_config, until
from tests.session_fixtures import append_lines, assistant_text, peer_message, write_transcript
from voice_coding_mcp.clock import now_ms
from voice_coding_mcp.config import SessionsConfig
from voice_coding_mcp.log import silent_logger
from voice_coding_mcp.sessions import ASK_SESSION_PROMPT, LiveSession, ReplyWatchOptions, ReplyWatchResult, SessionInfo
from voice_coding_mcp.tasks.manager import (
    Cancelled,
    Started,
    StartFailed,
    Task,
    TaskManager,
    TaskManagerOptions,
    format_for_voice,
)

_counter = count(1)

LIVE = LiveSession(pid=1, socket_path="/tmp/cc-socks/1.sock", status="idle", name="billing-fix")


@dataclass
class Setup:
    manager: TaskManager
    record: str
    state_dir: str


@dataclass
class Env:
    base: Path
    root: Path
    open: list[TaskManager] = field(default_factory=list)

    async def setup(self, **overrides: Any) -> Setup:
        n = next(_counter)
        record = str(self.base / f"record-{n}.jsonl")
        state_dir = str(self.base / f"state-{n}")
        fields: dict[str, Any] = {
            "state_dir": state_dir,
            "project_roots": [str(self.root)],
            "claude": claude_config(record),
            "tasks": tasks_config(),
            "sessions": SessionsConfig(days=7, reply_wait_minutes=1),
            "logger": silent_logger,
            "kill_grace_ms": 500,
            "reply_poll_ms": 20,
            **overrides,
        }
        manager = await TaskManager.create(TaskManagerOptions(**fields))
        self.open.append(manager)
        return Setup(manager, record, state_dir)

    def session(self, **overrides: Any) -> SessionInfo:
        session_id = f"22222222-0000-4000-8000-{next(_counter):012d}"
        cwd = str(self.root / "assistant")
        transcript = write_transcript(str(self.base / "home"), session_id, [assistant_text(cwd, "earlier work")])
        info = SessionInfo(
            id=session_id,
            name="billing-fix",
            titled=True,
            project="assistant",
            cwd=cwd,
            last_activity_ms=now_ms(),
            transcript_path=transcript,
            live=None,
        )
        return replace(info, **overrides)


@pytest.fixture
async def env(real_tmp: Path) -> AsyncIterator[Env]:
    root = real_tmp / "projects"
    for name in ("alpha", "assistant"):
        (root / name).mkdir(parents=True)
    e = Env(real_tmp, root)
    yield e
    await asyncio.gather(*(m.shutdown("test end") for m in e.open))


def read_projects(state_dir: str) -> Any:
    path = Path(state_dir, "projects.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"projects": {}}


def arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def mcp_config(argv: list[str]) -> Any:
    return json.loads(Path(arg(argv, "--mcp-config") or "").read_text(encoding="utf-8"))


def started(result: object) -> Task:
    assert isinstance(result, Started), f"start failed: {result}"
    return result.task


async def finished(manager: TaskManager, task_id: int) -> Task:
    done = await manager.wait(task_id, 5000)
    assert done is not None
    return done


def current(manager: TaskManager, task_id: int) -> Task:
    task = manager.get(task_id)
    assert task is not None
    return task


class TestAskSessionReadonly:
    async def test_runs_a_read_only_fork_in_its_cwd_with_the_question_on_stdin(self, env: Env) -> None:
        s = await env.setup()
        target = env.session()
        task = started(await s.manager.start_session("ask", target, "Na čem jsme skončili?\nRESULT U rozpočtu."))
        done = await finished(s.manager, task.id)
        assert format_for_voice(done) == "Task 1 for session billing-fix is done. U rozpočtu."
        [rec] = read_records(s.record)
        assert rec["cwd"] == target.cwd
        assert rec["stdin"] == "Na čem jsme skončili?\nRESULT U rozpočtu."
        assert arg(rec["argv"], "--resume") == target.id
        assert "--fork-session" in rec["argv"]
        assert arg(rec["argv"], "--permission-mode") == "plan"
        assert ASK_SESSION_PROMPT in (arg(rec["argv"], "--append-system-prompt") or "")
        assert arg(rec["argv"], "--tools") == "Read,Grep,Glob"
        assert "--strict-mcp-config" in rec["argv"]
        assert mcp_config(rec["argv"]) == {"mcpServers": {}}
        assert not any("Na čem" in a for a in rec["argv"])
        assert done.session_id
        assert done.session_id in s.manager.known_session_ids()
        # ask runs never become the project's session for run_task.
        projects = read_projects(s.state_dir)
        assert projects["projects"] == {}

    async def test_fails_without_a_rerun_when_the_session_transcript_is_gone(self, env: Env) -> None:
        s = await env.setup()
        task = started(await s.manager.start_session("ask", env.session(), "RESUME_MISSING"))
        done = await finished(s.manager, task.id)
        assert done.status == "failed"
        assert "transcript was not found" in format_for_voice(done)
        assert len(read_records(s.record)) == 1


class TestSendMessageToSession:
    async def test_refuses_a_session_that_is_not_running(self, env: Env) -> None:
        s = await env.setup()
        result = await s.manager.start_session("send", env.session(), "ahoj")
        assert isinstance(result, StartFailed)
        assert result.reason == "session_not_running"
        assert format_for_voice(result).startswith(
            "Session billing-fix is not running, so it cannot receive a message."
        )

    async def test_relays_the_message_on_stdin_with_a_narrow_relay_run_then_returns_the_reply(self, env: Env) -> None:
        calls: list[ReplyWatchOptions] = []

        async def watch(options: ReplyWatchOptions) -> ReplyWatchResult:
            calls.append(options)
            if options.on_arrived:
                options.on_arrived()
            return ReplyWatchResult(arrived=True, finished=True, reply="Přijato.", aborted=False)

        s = await env.setup(watch_reply=watch)
        target = env.session(live=LIVE)
        size = os.stat(target.transcript_path).st_size
        message = "Zkontroluj testy.\nRESULT SENT"
        task = started(await s.manager.start_session("send", target, message))
        done = await finished(s.manager, task.id)
        assert format_for_voice(done) == "Task 1 for session billing-fix is done. billing-fix replied: Přijato."
        [rec] = read_records(s.record)
        assert rec["cwd"] == target.cwd
        assert arg(rec["argv"], "--tools") == "SendMessage"
        assert arg(rec["argv"], "--name") == "vcm-relay-1"
        assert arg(rec["argv"], "--max-turns") == "3"
        assert arg(rec["argv"], "--resume") is None
        assert '"uds:/tmp/cc-socks/1.sock"' in (arg(rec["argv"], "--append-system-prompt") or "")
        assert "--strict-mcp-config" in rec["argv"]
        assert mcp_config(rec["argv"]) == {"mcpServers": {}}
        assert not any("Zkontroluj" in a for a in rec["argv"])
        match = re.search(r"----- (MESSAGE [0-9a-f]{12}) -----", rec["stdin"])
        assert match
        boundary = match.group(1)
        assert f"----- {boundary} -----\n{message}\n----- {boundary} -----" in rec["stdin"]
        assert 'session "billing-fix"' in rec["stdin"]
        assert len(calls) == 1
        assert calls[0].from_name == "vcm-relay-1"
        assert calls[0].offset == size
        assert calls[0].transcript_path == target.transcript_path
        assert done.prompt == message, "the task keeps the message itself"

    async def test_fails_when_the_relay_neither_sent_nor_confirmed(self, env: Env) -> None:
        async def watch(_options: ReplyWatchOptions) -> NoReturn:
            raise AssertionError("must not watch")

        s = await env.setup(watch_reply=watch)
        task = started(
            await s.manager.start_session(
                "send", env.session(live=LIVE), "SEND_MESSAGE none\nRESULT FAILED: no such session"
            )
        )
        done = await finished(s.manager, task.id)
        assert done.status == "failed"
        assert (
            format_for_voice(done)
            == "Task 1 for session billing-fix failed: could not deliver the message to billing-fix: the relay did not send it."
        )

    async def test_an_explicit_failed_answer_wins_over_a_successful_send(self, env: Env) -> None:
        async def watch(_options: ReplyWatchOptions) -> NoReturn:
            raise AssertionError("must not watch")

        s = await env.setup(watch_reply=watch)
        task = started(
            await s.manager.start_session("send", env.session(live=LIVE), "Ahoj.\nRESULT FAILED: no such session")
        )
        done = await finished(s.manager, task.id)
        assert done.status == "failed"
        assert done.error == "could not deliver the message to billing-fix: no such session"

    @pytest.mark.parametrize("ending", ["RESULT Done.", "FAIL"])
    async def test_a_verified_delivery_counts_as_sent_whatever_the_relay_answers(self, env: Env, ending: str) -> None:
        calls: list[ReplyWatchOptions] = []

        async def watch(options: ReplyWatchOptions) -> ReplyWatchResult:
            calls.append(options)
            return ReplyWatchResult(arrived=True, finished=True, reply="Přijato.", aborted=False)

        s = await env.setup(watch_reply=watch)
        task = started(await s.manager.start_session("send", env.session(live=LIVE), f"Ahoj.\n{ending}"))
        done = await finished(s.manager, task.id)
        assert done.status == "done", done.error
        assert done.error is None
        assert format_for_voice(done) == "Task 1 for session billing-fix is done. billing-fix replied: Přijato."
        assert len(calls) == 1

    async def test_a_failed_relay_run_without_a_verified_delivery_keeps_its_error(self, env: Env) -> None:
        async def watch(_options: ReplyWatchOptions) -> NoReturn:
            raise AssertionError("must not watch")

        s = await env.setup(watch_reply=watch)
        for how in ("none", "denied", "twice"):
            task = started(await s.manager.start_session("send", env.session(live=LIVE), f"SEND_MESSAGE {how}\nFAIL"))
            done = await finished(s.manager, task.id)
            assert done.status == "failed", how
            assert done.error == "API Error: something went wrong", how

    @pytest.mark.parametrize(
        ("how", "why"),
        [
            ("none", "the relay did not send it"),
            ("wrong_to", "the relay addressed it to a different session"),
            ("changed", "the relay changed the text"),
            ("twice", "the relay sent it 2 times instead of once"),
            ("denied", "the relay was not allowed to send it"),
        ],
    )
    async def test_fails_when_the_relay_did_not_send_exactly_the_message(self, env: Env, how: str, why: str) -> None:
        async def watch(_options: ReplyWatchOptions) -> NoReturn:
            raise AssertionError("must not watch")

        s = await env.setup(watch_reply=watch)
        task = started(
            await s.manager.start_session("send", env.session(live=LIVE), f"SEND_MESSAGE {how}\nRESULT SENT")
        )
        done = await finished(s.manager, task.id)
        assert done.status == "failed"
        text = format_for_voice(done)
        assert text.startswith(
            f"Task 1 for session billing-fix failed: could not deliver the message to billing-fix: {why}"
        )
        assert "SEND_MESSAGE" not in text

    async def test_a_denied_send_that_the_relay_retried_counts_once(self, env: Env) -> None:
        async def watch(_options: ReplyWatchOptions) -> ReplyWatchResult:
            return ReplyWatchResult(arrived=True, finished=True, reply="Přijato.", aborted=False)

        s = await env.setup(watch_reply=watch)
        task = started(
            await s.manager.start_session("send", env.session(live=LIVE), "SEND_MESSAGE denied_then_sent\nRESULT SENT")
        )
        done = await finished(s.manager, task.id)
        assert done.status == "done", done.error
        assert format_for_voice(done) == "Task 1 for session billing-fix is done. billing-fix replied: Přijato."

    async def test_run_task_keeps_the_configured_mcp_servers(self, env: Env) -> None:
        servers = {"docs": {"command": "docs-mcp"}}
        s = await env.setup()
        s2 = await env.setup(claude=claude_config(s.record, mcp_servers=servers))
        task = started(await s2.manager.start("alpha", "RESULT ok"))
        await finished(s2.manager, task.id)
        ask = started(await s2.manager.start_session("ask", env.session(), "RESULT ok"))
        await finished(s2.manager, ask.id)
        run, asked = read_records(s.record)
        assert mcp_config(run["argv"]) == {"mcpServers": servers}
        assert mcp_config(asked["argv"]) == {"mcpServers": {}}

    async def test_says_so_when_the_message_was_not_picked_up_or_the_reply_is_pending(self, env: Env) -> None:
        nxt = ReplyWatchResult(arrived=False, finished=False, reply=None, aborted=False)

        async def watch(_options: ReplyWatchOptions) -> ReplyWatchResult:
            return nxt

        s = await env.setup(watch_reply=watch)
        first = started(await s.manager.start_session("send", env.session(live=LIVE), "RESULT SENT"))
        quiet = await finished(s.manager, first.id)
        assert re.search(
            r"has not picked it up within 1 minute\. .*approve the message in the desktop app", format_for_voice(quiet)
        )
        nxt = ReplyWatchResult(arrived=True, finished=False, reply="Looking at it", aborted=False)
        second = started(await s.manager.start_session("send", env.session(live=LIVE), "RESULT SENT"))
        busy = await finished(s.manager, second.id)
        assert re.search(
            r"billing-fix received the message and is still working on it after 1 minute\. Its latest words: Looking at it",
            format_for_voice(busy),
        )

    async def test_frees_the_run_slot_while_waiting_and_can_be_cancelled_then(self, env: Env) -> None:
        s = await env.setup(tasks=tasks_config(max_concurrent=1))
        target = env.session(live=LIVE)
        sent = started(await s.manager.start_session("send", target, "RESULT SENT"))
        await until(lambda: current(s.manager, sent.id).progress[:1] == ["message sent, waiting for it to be read"])
        other = started(await s.manager.start("alpha", "RESULT other"))
        assert (await finished(s.manager, other.id)).status == "done"
        cancel = await s.manager.cancel(sent.id)
        assert isinstance(cancel, Cancelled)
        assert current(s.manager, sent.id).status == "cancelled"

    async def test_finds_the_reply_in_the_real_target_transcript(self, env: Env) -> None:
        s = await env.setup()
        target = env.session(live=LIVE)
        sent = started(await s.manager.start_session("send", target, "RESULT SENT"))
        await until(lambda: current(s.manager, sent.id).progress[:1] == ["message sent, waiting for it to be read"])
        append_lines(target.transcript_path, [peer_message(target.cwd, "vcm-relay-1", "RESULT SENT")])
        await until(lambda: current(s.manager, sent.id).progress[:1] == ["billing-fix is working on the message"])
        append_lines(target.transcript_path, [assistant_text(target.cwd, "Hotovo, testy prošly.")])
        done = await finished(s.manager, sent.id)
        assert (
            format_for_voice(done)
            == "Task 1 for session billing-fix is done. billing-fix replied: Hotovo, testy prošly."
        )


class TestDesktopAccessLeavesSessionRunsIsolated:
    async def test_ask_and_relay_keep_the_strict_empty_config_and_isolated_env(self, env: Env) -> None:
        async def watch(options: ReplyWatchOptions) -> ReplyWatchResult:
            return ReplyWatchResult(arrived=True, finished=True, reply="OK.", aborted=False)

        n = next(_counter)
        record = str(env.base / f"record-desktop-{n}.jsonl")
        s = await env.setup(
            claude=claude_config(record, access="desktop"), public_url="https://vcm.example.test", watch_reply=watch
        )
        ask = started(await s.manager.start_session("ask", env.session(), "RESULT Hotovo."))
        await finished(s.manager, ask.id)
        send = started(await s.manager.start_session("send", env.session(live=LIVE), "Ahoj\nRESULT SENT"))
        await finished(s.manager, send.id)
        records = read_records(record)
        assert len(records) == 2
        for rec in records:
            assert "--strict-mcp-config" in rec["argv"]
            assert mcp_config(rec["argv"]) == {"mcpServers": {}}
            assert arg(rec["argv"], "--setting-sources") == "user,local"
            assert "--settings" not in rec["argv"]
            assert rec["env"]["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
            assert rec["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
