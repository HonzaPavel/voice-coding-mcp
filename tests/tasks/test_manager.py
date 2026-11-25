"""The task manager: queue, sessions per project, persistence, cancel and voice texts."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from itertools import count
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import LogCapture, claude_config, read_records, tasks_config, until
from voice_coding_mcp.clock import now_ms, parse_iso, to_iso
from voice_coding_mcp.log import Logger, silent_logger
from voice_coding_mcp.notify import NotifyTask
from voice_coding_mcp.sessions import LiveSession, SessionInfo
from voice_coding_mcp.tasks.manager import (
    CancelFailed,
    Cancelled,
    Started,
    StartFailed,
    StartItem,
    Task,
    TaskManager,
    TaskManagerOptions,
    format_for_voice,
    session_task_state,
    task_head,
)
from voice_coding_mcp.tasks.runner import RunOptions, RunResult

_counter = count(1)


@dataclass
class Clock:
    offset: int = 0


@dataclass
class Setup:
    manager: TaskManager
    state_dir: str
    record: str
    notified: list[NotifyTask]
    clock: Clock


@dataclass
class Env:
    base: Path
    root: Path
    open: list[TaskManager] = field(default_factory=list)

    async def setup(
        self,
        *,
        tasks: dict[str, Any] | None = None,
        claude: dict[str, Any] | None = None,
        state_dir: str | None = None,
        record: str | None = None,
        run: Callable[[RunOptions], Awaitable[RunResult]] | None = None,
        live: Callable[[], set[str]] | None = None,
        logger: Logger = silent_logger,
        **extra: Any,
    ) -> Setup:
        n = next(_counter)
        state = state_dir or str(self.base / f"state-{n}")
        rec = record or str(self.base / f"record-{n}.jsonl")
        notified: list[NotifyTask] = []
        clock = Clock()

        async def notifier(task: NotifyTask) -> None:
            notified.append(task)

        manager = await TaskManager.create(
            TaskManagerOptions(
                state_dir=state,
                project_roots=[str(self.root)],
                claude=claude_config(rec, **(claude or {})),
                tasks=tasks_config(**(tasks or {})),
                logger=logger,
                notifier=notifier,
                now=lambda: now_ms() + clock.offset,
                kill_grace_ms=500,
                run=run,
                live_session_ids=live or set,
                **extra,
            )
        )
        self.open.append(manager)
        return Setup(manager, state, rec, notified, clock)


@pytest.fixture
async def env(real_tmp: Path) -> AsyncIterator[Env]:
    root = real_tmp / "projects"
    for name in ("alpha", "beta", "gamma", "voice-coding-mcp", "voice-notes"):
        (root / name).mkdir(parents=True)
    e = Env(real_tmp, root)
    yield e
    await asyncio.gather(*(m.shutdown("test end") for m in e.open))


async def start_ok(manager: TaskManager, project: str, prompt: str, new_session: bool = False) -> Task:
    r = await manager.start(project, prompt, new_session=new_session)
    assert isinstance(r, Started), f"start failed: {r}"
    return r.task


def resume_arg(argv: list[str]) -> str | None:
    if "--resume" not in argv:
        return None
    i = argv.index("--resume")
    return argv[i + 1] if i + 1 < len(argv) else None


def reason(result: object) -> str | None:
    return result.reason if isinstance(result, StartFailed) else None


def iso_ms(value: str | None) -> int:
    parsed = parse_iso(value)
    assert parsed is not None
    return parsed


def read_file_json(*parts: str) -> Any:
    return json.loads(Path(*parts).read_text(encoding="utf-8"))


def write_file_json(data: Any, *parts: str) -> None:
    Path(*parts).write_text(json.dumps(data), encoding="utf-8")


def read_task_file(state_dir: str, task_id: int) -> dict[str, Any]:
    data: dict[str, Any] = read_file_json(state_dir, "tasks", f"{task_id}.json")
    return data


def iso_now() -> str:
    return to_iso(now_ms())


def make_dir(*parts: str) -> None:
    Path(*parts).mkdir()


def exists(*parts: str) -> bool:
    return Path(*parts).exists()


async def test_returns_the_answer_when_the_task_finishes_within_the_wait(env: Env) -> None:
    s = await env.setup()
    task = await start_ok(s.manager, "Alpha", "RESULT Tests pass.")
    assert task.id == 1
    done = await s.manager.wait(task.id, 5000)
    assert done is not None
    assert done.status == "done"
    assert done.result == "Tests pass."
    assert format_for_voice(done) == "Task 1 in alpha is done. Tests pass."
    assert read_task_file(s.state_dir, 1)["status"] == "done"
    assert s.notified == []
    mcp = read_file_json(s.state_dir, "mcp-config.json")
    assert mcp == {"mcpServers": {}}


async def test_reports_a_running_task_with_progress_then_notifies_when_unobserved(env: Env) -> None:
    s = await env.setup()
    task = await start_ok(s.manager, "alpha", "PROGRESS 2\nSLEEP 800\nRESULT Done late.")
    early = await s.manager.wait(task.id, 300)
    assert early is not None
    assert early.status == "running"
    text = format_for_voice(early)
    assert text.startswith(
        "Task 1 in alpha is still running for less than a minute. Latest steps: Read config.py; Bash npm test"
    )
    assert "Ask for task 1 again" in text
    await asyncio.sleep(1.5)
    final = s.manager.get(task.id)
    assert final is not None
    assert final.status == "done"
    assert s.notified == [NotifyTask(id=1, project="alpha", status="done", result="Done late.")]


async def test_rejects_unknown_ambiguous_path_like_and_oversized_input(env: Env) -> None:
    s = await env.setup()
    nf = await s.manager.start("kubernetes", "x")
    assert reason(nf) == "not_found"
    assert (
        format_for_voice(nf)
        == 'No project matches "kubernetes". Available projects: alpha, beta, gamma, voice-coding-mcp, voice-notes.'
    )
    amb = await s.manager.start("voice", "x")
    assert format_for_voice(amb) == '"voice" matches several projects: voice-coding-mcp, voice-notes. Which one?'
    assert isinstance(await s.manager.start("../alpha", "x"), StartFailed)
    long = await s.manager.start("alpha", "x" * 20_001)
    assert reason(long) == "prompt_too_long"
    empty = await s.manager.start("alpha", "   ")
    assert reason(empty) == "empty_prompt"
    assert len(s.manager.list_tasks()) == 0


async def test_serializes_runs_per_project_and_runs_other_projects_alongside(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 2})
    a1 = await start_ok(s.manager, "alpha", "SLEEP 400")
    a2 = await start_ok(s.manager, "alpha", "RESULT second")
    b1 = await start_ok(s.manager, "beta", "RESULT beta")
    assert (t := s.manager.get(a2.id)) is not None
    assert t.status == "queued"
    assert (t := s.manager.get(b1.id)) is not None
    assert t.status == "running"
    assert s.manager.queue_position(a2.id) == 1
    r1, r2, r3 = await s.manager.wait_all([a1.id, a2.id, b1.id], 5000)
    assert r1 is not None
    assert r2 is not None
    assert r3 is not None
    assert (r1.status, r2.status, r3.status) == ("done", "done", "done")
    assert iso_ms(r2.started_at) >= iso_ms(r1.finished_at)
    assert iso_ms(r3.finished_at) <= iso_ms(r1.finished_at)


async def test_respects_max_concurrent_and_fifo_and_rejects_beyond_max_queued(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 1, "max_queued": 2})
    m = s.manager
    a = await start_ok(m, "alpha", "SLEEP 300")
    c = await start_ok(m, "gamma", "RESULT c")
    b = await start_ok(m, "beta", "RESULT b")
    full = await m.start("voice-notes", "RESULT d")
    assert reason(full) == "queue_full"
    assert "Too many tasks" in format_for_voice(full)
    queued = m.get(c.id)
    assert queued is not None
    assert "waiting in the queue, position 1" in format_for_voice(queued, queue_position=m.queue_position(c.id))
    ra, rc, rb = await m.wait_all([a.id, c.id, b.id], 5000)
    assert ra is not None
    assert rc is not None
    assert rb is not None
    assert iso_ms(rc.started_at) >= iso_ms(ra.finished_at)
    assert iso_ms(rb.started_at) >= iso_ms(rc.finished_at)
    assert [t.id for t in m.list_tasks()] == [3, 2, 1]
    assert [t.id for t in m.list_tasks(project="gamma")] == [2]
    assert (latest := m.latest()) is not None
    assert latest.id == 3
    assert (latest := m.latest("alpha")) is not None
    assert latest.id == 1


async def test_resumes_the_project_session_within_the_window_not_after_and_not_with_new_session(env: Env) -> None:
    s = await env.setup()
    m = s.manager

    async def run(prompt: str, new_session: bool = False) -> Task:
        t = await start_ok(m, "alpha", prompt, new_session)
        done = await m.wait(t.id, 5000)
        assert done is not None
        return done

    s1 = (await run("RESULT one")).session_id
    assert s1
    assert (await run("RESULT two")).resumed_session_id == s1
    s3 = (await run("RESULT three", True)).session_id
    assert s3
    assert s3 != s1
    assert (await run("RESULT four")).resumed_session_id == s3
    s.clock.offset = 13 * 3_600_000
    assert (await run("RESULT five")).resumed_session_id is None
    assert [resume_arg(r["argv"]) for r in read_records(s.record)] == [None, s1, None, s3, None]


async def test_decides_the_session_when_a_queued_task_starts(env: Env) -> None:
    s = await env.setup()
    t1 = await start_ok(s.manager, "beta", "SLEEP 200")
    t2 = await start_ok(s.manager, "beta", "RESULT follow-up")
    r1, r2 = await s.manager.wait_all([t1.id, t2.id], 5000)
    assert r1 is not None
    assert r2 is not None
    assert r1.session_id
    assert r2.resumed_session_id == r1.session_id


async def test_session_busy_while_a_task_uses_or_will_resume_the_session(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 1})
    m = s.manager
    first = await m.wait((await start_ok(m, "alpha", "RESULT one")).id, 5000)
    assert first is not None
    assert first.session_id
    sid = first.session_id
    assert not m.session_busy(sid)
    blocker = await start_ok(m, "beta", "SLEEP 300")
    queued = await start_ok(m, "alpha", "RESULT two")
    assert m.session_busy(sid)  # queued, will resume the project session
    assert not m.session_busy(sid, exclude_task=queued.id)
    fresh_sid = "not-a-session"
    assert not m.session_busy(fresh_sid)
    done = await m.wait_all([blocker.id, queued.id], 5000)
    assert all(t is not None and t.status == "done" for t in done)
    assert not m.session_busy(sid)


COPY_OF_SECOND = "44444444-0000-4000-8000-000000000002"


async def test_does_not_resume_a_session_that_is_open_in_remote_control(env: Env) -> None:
    live: set[str] = set()
    s = await env.setup(live=lambda: live)
    m = s.manager

    async def run(prompt: str) -> Task:
        done = await m.wait((await start_ok(m, "alpha", prompt)).id, 5000)
        assert done is not None
        return done

    first = await run("RESULT one")
    assert first.session_id
    live.add(first.session_id)
    second = await run("RESULT two")
    assert second.resumed_session_id is None
    assert second.resume_skipped is True
    assert second.session_id != first.session_id
    assert format_for_voice(second) == (
        "Task 2 in alpha (in a new conversation, because the previous one is open in the Claude app) is done. two"
    )
    assert read_task_file(s.state_dir, 2)["resumeSkipped"] is True
    assert "resumeSkipped" not in read_task_file(s.state_dir, 1)
    # A Remote Control copy of the project session counts too.
    assert second.session_id
    m.set_remote_session(2, COPY_OF_SECOND)
    assert read_task_file(s.state_dir, 2)["remoteSessionId"] == COPY_OF_SECOND
    assert COPY_OF_SECOND in m.known_session_ids()
    live.clear()
    live.add(COPY_OF_SECOND)
    third = await run("RESULT three")
    assert third.resume_skipped is True
    live.clear()
    fourth = await run("RESULT four")
    assert fourth.resumed_session_id == third.session_id
    assert fourth.resume_skipped is None


async def test_retries_once_without_resume_when_the_session_is_gone(env: Env) -> None:
    s = await env.setup()
    t1 = await start_ok(s.manager, "alpha", "RESULT first")
    r1 = await s.manager.wait(t1.id, 5000)
    assert r1 is not None
    s1 = r1.session_id
    t2 = await start_ok(s.manager, "alpha", "RESUME_MISSING\nRESULT fresh")
    r2 = await s.manager.wait(t2.id, 5000)
    assert r2 is not None
    assert r2.status == "done"
    assert r2.result == "fresh"
    assert r2.resumed_session_id is None
    assert r2.session_id != s1
    assert [resume_arg(r["argv"]) for r in read_records(s.record)] == [None, s1, None]


async def test_never_stores_the_session_id_of_a_resume_missing_run(env: Env) -> None:
    calls: list[str | None] = []

    async def run(opts: RunOptions) -> RunResult:
        calls.append(opts.resume_session_id)
        return RunResult(
            ok=False,
            outcome="error",
            exit_code=1,
            session_id=opts.resume_session_id or "gone",
            is_error=True,
            cost_usd=0,
            duration_ms=0,
            num_turns=0,
            stderr_tail="No conversation found with session ID: gone\n",
            resume_missing=True,
            error="error during execution",
        )

    s = await env.setup(run=run)
    t = await start_ok(s.manager, "alpha", "anything")
    r = await s.manager.wait(t.id, 5000)
    assert r is not None
    assert r.status == "failed"
    assert r.session_id is None
    assert len(calls) == 1
    await s.manager.shutdown("test end")
    assert not exists(s.state_dir, "projects.json")


async def test_does_not_rerun_a_resumed_task_that_did_work_and_failed_otherwise(env: Env) -> None:
    s = await env.setup()
    t1 = await start_ok(s.manager, "alpha", "RESULT first")
    r1 = await s.manager.wait(t1.id, 5000)
    assert r1 is not None
    s1 = r1.session_id
    t2 = await start_ok(s.manager, "alpha", "PROGRESS 2\nSTDERR session abc not found\nFAIL")
    r2 = await s.manager.wait(t2.id, 5000)
    assert r2 is not None
    assert r2.status == "failed"
    assert r2.resumed_session_id == s1
    assert r2.error == "API Error: something went wrong"
    assert [resume_arg(r["argv"]) for r in read_records(s.record)] == [None, s1]


async def test_leaves_no_stuck_task_when_a_state_write_fails_in_start(env: Env) -> None:
    s = await env.setup()
    # os.replace() of the temp file onto a directory fails
    make_dir(s.state_dir, "task-counter.json")
    r = await s.manager.start("alpha", "RESULT never")
    assert reason(r) == "state_error"
    assert "could not save the task" in format_for_voice(r)
    assert s.manager.list_tasks() == []
    assert s.manager.latest() is None
    assert s.manager.queue_position(1) is None
    assert read_records(s.record) == []


async def test_marks_failures_and_timeouts(env: Env) -> None:
    s = await env.setup(tasks={"timeout_minutes": 1})
    f = await start_ok(s.manager, "alpha", "FAIL")
    rf = await s.manager.wait(f.id, 5000)
    assert rf is not None
    assert rf.status == "failed"
    assert format_for_voice(rf) == "Task 1 in alpha failed: API Error: something went wrong."
    timeout = rf.model_copy(update={"status": "timeout", "error": "timed out after 900 s"})
    assert format_for_voice(timeout) == "Task 1 in alpha timed out and was stopped."


async def test_cancels_queued_and_running_tasks(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 1})
    m = s.manager
    running = await start_ok(m, "alpha", "SLEEP 60000")
    queued = await start_ok(m, "beta", "RESULT never")
    cq = await m.cancel(queued.id)
    assert isinstance(cq, Cancelled)
    assert cq.task is not None
    assert cq.task.status == "cancelled"
    await asyncio.sleep(0.2)
    started = time.monotonic()
    cr = await m.cancel(running.id)
    assert isinstance(cr, Cancelled)
    assert cr.task is not None
    assert cr.task.status == "cancelled"
    assert time.monotonic() - started < 5
    after = m.get(running.id)
    assert after is not None
    assert format_for_voice(after) == "Task 1 in alpha was cancelled."
    again = await m.cancel(running.id)
    assert isinstance(again, CancelFailed)
    assert again.reason == "already_finished"
    assert isinstance(await m.cancel(99), CancelFailed)
    assert s.notified == []
    assert len(read_records(s.record)) == 1


def test_cuts_long_answers_for_voice() -> None:
    task = Task(
        id=4,
        project="alpha",
        project_path="/x",
        prompt="p",
        status="done",
        new_session=False,
        created_at="",
        started_at=None,
        finished_at=None,
        resumed_session_id=None,
        session_id=None,
        result="a" * 5000,
        error=None,
        progress=[],
        cost_usd=None,
        duration_ms=None,
        num_turns=None,
        permission_denials=0,
        exit_code=None,
    )
    text = format_for_voice(task)
    assert text.startswith("Task 4 in alpha is done. aaa")
    assert len(text) < 2100
    assert "Answer shortened" in text
    assert format_for_voice(task.model_copy(update={"result": ""})) == "Task 4 in alpha is done, without a text answer."


async def test_continues_ids_fails_interrupted_tasks_and_prunes_old_files_after_a_restart(env: Env) -> None:
    first = await env.setup()
    t1 = await start_ok(first.manager, "alpha", "RESULT persisted")
    await first.manager.wait(t1.id, 5000)
    await first.manager.shutdown("restart")
    tasks_dir = Path(first.state_dir, "tasks")
    persisted = first.manager.get(t1.id)
    assert persisted is not None

    def stale(task_id: int, status: str) -> str:
        data = {**persisted.to_json(), "id": task_id, "status": status, "result": None, "finishedAt": None}
        return json.dumps(data)

    (tasks_dir / "2.json").write_text(stale(2, "running"))
    (tasks_dir / "3.json").write_text(stale(3, "queued"))
    (tasks_dir / "4.json").write_text(stale(4, "done"))
    old = time.time() - 20 * 86_400
    os.utime(tasks_dir / "4.json", (old, old))

    second = await env.setup(state_dir=first.state_dir)
    m = second.manager
    assert (t := m.get(1)) is not None
    assert t.result == "persisted"
    assert (t := m.get(2)) is not None
    assert t.status == "failed"
    assert t.error == "server restarted"
    assert (t := m.get(3)) is not None
    assert t.status == "failed"
    assert m.get(4) is None
    assert "4.json" not in os.listdir(tasks_dir)
    assert json.loads((tasks_dir / "2.json").read_text(encoding="utf-8"))["status"] == "failed"

    nxt = await start_ok(m, "alpha", "RESULT next")
    assert nxt.id == 5
    projects = read_file_json(first.state_dir, "projects.json")
    assert projects["projects"][str(env.root / "alpha")]["lastSessionId"]


async def test_shutdown_fails_queued_and_running_tasks_and_refuses_new_ones(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 1})
    m = s.manager
    a = await start_ok(m, "alpha", "SLEEP 60000")
    b = await start_ok(m, "beta", "RESULT b")
    await asyncio.sleep(0.2)
    await m.shutdown("server stopping")
    assert (t := m.get(a.id)) is not None
    assert t.status == "failed"
    assert t.error == "server stopping"
    assert (t := m.get(b.id)) is not None
    assert t.status == "failed"
    r = await m.start("alpha", "x")
    assert reason(r) == "shutting_down"


async def test_start_many_starts_each_item_independently(env: Env) -> None:
    s = await env.setup()
    results = await s.manager.start_many(
        [
            StartItem(project="alpha", prompt="RESULT a"),
            StartItem(project="nope-nope-nope", prompt="RESULT b"),
            StartItem(project="beta", prompt="RESULT c"),
        ]
    )
    assert [isinstance(r, Started) for r in results] == [True, False, True]
    ids = [r.task.id for r in results if isinstance(r, Started)]
    done = await s.manager.wait_all(ids, 5000)
    assert [t.result if t else None for t in done] == ["a", "c"]


async def test_concurrent_starts_never_exceed_max_queued(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 1, "max_queued": 1})
    m = s.manager
    projects = ["alpha", "beta", "gamma", "voice-notes", "voice-coding-mcp"]
    results = await asyncio.gather(*(m.start(p, "SLEEP 300") for p in projects))
    ok = [r for r in results if isinstance(r, Started)]
    assert len(ok) == 2, results
    assert [reason(r) for r in results].count("queue_full") == 3
    assert len(read_records(s.record)) <= 1
    done = await m.wait_all([r.task.id for r in ok], 5000)
    assert all(t is not None and t.status == "done" for t in done)
    assert len(read_records(s.record)) == 2


async def test_concurrent_starts_for_one_project_queue_behind_each_other(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 3, "max_queued": 1})
    results = await asyncio.gather(*(s.manager.start("alpha", "SLEEP 200") for _ in range(3)))
    assert [isinstance(r, Started) for r in results].count(True) == 2
    assert [reason(r) for r in results].count("queue_full") == 1


async def test_starts_for_one_project_run_in_order(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 3, "max_queued": 3})
    first, second = await asyncio.gather(
        s.manager.start("alpha", "SLEEP 200\nRESULT first"), s.manager.start("alpha", "RESULT second")
    )
    assert isinstance(first, Started)
    assert isinstance(second, Started)
    assert second.task.status == "queued"
    done = await s.manager.wait_all([first.task.id, second.task.id], 5000)
    assert [t.result if t else None for t in done] == ["first", "second"]
    assert [rec["stdin"].strip() for rec in read_records(s.record)] == ["SLEEP 200\nRESULT first", "RESULT second"]


async def test_prunes_old_task_files_periodically(env: Env) -> None:
    s = await env.setup(tasks={"keep_days": 1}, prune_interval_ms=50)
    m = s.manager
    done = await start_ok(m, "alpha", "RESULT old")
    await m.wait(done.id, 5000)
    tasks_dir = os.path.join(s.state_dir, "tasks")
    name = f"{done.id}.json"
    assert name in os.listdir(tasks_dir)
    old = time.time() - 2 * 86_400
    os.utime(os.path.join(tasks_dir, name), (old, old))

    def pruned() -> bool:
        return name not in os.listdir(tasks_dir)

    await until(pruned)
    nxt = await start_ok(m, "alpha", "RESULT new")
    assert nxt.id == done.id + 1


async def test_logs_stderr_only_as_its_length_unless_log_prompts(env: Env) -> None:
    quiet = LogCapture()
    s = await env.setup(logger=quiet.logger)
    t = await start_ok(s.manager, "alpha", "STDERR secret prompt text\nFAIL")
    await s.manager.wait(t.id, 5000)
    [entry] = [r for r in quiet.records if r["msg"] == "task stderr"]
    assert entry["stderrChars"] == len("secret prompt text\n")
    assert "stderrTail" not in entry
    assert "secret prompt text" not in quiet.text()

    loud = LogCapture()
    s2 = await env.setup(logger=loud.logger, log_prompts=True)
    t2 = await start_ok(s2.manager, "alpha", "STDERR visible text\nFAIL")
    await s2.manager.wait(t2.id, 5000)
    [entry2] = [r for r in loud.records if r["msg"] == "task stderr"]
    assert entry2["stderrTail"] == "visible text"


BAD_ID = "--dangerous-flag"


def invalid_id_warnings(log: LogCapture) -> list[dict[str, Any]]:
    return [r for r in log.records if r["msg"] == "invalid session id ignored"]


async def test_never_resumes_an_invalid_last_session_id_from_projects_json(env: Env) -> None:
    first = await env.setup()
    await first.manager.shutdown("restart")
    alpha = str(env.root / "alpha")
    projects = {"projects": {alpha: {"name": "alpha", "lastSessionId": BAD_ID, "lastActivityAt": iso_now()}}}
    write_file_json(projects, first.state_dir, "projects.json")
    log = LogCapture()
    s = await env.setup(state_dir=first.state_dir, logger=log.logger)
    done = await s.manager.wait((await start_ok(s.manager, "alpha", "RESULT fresh")).id, 5000)
    assert done is not None
    assert done.status == "done"
    assert done.resumed_session_id is None
    assert [resume_arg(r["argv"]) for r in read_records(s.record)] == [None]
    assert len(invalid_id_warnings(log)) == 1
    assert BAD_ID not in log.text()


async def test_never_stores_an_invalid_session_id_from_a_run_result(env: Env) -> None:
    async def run(_options: RunOptions) -> RunResult:
        return RunResult(ok=True, outcome="success", exit_code=0, session_id=BAD_ID, result="done")

    log = LogCapture()
    s = await env.setup(run=run, logger=log.logger)
    done = await s.manager.wait((await start_ok(s.manager, "alpha", "x")).id, 5000)
    assert done is not None
    assert done.status == "done"
    assert done.session_id is None
    assert not exists(s.state_dir, "projects.json") or read_file_json(s.state_dir, "projects.json")["projects"] == {}
    assert len(invalid_id_warnings(log)) == 1
    assert BAD_ID not in log.text()


async def test_drops_invalid_session_ids_from_task_files_and_remote_control(env: Env) -> None:
    first = await env.setup()
    t1 = await start_ok(first.manager, "alpha", "RESULT persisted")
    await first.manager.wait(t1.id, 5000)
    await first.manager.shutdown("restart")
    data = read_task_file(first.state_dir, 1)
    data.update({"sessionId": BAD_ID, "remoteSessionId": "x", "resumedSessionId": "y"})
    write_file_json(data, first.state_dir, "tasks", "1.json")
    log = LogCapture()
    s = await env.setup(state_dir=first.state_dir, logger=log.logger)
    assert (t := s.manager.get(1)) is not None
    assert (t.session_id, t.remote_session_id, t.resumed_session_id) == (None, None, None)
    assert len(invalid_id_warnings(log)) == 3
    s.manager.set_remote_session(1, BAD_ID)
    assert (t := s.manager.get(1)) is not None
    assert t.remote_session_id is None
    assert len(invalid_id_warnings(log)) == 4
    assert BAD_ID not in log.text()


async def test_names_the_sessions_opened_for_remote_control(env: Env) -> None:
    s = await env.setup()
    done = await s.manager.wait((await start_ok(s.manager, "alpha", "RESULT x")).id, 5000)
    assert done is not None
    assert s.manager.remote_session_names() == {}
    remote = "55555555-0000-4000-8000-000000000001"
    s.manager.set_remote_session(done.id, remote)
    assert s.manager.remote_session_names() == {remote: f"alpha (voice {done.id})"}


def name_arg(argv: list[str]) -> str | None:
    return argv[argv.index("--name") + 1] if "--name" in argv else None


async def test_names_tasks_by_title_else_by_project_and_passes_the_name_to_headless_runs(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 3})
    m = s.manager
    titled = await start_ok_titled(m, "alpha", "RESULT a", "  Oprava\npřihlášení ")
    untitled = await start_ok(m, "beta", "RESULT secret prompt words")
    many = await m.start_many([StartItem(project="gamma", prompt="RESULT g", title="-Gamma check")])
    assert isinstance(many[0], Started)
    remote = m.start_remote("alpha", "Oprav testy", title="Opravit testy")
    assert isinstance(remote, Started)
    assert titled.title == "Oprava přihlášení"
    assert titled.name == f"Oprava přihlášení (voice {titled.id})"
    assert untitled.title is None
    assert untitled.name == f"beta (voice {untitled.id})"
    assert many[0].task.name == f"Gamma check (voice {many[0].task.id})"
    assert remote.task.name == f"Opravit testy (voice {remote.task.id})"
    await m.wait_all([titled.id, untitled.id, many[0].task.id], 5000)
    names = sorted(name_arg(r["argv"]) or "" for r in read_records(s.record))
    assert names == sorted([titled.name, untitled.name, many[0].task.name])
    assert all("secret" not in " ".join(r["argv"]) for r in read_records(s.record))
    # The name survives a restart (task file).
    again = await env.setup(state_dir=s.state_dir, record=s.record)
    reloaded = again.manager.get(titled.id)
    assert reloaded is not None
    assert (reloaded.title, reloaded.name) == (titled.title, titled.name)


async def start_ok_titled(manager: TaskManager, project: str, prompt: str, title: str) -> Task:
    r = await manager.start(project, prompt, title=title)
    assert isinstance(r, Started), f"start failed: {r}"
    return r.task


async def test_a_later_start_for_another_project_waits_for_the_free_slot_in_order(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 1, "max_queued": 3})
    results = await asyncio.gather(*(s.manager.start(p, f"RESULT {p}") for p in ("alpha", "beta", "gamma")))
    assert all(isinstance(r, Started) for r in results)
    done = await s.manager.wait_all([r.task.id for r in results if isinstance(r, Started)], 5000)
    assert [t.result if t else None for t in done] == ["alpha", "beta", "gamma"]
    assert [rec["stdin"].strip() for rec in read_records(s.record)] == ["RESULT alpha", "RESULT beta", "RESULT gamma"]


def session_info(env: Env) -> SessionInfo:
    return SessionInfo(
        id="77777777-0000-4000-8000-000000000001",
        name="billing-fix",
        titled=True,
        project="alpha",
        cwd=str(env.root / "alpha"),
        last_activity_ms=now_ms(),
        transcript_path=str(env.base / "none.jsonl"),
        live=None,
    )


async def test_forgets_the_session_details_of_cancelled_and_shut_down_queued_session_tasks(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 1, "max_queued": 5})
    m = s.manager
    await start_ok(m, "beta", "SLEEP 60000")
    cancelled = await m.start_session("ask", session_info(env), "RESULT never")
    stopped = await m.start_session("ask", session_info(env), "RESULT never either")
    assert isinstance(cancelled, Started)
    assert isinstance(stopped, Started)
    assert m.queue_position(cancelled.task.id) == 1
    assert isinstance(await m.cancel(cancelled.task.id), Cancelled)
    assert cancelled.task.id not in m._specs
    assert stopped.task.id in m._specs
    await m.shutdown("test end")
    assert m._specs == {}


def test_task_head_names_the_session_or_the_project() -> None:
    task = Task(id=3, project="alpha", project_path="/x", prompt="p", status="done", new_session=False, created_at="")
    assert task_head(task) == "Task 3 in alpha"
    assert task_head(task.model_copy(update={"target_session_name": "billing-fix"})) == "Task 3 for session billing-fix"


# --- session tasks (run_task with remote_control) -------------------------------------------------

SESSION_ID = "77777777-0000-4000-8000-000000000001"


def live_entry(status: str, waiting_for: str | None = None) -> dict[str, LiveSession]:
    return {
        SESSION_ID: LiveSession(pid=1, socket_path=None, status=status, name="alpha (voice 1)", waiting_for=waiting_for)
    }


async def test_session_tasks_take_no_slot_survive_a_restart_and_end_when_the_session_is_gone(env: Env) -> None:
    s = await env.setup(tasks={"max_concurrent": 1})
    m = s.manager
    started = m.start_remote("alpha", "Oprav testy")
    assert isinstance(started, Started)
    task_id = started.task.id
    m.remote_launched(task_id, session_id=SESSION_ID, bg_id="77777777")
    assert m.remote_task_ids() == [task_id]
    # No run slot: a headless task still starts right away.
    other = await start_ok(m, "beta", "RESULT b")
    assert (await m.wait(other.id, 5000) or other).status == "done"

    m.refresh_remote(live_entry("waiting", "permission prompt"))
    assert (t := m.get(task_id)) is not None
    assert t.session_state == "approval"
    write_file_json({"session_id": SESSION_ID, "last_assistant_message": " Hotovo. "}, m.session_answer_path(task_id))
    m.refresh_remote(live_entry("idle"))
    assert (t := m.get(task_id)) is not None
    assert (t.status, t.session_state, t.result) == ("running", "idle", "Hotovo.")
    assert read_task_file(s.state_dir, task_id)["sessionState"] == "idle"

    # The answer of another session is ignored.
    write_file_json({"session_id": "x", "last_assistant_message": "cizí"}, m.session_answer_path(task_id))
    m.refresh_remote(live_entry("busy"))
    assert (t := m.get(task_id)) is not None
    assert (t.session_state, t.result) == ("working", "Hotovo.")

    restarted = (await env.setup(state_dir=s.state_dir)).manager
    assert (t := restarted.get(task_id)) is not None
    assert t.status == "running"  # not "server restarted": the session runs on its own
    restarted.refresh_remote({})
    assert (t := restarted.get(task_id)) is not None
    assert (t.status, t.session_state, t.result) == ("done", None, "Hotovo.")
    assert format_for_voice(t).startswith("Task 1 in alpha is done; session alpha (voice 1) has ended.")


async def test_a_session_task_without_a_known_session_ends_after_a_grace_period(env: Env) -> None:
    s = await env.setup()
    m = s.manager
    started = m.start_remote("alpha", "Ahoj")
    assert isinstance(started, Started)
    m.remote_launched(started.task.id, session_id=None, bg_id="77777777")
    m.refresh_remote({})
    assert (t := m.get(started.task.id)) is not None
    assert t.status == "running"
    m.refresh_remote(live_entry("busy"))  # found by its id8
    assert (t := m.get(started.task.id)) is not None
    assert t.session_id == SESSION_ID
    s.clock.offset = 2 * 60_000
    other = m.start_remote("beta", "Ahoj")
    assert isinstance(other, Started)
    m.remote_launched(other.task.id, session_id=None, bg_id="not-hex!")
    assert (t := m.get(other.task.id)) is not None
    assert t.bg_id is None
    s.clock.offset = 4 * 60_000
    m.refresh_remote({})
    assert (t := m.get(other.task.id)) is not None
    assert t.status == "done"


async def test_a_session_task_that_lost_remote_control_reads_as_disconnected(env: Env) -> None:
    s = await env.setup()
    m = s.manager
    started = m.start_remote("alpha", "Ahoj")
    assert isinstance(started, Started)
    m.remote_launched(started.task.id, session_id=SESSION_ID, bg_id="77777777")
    now = now_ms() + s.clock.offset

    def bg(started_at: int | None, bridge: str | None = None) -> dict[str, LiveSession]:
        entry = LiveSession(
            pid=1, socket_path=None, status="idle", name="77777777", kind="bg", bridge_session_id=bridge
        )
        return {SESSION_ID: replace(entry, started_at_ms=started_at)}

    # Still starting up, or no startedAt to judge by: not disconnected yet.
    for live in (bg(now - 5_000), bg(None), bg(now - 7_200_000, "session_01X")):
        m.refresh_remote(live)
        assert (t := m.get(started.task.id)) is not None
        assert t.session_state == "idle"
    m.refresh_remote(bg(now - 7_200_000))
    assert (t := m.get(started.task.id)) is not None
    assert (t.status, t.session_state) == ("running", "disconnected")
    assert format_for_voice(t).startswith(
        "Task 1 in alpha: session alpha (voice 1) lost its connection to the Claude app and no longer shows there."
    )
    assert session_task_state(t) == "session alpha (voice 1) lost its connection to the Claude app"


async def test_set_remote_session_keeps_the_launcher_id8(env: Env) -> None:
    s = await env.setup()
    m = s.manager
    done = await start_ok(m, "alpha", "RESULT a")
    await m.wait(done.id, 5000)
    m.set_remote_session(done.id, SESSION_ID, bg_id="77777777")
    assert (t := m.get(done.id)) is not None
    assert (t.remote_session_id, t.bg_id) == (SESSION_ID, "77777777")
    assert m.remote_session_tasks()[SESSION_ID].id == done.id


DESKTOP_COPY = "44444444-0000-4000-8000-000000000003"


async def test_records_the_desktop_handoff_and_clears_it_when_reopened(env: Env) -> None:
    live: set[str] = set()
    s = await env.setup(live=lambda: live)
    m = s.manager
    first = await m.wait((await start_ok(m, "alpha", "RESULT one")).id, 5000)
    assert first is not None
    assert first.session_id
    m.set_remote_session(first.id, DESKTOP_COPY, bg_id=DESKTOP_COPY[:8], launched=True)
    m.handed_over(first.id, DESKTOP_COPY)
    m.handed_over(first.id, "not-a-uuid")  # ignored
    saved = read_task_file(s.state_dir, first.id)
    assert saved["desktopSessionId"] == DESKTOP_COPY
    assert isinstance(saved["handedOverAt"], str)
    assert format_for_voice(m.get(first.id) or first) == "Task 1 in alpha is done. one"

    # The owner continues the handed-over copy in the desktop app: a follow-up never resumes the
    # project session (that would make a second writer), it starts a fresh conversation.
    live.add(DESKTOP_COPY)
    second = await m.wait((await start_ok(m, "alpha", "RESULT two")).id, 5000)
    assert second is not None
    assert second.resumed_session_id is None
    assert second.resume_skipped is True
    assert "--resume" not in read_records(s.record)[-1]["argv"]

    # Reopening with the same id but no launch keeps the marker; a launch clears it.
    m.set_remote_session(first.id, DESKTOP_COPY, bg_id=DESKTOP_COPY[:8])
    assert (m.get(first.id) or first).handed_over_at is not None
    m.set_remote_session(first.id, DESKTOP_COPY, bg_id=DESKTOP_COPY[:8], launched=True)
    after = read_task_file(s.state_dir, first.id)
    assert "handedOverAt" not in after
    assert "desktopSessionId" not in after
