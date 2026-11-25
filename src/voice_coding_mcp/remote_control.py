"""Reopen a finished task's Claude Code session with Remote Control, so the owner can continue it by
typing in the Code section of the Claude app.

One launch is `claude --bg --resume <session id> --remote-control <name> ...` in the task's project
directory, stdin /dev/null, no shell. The launcher prints "backgrounded · <id8>" and exits at once;
the background session then shows up in the live registry (`<claudeHome>/sessions/<pid>.json`) with
`kind: "bg"` and, a few seconds later, `bridgeSessionId`. The registry `name` is the id8, not the
Remote Control name. The app link is `https://claude.ai/code/<bridgeSessionId>`. Launcher output
goes to a temp file, not a pipe: the background session may inherit it and would hold a pipe open.

A session that is still in the CLI's background list (`claude agents --all`, also once stopped)
is copied by `--bg --resume`: a new id ("backgrounded · <new id8>" plus a note). The server
therefore runs `claude rm <id8>` first (it keeps the conversation), so the resume continues the
session under its own id; when `rm` refuses, a copy results. The opened id is
therefore taken from the launcher output and returned, and the caller persists it on the task
(`Task.remote_session_id`); the limit counts the sessions among those persisted ids that are
connected to Remote Control (or still starting up).

A bg session can lose Remote Control while its process stays alive (no `bridgeSessionId` any more,
`claude logs` ends with "Remote Control disconnected ... (code 4090)"); the owner no longer sees it
in the app. Such a stale session (`rc_disconnected`) is not counted, and reopening its task stops
it first (`claude stop <id8>`), waits until the registry no longer lists it (else it fails with a
retry hint, a launch would only make a copy) and launches again. `close` stops a task's live
session for cancel_task and the idle close, also when the task itself has finished.

`open`, `start_session` and `close` are serialized by one lock and re-read the task
(`get_task`) once they hold it, so an idle stop and a reopen of the same task never interleave.
The idle close passes `only_if`, which is checked under the lock against the session's current
registry entry: a session that is no longer idle is not stopped (`SKIPPED`).

`start_session` starts a new task as a Remote Control session instead (`run_task` with
`remote_control`): `claude --bg --remote-control <name> --name <name> ...` with the prompt on stdin
(never in argv) and the task's access. Its `--settings` JSON (one, merged with the access settings)
adds a Stop hook that writes the hook input, which carries `last_assistant_message`, to
`stateDir/session-answers/<task id>.json`; that is the task's latest answer. `close` ends such a
session with `claude stop <id8>` (the conversation is kept). A reopened session (`--resume`) gets
the same Stop hook, so the latest answer of a task continued in the app is known too.

Handoff to the desktop app (`remoteControl.handoffToDesktop`, macOS): the desktop app lists its own
records plus only recent or live CLI sessions, so a stopped voice session drops out of its list.
`open claude://resume?session=<id>` makes the app adopt the transcript into its own list, with the
title the session was given (`--name`). After `close` succeeded, a background job waits
until the registry no longer lists the session (the app does not check whether another process
owns the transcript), then runs `/usr/bin/open <url>` (argv, no shell). Failures are only logged;
a success is recorded on the task (`on_handoff`). At shutdown pending handoffs get a few seconds
(`shutdown_wait_ms`) to finish before they are cancelled. A session that is live in a process the server did
not start (the desktop app, a terminal: any entry that is not `kind: "bg"`) is never resumed.

Verified with Claude Code 2.1.331 on macOS; these are internal files and flags, best effort.
`--bg` works only in a trusted folder; the launcher then prints "backgrounded · <id8> · <name>".
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from .clock import now_ms
from .config import ClaudeConfig
from .json_file import loads_json
from .log import Logger
from .sessions import LiveSession, clip_words, is_session_id, rc_disconnected, read_live_sessions
from .tasks.access import RunAccess, isolated_access, task_access
from .tasks.manager import StartFailed, StartResult, Task, is_terminal, live_bg_session, session_name, valid_bg_id
from .tasks.prompt import REMOTE_SESSION_PROMPT
from .tasks.runner import build_env
from .text import plural

LINK_BASE = "https://claude.ai/code/"
DEFAULT_LINK_WAIT_MS = 20_000
DEFAULT_POLL_MS = 500
DEFAULT_LAUNCH_TIMEOUT_MS = 15_000
DEFAULT_STOP_WAIT_MS = 5_000
"""After `claude stop`, how long to wait for the registry entry to go before launching again."""
DEFAULT_HANDOFF_WAIT_MS = 30_000
"""How long a handoff waits for the stopped session to leave the registry before it gives up."""
DEFAULT_SHUTDOWN_WAIT_MS = 5_000
"""How long `aclose` lets pending handoffs finish before it cancels them."""
SKIPPED = "skipped"
"""`close` with `only_if`: the session was not stopped, it is no longer in that state."""
CloseReason = Literal["cancel", "idle"]
DESKTOP_RESUME_URL = "claude://resume?session="
OPEN_BIN = "/usr/bin/open"
OUTPUT_LIMIT = 8192
AGENTS_OUTPUT_LIMIT = 4 * 1024 * 1024
"""`claude agents --json --all` lists every background session the CLI remembers."""
UNTRUSTED_TEXT = "Workspace not trusted"
"""What `claude --bg` prints when the folder's trust prompt was never accepted (`claude -p` does not check)."""
RESUME_MISSING_TEXT = "No conversation found"
NO_RC_ENV: tuple[str, ...] = ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",)
"""Dropped from the task environment for the launch: with it Remote Control never connects
(verified with 2.1.331, the bg session starts but gets no `bridgeSessionId`)."""

_BACKGROUNDED_RE = re.compile(r"backgrounded\s+\S+\s+([0-9a-f]{8})\b")


def launched_id8(output: str) -> str | None:
    """The id8 after "backgrounded · " in the launcher output (a copy has a new one)."""
    match = _BACKGROUNDED_RE.search(output)
    return match.group(1) if match else None


def link_for(live: LiveSession) -> str | None:
    return f"{LINK_BASE}{live.bridge_session_id}" if live.bridge_session_id else None


@dataclass(frozen=True)
class RcOpened:
    name: str
    project: str
    link: str | None
    """None while Remote Control has not connected yet."""
    already_open: bool
    session_id: str | None
    """Full id of the Remote Control session (may be a copy), to persist; None when not found in time."""
    bg_id: str | None = None
    """The id8 the launcher printed (what `claude stop` takes); None when nothing was launched."""
    replaced_stale: bool = False
    """A stale session of the task (alive, but disconnected from the app) was stopped first."""


RcFailureReason = Literal[
    "disabled",
    "not_finished",
    "relay_task",
    "no_session",
    "session_busy",
    "limit",
    "untrusted",
    "launch_failed",
    "stop_failed",
    "still_stopping",
    "open_elsewhere",
]


@dataclass(frozen=True)
class RcFailed:
    reason: RcFailureReason
    message: str
    """Plain words for the voice model."""


RcResult = RcOpened | RcFailed


@dataclass(frozen=True)
class RcStarted:
    """start_session created the task; `outcome` says whether its session started."""

    task: Task
    outcome: RcResult


SessionStartResult = RcStarted | RcFailed | StartFailed
"""RcFailed here (only "disabled" or "limit") means no task was created."""


@dataclass
class RemoteControlOptions:
    claude: ClaudeConfig
    claude_home: str
    """Where the live registry is (`claude.config_dir` or `~/.claude`)."""
    mcp_config_path: str
    """The task runs' MCP config (`stateDir/mcp-config.json`, written at startup)."""
    max_sessions: int
    logger: Logger
    base_env: Mapping[str, str] | None = None
    """Environment for the allowlist, default `os.environ`."""
    link_wait_ms: int = DEFAULT_LINK_WAIT_MS
    """How long one call waits in total for the Remote Control link."""
    poll_ms: int = DEFAULT_POLL_MS
    launch_timeout_ms: int = DEFAULT_LAUNCH_TIMEOUT_MS
    read_live: Callable[[str], dict[str, LiveSession]] | None = None
    """Registry reader, for tests (default `read_live_sessions`)."""
    now: Callable[[], int] | None = None
    """Wall clock in ms, for tests (default `now_ms`); judges `startedAt` in the registry."""
    stop_wait_ms: int = DEFAULT_STOP_WAIT_MS
    public_url: str | None = None
    """The server's public URL; `claude.access` desktop denies it (required there)."""
    home_dir: str | None = None
    """Where `.claude.json` is looked for without `claude.configDir`, default `~` (for tests)."""
    answer_path: Callable[[int], str] | None = None
    """Where a session task's Stop hook writes (`TaskManager.session_answer_path`); None adds no hook."""
    handoff_to_desktop: bool = False
    """Open stopped sessions in the Claude desktop app (macOS only; the caller checks the platform)."""
    handoff_wait_ms: int = DEFAULT_HANDOFF_WAIT_MS
    open_url: Callable[[str], Awaitable[str | None]] | None = None
    """Opens a URL, None when it worked, else why not; for tests (default `/usr/bin/open <url>`)."""
    on_handoff: Callable[[int, str], None] | None = None
    """Called with (task id, session id) after a handoff opened the session in the desktop app
    (`TaskManager.handed_over`, which records it on the task)."""
    get_task: Callable[[int], Task | None] | None = None
    """The task's current state (`TaskManager.get`), read again once a call holds the lock; None
    uses the task as passed."""
    shutdown_wait_ms: int = DEFAULT_SHUTDOWN_WAIT_MS


def desktop_resume_url(session_id: str) -> str | None:
    """`claude://resume?session=<id>` for a UUID-shaped session id, else None."""
    return f"{DESKTOP_RESUME_URL}{session_id}" if is_session_id(session_id) else None


def stop_hook_settings(answer_path: str) -> dict[str, Any]:
    """`--settings` part with a Stop hook that saves its input (with `last_assistant_message`)
    atomically to `answer_path`, readable by the owner only."""
    target = shlex.quote(answer_path)
    temp = shlex.quote(f"{answer_path}.tmp")
    command = f"umask 077; cat > {temp} && mv -f {temp} {target}"
    return {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": command}]}]}}


def _session_args(
    claude: ClaudeConfig,
    *,
    name: str,
    access: RunAccess,
    answer_path: str | None,
    resume: str | None = None,
    system_prompt: str | None = None,
) -> list[str]:
    """The shared argv of `build_start_args` and `build_resume_args`: `--bg` (with `--resume`),
    Remote Control and session name, permission mode, the access args with the Stop hook merged
    into the one `--settings`, the system prompt when given, model and effort."""
    if answer_path is not None:
        access = replace(access, settings={**access.settings, **stop_hook_settings(answer_path)})
    args = [
        "--bg",
        *(["--resume", resume] if resume is not None else []),
        "--remote-control",
        name,
        "--name",
        name,
        "--permission-mode",
        claude.permission_mode,
        *access.args(),
    ]
    if system_prompt is not None:
        args += ["--append-system-prompt", system_prompt]
    if claude.model:
        args += ["--model", claude.model]
    if claude.effort:
        args += ["--effort", claude.effort]
    return args


def build_start_args(
    claude: ClaudeConfig, *, name: str, access: RunAccess, answer_path: str | None = None
) -> list[str]:
    """argv without the executable for a new session task: no prompt (it goes to stdin). The
    task's permission mode and access, one `--settings` with the Stop hook merged in, the session
    prompt, model and effort; no `-p` flags and no `extraArgs` (like `build_resume_args`)."""
    return _session_args(claude, name=name, access=access, answer_path=answer_path, system_prompt=REMOTE_SESSION_PROMPT)


def build_resume_args(
    claude: ClaudeConfig,
    *,
    session_id: str,
    name: str,
    mcp_config_path: str,
    access: RunAccess | None = None,
    answer_path: str | None = None,
) -> list[str]:
    """argv without the executable. `--name` keeps the session title (the desktop app shows it). The
    same permission mode and access (MCP config, setting sources, `--settings`; default isolated
    with `mcp_config_path`) as the task runs, plus model and effort; with `answer_path` the Stop hook
    merged into the one `--settings` (like `build_start_args`). Not `-p` flags (`--output-format`,
    `--permission-prompts`, `--max-turns`, the spoken-answer prompt) and not `extraArgs`, which are
    meant for `-p` runs: the owner now types in an interactive session and approves prompts there."""
    access = access if access is not None else isolated_access(claude, mcp_config_path)
    return _session_args(claude, name=name, access=access, answer_path=answer_path, resume=session_id)


@dataclass(frozen=True)
class _Launch:
    exit_code: int | None
    output: str
    error: str | None = None


class RemoteControl:
    """Opens Remote Control sessions for finished tasks. Calls are serialized, so two calls for the
    same task never start two sessions."""

    def __init__(self, options: RemoteControlOptions) -> None:
        self._opts = options
        self._read_live = options.read_live or read_live_sessions
        self._now = options.now or now_ms
        self._lock = asyncio.Lock()
        self._handoffs: set[asyncio.Task[None]] = set()

    def disconnected(self, live: LiveSession) -> bool:
        """The session's process lives, but the owner does not see it in the app (`rc_disconnected`)."""
        return rc_disconnected(live, self._now())

    def _counted(self, live: Mapping[str, LiveSession], opened: Mapping[str, Task]) -> list[Task]:
        """The tasks of the opened sessions that count toward the limit: live background sessions that
        are not disconnected. A session the owner continues in the desktop app or a terminal (an
        `interactive` entry with the same id, for example after the desktop handoff) is theirs, not
        the server's, and does not count."""
        tasks = {
            sid: opened[sid]
            for sid, entry in live.items()
            if sid in opened and entry.is_bg and not self.disconnected(entry)
        }
        return sorted(tasks.values(), key=lambda t: t.id)

    def _limit_failure(self, counted: Sequence[Task]) -> RcFailed:
        """Names the sessions as the app shows them, with their task numbers for cancel_task."""
        names = ", ".join(f"{session_name(t)} (task {t.id})" for t in counted)
        hint = f" To close one, ask to cancel its task, for example cancel task {counted[0].id}." if counted else ""
        return RcFailed(
            "limit",
            f"There {'is' if len(counted) == 1 else 'are'} already {plural(len(counted), 'session')} open in the Claude app: {names}.{hint}",
        )

    def _current(self, task: Task) -> Task:
        """The task as the manager has it now (`get_task`), else as passed."""
        get = self._opts.get_task
        return (get(task.id) if get is not None else None) or task

    def read_live(self) -> dict[str, LiveSession]:
        """The whole live registry, by session id (synchronous file reads: from async code use
        `_live`)."""
        return self._read_live(self._opts.claude_home)

    async def _live(self) -> dict[str, LiveSession]:
        """`read_live` off the event loop."""
        return await asyncio.to_thread(self.read_live)

    def live_sessions(self, opened: Mapping[str, object]) -> dict[str, LiveSession]:
        """The running sessions among `opened` (session ids this server opened)."""
        return {sid: live for sid, live in self.read_live().items() if sid in opened}

    async def open(self, task: Task, *, session_busy: bool, opened: Mapping[str, Task]) -> RcResult:
        """`session_busy`: another queued or running task still uses or may resume the task's
        session. `opened`: session ids this server opened for Remote Control, with their tasks. The
        task is read again under the lock (`get_task`): an idle stop may have ended it meanwhile."""
        async with self._lock:
            return await self._open(self._current(task), session_busy=session_busy, opened=opened)

    async def _open(self, task: Task, *, session_busy: bool, opened: Mapping[str, Task]) -> RcResult:
        opts = self._opts
        logger = opts.logger
        if task.kind == "send":
            target = task.target_session_name or "the target session"
            return RcFailed(
                "relay_task",
                f"Task {task.id} only relayed a message to {target}. Continue in {target} itself, it is listed in the Code section of the Claude app if it uses Remote Control.",
            )
        live = await self._live()
        if task.is_session_task:
            current = live.get(task.session_id or "")
            if current is None or not self.disconnected(current):
                # Already a Remote Control session: name it instead of starting another.
                return RcOpened(
                    name=session_name(task),
                    project=task.project,
                    link=link_for(current) if current else None,
                    already_open=True,
                    session_id=task.session_id,
                )
            # Its process lives but the app lost it: stopped and launched again below.
        elif not is_terminal(task.status):
            return RcFailed(
                "not_finished", f"Task {task.id} is still running. Wait for its result, then ask again to continue it."
            )
        # Task session ids are validated by the model (`SessionId`): only UUID-shaped ids get here.
        session_id = task.session_id
        if not session_id:
            return RcFailed("no_session", f"Task {task.id} has no Claude Code session to continue.")
        if session_busy:
            return RcFailed(
                "session_busy",
                f"Another task in {task.project} is still using this conversation. Wait for its result, then ask again.",
            )

        name = session_name(task)
        # The Remote Control session opened earlier holds the latest conversation from the phone.
        candidates = list(dict.fromkeys(sid for sid in (task.remote_session_id, session_id) if sid))
        foreign = self._open_elsewhere(
            task, live, [*candidates, *([task.desktop_session_id] if task.desktop_session_id else [])]
        )
        if foreign is not None:
            return foreign
        stale: list[str] = []
        for sid in candidates:
            current = live.get(sid)
            if current is None:
                continue
            if self.disconnected(current):
                stale.append(sid)
                continue
            logger.info("remote control already open", {"taskId": task.id, "pid": current.pid})
            return RcOpened(name=name, project=task.project, link=link_for(current), already_open=True, session_id=sid)

        if opts.max_sessions <= 0:
            return RcFailed("disabled", "Continuing tasks in the Claude app is turned off on this server.")
        counted = self._counted(live, {sid: t for sid, t in opened.items() if sid not in stale})
        if len(counted) >= opts.max_sessions:
            return self._limit_failure(counted)

        for sid in stale:
            failure = await self._stop_stale(task, sid)
            if failure is not None:
                return failure

        started = time.monotonic()
        resume_id = candidates[0]
        access = await self._access(task)
        launch = await self._launch(task, resume_id, name, access)
        if resume_id != session_id and RESUME_MISSING_TEXT in launch.output:
            # The earlier Remote Control transcript is gone: fall back to the task's own session.
            logger.warn("remote control session gone, resuming the task session", {"taskId": task.id})
            resume_id = session_id
            launch = await self._launch(task, resume_id, name, access)
        failure = self._launch_failure(
            task,
            launch,
            launched=True,
            refused_log="remote control refused: workspace not trusted",
            untrusted=f"Claude Code does not trust the folder {task.project} yet. Open Claude Code once in that project on the Mac and confirm the trust question, then try again.",
            failed_log="remote control launch failed",
            failed=f"Claude Code could not reopen task {task.id}",
        )
        if failure is not None:
            return failure

        id8 = launched_id8(launch.output) or resume_id[:8]
        found = await self._wait_for_link(id8, started)
        # Registry ids are UUID-shaped (read_live_sessions skips others).
        opened_id = found[0] if found else resume_id if resume_id.startswith(id8) else None
        logger.info(
            "remote control opened",
            {
                "taskId": task.id,
                "copy": not resume_id.startswith(id8),
                "pid": found[1].pid if found else None,
                "linkReady": bool(found and found[1].bridge_session_id),
            },
        )
        return RcOpened(
            name=name,
            project=task.project,
            link=link_for(found[1]) if found else None,
            already_open=False,
            session_id=opened_id,
            bg_id=id8,
            replaced_stale=bool(stale),
        )

    def _open_elsewhere(self, task: Task, live: Mapping[str, LiveSession], ids: Sequence[str]) -> RcResult | None:
        """A session of the task that runs in a process the server did not start (the desktop app
        after the handoff, or a terminal; any registry entry that is not `kind: "bg"`): never resumed
        a second time, two processes would write one transcript. With Remote Control switched on
        there it is named like an open session, else the owner has to switch it on there."""
        for sid in dict.fromkeys(ids):
            current = live.get(sid)
            if current is None or current.is_bg:
                continue
            name = session_name(task)
            desktop = task.handed_over_at is not None and current.kind == "interactive"
            self._opts.logger.info(
                "remote control: session open outside the server",
                {"taskId": task.id, "kind": current.kind, "bridge": bool(current.bridge_session_id)},
            )
            if current.bridge_session_id:
                return RcOpened(
                    name=name, project=task.project, link=link_for(current), already_open=True, session_id=sid
                )
            where = (
                "in the Claude desktop app on the Mac"
                if desktop
                else "in Claude Code on the Mac (the desktop app or a terminal)"
            )
            return RcFailed(
                "open_elsewhere",
                f"Task {task.id}'s session {name} is open {where}, so it is not opened a second time. To continue it on the phone, switch on Remote Control for that session there; it then shows in the Code section of the Claude app.",
            )
        return None

    async def _stop_stale(self, task: Task, session_id: str) -> RcFailed | None:
        """Stops a session of the task that lives but lost Remote Control, and waits for it to go.
        When it is still listed after `stop_wait_ms`, fails with a retry hint: launching then would
        only start a copy next to it (the CLI copies a session that is still running)."""
        failure = await self._stop_id(task, session_id[:8], reason="disconnected")
        if failure is not None:
            return RcFailed(
                "stop_failed",
                f"Session {session_name(task)} of task {task.id} lost its connection to the Claude app, and it could not be closed to open it again: {failure}",
            )
        if not await self._wait_gone(session_id, self._opts.stop_wait_ms):
            self._opts.logger.warn("stale session still running after stop", {"taskId": task.id})
            return RcFailed(
                "still_stopping",
                f"Session {session_name(task)} of task {task.id} lost its connection to the Claude app and is being closed, but it has not ended yet. Ask again in a moment to open it again.",
            )
        return None

    async def _wait_gone(self, session_id: str, wait_ms: int) -> bool:
        """Polls the registry until it no longer lists `session_id`; False when it still does after `wait_ms`."""
        deadline = time.monotonic() + wait_ms / 1000
        while session_id in await self._live():
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self._opts.poll_ms / 1000)
        return True

    def _launch_failure(
        self,
        task: Task,
        launch: _Launch,
        *,
        launched: bool,
        refused_log: str,
        untrusted: str,
        failed_log: str,
        failed: str,
    ) -> RcFailed | None:
        """Why a `--bg` launch did not start a session (logged), else None: the folder is not
        trusted (`untrusted` is the message), or it failed (`failed` plus the clipped detail).
        `launched` False (no id8 printed) counts as failed too."""
        logger = self._opts.logger
        if UNTRUSTED_TEXT in launch.output:
            logger.warn(refused_log, {"taskId": task.id, "project": task.project})
            return RcFailed("untrusted", untrusted)
        if launch.error is not None or launch.exit_code != 0 or not launched:
            detail = launch.error or clip_words(launch.output, 200) or f"exit code {launch.exit_code}"
            logger.warn(failed_log, {"taskId": task.id, "exitCode": launch.exit_code})
            return RcFailed("launch_failed", f"{failed}: {detail}")
        return None

    async def _access(self, task: Task) -> RunAccess | _Launch:
        opts = self._opts
        try:
            return await asyncio.to_thread(
                task_access,
                opts.claude,
                project_path=task.project_path,
                public_url=opts.public_url,
                mcp_config_path=opts.mcp_config_path,
                home_dir=opts.home_dir,
            )
        except ValueError as err:
            return _Launch(exit_code=None, output="", error=str(err))

    async def _forget(self, task: Task, session_id: str) -> None:
        """`claude rm <id8>` for a stopped background session with this id, so the resume continues
        it under its own id: the CLI starts a copy of any session still in its background list, also
        a stopped one (verified with 2.1.331), and every copy would get one more desktop record after
        the handoff. `rm` keeps the conversation and refuses when that is not safe (a worktree with
        unpushed work); the resume then makes a copy as before. Never for a live session."""
        cwd = task.project_path
        listing = await self._run_cli(["agents", "--json", "--all"], cwd=cwd, isolated=True, limit=AGENTS_OUTPUT_LIMIT)
        if listing.error is not None or listing.exit_code != 0:
            return
        try:
            data: Any = loads_json(listing.output, containers_only=True)
        except ValueError:
            return
        entries = data if isinstance(data, list) else []
        ids = [
            valid_bg_id(e.get("id"))
            for e in entries
            if isinstance(e, dict) and e.get("sessionId") == session_id and e.get("kind") == "background"
        ]
        if len(ids) != 1 or ids[0] is None or session_id in await self._live():
            return
        result = await self._run_cli(["rm", ids[0]], cwd=cwd, isolated=True)
        self._opts.logger.info(
            "background session removed before resume",
            {"taskId": task.id, "ok": result.error is None and result.exit_code == 0, "exitCode": result.exit_code},
        )

    async def _launch(self, task: Task, session_id: str, name: str, access: RunAccess | _Launch) -> _Launch:
        """`claude rm` (`_forget`) and the `--bg --resume` launcher for `session_id`; `access` from
        `_access` (a `_Launch` there is its failure, returned as it is)."""
        opts = self._opts
        if isinstance(access, _Launch):
            return access
        await self._forget(task, session_id)
        args = build_resume_args(
            opts.claude,
            session_id=session_id,
            name=name,
            mcp_config_path=opts.mcp_config_path,
            access=access,
            answer_path=opts.answer_path(task.id) if opts.answer_path is not None else None,
        )
        return await self._run_cli(args, cwd=task.project_path, isolated=access.isolated)

    async def _run_cli(
        self, args: Sequence[str], *, cwd: str, isolated: bool, stdin: bytes | None = None, limit: int = OUTPUT_LIMIT
    ) -> _Launch:
        """One short `claude` CLI call (a `--bg` launcher or `claude stop`): output to a temp file,
        `stdin` written and closed (else /dev/null), no shell."""
        opts = self._opts
        env = build_env(opts.claude, opts.base_env if opts.base_env is not None else os.environ, isolated=isolated)
        for key in NO_RC_ENV:  # also when claude.env sets it: Remote Control cannot work with it
            env.pop(key, None)
        with tempfile.TemporaryFile() as out:
            try:
                proc = await asyncio.create_subprocess_exec(
                    opts.claude.bin,
                    *args,
                    cwd=cwd,
                    env=env,
                    stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                    stdout=out,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as err:
                return _Launch(exit_code=None, output="", error=f"spawn failed: {err}")
            try:
                # communicate() writes stdin (a vanished reader is ignored) and closes it.
                await asyncio.wait_for(proc.communicate(stdin), opts.launch_timeout_ms / 1000)
                code: int | None = proc.returncode
                error = None
            except TimeoutError:
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), 2)
                code, error = None, "the launcher did not finish"
            out.seek(0)
            output = out.read(limit).decode("utf-8", "replace")
        return _Launch(exit_code=code, output=output, error=error)

    async def start_session(
        self, create: Callable[[], StartResult], *, opened: Mapping[str, Task]
    ) -> SessionStartResult:
        """run_task with remote_control: checks the session limit, then calls `create` (which makes
        the task) and starts the task's prompt as a new Remote Control session. `opened`: the
        sessions this server opened (resumed or started), with their names; they share
        `max_sessions`. Serialized with `open`, so the limit holds."""
        async with self._lock:
            opts = self._opts
            if opts.max_sessions <= 0:
                return RcFailed("disabled", "Sessions for the Claude app are turned off on this server.")
            counted = self._counted(await self._live(), opened)
            if len(counted) >= opts.max_sessions:
                return self._limit_failure(counted)
            created = create()
            if isinstance(created, StartFailed):
                return created
            task = created.task
            return RcStarted(task=task, outcome=await self._start(task))

    async def _start(self, task: Task) -> RcResult:
        opts = self._opts
        logger = opts.logger
        name = session_name(task)
        started = time.monotonic()
        access = await self._access(task)
        if isinstance(access, _Launch):
            launch = access
        else:
            answer_path = opts.answer_path(task.id) if opts.answer_path is not None else None
            args = build_start_args(opts.claude, name=name, access=access, answer_path=answer_path)
            launch = await self._run_cli(
                args, cwd=task.project_path, isolated=access.isolated, stdin=task.prompt.encode("utf-8")
            )
        id8 = launched_id8(launch.output)
        failure = self._launch_failure(
            task,
            launch,
            launched=id8 is not None,
            refused_log="session start refused: workspace not trusted",
            untrusted=f"Claude Code does not trust the folder {task.project} yet, so it cannot start a session there. Run claude once in that folder on the Mac and confirm the trust question, then try again, or run the task without the Claude app.",
            failed_log="session start failed",
            failed=f"Claude Code could not start task {task.id} as a session",
        )
        if failure is not None:
            return failure
        assert id8 is not None  # without an id8 `_launch_failure` failed
        found = await self._wait_for_link(id8, started)
        logger.info(
            "session started",
            {
                "taskId": task.id,
                "pid": found[1].pid if found else None,
                "linkReady": bool(found and found[1].bridge_session_id),
            },
        )
        return RcOpened(
            name=name,
            project=task.project,
            link=link_for(found[1]) if found else None,
            already_open=False,
            session_id=found[0] if found else None,
            bg_id=id8,
        )

    async def close(
        self, task: Task, *, reason: CloseReason = "cancel", only_if: Callable[[str, LiveSession], bool] | None = None
    ) -> Literal["nothing", "closed"] | str:
        """cancel_task (or the idle close): stops the task's live Remote Control session
        (`claude stop <id8>`, the conversation is kept), also when the task itself finished, and
        hands it over to the desktop app when that is on. Under the lock, with the task read again
        (`get_task`), so it never interleaves with a reopen. The session is `live_bg_session`; a
        running session task whose session the registry does not list (yet) is stopped by its
        `bg_id` (else its session id's first 8 characters). The caller then ends a session task
        (`TaskManager.session_closed`).

        "closed", "nothing" when no session is live, `SKIPPED` when `only_if` (the idle close,
        checked under the lock with the live session's id and registry entry) rejects it or there
        is no live session to check, else why not (plain words)."""
        async with self._lock:
            task = self._current(task)
            found = live_bg_session(task, await self._live())
            if only_if is not None and (found is None or not only_if(*found)):
                return SKIPPED
            session_id: str | None
            if found is not None:
                bg_id, session_id = found[0][:8], found[0]
            elif task.is_session_task:
                fallback = task.bg_id or (task.session_id[:8] if task.session_id else None)
                if not fallback:
                    return "its session id is not known"
                bg_id, session_id = fallback, task.session_id
            else:
                return "nothing"
            failure = await self._stop_id(task, bg_id, reason=reason)
            if failure is not None:
                return failure
            if session_id:
                self._schedule_handoff(task, session_id)
            return "closed"

    async def _stop_id(self, task: Task, bg_id: str, *, reason: str) -> str | None:
        result = await self._run_cli(["stop", bg_id], cwd=task.project_path, isolated=True)
        ok = result.error is None and result.exit_code == 0
        self._opts.logger.info(
            "session stop", {"taskId": task.id, "reason": reason, "ok": ok, "exitCode": result.exit_code}
        )
        if ok:
            return None
        return result.error or clip_words(result.output, 200) or f"exit code {result.exit_code}"

    def live_session_of(self, task: Task) -> str | None:
        """The id of the task's live background session opened for Remote Control (connected or
        not), see `live_bg_session` (synchronous file reads)."""
        found = live_bg_session(task, self.read_live())
        return found[0] if found else None

    def _schedule_handoff(self, task: Task, session_id: str) -> None:
        """Starts the handoff of a stopped session in the background (the tool call does not wait)."""
        if not self._opts.handoff_to_desktop:
            return
        job = asyncio.get_running_loop().create_task(self._handoff(task.id, session_id))
        self._handoffs.add(job)
        job.add_done_callback(self._handoffs.discard)

    async def _handoff(self, task_id: int, session_id: str) -> None:
        """Waits until the session left the registry, then opens it in the desktop app."""
        opts = self._opts
        logger = opts.logger
        url = desktop_resume_url(session_id)
        if url is None:
            logger.warn("desktop handoff skipped: invalid session id", {"taskId": task_id})
            return
        if not await self._wait_gone(session_id, opts.handoff_wait_ms):
            logger.warn("desktop handoff skipped: session still running", {"taskId": task_id})
            return
        try:
            failure = await (opts.open_url or _open_url)(url)
        except Exception as err:
            failure = str(err)
        if failure is None:
            logger.info("desktop handoff", {"taskId": task_id, "ok": True})
            if opts.on_handoff is not None:
                opts.on_handoff(task_id, session_id)
        else:
            logger.warn("desktop handoff", {"taskId": task_id, "ok": False, "error": clip_words(failure, 200)})

    async def wait_handoffs(self) -> None:
        """Waits for the pending handoffs (for tests and shutdown)."""
        while self._handoffs:
            await asyncio.gather(*list(self._handoffs), return_exceptions=True)

    async def aclose(self) -> None:
        """Server shutdown: lets pending handoffs finish for at most `shutdown_wait_ms` (a session
        stopped just before usually leaves the registry within a second or two), then cancels the rest."""
        jobs = list(self._handoffs)
        if not jobs:
            return
        _, pending = await asyncio.wait(jobs, timeout=self._opts.shutdown_wait_ms / 1000)
        for job in pending:
            job.cancel()
        if pending:
            self._opts.logger.warn("desktop handoff cancelled at shutdown", {"count": len(pending)})
        await asyncio.gather(*pending, return_exceptions=True)

    async def _wait_for_link(self, id8: str, started: float) -> tuple[str, LiveSession] | None:
        """(session id, registry entry) of the launched session, whose id starts with `id8`, once it
        has a Remote Control link or at the deadline."""
        opts = self._opts
        deadline = started + opts.link_wait_ms / 1000
        while True:
            live = await self._live()
            matches: Sequence[tuple[str, LiveSession]] = [(sid, e) for sid, e in live.items() if sid.startswith(id8)]
            found = matches[0] if len(matches) == 1 else None
            if found is not None and found[1].bridge_session_id:
                return found
            left = deadline - time.monotonic()
            if left <= 0:
                return found
            await asyncio.sleep(min(opts.poll_ms / 1000, left))


async def _open_url(url: str) -> str | None:
    """`/usr/bin/open <url>` (argv, no shell, output discarded); None when it exited 0."""
    try:
        proc = await asyncio.create_subprocess_exec(
            OPEN_BIN,
            url,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as err:
        return f"spawn failed: {err}"
    try:
        code = await asyncio.wait_for(proc.wait(), 10)
    except TimeoutError:
        with contextlib.suppress(OSError):
            proc.kill()
        return "open did not finish"
    return None if code == 0 else f"open exited with code {code}"


def format_session_started(task: Task, opened: RcOpened) -> str:
    """run_task with remote_control: the name to look for first, the link last (not to be read aloud)."""
    head = f"Task {task.id} in {task.project} started as session {opened.name}."
    where = f" In the Claude app, open Code and pick {opened.name} to watch, approve or continue it. Ask for task {task.id} for its latest answer."
    if opened.link:
        return f"{head}{where} Link (do not read it aloud): {opened.link}"
    return f"{head}{where} The session shows up in the app under that name in a moment."


def format_opened(task_id: int, opened: RcOpened) -> str:
    """Tool response: the name to look for first, the link last (not to be read aloud)."""
    if opened.already_open:
        head = f"Task {task_id} is already open as session {opened.name} in {opened.project}."
    elif opened.replaced_stale:
        head = f"The earlier session of task {task_id} had lost its connection to the Claude app, so it was closed. Task {task_id} is now open again as session {opened.name} in {opened.project}."
    else:
        head = f"Task {task_id} is now open as session {opened.name} in {opened.project}."
    where = f" In the Claude app, open Code and pick {opened.name} to continue by typing."
    if opened.link:
        return f"{head}{where} Link (do not read it aloud): {opened.link}"
    return f"{head}{where} The link is not ready yet; the session shows up in the app under that name in a moment."
