"""Idle close (`remoteControl.idleCloseMinutes`): stops the server's own Remote Control sessions
that have sat idle, or disconnected from the Claude app, for too long.

The sessions checked are the live background sessions of this server's tasks (`remoteSessionId`
of a session task or of a task reopened with resume_task_in_remote_control). A session counts as
idle while its registry `status` is "idle" (its turn is done, it waits for nothing); the idle time
starts at `statusUpdatedAt`, or, without it, when this check first saw it idle. A session that is
alive but disconnected from the app (`rc_disconnected`) is timed from when the check first saw it
so. Sessions waiting for an approval or an answer, or working, are never closed.

The session is stopped with `RemoteControl.close`, which takes the RemoteControl lock (so a reopen
never interleaves) and checks again under it, with `only_if`, that the task's live session is still
the one seen and still idle (or disconnected) long enough; otherwise nothing is stopped. It hands
the session over to the desktop app when `remoteControl.handoffToDesktop` is on. A session task
then ends as "done" (`TaskManager.session_closed`). The state is in memory only: after a restart a
session without `statusUpdatedAt` starts its idle time again.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from .clock import MINUTE_MS, Clock, elapsed_minutes, now_ms
from .log import Logger
from .remote_control import SKIPPED, CloseReason
from .sessions import LiveSession
from .tasks.manager import Task, TaskManager, live_bg_session, session_state

CHECK_INTERVAL_MS = MINUTE_MS
"""How often the idle close looks at the live registry."""

IdleKind = Literal["idle", "disconnected"]


@dataclass
class _Seen:
    kind: IdleKind
    since_ms: int
    """When the check first saw the session in this state."""


class SessionStopper(Protocol):
    """What the idle close needs of `RemoteControl` (a stub in tests)."""

    def read_live(self) -> dict[str, LiveSession]: ...

    async def close(
        self, task: Task, *, reason: CloseReason = ..., only_if: Callable[[str, LiveSession], bool] | None = ...
    ) -> str: ...


class IdleCloser:
    """Call `check` directly (tests) or `start` the periodic loop and `stop` it at shutdown."""

    def __init__(
        self,
        manager: TaskManager,
        remote: SessionStopper,
        *,
        idle_minutes: int,
        logger: Logger,
        now: Clock | None = None,
        interval_ms: int = CHECK_INTERVAL_MS,
    ) -> None:
        self._manager = manager
        self._remote = remote
        self._idle_ms = idle_minutes * MINUTE_MS
        self._logger = logger
        self._now = now or now_ms
        self._interval_ms = interval_ms
        self._seen: dict[str, _Seen] = {}
        self._loop: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._idle_ms > 0

    def start(self) -> None:
        if self.enabled and self._loop is None:
            self._loop = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        loop, self._loop = self._loop, None
        if loop is not None:
            loop.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval_ms / 1000)
            try:
                await self.check()
            except Exception as err:
                self._logger.error("idle close check failed", {"error": err})

    async def check(self) -> list[int]:
        """One pass: stops the sessions idle (or disconnected) for at least `idle_minutes`.
        Returns the ids of the tasks whose session was stopped."""
        if not self.enabled:
            return []
        live = await asyncio.to_thread(self._remote.read_live)
        self._manager.refresh_remote(live)
        now = self._now()
        closed: list[int] = []
        tracked: set[str] = set()
        for task in self._manager.remote_session_tasks().values():
            found = live_bg_session(task, live)
            if found is None:
                continue
            sid, entry = found
            kind = self._kind(entry)
            if kind is None:
                continue
            tracked.add(sid)
            seen = self._seen.get(sid)
            if seen is None or seen.kind != kind:
                seen = self._seen[sid] = _Seen(kind=kind, since_ms=now)
            idle_for = self._idle_for(sid, entry, now)
            if idle_for is None or idle_for < self._idle_ms:
                continue
            fields = {"taskId": task.id, "state": kind, "idleMinutes": elapsed_minutes(idle_for)}

            def still_due(current: str, live_entry: LiveSession, *, seen_sid: str = sid) -> bool:
                """Under the RemoteControl lock: the same session, still idle (or disconnected) long enough."""
                if current != seen_sid:
                    return False
                left = self._idle_for(current, live_entry, self._now())
                return left is not None and left >= self._idle_ms

            outcome = await self._remote.close(task, reason="idle", only_if=still_due)
            if outcome == SKIPPED:
                self._logger.info("idle close skipped: session changed", fields)
                continue
            if outcome == "closed":
                self._manager.session_closed(task.id, "idle")
                self._logger.info("idle session closed", fields)
                closed.append(task.id)
                tracked.discard(sid)
            else:
                self._logger.warn("idle session close failed", fields)
        for sid in list(self._seen):
            if sid not in tracked:
                del self._seen[sid]
        return closed

    def _kind(self, entry: LiveSession) -> IdleKind | None:
        """ "idle" or "disconnected" by the manager's `session_state`, else None (also for a
        session the server did not start)."""
        if not entry.is_bg:
            return None
        state = session_state(entry, self._now())
        return "idle" if state == "idle" else "disconnected" if state == "disconnected" else None

    def _idle_for(self, sid: str, entry: LiveSession, now: int) -> int | None:
        """How long the session has been in its idle (or disconnected) state; None when it is in
        neither or the check has not seen it so (no start time to judge by)."""
        kind = self._kind(entry)
        seen = self._seen.get(sid)
        if kind is None or seen is None or seen.kind != kind:
            return None
        if kind == "idle" and entry.status_updated_at_ms is not None:
            return now - entry.status_updated_at_ms
        return now - seen.since_ms
