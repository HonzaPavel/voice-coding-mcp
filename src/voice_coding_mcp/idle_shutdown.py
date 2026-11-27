"""Idle shutdown (`idleShutdownMinutes`, `serve --idle-shutdown-minutes`): the server stops itself
after a period without use, so a forgotten public URL does not stay reachable.

Activity is only an authenticated MCP `tools/call` (`Activity.call`, wrapped around the request
behind the bearer check). `initialize`, `tools/list`, unauthenticated requests and the OAuth
endpoints never count, so scanners probing the public URL keep nothing alive. The idle time starts
when the server starts.

The server never stops while work is in progress: a tools/call still being handled, or a headless
task in the task manager (queued, running, or a relay waiting for the session's reply; ask and
relay runs are tasks too). Each check that sees such work moves the last activity to now, so the
idle time counts from when the work ended. Remote Control session tasks have no run slot and run
on without the server, so they do not keep it up.

When the idle time is reached the check logs `idle shutdown` and calls `on_idle` once; the server
then stops the same way as on SIGTERM.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterator

from .clock import MINUTE_MS, Clock, elapsed_minutes, now_ms
from .log import Logger

CHECK_INTERVAL_MS = MINUTE_MS
"""How often the idle shutdown checks for activity."""


class Activity:
    """The last authenticated tools/call and the number still being handled."""

    def __init__(self, now: Clock | None = None) -> None:
        self._now = now or now_ms
        self.last_ms = self._now()
        self.in_flight = 0

    def touch(self, at_ms: int | None = None) -> None:
        self.last_ms = self._now() if at_ms is None else at_ms

    @contextlib.contextmanager
    def call(self) -> Iterator[None]:
        """Wraps the handling of one tools/call: activity at its start and at its end."""
        self.touch()
        self.in_flight += 1
        try:
            yield
        finally:
            self.in_flight -= 1
            self.touch()


class IdleShutdown:
    """Call `check` directly (tests) or `start` the periodic loop and `stop` it at shutdown."""

    def __init__(
        self,
        activity: Activity,
        *,
        busy: Callable[[], bool],
        idle_minutes: int,
        on_idle: Callable[[], None],
        logger: Logger,
        now: Clock | None = None,
        interval_ms: int = CHECK_INTERVAL_MS,
    ) -> None:
        self._activity = activity
        self._busy = busy
        self._idle_ms = idle_minutes * MINUTE_MS
        self._on_idle = on_idle
        self._logger = logger
        self._now = now or now_ms
        self._interval_ms = interval_ms
        self._fired = False
        self._loop: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._idle_ms > 0

    def start(self) -> None:
        if self.enabled and self._loop is None:
            self._loop = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        loop, self._loop = self._loop, None
        if loop is not None and loop is not asyncio.current_task():
            loop.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop

    async def _run(self) -> None:
        while not self._fired:
            await asyncio.sleep(self._interval_ms / 1000)
            try:
                self.check()
            except Exception as err:
                self._logger.error("idle shutdown check failed", {"error": err})

    def check(self) -> bool:
        """One pass: True (and `on_idle` called, once) when the server has been idle long enough."""
        if not self.enabled or self._fired:
            return False
        now = self._now()
        if self._activity.in_flight > 0 or self._busy():
            self._activity.touch(now)
            return False
        idle_for = now - self._activity.last_ms
        if idle_for < self._idle_ms:
            return False
        self._fired = True
        self._logger.info("idle shutdown", {"idleMinutes": elapsed_minutes(idle_for)})
        self._on_idle()
        return True
