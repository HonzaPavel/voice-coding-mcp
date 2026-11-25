"""Pairing window: one-time code that opens /register and /authorize.

State file `stateDir/pairing.json` with the keys
`{"codeHash", "createdAt", "expiresAt", "failures"}` (times in epoch ms). Wrong codes per client
IP (IPv6 by /64, see `web.client_key`) are kept in memory, keyed by the window's code hash;
verifications are serialized with a module-level `asyncio.Lock`. Every read-modify-write of
pairing.json (open, verify, expiry and removal) also holds an exclusive `fcntl.flock` on
`stateDir/pairing.lock`, shared with other processes such as the CLI.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import secrets
import weakref
from dataclasses import dataclass
from typing import Literal

from ..clock import MINUTE_MS, now_ms
from ..json_file import StateModel, file_lock, read_json, write_json_atomic
from ..web import client_key as _client_key
from .crypto import safe_equal, sha256_hex

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
"""Crockford base32 alphabet (no I, L, O, U)."""
CODE_LENGTH = 10
PAIRING_ATTEMPTS_PER_CLIENT = 5
"""Wrong codes one client (IP) may send per window before it is locked out for the rest of the window."""
PAIRING_MAX_FAILURES = 50
"""Wrong codes from all clients together per window before the window closes."""
PAIRING_LOCK_FILE = "pairing.lock"
"""Lock file next to pairing.json; every process that reads and changes pairing.json holds an exclusive flock on it."""


class PairingFile(StateModel):
    """`pairing.json`. Readers tolerate missing `createdAt`/`failures` (0) and treat a file without
    a string `codeHash` or numeric `expiresAt` as no window."""

    code_hash: str
    created_at: int
    expires_at: int
    failures: int


@dataclass(frozen=True)
class PairingWindow:
    expires_at: int
    attempts_left: int
    """Wrong codes left in this window, for all clients together."""


@dataclass(frozen=True)
class OpenedPairing:
    code: str
    """Formatted `XXXXX-XXXXX`, shown once by the CLI."""
    expires_at: int


PairingStatus = Literal["ok", "invalid", "locked", "exhausted", "closed"]


@dataclass(frozen=True)
class VerifyPairingResult:
    """ok; invalid (wrong code, window stays open, `attempts_left` is for this client); locked (this
    client sent too many wrong codes); exhausted (global cap reached, window now closed); closed
    (no window, or it expired)."""

    status: PairingStatus
    attempts_left: int | None = None


_SPACE_OR_DASH = re.compile(r"[\s-]+")
_LOOK_ALIKE_ONE = re.compile("[IL]")


def pairing_path(state_dir: str) -> str:
    return os.path.join(state_dir, "pairing.json")


def _file_lock(state_dir: str) -> contextlib.AbstractContextManager[None]:
    """Exclusive flock on `stateDir/pairing.lock` (mode 0600), held for one small read-modify-write.
    Creates `state_dir` (0700) when needed."""
    return file_lock(os.path.join(state_dir, PAIRING_LOCK_FILE))


def format_pairing_code(code: str) -> str:
    """Formats a 10 character code as `XXXXX-XXXXX`."""
    return f"{code[:5]}-{code[5:]}"


def normalize_pairing_code(text: str) -> str:
    """Uppercases, drops dashes and whitespace and maps Crockford look-alikes (I, L -> 1, O -> 0)."""
    return _LOOK_ALIKE_ONE.sub("1", _SPACE_OR_DASH.sub("", text.upper())).replace("O", "0")


def _generate_code() -> str:
    # 32 divides 256, so masking each byte to 5 bits is uniform.
    return "".join(ALPHABET[byte & 31] for byte in secrets.token_bytes(CODE_LENGTH))


def _remove_file(state_dir: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(pairing_path(state_dir))


def _number(value: object) -> int | float | None:
    """`value` when it is a JSON number (int or float, not bool), else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value


def _read_pairing(state_dir: str) -> PairingFile | None:
    data = read_json(pairing_path(state_dir), None)
    if not isinstance(data, dict):
        return None
    code_hash = data.get("codeHash")
    expires_at = _number(data.get("expiresAt"))
    if not isinstance(code_hash, str) or expires_at is None:
        return None
    created_at = _number(data.get("createdAt"))
    failures = _number(data.get("failures"))
    return PairingFile(
        code_hash=code_hash,
        created_at=int(created_at) if created_at is not None else 0,
        expires_at=int(expires_at),
        failures=int(failures) if failures is not None and failures >= 0 else 0,
    )


def _is_open(file: PairingFile, now: int) -> bool:
    return file.expires_at > now and file.failures < PAIRING_MAX_FAILURES


@dataclass
class _ClientFailures:
    code_hash: str
    by_client: dict[str, int]


_client_failures: dict[str, _ClientFailures] = {}
"""Wrong codes per client, in memory, for the current window of each state dir. Keyed by the
window's code hash, so a new window starts with a clean slate. At most PAIRING_MAX_FAILURES entries
per window, because every entry costs at least one global failure."""


def _failures_for(state_dir: str, code_hash: str) -> dict[str, int]:
    entry = _client_failures.get(state_dir)
    if entry is None or entry.code_hash != code_hash:
        entry = _ClientFailures(code_hash=code_hash, by_client={})
        _client_failures[state_dir] = entry
    return entry.by_client


_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()


def _lock() -> asyncio.Lock:
    """Serializes verifications in this process so concurrent requests cannot share one attempt. One
    lock per event loop (an asyncio.Lock must not be shared between loops, e.g. across tests)."""
    loop = asyncio.get_running_loop()
    lock = _locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _locks[loop] = lock
    return lock


async def open_pairing(state_dir: str, minutes: float, now: int | None = None) -> OpenedPairing:
    """Writes the hash of a fresh code to pairing.json (replacing any earlier window) and returns the
    plain code. Ensures `state_dir` exists (0700)."""
    t = now_ms() if now is None else now
    code = _generate_code()
    file = PairingFile(code_hash=sha256_hex(code), created_at=t, expires_at=t + round(minutes * MINUTE_MS), failures=0)
    with _file_lock(state_dir):
        write_json_atomic(pairing_path(state_dir), file.to_json())
    return OpenedPairing(code=format_pairing_code(code), expires_at=file.expires_at)


async def get_pairing(state_dir: str, now: int | None = None) -> PairingWindow | None:
    """The open pairing window, or None. An expired or exhausted window is deleted."""
    return _get_pairing(state_dir, now_ms() if now is None else now)


def _get_pairing(state_dir: str, t: int) -> PairingWindow | None:
    # Without a pairing file there is nothing to lock (and no reason to create the lock file).
    if not os.path.exists(pairing_path(state_dir)):
        return None
    with _file_lock(state_dir):
        file = _read_pairing(state_dir)
        if file is None:
            return None
        if not _is_open(file, t):
            _remove_file(state_dir)
            return None
    return PairingWindow(expires_at=file.expires_at, attempts_left=PAIRING_MAX_FAILURES - file.failures)


async def close_pairing(state_dir: str) -> None:
    """Closes the pairing window (`unpair`); no error when none is open."""
    _close_pairing(state_dir)


def _close_pairing(state_dir: str) -> None:
    if not os.path.exists(pairing_path(state_dir)):
        return
    with _file_lock(state_dir):
        _remove_file(state_dir)


async def verify_pairing_code(
    state_dir: str,
    text: str,
    now: int | None = None,
    client_key: str = "unknown",
) -> VerifyPairingResult:
    """Checks a code typed by the owner. `client_key` identifies the sender (its IP; IPv6 counts by
    its /64, IPv4-mapped IPv6 as the IPv4 address). Success deletes
    the window. A wrong code counts against the client (in memory) and against the window (in the
    file): after PAIRING_ATTEMPTS_PER_CLIENT wrong codes the client is locked out for the rest of the
    window, even with the right code; after PAIRING_MAX_FAILURES wrong codes from anyone the window
    closes. So a watcher who sees the window open cannot close it with a handful of wrong codes."""
    t = now_ms() if now is None else now
    key = _client_key(client_key)
    async with _lock():
        with _file_lock(state_dir):
            return _verify_locked(state_dir, text, t, key)


def _verify_locked(state_dir: str, text: str, t: int, client_key: str) -> VerifyPairingResult:
    """`verify_pairing_code` under both locks."""
    file = _read_pairing(state_dir)
    if file is None:
        return VerifyPairingResult(status="closed")
    if not _is_open(file, t):
        _remove_file(state_dir)
        return VerifyPairingResult(status="closed")
    by_client = _failures_for(state_dir, file.code_hash)
    client_fails = by_client.get(client_key, 0)
    if client_fails >= PAIRING_ATTEMPTS_PER_CLIENT:
        return VerifyPairingResult(status="locked")
    normalized = normalize_pairing_code(text)[:64]
    if safe_equal(sha256_hex(normalized), file.code_hash):
        _remove_file(state_dir)
        _client_failures.pop(state_dir, None)
        return VerifyPairingResult(status="ok")
    by_client[client_key] = client_fails + 1
    failures = file.failures + 1
    if failures >= PAIRING_MAX_FAILURES:
        _remove_file(state_dir)
        _client_failures.pop(state_dir, None)
        return VerifyPairingResult(status="exhausted")
    write_json_atomic(pairing_path(state_dir), file.model_copy(update={"failures": failures}).to_json())
    attempts_left = PAIRING_ATTEMPTS_PER_CLIENT - (client_fails + 1)
    if attempts_left <= 0:
        return VerifyPairingResult(status="locked")
    return VerifyPairingResult(status="invalid", attempts_left=attempts_left)
