"""Tests of the pairing window."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import threading
from pathlib import Path

import pytest

from tests.auth.helpers import exists, read_text
from tests.helpers import trace_lock_io
from voice_coding_mcp import json_file
from voice_coding_mcp.auth import (
    PairingWindow,
    VerifyPairingResult,
    close_pairing,
    get_pairing,
    normalize_pairing_code,
    open_pairing,
    pairing_path,
    verify_pairing_code,
)
from voice_coding_mcp.auth import pairing as pairing_module
from voice_coding_mcp.auth.crypto import sha256_hex

T0 = 1_800_000_000_000
OK = VerifyPairingResult("ok")
CLOSED = VerifyPairingResult("closed")
LOCKED = VerifyPairingResult("locked")


@pytest.fixture
def state_dir(real_tmp: Path) -> str:
    return str(real_tmp)


async def test_opens_a_window_with_a_formatted_code_and_stores_only_its_hash(state_dir: str) -> None:
    opened = await open_pairing(state_dir, 10, T0)
    assert re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{5}-[0-9A-HJKMNP-TV-Z]{5}", opened.code)
    assert opened.expires_at == T0 + 600_000
    text = read_text(pairing_path(state_dir))
    assert opened.code.replace("-", "") not in text
    assert opened.code not in text
    assert list(json.loads(text)) == ["codeHash", "createdAt", "expiresAt", "failures"]
    assert await get_pairing(state_dir, T0) == PairingWindow(expires_at=opened.expires_at, attempts_left=50)


async def test_accepts_the_code_in_any_case_with_dashes_and_spaces(state_dir: str) -> None:
    opened = await open_pairing(state_dir, 10, T0)
    typed = f" {opened.code.lower().replace('-', ' - ')} "
    assert await verify_pairing_code(state_dir, typed, T0 + 1000) == OK
    assert not exists(pairing_path(state_dir))
    assert await verify_pairing_code(state_dir, opened.code, T0 + 1000) == CLOSED


def test_normalizes_crockford_look_alikes() -> None:
    assert normalize_pairing_code("ab-cd eio l") == "ABCDE101"


async def test_locks_out_one_client_after_five_wrong_codes(state_dir: str) -> None:
    opened = await open_pairing(state_dir, 10, T0)
    for left in range(4, 0, -1):
        assert await verify_pairing_code(state_dir, "WRONG-CODE0", T0, "attacker") == VerifyPairingResult(
            "invalid", left
        )
    assert await verify_pairing_code(state_dir, "WRONG-CODE0", T0, "attacker") == LOCKED
    # Locked for the rest of the window, even with the right code.
    assert await verify_pairing_code(state_dir, opened.code, T0, "attacker") == LOCKED
    assert exists(pairing_path(state_dir))
    assert await get_pairing(state_dir, T0) == PairingWindow(expires_at=T0 + 600_000, attempts_left=45)
    assert await verify_pairing_code(state_dir, opened.code, T0, "owner") == OK
    assert not exists(pairing_path(state_dir))


async def test_a_new_window_clears_the_per_client_lockout(state_dir: str) -> None:
    await open_pairing(state_dir, 10, T0)
    for _ in range(5):
        await verify_pairing_code(state_dir, "nope", T0, "owner")
    opened = await open_pairing(state_dir, 10, T0)
    assert await verify_pairing_code(state_dir, opened.code, T0, "owner") == OK


async def test_closes_the_window_after_50_wrong_codes(state_dir: str) -> None:
    opened = await open_pairing(state_dir, 10, T0)
    for i in range(49):
        result = await verify_pairing_code(state_dir, "nope", T0, f"ip-{i // 5}")
        assert result.status != "exhausted"
    assert await get_pairing(state_dir, T0) == PairingWindow(expires_at=T0 + 600_000, attempts_left=1)
    assert await verify_pairing_code(state_dir, "nope", T0, "ip-last") == VerifyPairingResult("exhausted")
    assert not exists(pairing_path(state_dir))
    assert await verify_pairing_code(state_dir, opened.code, T0, "owner") == CLOSED


async def test_does_not_let_concurrent_attempts_share_one_attempt(state_dir: str) -> None:
    await open_pairing(state_dir, 10, T0)
    results = await asyncio.gather(*(verify_pairing_code(state_dir, "nope", T0, "one-ip") for _ in range(7)))
    assert len([r for r in results if r.status == "invalid"]) == 4
    assert len([r for r in results if r.status == "locked"]) == 3
    assert await get_pairing(state_dir, T0) == PairingWindow(expires_at=T0 + 600_000, attempts_left=45)


async def test_treats_an_expired_window_as_closed_and_removes_it(state_dir: str) -> None:
    opened = await open_pairing(state_dir, 10, T0)
    assert await get_pairing(state_dir, T0 + 600_000) is None
    assert not exists(pairing_path(state_dir))
    await open_pairing(state_dir, 1, T0)
    assert await verify_pairing_code(state_dir, opened.code, T0 + 60_001) == CLOSED


async def test_close_pairing_removes_the_window_and_is_idempotent(state_dir: str) -> None:
    await open_pairing(state_dir, 10, T0)
    await close_pairing(state_dir)
    await close_pairing(state_dir)
    assert await get_pairing(state_dir, T0) is None


async def test_counts_ipv6_failures_per_64_and_ipv4_mapped_as_ipv4(state_dir: str) -> None:
    opened = await open_pairing(state_dir, 10, T0)
    for i in range(5):
        await verify_pairing_code(state_dir, "nope", T0, f"2001:db8::{i + 1}")
    assert await verify_pairing_code(state_dir, opened.code, T0, "2001:db8::ffff:1") == LOCKED
    for _ in range(5):
        await verify_pairing_code(state_dir, "nope", T0, "198.51.100.7")
    assert await verify_pairing_code(state_dir, opened.code, T0, "::ffff:198.51.100.7") == LOCKED
    assert await verify_pairing_code(state_dir, opened.code, T0, "2001:db8:0:1::1") == OK


async def test_holds_the_pairing_lock_around_open_verify_and_expiry(
    state_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = os.path.join(state_dir, "pairing.lock")
    events = trace_lock_io(monkeypatch, pairing_module, lock_path)
    await open_pairing(state_dir, 10, T0)
    assert events == ["lock", "write", "unlock"]
    events.clear()
    await verify_pairing_code(state_dir, "nope", T0, "ip")
    assert events == ["lock", "read", "write", "unlock"]
    events.clear()
    assert await get_pairing(state_dir, T0 + 600_000) is None
    assert events == ["lock", "read", "unlock"]
    assert os.stat(lock_path).st_mode & 0o777 == 0o600


def test_verify_waits_for_another_process_holding_the_pairing_lock(state_dir: str) -> None:
    """The CLI (simulated by a thread with its own fd) reopens the window while holding the lock; the
    server's verification waits and then checks against the new window."""
    asyncio.run(open_pairing(state_dir, 10, T0))
    fd = os.open(os.path.join(state_dir, "pairing.lock"), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    results: list[VerifyPairingResult] = []
    done = threading.Event()
    holder: dict[str, str] = {}

    def server_op() -> None:
        while "code" not in holder:
            pass
        results.append(asyncio.run(verify_pairing_code(state_dir, holder["code"], T0, "ip")))
        done.set()

    thread = threading.Thread(target=server_op)
    thread.start()
    try:
        # Write a new window directly, the way open_pairing does under the lock.
        new = pairing_module.PairingFile(
            code_hash=sha256_hex("ABCDEFGHJK"), created_at=T0, expires_at=T0 + 600_000, failures=0
        )
        json_file.write_json_atomic(pairing_path(state_dir), new.to_json())
        holder["code"] = "ABCDE-FGHJK"
        assert not done.wait(0.2), "the server verified while another process held the lock"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    thread.join(5)
    assert results == [OK]
