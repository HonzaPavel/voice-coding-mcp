"""Tests of the OAuth state file store."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import threading
from pathlib import Path

import pytest

from tests.auth.helpers import read_text, write_text
from tests.helpers import FakeClock, create_grant, trace_lock_io
from voice_coding_mcp.auth import ClientLimitError, OAuthStore
from voice_coding_mcp.auth import store as store_module
from voice_coding_mcp.auth.store import IssuedTokens, NewClient, NewCode, NewGrant, RevokeAllResult, StoredCode
from voice_coding_mcp.json_file import FileIdentity, write_json_atomic_identity

R = "https://vcm.example.com/mcp"


@pytest.fixture
def now() -> FakeClock:
    return FakeClock(1_800_000_000_000)


@pytest.fixture
def state_dir(real_tmp: Path) -> str:
    return str(real_tmp)


async def grant_for(store: OAuthStore, client_id: str) -> IssuedTokens:
    return await create_grant(store, NewGrant(client_id=client_id, client_name="x", resource=R, scope="mcp"))


def new_client(name: str, uris: list[str] | None = None, method: str = "none") -> NewClient:
    return NewClient(client_name=name, redirect_uris=uris or ["https://a/cb"], token_endpoint_auth_method=method)  # type: ignore[arg-type]


async def test_stores_only_hashes_in_a_0600_file(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now)
    registered = await store.register_client(
        new_client("Claude", ["https://claude.ai/api/mcp/auth_callback"], "client_secret_post")
    )
    code = await store.create_authorization_code(
        NewCode(
            client_id=registered.client.client_id,
            redirect_uri="x",
            code_challenge="c",
            resource=R,
            scope="mcp",
            client_name="Claude",
        )
    )
    tokens = await grant_for(store, registered.client.client_id)
    text = read_text(store.path)
    assert registered.client_secret
    for secret in [registered.client_secret, code, tokens.access_token, tokens.refresh_token]:
        assert secret not in text
    assert os.stat(store.path).st_mode & 0o777 == 0o600


async def test_a_failed_save_does_not_poison_later_operations(
    state_dir: str, now: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = OAuthStore(state_dir, now=now)
    kept = await store.register_client(new_client("Kept"))

    def failing_write(path: str, data: object) -> FileIdentity:
        raise OSError("disk full")

    monkeypatch.setattr(store_module, "write_json_atomic_identity", failing_write)
    with pytest.raises(OSError, match="disk full"):
        await store.register_client(new_client("Lost"))
    monkeypatch.setattr(store_module, "write_json_atomic_identity", write_json_atomic_identity)
    names = [c.client_name for c in (await store.status()).clients]
    assert names == ["Kept"]
    later = await store.register_client(new_client("Later"))
    assert await store.get_client(later.client.client_id) is not None
    assert await store.get_client(kept.client.client_id) is not None
    reread = OAuthStore(state_dir, now=now)
    assert sorted(c.client_name for c in (await reread.status()).clients) == ["Kept", "Later"]


async def test_reloads_when_another_process_rewrites_the_file(state_dir: str, now: FakeClock) -> None:
    server = OAuthStore(state_dir, now=now)
    tokens = await grant_for(server, "c1")
    assert await server.use_access_token(tokens.access_token, R)
    await asyncio.sleep(0.015)
    cli = OAuthStore(state_dir, now=now)
    assert await cli.revoke_all() == RevokeAllResult(clients=0, grants=1)
    assert await server.use_access_token(tokens.access_token, R) is None


async def test_sees_grants_written_by_another_instance(state_dir: str, now: FakeClock) -> None:
    a = OAuthStore(state_dir, now=now)
    b = OAuthStore(state_dir, now=now)
    assert (await b.status()).live_grants == 0
    await asyncio.sleep(0.015)
    await grant_for(a, "c1")
    assert (await b.status()).live_grants == 1


async def test_expires_access_tokens_and_prunes_dead_grants_and_codes(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now, access_token_hours=1, refresh_token_days=1)
    tokens = await grant_for(store, "c1")
    await store.create_authorization_code(
        NewCode(client_id="c1", redirect_uri="x", code_challenge="c", resource=R, scope="mcp", client_name="Claude")
    )
    now.t += 3_600_000
    assert await store.use_access_token(tokens.access_token, R) is None
    assert (await store.status()).live_grants == 1
    now.t += 86_400_000
    await grant_for(store, "c2")  # any write prunes
    data = json.loads(read_text(store.path))
    assert len(data["grants"]) == 1
    assert len(data["codes"]) == 0


async def test_keeps_at_most_max_clients_evicting_the_oldest_without_live_grants(
    state_dir: str, now: FakeClock
) -> None:
    store = OAuthStore(state_dir, now=now, max_clients=3)
    ids: list[str] = []
    for i in range(3):
        now.t += 1000
        ids.append((await store.register_client(new_client(f"c{i}"))).client.client_id)
    await grant_for(store, ids[0])  # oldest, but has a live grant
    await store.register_client(new_client("c3"))
    assert await store.get_client(ids[0])
    assert await store.get_client(ids[1]) is None
    assert await store.get_client(ids[2])
    assert len((await store.status()).clients) == 3


async def test_never_evicts_a_client_with_a_live_grant(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now, max_clients=2)
    ids: list[str] = []
    for i in range(2):
        now.t += 1000
        client = (await store.register_client(new_client(f"c{i}"))).client
        ids.append(client.client_id)
        await grant_for(store, client.client_id)
    with pytest.raises(ClientLimitError):
        await store.register_client(new_client("c2"))
    for client_id in ids:
        assert await store.get_client(client_id)
    assert (await store.status()).live_grants == 2


async def test_status_reports_clients_without_secrets(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now)
    registered = await store.register_client(
        new_client("Inspector", ["http://localhost:6274/oauth/callback"], "client_secret_basic")
    )
    tokens = await grant_for(store, registered.client.client_id)
    await create_grant(
        store,
        NewGrant(
            client_id="https://claude.ai/oauth/mcp-oauth-client-metadata", client_name="Claude", resource=R, scope="mcp"
        ),
    )
    now.t += 10 * 60_000
    await store.use_access_token(tokens.access_token, R)
    status = await store.status()
    assert status.live_grants == 2
    dcr = next(c for c in status.clients if c.kind == "dcr")
    assert (dcr.client_name, dcr.redirect_hosts, dcr.live_grants, dcr.last_used_at) == (
        "Inspector",
        ["localhost:6274"],
        1,
        now.t,
    )
    assert next(c for c in status.clients if c.kind == "cimd").client_name == "Claude"
    text = repr(status)
    assert registered.client_secret
    for secret in [registered.client_secret, tokens.access_token, tokens.refresh_token]:
        assert secret not in text
    assert "hash" not in text.lower()


async def test_writes_last_used_at_at_most_every_few_minutes(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now)
    tokens = await grant_for(store, "c1")
    before = os.stat(store.path).st_mtime_ns
    await asyncio.sleep(0.015)
    now.t += 1000
    await store.use_access_token(tokens.access_token, R)
    assert os.stat(store.path).st_mtime_ns == before
    now.t += 5 * 60_000
    await store.use_access_token(tokens.access_token, R)
    assert os.stat(store.path).st_mtime_ns != before


async def test_round_trips_a_file_with_unknown_keys(state_dir: str, now: FakeClock) -> None:
    """A file with an unknown key is read and written back in the same shape."""
    grant = {
        "id": "abcdefghijkl",
        "clientId": "https://claude.ai/oauth/mcp-oauth-client-metadata",
        "clientName": "Claude",
        "resource": R,
        "scope": "mcp",
        "createdAt": now.t,
        "lastUsedAt": now.t,
        "accessTokenHash": "a" * 64,
        "accessExpiresAt": now.t + 3_600_000,
        "refreshTokenHash": "b" * 64,
        "refreshExpiresAt": now.t + 86_400_000,
        "rotatedRefreshHashes": ["c" * 64],
        "futureField": {"x": 1},
    }
    client = {
        "clientId": "vcm-x",
        "clientName": "Inspector",
        "redirectUris": ["http://localhost:6274/oauth/callback"],
        "tokenEndpointAuthMethod": "none",
        "clientSecretHash": None,
        "createdAt": now.t,
    }
    original = {"version": 1, "clients": {"vcm-x": client}, "codes": {}, "grants": {grant["id"]: grant}}
    write_text(os.path.join(state_dir, "oauth.json"), json.dumps(original, indent=2) + "\n")
    store = OAuthStore(state_dir, now=now)
    assert (await store.status()).live_grants == 1
    await store.revoke_token("not-a-token")  # no write
    await store.create_authorization_code(
        NewCode(client_id="vcm-x", redirect_uri="x", code_challenge="c", resource=R, scope="mcp", client_name="Claude")
    )
    data = json.loads(read_text(store.path))
    assert data["grants"] == original["grants"]
    assert data["clients"] == original["clients"]
    assert list(data) == ["version", "clients", "codes", "grants"]
    (code,) = data["codes"].values()
    assert list(code) == [
        "clientId",
        "redirectUri",
        "codeChallenge",
        "resource",
        "scope",
        "createdAt",
        "expiresAt",
        "usedAt",
        "grantId",
        "clientName",
    ]


async def test_evicts_recent_clients_without_a_pending_code_oldest_first(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now, max_clients=3, code_ttl_ms=60_000)
    ids: list[str] = []
    for i in range(3):
        now.t += 1000
        ids.append((await store.register_client(new_client(f"c{i}"))).client.client_id)
    # The owner's client (the oldest) has a pending code; the rest is a fresh flood.
    await store.create_authorization_code(
        NewCode(
            client_id=ids[0],
            redirect_uri="https://a/cb",
            code_challenge="c",
            resource=R,
            scope="mcp",
            client_name="Claude",
        )
    )
    now.t += 1000
    await store.register_client(new_client("flood1"))
    assert await store.get_client(ids[1]) is None
    now.t += 1000
    await store.register_client(new_client("flood2"))
    assert await store.get_client(ids[2]) is None
    assert await store.get_client(ids[0])
    assert len((await store.status()).clients) == 3


async def test_never_evicts_a_client_with_a_pending_authorization_code(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now, max_clients=2, code_ttl_ms=60_000)
    ids = [(await store.register_client(new_client(f"c{i}"))).client.client_id for i in range(2)]
    await store.create_authorization_code(
        NewCode(
            client_id=ids[0],
            redirect_uri="https://a/cb",
            code_challenge="c",
            resource=R,
            scope="mcp",
            client_name="Claude",
        )
    )
    now.t += 1000
    await store.register_client(new_client("c2"))
    assert await store.get_client(ids[0])
    assert await store.get_client(ids[1]) is None
    # Once the code expires, the client is evictable again.
    now.t += 60_000
    await store.register_client(new_client("c3"))
    assert await store.get_client(ids[0]) is None


async def test_holds_the_file_lock_around_the_whole_read_modify_write(
    state_dir: str, now: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = OAuthStore(state_dir, now=now)
    events = trace_lock_io(monkeypatch, store_module, store.lock_path)
    await store.register_client(new_client("c0"))
    assert events == ["lock", "write", "unlock"]
    events.clear()
    other = OAuthStore(state_dir, now=now)
    await other.revoke_all()
    assert events == ["lock", "read", "write", "unlock"]
    assert os.stat(store.lock_path).st_mode & 0o777 == 0o600


async def test_identity_comes_from_the_written_file(
    state_dir: str, now: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = OAuthStore(state_dir, now=now)
    await store.register_client(new_client("c0"))
    # A later stat of the path must not be used to record our own write.
    monkeypatch.setattr(store_module, "file_identity", lambda _path: FileIdentity(1, 1, 1))
    await store.register_client(new_client("c1"))
    info = os.stat(store.path)
    assert store._loaded_identity == FileIdentity(info.st_ino, info.st_mtime_ns, info.st_size)


async def test_reloads_an_external_write_with_the_same_mtime(state_dir: str, now: FakeClock) -> None:
    """Coarse timestamps: the CLI's revoke-all lands in the same mtime tick as the server's last
    write. The new inode still triggers a reload, so the server never writes revoked grants back."""
    server = OAuthStore(state_dir, now=now)
    registered = await server.register_client(new_client("c0"))
    tokens = await grant_for(server, registered.client.client_id)
    server_mtime_ns = os.stat(server.path).st_mtime_ns
    # The CLI (another store, another process) revokes everything in the same mtime tick.
    cli = OAuthStore(state_dir, now=now)
    await cli.revoke_all()
    os.utime(server.path, ns=(server_mtime_ns, server_mtime_ns))
    assert os.stat(server.path).st_mtime_ns == server_mtime_ns
    assert await server.use_access_token(tokens.access_token, R) is None
    await server.register_client(new_client("c1"))
    data = json.loads(read_text(server.path))
    assert data["grants"] == {}
    assert [c["clientName"] for c in data["clients"].values()] == ["c1"]


async def test_reloads_when_only_the_inode_differs(state_dir: str, now: FakeClock) -> None:
    server = OAuthStore(state_dir, now=now)
    await server.register_client(new_client("c0"))
    before = os.stat(server.path)
    # Same mtime and same size, new inode: an identical-looking file with other content.
    data = json.loads(read_text(server.path))
    client = next(iter(data["clients"].values()))
    client["clientName"] = "zz"
    identity = write_json_atomic_identity(server.path, data)
    os.utime(server.path, ns=(before.st_mtime_ns, before.st_mtime_ns))
    assert identity.ino != before.st_ino
    assert os.stat(server.path).st_size == before.st_size
    assert [c.client_name for c in (await server.status()).clients] == ["zz"]


async def test_exchange_redeems_and_grants_in_one_operation(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now)
    client_id = (await store.register_client(new_client("c0"))).client.client_id
    code = await store.create_authorization_code(
        NewCode(
            client_id=client_id,
            redirect_uri="https://a/cb",
            code_challenge="c",
            resource=R,
            scope="mcp",
            client_name="Claude",
        )
    )
    events: list[str] = []
    real_run = store._run

    async def run(fn: object) -> object:
        events.append("op")
        return await real_run(fn)  # type: ignore[arg-type]

    store._run = run  # type: ignore[method-assign,assignment]
    result = await store.exchange_authorization_code(code, lambda _stored: None)
    assert events == ["op"]
    assert result.status == "ok"
    grant = await store.use_access_token(result.tokens.access_token, R)
    assert grant is not None
    assert grant.client_name == "Claude"
    # A reuse right after always finds the grant and revokes it.
    reused = await store.exchange_authorization_code(code, lambda _stored: None)
    assert reused.status == "reused"
    assert reused.revoked_grant
    assert await store.use_access_token(result.tokens.access_token, R) is None


async def test_concurrent_exchanges_of_one_code_leave_no_grant(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now)
    client_id = (await store.register_client(new_client("c0"))).client.client_id
    code = await store.create_authorization_code(
        NewCode(
            client_id=client_id,
            redirect_uri="https://a/cb",
            code_challenge="c",
            resource=R,
            scope="mcp",
            client_name="Claude",
        )
    )
    results = await asyncio.gather(*(store.exchange_authorization_code(code, lambda _stored: None) for _ in range(2)))
    assert sorted(r.status for r in results) == ["ok", "reused"]
    assert (await store.status()).live_grants == 0


async def test_exchange_burns_a_code_the_check_rejects(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now)
    client_id = (await store.register_client(new_client("c0"))).client.client_id
    code = await store.create_authorization_code(
        NewCode(
            client_id=client_id,
            redirect_uri="https://a/cb",
            code_challenge="c",
            resource=R,
            scope="mcp",
            client_name="Claude",
        )
    )
    seen: list[StoredCode] = []

    def check(stored: StoredCode) -> str | None:
        seen.append(stored)
        return "PKCE verification failed"

    rejected = await store.exchange_authorization_code(code, check)
    assert rejected.status == "rejected"
    assert rejected.reason == "PKCE verification failed"
    assert seen[0].client_id == client_id
    assert (await store.status()).live_grants == 0
    again = await store.exchange_authorization_code(code, lambda _stored: None)
    assert again.status == "reused"
    assert not again.revoked_grant
    assert (await store.exchange_authorization_code("nope", lambda _stored: None)).status == "invalid"


def test_client_limit_error_names_the_eviction_rule() -> None:
    assert "every client has a live grant or a pending authorization code" in str(ClientLimitError(3))


def test_waits_for_another_process_holding_the_lock_and_keeps_its_write(state_dir: str, now: FakeClock) -> None:
    """Another process (simulated by a thread with its own fd) holds the lock while it rewrites the
    file; the server's operation waits, then builds on that write instead of overwriting it."""
    server = OAuthStore(state_dir, now=now)
    asyncio.run(server.register_client(new_client("before")))
    fd = os.open(server.lock_path, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    done = threading.Event()

    def server_op() -> None:
        asyncio.run(server.register_client(new_client("server")))
        done.set()

    thread = threading.Thread(target=server_op)
    thread.start()
    try:
        assert not done.wait(0.2), "the server wrote while another process held the lock"
        cli_file = {
            "version": 1,
            "clients": {
                "cli": {
                    "clientId": "cli",
                    "clientName": "cli",
                    "redirectUris": ["https://a/cb"],
                    "tokenEndpointAuthMethod": "none",
                    "clientSecretHash": None,
                    "createdAt": now.t,
                }
            },
            "codes": {},
            "grants": {},
        }
        write_text(server.path, json.dumps(cli_file))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    thread.join(5)
    assert done.is_set()
    names = sorted(c["clientName"] for c in json.loads(read_text(server.path))["clients"].values())
    assert names == ["cli", "server"]


async def test_bearer_lookups_skip_the_flock_while_the_file_is_unchanged(
    state_dir: str, now: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = OAuthStore(state_dir, now=now)
    tokens = await grant_for(store, "c1")
    locks: list[int] = []
    real_flock = fcntl.flock

    def flock(fd: int, op: int) -> None:
        locks.append(op)
        real_flock(fd, op)

    monkeypatch.setattr(fcntl, "flock", flock)
    now.t += 1000
    grant = await store.use_access_token(tokens.access_token, R)
    assert grant is not None
    assert grant.last_used_at == now.t
    assert await store.use_access_token("unknown", R) is None
    assert locks == []
    now.t += 5 * 60_000  # lastUsedAt is due: written under the flock
    assert await store.use_access_token(tokens.access_token, R) is not None
    assert locks == [fcntl.LOCK_EX, fcntl.LOCK_UN]


async def test_tokens_only_work_for_the_resource_of_their_grant(state_dir: str, now: FakeClock) -> None:
    store = OAuthStore(state_dir, now=now)
    other = "https://other.example/mcp"
    tokens = await create_grant(store, NewGrant(client_id="c1", client_name="x", resource=other, scope="mcp"))
    assert await store.use_access_token(tokens.access_token, R) is None
    refreshed = await store.refresh_grant(tokens.refresh_token, "c1", R)
    assert refreshed.status == "resource_mismatch"
    assert await store.use_access_token(tokens.access_token, other) is not None
    assert (await store.refresh_grant(tokens.refresh_token, "c1", other)).status == "ok"


async def test_bearer_lookups_see_a_revocation_by_another_process_at_once(state_dir: str, now: FakeClock) -> None:
    server = OAuthStore(state_dir, now=now)
    tokens = await grant_for(server, "c1")
    other = await grant_for(server, "c2")
    assert await server.use_access_token(tokens.access_token, R) is not None
    # The CLI is another process with its own store; no sleep, so the write may share an mtime tick.
    assert await OAuthStore(state_dir, now=now).revoke_token(tokens.access_token)
    assert await server.use_access_token(tokens.access_token, R) is None
    assert await server.use_access_token(other.access_token, R) is not None
    await OAuthStore(state_dir, now=now).revoke_all()
    assert await server.use_access_token(other.access_token, R) is None
