"""OAuth state in `stateDir/oauth.json`: DCR clients, authorization codes and grants.

File format: `{"version": 1, "clients": {clientId: StoredClient}, "codes": {sha256(code): StoredCode},
"grants": {grantId: StoredGrant}}`, camelCase keys, times in epoch ms. Only SHA-256 hashes of
tokens, codes and secrets are stored.

Every public method runs under one `asyncio.Lock` (operations of this process) and an exclusive
`fcntl.flock` on `stateDir/oauth.lock` (operations of other processes, such as the CLI editing the
file next to the running server). Under both locks the file is reloaded when its identity (inode, mtime, size) changed,
the operation runs, and any write completes before the locks are released, so a read-modify-write
never interleaves with another process. The one exception is the bearer check (`use_access_token`):
while the file is unchanged since we last read or wrote it, it answers from memory without the
flock, which it takes only to reload the file or to persist `lastUsedAt`. Methods return copies
(`model_copy(deep=True)`), never the stored objects.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import ValidationError

from ..clock import DAY_MS, HOUR_MS, MINUTE_MS, Clock, now_ms
from ..json_file import (
    FileIdentity,
    StateModel,
    ensure_dir,
    file_identity,
    file_lock,
    read_json,
    write_json_atomic_identity,
)
from ..urls import host_of
from .crypto import base64url, random_token, sha256_hex

TokenEndpointAuthMethod = Literal["none", "client_secret_post", "client_secret_basic"]

TOUCH_PERSIST_MS = 5 * MINUTE_MS
"""lastUsedAt is kept in memory on every use but written at most this often per grant."""
MAX_ROTATED_HASHES = 20
LOCK_FILE = "oauth.lock"
"""Lock file next to oauth.json; every process that changes oauth.json holds an exclusive flock on it."""
REFRESH_GRACE_MS = MINUTE_MS
"""A repeat of a just-rotated refresh token within this window gets the same response, not a revocation."""


class StoredClient(StateModel):
    """A client registered through DCR. CIMD clients are not stored, only their grants."""

    client_id: str
    client_name: str
    redirect_uris: list[str]
    token_endpoint_auth_method: TokenEndpointAuthMethod
    client_secret_hash: str | None
    """SHA-256 hex of the client secret, None for public clients."""
    created_at: int


class StoredCode(StateModel):
    client_id: str
    redirect_uri: str
    code_challenge: str
    resource: str
    scope: str
    created_at: int
    expires_at: int
    used_at: int | None
    """Set when the code is redeemed. A second redemption is a reuse."""
    grant_id: str | None
    """Grant issued from this code, revoked when the code is reused."""
    client_name: str = ""
    """Client name shown at consent, for the grant (empty in codes stored before it was kept)."""


class StoredGrant(StateModel):
    """One authorization (token family): current access and refresh tokens plus rotated refresh hashes."""

    id: str
    client_id: str
    client_name: str
    resource: str
    scope: str
    created_at: int
    last_used_at: int
    access_token_hash: str
    access_expires_at: int
    refresh_token_hash: str
    refresh_expires_at: int
    rotated_refresh_hashes: list[str]
    """Hashes of refresh tokens already rotated away; presenting one revokes the grant."""


@dataclass(frozen=True)
class NewClient:
    client_name: str
    redirect_uris: list[str]
    token_endpoint_auth_method: TokenEndpointAuthMethod


@dataclass(frozen=True)
class NewCode:
    client_id: str
    redirect_uri: str
    code_challenge: str
    resource: str
    scope: str
    client_name: str


@dataclass(frozen=True)
class NewGrant:
    client_id: str
    client_name: str
    resource: str
    scope: str


@dataclass(frozen=True)
class IssuedTokens:
    grant_id: str
    access_token: str
    refresh_token: str
    expires_in: int
    """Seconds until the access token expires."""
    scope: str


@dataclass(frozen=True)
class RegisteredClient:
    client: StoredClient
    client_secret: str | None
    """The plain secret, returned once; None for public clients."""


@dataclass(frozen=True)
class RedeemOk:
    code_hash: str
    code: StoredCode
    status: Literal["ok"] = "ok"


@dataclass(frozen=True)
class RedeemFailed:
    status: Literal["invalid", "expired"]


@dataclass(frozen=True)
class RedeemReused:
    revoked_grant: bool
    status: Literal["reused"] = "reused"


RedeemCodeResult = RedeemOk | RedeemFailed | RedeemReused


@dataclass(frozen=True)
class ExchangeOk:
    tokens: IssuedTokens
    status: Literal["ok"] = "ok"


@dataclass(frozen=True)
class ExchangeRejected:
    """The code was valid but `check` refused it (PKCE, client, redirect URI); the code is burned."""

    reason: str
    status: Literal["rejected"] = "rejected"


ExchangeCodeResult = ExchangeOk | ExchangeRejected | RedeemFailed | RedeemReused


@dataclass(frozen=True)
class RefreshOk:
    tokens: IssuedTokens
    grant: StoredGrant
    status: Literal["ok"] = "ok"


@dataclass(frozen=True)
class RefreshFailed:
    status: Literal["invalid", "expired", "client_mismatch", "resource_mismatch"]


@dataclass(frozen=True)
class RefreshReused:
    grant_id: str
    status: Literal["reused"] = "reused"


RefreshResult = RefreshOk | RefreshFailed | RefreshReused


@dataclass(frozen=True)
class ClientStatus:
    client_id: str
    client_name: str
    kind: Literal["dcr", "cimd"]
    redirect_hosts: list[str]
    created_at: int
    last_used_at: int | None
    live_grants: int


@dataclass(frozen=True)
class OAuthStatus:
    clients: list[ClientStatus]
    """Sorted by created_at."""
    live_grants: int


@dataclass(frozen=True)
class RevokeAllResult:
    clients: int
    grants: int


class ClientLimitError(Exception):
    """Raised by register_client when the client cap is reached and no stored client can be evicted."""

    def __init__(self, max_clients: int) -> None:
        super().__init__(
            f"client limit of {max_clients} reached and every client has a live grant or a pending authorization code"
        )


@dataclass(frozen=True)
class _RecentRefresh:
    client_id: str
    grant_id: str
    tokens: IssuedTokens
    expires_at: int


@dataclass
class _OAuthFile:
    clients: dict[str, StoredClient]
    codes: dict[str, StoredCode]
    grants: dict[str, StoredGrant]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "clients": {key: client.to_json() for key, client in self.clients.items()},
            "codes": {key: code.to_json() for key, code in self.codes.items()},
            "grants": {key: grant.to_json() for key, grant in self.grants.items()},
        }


def _empty_file() -> _OAuthFile:
    return _OAuthFile(clients={}, codes={}, grants={})


def _records[M: StateModel](raw: object, model: type[M]) -> dict[str, M]:
    """Validates one section of the file. Entries that are not valid records are dropped (they could
    only break later requests)."""
    out: dict[str, M] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        try:
            out[str(key)] = model.model_validate(value)
        except ValidationError:
            continue
    return out


def _normalize_file(data: object) -> _OAuthFile:
    if not isinstance(data, dict):
        return _empty_file()
    return _OAuthFile(
        clients=_records(data.get("clients"), StoredClient),
        codes=_records(data.get("codes"), StoredCode),
        grants=_records(data.get("grants"), StoredGrant),
    )


class OAuthStore:
    def __init__(
        self,
        state_dir: str,
        *,
        access_token_hours: float = 24,
        refresh_token_days: float = 90,
        code_ttl_ms: int = MINUTE_MS,
        max_clients: int = 20,
        now: Clock | None = None,
    ) -> None:
        """The file is `state_dir/oauth.json` (`self.path`); it is read lazily on the first call, which
        also creates `state_dir` (0700) when needed."""
        self._state_dir = state_dir
        self._dir_ready = False
        self._path = os.path.join(state_dir, "oauth.json")
        self._lock_path = os.path.join(state_dir, LOCK_FILE)
        self._access_ttl_ms = round(access_token_hours * HOUR_MS)
        self._refresh_ttl_ms = round(refresh_token_days * DAY_MS)
        self._code_ttl_ms = code_ttl_ms
        self._max_clients = max_clients
        self._now: Clock = now or now_ms
        self._data = _empty_file()
        self._loaded = False
        self._loaded_identity: FileIdentity | None = None
        """Identity of the file as last read or written by us (None: no file)."""
        self._lock = asyncio.Lock()
        self._recent_refreshes: dict[str, _RecentRefresh] = {}
        """Refresh token hash -> the exact response its rotation produced, for REFRESH_GRACE_MS. Claude
        may send two refreshes with the same token at once; the second must not count as reuse. In
        memory only, so plain tokens never reach the file."""

    @property
    def path(self) -> str:
        return self._path

    @property
    def lock_path(self) -> str:
        return self._lock_path

    async def _run[T](self, fn: Callable[[], T]) -> T:
        """Runs `fn` after all earlier operations, with the file freshly synced, under both locks."""
        async with self._lock:
            return self._run_file_locked(fn)

    def _run_file_locked[T](self, fn: Callable[[], T]) -> T:
        """`_run` once the asyncio lock is held: takes the flock (held only for this one small
        read-modify-write), syncs and runs `fn`. If `fn` raises (e.g. `_save` cannot write the file),
        any change it made in memory is dropped: the next operation reloads the file, so a failed
        write cannot poison later ones. The state dir is ensured on the first call only."""
        if not self._dir_ready:
            ensure_dir(self._state_dir)
            self._dir_ready = True
        with file_lock(self._lock_path, ensure_parent=False):
            self._sync()
            try:
                return fn()
            except BaseException:
                self._loaded = False
                self._loaded_identity = None
                raise

    def _is_synced(self) -> bool:
        """True when the file is known to be what we last read or wrote (same identity)."""
        return self._loaded and file_identity(self._path) == self._loaded_identity

    def _sync(self) -> None:
        """Reloads the file when its identity (inode, mtime in ns, size) differs from what we last saw
        (e.g. the CLI changed it). The inode catches a write in the same timestamp tick as ours."""
        identity = file_identity(self._path)
        if self._loaded and identity == self._loaded_identity:
            return
        self._data = _empty_file() if identity is None else _normalize_file(read_json(self._path, None))
        self._loaded = True
        self._loaded_identity = identity

    def _save(self) -> None:
        self._prune()
        # The identity of the file just written (from its own fd), so a later write by another
        # process can never be mistaken for ours.
        self._loaded_identity = write_json_atomic_identity(self._path, self._data.to_json())

    @staticmethod
    def _is_live(grant: StoredGrant, now: int) -> bool:
        return grant.refresh_expires_at > now or grant.access_expires_at > now

    def _prune(self) -> None:
        """Drops expired codes and grants. Clients are evicted only by register_client, never with live grants."""
        now = self._now()
        for key in [k for k, code in self._data.codes.items() if code.expires_at <= now]:
            del self._data.codes[key]
        for key in [k for k, grant in self._data.grants.items() if not self._is_live(grant, now)]:
            del self._data.grants[key]

    def _make_room_for_client(self) -> bool:
        """Evicts the oldest clients until there is room for one more. Never evicted: a client with a
        live grant and one with a pending (unredeemed, unexpired) authorization code. Age gives no
        protection, so a flood of fresh registrations cannot lock out the owner's registration (it
        pushes out the flood instead). False when no client can be evicted."""
        self._prune()
        now = self._now()
        protected = {g.client_id for g in self._data.grants.values() if self._is_live(g, now)}
        protected |= {c.client_id for c in self._data.codes.values() if c.used_at is None and c.expires_at > now}
        evictable = sorted(
            (c for c in self._data.clients.values() if c.client_id not in protected),
            key=lambda c: c.created_at,
        )
        count = len(self._data.clients)
        while count >= self._max_clients:
            if not evictable:
                return False
            self._remove_client(evictable.pop(0).client_id)
            count -= 1
        return True

    def _remove_client(self, client_id: str) -> None:
        self._data.clients.pop(client_id, None)
        for key in [k for k, g in self._data.grants.items() if g.client_id == client_id]:
            del self._data.grants[key]
        for key in [k for k, c in self._data.codes.items() if c.client_id == client_id]:
            del self._data.codes[key]

    def _new_tokens(self, grant: StoredGrant, now: int) -> IssuedTokens:
        access_token = random_token()
        refresh_token = random_token()
        grant.access_token_hash = sha256_hex(access_token)
        grant.access_expires_at = now + self._access_ttl_ms
        grant.refresh_token_hash = sha256_hex(refresh_token)
        grant.refresh_expires_at = now + self._refresh_ttl_ms
        return IssuedTokens(
            grant_id=grant.id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self._access_ttl_ms // 1000,
            scope=grant.scope,
        )

    async def register_client(self, new: NewClient) -> RegisteredClient:
        """Registers a DCR client (`clientId` = `vcm-` + 16 random bytes base64url). When full, evicts the
        oldest clients that have no live grant and no pending authorization code; raises ClientLimitError
        if none can be evicted."""

        def op() -> RegisteredClient:
            if not self._make_room_for_client():
                raise ClientLimitError(self._max_clients)
            client_secret = None if new.token_endpoint_auth_method == "none" else random_token()
            client = StoredClient(
                client_id=f"vcm-{base64url(secrets.token_bytes(16))}",
                client_name=new.client_name,
                redirect_uris=list(new.redirect_uris),
                token_endpoint_auth_method=new.token_endpoint_auth_method,
                client_secret_hash=None if client_secret is None else sha256_hex(client_secret),
                created_at=self._now(),
            )
            self._data.clients[client.client_id] = client
            self._save()
            return RegisteredClient(client=client.model_copy(deep=True), client_secret=client_secret)

        return await self._run(op)

    async def get_client(self, client_id: str) -> StoredClient | None:
        def op() -> StoredClient | None:
            client = self._data.clients.get(client_id)
            return client.model_copy(deep=True) if client else None

        return await self._run(op)

    async def create_authorization_code(self, new: NewCode) -> str:
        """Stores a new authorization code (as hash) and returns the plain code."""

        def op() -> str:
            code = random_token()
            now = self._now()
            self._data.codes[sha256_hex(code)] = StoredCode(
                client_id=new.client_id,
                redirect_uri=new.redirect_uri,
                code_challenge=new.code_challenge,
                resource=new.resource,
                scope=new.scope,
                created_at=now,
                expires_at=now + self._code_ttl_ms,
                used_at=None,
                grant_id=None,
                client_name=new.client_name,
            )
            self._save()
            return code

        return await self._run(op)

    def _redeem(self, code: str) -> RedeemCodeResult:
        """Marks a code as used in memory (the caller saves). A reuse revokes the grant issued from it."""
        code_hash = sha256_hex(code)
        stored = self._data.codes.get(code_hash)
        if stored is None:
            return RedeemFailed(status="invalid")
        now = self._now()
        if stored.used_at is not None:
            revoked_grant = False
            if stored.grant_id and stored.grant_id in self._data.grants:
                del self._data.grants[stored.grant_id]
                revoked_grant = True
            return RedeemReused(revoked_grant=revoked_grant)
        if stored.expires_at <= now:
            del self._data.codes[code_hash]
            return RedeemFailed(status="expired")
        stored.used_at = now
        return RedeemOk(code_hash=code_hash, code=stored.model_copy(deep=True))

    def _add_grant(self, new: NewGrant, code_hash: str | None) -> IssuedTokens:
        """Creates a grant in memory (the caller saves), linked to the code it came from."""
        now = self._now()
        grant = StoredGrant(
            id=base64url(secrets.token_bytes(9)),
            client_id=new.client_id,
            client_name=new.client_name,
            resource=new.resource,
            scope=new.scope,
            created_at=now,
            last_used_at=now,
            access_token_hash="",
            access_expires_at=0,
            refresh_token_hash="",
            refresh_expires_at=0,
            rotated_refresh_hashes=[],
        )
        tokens = self._new_tokens(grant, now)
        self._data.grants[grant.id] = grant
        code = self._data.codes.get(code_hash) if code_hash else None
        if code is not None:
            code.grant_id = grant.id
        return tokens

    async def exchange_authorization_code(
        self,
        code: str,
        check: Callable[[StoredCode], str | None],
    ) -> ExchangeCodeResult:
        """Redeems a code and creates its grant in one locked operation, so a concurrent reuse of the
        code always finds (and revokes) the grant. `check` gets the stored code and returns a reason to
        refuse it (PKCE, client, redirect URI), or None; a refused code stays burned. The grant takes
        client, client name, resource and scope from the code."""

        def op() -> ExchangeCodeResult:
            redeemed = self._redeem(code)
            if not isinstance(redeemed, RedeemOk):
                if not (isinstance(redeemed, RedeemFailed) and redeemed.status == "invalid"):
                    self._save()
                return redeemed
            stored = redeemed.code
            reason = check(stored.model_copy(deep=True))
            if reason is not None:
                self._save()
                return ExchangeRejected(reason=reason)
            tokens = self._add_grant(
                NewGrant(
                    client_id=stored.client_id,
                    client_name=stored.client_name or stored.client_id,
                    resource=stored.resource,
                    scope=stored.scope,
                ),
                redeemed.code_hash,
            )
            self._save()
            return ExchangeOk(tokens=tokens)

        return await self._run(op)

    async def refresh_grant(self, refresh_token: str, client_id: str, resource: str) -> RefreshResult:
        """Rotates a refresh token of a grant issued for `resource`. Reuse of a rotated token revokes the
        whole grant, except that a repeat within REFRESH_GRACE_MS of its rotation (parallel refreshes)
        gets the same response again, as long as the tokens from that response are still the grant's
        current ones. A grant for another resource (the public URL changed) is `resource_mismatch`."""

        def op() -> RefreshResult:
            token_hash = sha256_hex(refresh_token)
            now = self._now()
            for key in [k for k, e in self._recent_refreshes.items() if e.expires_at <= now]:
                del self._recent_refreshes[key]
            recent = self._recent_refreshes.get(token_hash)
            if recent is not None:
                current = self._data.grants.get(recent.grant_id)
                if current is not None and current.refresh_token_hash == sha256_hex(recent.tokens.refresh_token):
                    if recent.client_id != client_id:
                        return RefreshFailed(status="client_mismatch")
                    return RefreshOk(tokens=replace(recent.tokens), grant=current.model_copy(deep=True))
                del self._recent_refreshes[token_hash]
            for grant in list(self._data.grants.values()):
                if token_hash in grant.rotated_refresh_hashes:
                    del self._data.grants[grant.id]
                    self._save()
                    return RefreshReused(grant_id=grant.id)
                if grant.refresh_token_hash != token_hash:
                    continue
                if grant.client_id != client_id:
                    return RefreshFailed(status="client_mismatch")
                if grant.resource != resource:
                    return RefreshFailed(status="resource_mismatch")
                if grant.refresh_expires_at <= now:
                    return RefreshFailed(status="expired")
                grant.rotated_refresh_hashes = [*grant.rotated_refresh_hashes, token_hash][-MAX_ROTATED_HASHES:]
                grant.last_used_at = now
                tokens = self._new_tokens(grant, now)
                self._save()
                self._recent_refreshes[token_hash] = _RecentRefresh(
                    client_id=client_id, grant_id=grant.id, tokens=tokens, expires_at=now + REFRESH_GRACE_MS
                )
                return RefreshOk(tokens=tokens, grant=grant.model_copy(deep=True))
            return RefreshFailed(status="invalid")

        return await self._run(op)

    def _live_access_grant(self, token_hash: str, resource: str, now: int) -> StoredGrant | None:
        grant = next((g for g in self._data.grants.values() if g.access_token_hash == token_hash), None)
        if grant is None or grant.access_expires_at <= now or grant.resource != resource:
            return None
        return grant

    async def use_access_token(self, access_token: str, resource: str) -> StoredGrant | None:
        """Looks up a live access token issued for `resource` and records its use. None for unknown or
        expired tokens and for tokens of another resource, so the caller needs no checks of its own.

        Runs on every /mcp request, so while the file is unchanged (same identity) it answers from
        memory without the flock. It takes the flock only to reload a changed file or when
        `lastUsedAt` is due to be written (TOUCH_PERSIST_MS). A revocation by another process
        changes the file, so the next request sees it."""
        token_hash = sha256_hex(access_token)

        def op() -> StoredGrant | None:
            now = self._now()
            grant = self._live_access_grant(token_hash, resource, now)
            if grant is None:
                return None
            persist = now - grant.last_used_at >= TOUCH_PERSIST_MS
            grant.last_used_at = now
            if persist:
                self._save()
            return grant.model_copy(deep=True)

        async with self._lock:
            if self._is_synced():
                now = self._now()
                grant = self._live_access_grant(token_hash, resource, now)
                if grant is None:
                    return None
                if now - grant.last_used_at < TOUCH_PERSIST_MS:
                    grant.last_used_at = now
                    return grant.model_copy(deep=True)
            return self._run_file_locked(op)

    async def revoke_token(self, token: str, client_id: str | None = None) -> bool:
        """Revokes the grant owning an access, refresh or rotated refresh token (RFC 7009). With
        `client_id`, a token of another client is left alone. Returns whether a grant was revoked."""

        def op() -> bool:
            token_hash = sha256_hex(token)
            grant = next(
                (
                    g
                    for g in self._data.grants.values()
                    if token_hash in (g.access_token_hash, g.refresh_token_hash)
                    or token_hash in g.rotated_refresh_hashes
                ),
                None,
            )
            if grant is None or (client_id is not None and grant.client_id != client_id):
                return False
            del self._data.grants[grant.id]
            self._save()
            return True

        return await self._run(op)

    async def status(self) -> OAuthStatus:
        """Overview without secrets, for the CLI."""

        def op() -> OAuthStatus:
            now = self._now()
            by_client: dict[str, ClientStatus] = {}
            for client in self._data.clients.values():
                by_client[client.client_id] = ClientStatus(
                    client_id=client.client_id,
                    client_name=client.client_name,
                    kind="dcr",
                    redirect_hosts=list(dict.fromkeys(host_of(uri) for uri in client.redirect_uris)),
                    created_at=client.created_at,
                    last_used_at=None,
                    live_grants=0,
                )
            live_grants = 0
            for grant in self._data.grants.values():
                if not self._is_live(grant, now):
                    continue
                live_grants += 1
                entry = by_client.get(grant.client_id) or ClientStatus(
                    client_id=grant.client_id,
                    client_name=grant.client_name,
                    kind="cimd",
                    redirect_hosts=[],
                    created_at=grant.created_at,
                    last_used_at=None,
                    live_grants=0,
                )
                by_client[grant.client_id] = replace(
                    entry,
                    live_grants=entry.live_grants + 1,
                    created_at=min(entry.created_at, grant.created_at),
                    last_used_at=max(entry.last_used_at or 0, grant.last_used_at),
                )
            clients = sorted(by_client.values(), key=lambda e: e.created_at)
            return OAuthStatus(clients=clients, live_grants=live_grants)

        return await self._run(op)

    async def revoke_all(self) -> RevokeAllResult:
        """Removes all clients, codes and grants. Every connected app must pair again."""

        def op() -> RevokeAllResult:
            result = RevokeAllResult(clients=len(self._data.clients), grants=len(self._data.grants))
            self._data = _empty_file()
            self._recent_refreshes.clear()
            self._save()
            return result

        return await self._run(op)
