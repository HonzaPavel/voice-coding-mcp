"""Bearer token check for /mcp, as an ASGI middleware.

It accepts only a live access token issued for `config.resource_url`, stores an `AuthInfo` in
`scope["state"]["auth"]` (read it with `request_auth`), and answers anything else with 401 and a WWW-Authenticate challenge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Receive, Scope, Send

from ..config import Config
from ..log import Logger, fingerprint
from ..web import client_ip, json_response
from .router import SCOPE, resource_metadata_url
from .store import OAuthStore

# re.ASCII: ASCII case folding only (otherwise e.g. the Kelvin sign would match k). Trailing
# spaces and tabs are optional white space (RFC 9110).
_BEARER = re.compile(r"Bearer[ ]+([A-Za-z0-9\-._~+/]+=*)[ \t]*", re.IGNORECASE | re.ASCII)


def bearer_token(header: str | None) -> str | None:
    """The token of a well-formed `Authorization: Bearer <token>` header, else None."""
    match = _BEARER.fullmatch(header) if header else None
    return match.group(1) if match else None


@dataclass(frozen=True)
class BearerError:
    code: str
    description: str


@dataclass(frozen=True)
class AuthInfo:
    """The authenticated caller of an /mcp request, as the MCP tools see it."""

    token: str
    client_id: str
    scopes: list[str]
    expires_at: int
    """Seconds since the epoch."""
    resource: str
    resource_metadata_url: str
    extra: dict[str, Any] = field(default_factory=dict)
    """`grantId`, `clientName`."""


def www_authenticate(config: Config, error: BearerError | None = None) -> str:
    """`Bearer resource_metadata="<issuer>/.well-known/oauth-protected-resource/mcp", scope="mcp"`
    plus `error` and `error_description` when given."""
    parts = [f'resource_metadata="{resource_metadata_url(config)}"', f'scope="{SCOPE}"']
    if error is not None:
        parts += [f'error="{error.code}"', f'error_description="{error.description}"']
    return f"Bearer {', '.join(parts)}"


def request_auth(conn: HTTPConnection) -> AuthInfo | None:
    """The AuthInfo the middleware stored for this request, if any."""
    state = conn.scope.get("state")
    auth = state.get("auth") if isinstance(state, dict) else None
    return auth if isinstance(auth, AuthInfo) else None


class BearerMiddleware:
    """`Middleware(BearerMiddleware, store=..., config=..., logger=...)` or wrap an ASGI app directly.

    Header must match `^Bearer[ ]+([A-Za-z0-9\\-._~+/]+=*)\\s*$` (case-insensitive); otherwise 401
    `invalid_request`. Unknown or expired token, or one issued for another resource: 401 `invalid_token` (logged with the fingerprint only).
    A store error propagates (500). Responses carry `Cache-Control: no-store`."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        store: OAuthStore,
        config: Config,
        logger: Logger,
    ) -> None:
        self._app = app
        self._store = store
        self._config = config
        self._logger = logger.child({"component": "bearer"})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        conn = HTTPConnection(scope)

        async def reject(error: BearerError | None = None) -> None:
            response = json_response(
                401,
                {
                    "error": error.code if error else "invalid_token",
                    "error_description": error.description if error else "authorization required",
                },
                {"WWW-Authenticate": www_authenticate(self._config, error)},
            )
            await response(scope, receive, send)

        header = conn.headers.get("authorization")
        if not header:
            await reject()
            return
        token = bearer_token(header)
        if token is None:
            await reject(BearerError("invalid_request", "malformed Authorization header"))
            return
        # The store checks expiry (its clock) and the resource, so a grant it returns is valid here.
        grant = await self._store.use_access_token(token, self._config.resource_url)
        if grant is None:
            self._logger.info("invalid or expired access token", {"token": fingerprint(token), "ip": client_ip(conn)})
            await reject(BearerError("invalid_token", "access token invalid or expired"))
            return
        auth = AuthInfo(
            token=token,
            client_id=grant.client_id,
            scopes=[s for s in grant.scope.split(" ") if s],
            expires_at=grant.access_expires_at // 1000,
            resource=grant.resource,
            resource_metadata_url=resource_metadata_url(self._config),
            extra={"grantId": grant.id, "clientName": grant.client_name},
        )
        scope.setdefault("state", {})["auth"] = auth
        await self._app(scope, receive, send)
