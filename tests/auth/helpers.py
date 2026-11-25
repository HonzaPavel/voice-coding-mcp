"""Auth test helpers.

The auth routes and a `/mcp` echo route wrapped in `BearerMiddleware` sit on a bare Starlette app,
driven through `httpx2.ASGITransport` (client address 127.0.0.1).
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, cast
from urllib.parse import parse_qs, urlsplit

import httpx2
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tests.helpers import FakeClock, LogCapture, config_data
from voice_coding_mcp.auth import (
    AuthRouterOptions,
    BearerMiddleware,
    OAuthStore,
    RateLimitSettings,
    create_auth_routes,
    open_pairing,
    request_auth,
)
from voice_coding_mcp.config import Config, parse_config

PUBLIC_URL = "https://vcm.example.com"
RESOURCE = f"{PUBLIC_URL}/mcp"
CLAUDE_REDIRECT = "https://claude.ai/api/mcp/auth_callback"
CIMD_URL = "https://claude.ai/oauth/mcp-oauth-client-metadata"
T0 = 1_795_348_800_000
"""Date.UTC(2026, 10, 22, 12, 0, 0)."""


@dataclass
class Ctx:
    config: Config
    state_dir: str
    store: OAuthStore
    clock: FakeClock
    log: LogCapture
    http: httpx2.AsyncClient
    secrets: list[str] = field(default_factory=list)
    """Every secret the server handed out during the test (for the log leak check)."""

    @property
    def log_lines(self) -> list[str]:
        return self.log.lines


def make_config(root: Path, oauth: dict[str, Any] | None = None) -> Config:
    state_dir = root / "state"
    projects = root / "projects"
    projects.mkdir(parents=True, exist_ok=True)
    return parse_config(
        config_data(PUBLIC_URL, str(state_dir), str(projects), oauth=oauth or {}), home_dir="/nonexistent-home-dir"
    )


async def _mcp_echo(request: Request) -> Response:
    auth = request_auth(request)
    assert auth is not None
    return JSONResponse(
        {"clientId": auth.client_id, "scopes": auth.scopes, "resource": auth.resource, "expiresAt": auth.expires_at}
    )


def setup(
    root: Path,
    *,
    oauth: dict[str, Any] | None = None,
    http_client: httpx2.AsyncClient | None = None,
    rate_limit: RateLimitSettings | None = None,
) -> Ctx:
    config = make_config(root, oauth)
    clock = FakeClock(T0)
    log = LogCapture()
    store = OAuthStore(
        config.state_dir,
        access_token_hours=config.oauth.access_token_hours,
        refresh_token_days=config.oauth.refresh_token_days,
        now=clock.now,
    )
    routes = create_auth_routes(
        AuthRouterOptions(
            config=config,
            store=store,
            state_dir=config.state_dir,
            logger=log.logger,
            now=clock.now,
            http_client=http_client,
            rate_limit=rate_limit,
        )
    )
    mcp = Route(
        "/mcp",
        _mcp_echo,
        methods=["POST"],
        middleware=[Middleware(BearerMiddleware, store=store, config=config, logger=log.logger)],
    )
    app = Starlette(routes=[*routes, mcp])
    http = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app, client=("127.0.0.1", 50123)),
        base_url=PUBLIC_URL,
        follow_redirects=False,
    )
    return Ctx(config=config, state_dir=config.state_dir, store=store, clock=clock, log=log, http=http)


# Deliberately not `auth.crypto`: the tests compute PKCE on their own, as an independent check of
# the server's S256 verification.
def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def pkce() -> tuple[str, str]:
    """(verifier, challenge)."""
    verifier = b64url(secrets.token_bytes(32))
    return verifier, b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def location_params(res: httpx2.Response) -> dict[str, str]:
    location = res.headers["location"]
    return {k: v[0] for k, v in parse_qs(urlsplit(location).query, keep_blank_values=True).items()}


async def pair(ctx: Ctx) -> str:
    opened = await open_pairing(ctx.state_dir, 10, ctx.clock.t)
    ctx.secrets += [opened.code, opened.code.replace("-", "")]
    return opened.code


async def register(ctx: Ctx, body: dict[str, Any]) -> httpx2.Response:
    return await ctx.http.post("/register", json=body)


class RegisteredClient(TypedDict, total=False):
    client_id: str
    client_secret: str
    redirect_uris: list[str]


async def register_client(ctx: Ctx, **body: Any) -> RegisteredClient:
    res = await register(
        ctx,
        {"client_name": "Claude", "redirect_uris": [CLAUDE_REDIRECT], "token_endpoint_auth_method": "none", **body},
    )
    if res.status_code != 201:
        raise AssertionError(f"register failed: {res.status_code} {res.text}")
    data = cast(RegisteredClient, res.json())
    if data.get("client_secret"):
        ctx.secrets.append(data["client_secret"])
    return data


def authorize_params(client_id: str, challenge: str, /, **extra: str | None) -> dict[str, str]:
    """Extra values override; None drops a parameter."""
    params: dict[str, str | None] = {
        "client_id": client_id,
        "redirect_uri": CLAUDE_REDIRECT,
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "st-123",
        "resource": RESOURCE,
        "scope": "mcp",
        **extra,
    }
    return {k: v for k, v in params.items() if v is not None}


async def get_authorize(ctx: Ctx, params: dict[str, str]) -> httpx2.Response:
    return await ctx.http.get("/authorize", params=params)


async def post_authorize(ctx: Ctx, params: dict[str, str]) -> httpx2.Response:
    return await ctx.http.post("/authorize", data=params)


async def post_form(
    ctx: Ctx, path: str, params: dict[str, str], headers: dict[str, str] | None = None
) -> httpx2.Response:
    return await ctx.http.post(path, data=params, headers=headers or {})


async def get_code(ctx: Ctx, client_id: str, challenge: str, **extra: str | None) -> str:
    """Pairs, authorizes and returns the authorization code from the redirect."""
    pairing_code = await pair(ctx)
    res = await post_authorize(ctx, {**authorize_params(client_id, challenge, **extra), "pairing_code": pairing_code})
    if res.status_code != 302:
        raise AssertionError(f"authorize failed: {res.status_code} {res.text}")
    code = location_params(res).get("code")
    if not code:
        raise AssertionError(f"no code in {res.headers['location']}")
    ctx.secrets.append(code)
    return code


class TokenResponse(TypedDict):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    scope: str


async def exchange(ctx: Ctx, client_id: str, code: str, verifier: str, **extra: str) -> httpx2.Response:
    return await post_form(
        ctx,
        "/token",
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": CLAUDE_REDIRECT,
            "resource": RESOURCE,
            **extra,
        },
    )


def token_json(ctx: Ctx, res: httpx2.Response) -> TokenResponse:
    if res.status_code != 200:
        raise AssertionError(f"token failed: {res.status_code} {res.text}")
    data = cast(TokenResponse, res.json())
    ctx.secrets += [data["access_token"], data["refresh_token"]]
    return data


async def full_flow(ctx: Ctx) -> tuple[str, TokenResponse]:
    """Full flow for a public DCR client; returns client id and tokens."""
    await pair(ctx)
    client = await register_client(ctx)
    verifier, challenge = pkce()
    code = await get_code(ctx, client["client_id"], challenge)
    tokens = token_json(ctx, await exchange(ctx, client["client_id"], code, verifier))
    return client["client_id"], tokens


async def call_mcp(ctx: Ctx, token: str | None = None, raw_header: str | None = None) -> httpx2.Response:
    headers = {"content-type": "application/json"}
    if raw_header is not None:
        headers["authorization"] = raw_header
    elif token is not None:
        headers["authorization"] = f"Bearer {token}"
    return await ctx.http.post("/mcp", headers=headers, content=b"{}")


def exists(path: str) -> bool:
    """Synchronous file checks, kept out of the async test bodies (ruff ASYNC240)."""
    return os.path.exists(path)


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
