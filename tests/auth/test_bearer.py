"""Tests of the bearer token check on /mcp."""

from __future__ import annotations

import asyncio
import re

from tests.auth.conftest import MakeCtx
from tests.auth.helpers import PUBLIC_URL, call_mcp, full_flow
from tests.helpers import create_grant
from voice_coding_mcp.auth import OAuthStore, www_authenticate
from voice_coding_mcp.auth.store import NewGrant

CHALLENGE = f'Bearer resource_metadata="{PUBLIC_URL}/.well-known/oauth-protected-resource/mcp", scope="mcp"'


async def test_answers_a_request_without_a_token_with_401_and_the_exact_challenge(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    res = await call_mcp(ctx)
    assert res.status_code == 401
    assert res.headers["www-authenticate"] == CHALLENGE
    assert res.headers["cache-control"] == "no-store"
    assert res.json() == {"error": "invalid_token", "error_description": "authorization required"}
    assert www_authenticate(ctx.config) == CHALLENGE


async def test_rejects_malformed_headers_unknown_tokens_and_other_schemes(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    for header in ["Bearer", "Bearer a b", "Basic dXNlcjpwYXNz", "Bearer not-issued-token"]:
        res = await call_mcp(ctx, raw_header=header)
        assert res.status_code == 401, header
        assert res.headers["www-authenticate"].startswith(f'{CHALLENGE}, error="'), header


async def test_accepts_a_valid_token_and_the_scheme_case_insensitively(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    _, tokens = await full_flow(ctx)
    assert (await call_mcp(ctx, raw_header=f"bearer {tokens['access_token']}")).status_code == 200


async def test_rejects_an_expired_access_token(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    _, tokens = await full_flow(ctx)
    ctx.clock.t += 24 * 3_600_000 - 1
    assert (await call_mcp(ctx, tokens["access_token"])).status_code == 200
    ctx.clock.t += 1
    res = await call_mcp(ctx, tokens["access_token"])
    assert res.status_code == 401
    assert re.search(r'error="invalid_token"', res.headers["www-authenticate"])


async def test_rejects_a_token_issued_for_another_audience(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    store = OAuthStore(ctx.state_dir, now=ctx.clock.now)
    tokens = await create_grant(
        store, NewGrant(client_id="c", client_name="c", resource="https://other.example/mcp", scope="mcp")
    )
    ctx.secrets.append(tokens.access_token)
    await asyncio.sleep(0.015)
    res = await call_mcp(ctx, tokens.access_token)
    assert res.status_code == 401


async def test_rejects_tokens_after_revoke_all_from_another_process(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    _, tokens = await full_flow(ctx)
    await asyncio.sleep(0.015)
    await OAuthStore(ctx.state_dir).revoke_all()
    assert (await call_mcp(ctx, tokens["access_token"])).status_code == 401


async def test_logs_only_fingerprints_of_rejected_tokens(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    token = "A" * 43
    await call_mcp(ctx, token)
    text = "\n".join(ctx.log_lines)
    assert token not in text
    assert re.search(r"len=43 sha256=", text)
