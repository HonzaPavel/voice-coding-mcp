"""Tests of the HTTP app: Host/Origin checks, routing details, OAuth routes and /mcp."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from tests.app_helpers import AppCtx, start_app
from tests.auth.helpers import pkce
from tests.helpers import create_grant
from voice_coding_mcp.auth import open_pairing
from voice_coding_mcp.auth.store import NewClient, NewGrant

INIT = json.dumps(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    }
)
JSON = {"content-type": "application/json"}


@pytest.fixture
async def ctx(real_tmp: Path) -> AsyncIterator[AppCtx]:
    async with start_app(real_tmp) as app:
        yield app


async def test_serves_healthz_without_auth_even_on_another_host(ctx: AppCtx) -> None:
    res = await ctx.http.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"ok": True}
    assert res.headers["cache-control"] == "no-store"
    local = await ctx.http.get("/healthz", headers={"host": "localhost:8787"})
    assert local.status_code == 200


async def test_refuses_a_host_other_than_the_public_host(ctx: AppCtx) -> None:
    for path in ["/mcp", "/.well-known/oauth-protected-resource/mcp", "/authorize"]:
        res = await ctx.http.get(path, headers={"host": "evil.example.com"})
        assert res.status_code == 403, path
        assert res.text == "Forbidden: unknown host."
    ok = await ctx.http.get("/.well-known/oauth-protected-resource/mcp", headers={"host": f"127.0.0.1:{ctx.port}"})
    assert ok.status_code == 200


async def test_allows_only_the_public_origin_and_the_claude_origins(ctx: AppCtx) -> None:
    path = "/.well-known/oauth-authorization-server"
    for origin in ["https://claude.ai", "https://claude.com", ctx.base]:
        assert (await ctx.http.get(path, headers={"origin": origin})).status_code == 200, origin
    for origin in ["https://evil.example.com", "https://claude.ai.evil.com", "null"]:
        assert (await ctx.http.get(path, headers={"origin": origin})).status_code == 403, origin
    mcp = await ctx.http.post("/mcp", headers={"origin": "https://evil.example.com", **JSON}, content=INIT)
    assert mcp.status_code == 403
    refused = [r for r in ctx.log.records if r["msg"] == "origin refused"]
    assert refused[0]["origin"] == "https://evil.example.com"


async def test_accepts_the_consent_form_post_with_the_public_origin_and_refuses_origin_null(ctx: AppCtx) -> None:
    opened = await open_pairing(ctx.state_dir, 10)
    reg = await ctx.http.post(
        "/register",
        json={
            "client_name": "Claude",
            "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert reg.status_code == 201
    client_id = reg.json()["client_id"]
    _verifier, challenge = pkce()
    form = {
        "client_id": client_id,
        "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "s",
        "pairing_code": opened.code,
    }
    refused = await ctx.http.post("/authorize", headers={"origin": "null"}, data=form)
    assert refused.status_code == 403
    assert "origin not allowed" in refused.text
    ok = await ctx.http.post("/authorize", headers={"origin": ctx.base}, data=form)
    assert ok.status_code == 302
    assert re.match(r"^https://claude\.ai/api/mcp/auth_callback\?code=", ok.headers.get("location", ""))


async def test_answers_post_mcp_without_a_token_with_401_and_the_resource_metadata_challenge(ctx: AppCtx) -> None:
    res = await ctx.http.post("/mcp", headers=JSON, content=INIT)
    assert res.status_code == 401
    assert (
        res.headers["www-authenticate"]
        == f'Bearer resource_metadata="{ctx.base}/.well-known/oauth-protected-resource/mcp", scope="mcp"'
    )
    bad = await ctx.http.post("/mcp", headers={**JSON, "authorization": "Bearer nope"}, content=INIT)
    assert bad.status_code == 401
    assert 'error="invalid_token"' in bad.headers["www-authenticate"]


async def test_answers_get_and_delete_mcp_with_405_and_allow_post(ctx: AppCtx) -> None:
    for method in ["GET", "DELETE", "HEAD"]:
        res = await ctx.http.request(method, "/mcp")
        assert res.status_code == 405, method
        assert res.headers["allow"] == "POST"
        assert res.headers["cache-control"] == "no-store"
        if method != "HEAD":
            assert res.json() == {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": "Method not allowed."},
                "id": None,
            }


async def test_answers_unknown_paths_with_404(ctx: AppCtx) -> None:
    res = await ctx.http.get("/nope?x=1")
    assert res.status_code == 404
    assert res.text == "Not found."


async def test_matches_paths_case_insensitively_with_optional_slash(ctx: AppCtx) -> None:
    """Case-insensitive, one optional trailing slash, raw path; a wrong method is 404, not 405."""
    for path in ["/HEALTHZ", "/healthz/"]:
        assert (await ctx.http.get(path)).status_code == 200, path
    for path in ["/MCP", "/mcp/"]:
        res = await ctx.http.post(path, headers=JSON, content=INIT)
        assert res.status_code == 401, path
    assert (await ctx.http.get("/.well-known/OAuth-Authorization-Server/")).status_code == 200
    for method, path in [
        ("GET", "/register"),
        ("GET", "/token"),
        ("PUT", "/mcp"),
        ("POST", "/healthz"),
        ("POST", "/.well-known/oauth-authorization-server"),
        ("GET", "/mcp//"),
        ("POST", "/%6Dcp"),
    ]:
        res = await ctx.http.request(method, path)
        assert res.status_code == 404, (method, path)
        assert res.text == "Not found."
    # The Host check still comes first for every path but GET /healthz.
    assert (await ctx.http.post("/healthz", headers={"host": "evil.example.com"})).status_code == 403


async def test_logs_requests_without_query_strings_headers_or_tokens(ctx: AppCtx) -> None:
    await ctx.http.get("/nope?secret=abc123", headers={"authorization": "Bearer my-very-secret-token"})
    lines = [r for r in ctx.log.records if r["msg"] == "request"]
    line = next((r for r in lines if r["path"] == "/nope"), None)
    assert line is not None, "request line logged"
    assert line["method"] == "GET"
    assert line["status"] == 404
    assert isinstance(line["ms"], int)
    assert line["ip"] == "127.0.0.1"
    assert re.match(r"^len=\d+ sha256=[0-9a-f]{8}$", str(line["token"]))
    text = ctx.log.text()
    assert "abc123" not in text
    assert "my-very-secret-token" not in text


async def test_does_not_leak_stack_traces_on_malformed_oauth_bodies(ctx: AppCtx) -> None:
    res = await ctx.http.post("/register", headers=JSON, content="{not json")
    assert res.status_code in (400, 403)
    assert " at " not in res.text
    assert "Traceback" not in res.text


async def _token(ctx: AppCtx) -> str:
    client = await ctx.store.register_client(
        NewClient(
            client_name="Test",
            redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
            token_endpoint_auth_method="none",
        )
    )
    tokens = await create_grant(
        ctx.store,
        NewGrant(client_id=client.client.client_id, client_name="Test", scope="mcp", resource=ctx.config.resource_url),
    )
    return tokens.access_token


async def test_mcp_body_errors_are_json_rpc_errors(ctx: AppCtx) -> None:
    auth = {"authorization": f"Bearer {await _token(ctx)}"}
    bad = await ctx.http.post("/mcp", headers={**JSON, **auth}, content="{not json")
    assert bad.status_code == 400
    assert bad.json() == {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error."}, "id": None}
    scalar = await ctx.http.post("/mcp", headers={**JSON, **auth}, content='"just a string"')
    assert scalar.status_code == 400
    big = await ctx.http.post("/mcp", headers={**JSON, **auth}, content=b"[" + b" " * (1024 * 1024) + b"]")
    assert big.status_code == 413
    assert big.json() == {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Request body too large."}, "id": None}
    charset = await ctx.http.post(
        "/mcp", headers={"content-type": "application/json; charset=latin1", **auth}, content="{}"
    )
    assert charset.status_code == 415
    utf16 = await ctx.http.post(
        "/mcp", headers={"content-type": "application/json; charset=utf-16", **auth}, content="{}".encode("utf-16")
    )
    assert utf16.status_code == 415  # UTF-8 only
    gzipped = await ctx.http.post("/mcp", headers={**JSON, "content-encoding": "gzip", **auth}, content=b"{}")
    assert gzipped.status_code == 415
    # Without a token the bearer check answers first.
    assert (await ctx.http.post("/mcp", headers=JSON, content="{not json")).status_code == 401


async def test_mcp_answers_2025_requests_with_json(ctx: AppCtx) -> None:
    headers = {**JSON, "accept": "application/json, text/event-stream", "authorization": f"Bearer {await _token(ctx)}"}
    init = await ctx.http.post("/mcp", headers=headers, content=INIT)
    assert init.status_code == 200
    assert init.headers["content-type"].startswith("application/json")
    assert init.json()["result"]["serverInfo"]["name"] == "voice-coding-mcp"
    call = await ctx.http.post(
        "/mcp",
        headers={**headers, "mcp-protocol-version": "2025-11-25"},
        content=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_projects"}}),
    )
    assert call.status_code == 200
    assert call.headers["content-type"].startswith("application/json")
    # Plain text content only (no structuredContent); the SDK always writes isError.
    assert call.json()["result"] == {
        "content": [{"type": "text", "text": "Projects: alpha-app, beta-site."}],
        "isError": False,
    }


async def test_warns_at_startup_when_loopback_redirects_are_allowed(real_tmp: Path) -> None:
    async with start_app(real_tmp, oauth={"allowLoopbackRedirects": True}) as app:
        warnings = [r for r in app.log.records if r["msg"] == "config warning"]
        assert len(warnings) == 1
        assert warnings[0]["level"] == "warn"
        assert "oauth.allowLoopbackRedirects" in warnings[0]["warning"]


async def test_no_config_warning_by_default(ctx: AppCtx) -> None:
    assert not [r for r in ctx.log.records if r["msg"] == "config warning"]
