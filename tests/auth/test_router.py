"""Tests of the OAuth authorization server endpoints."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from typing import Any
from urllib.parse import quote, urlsplit

import httpx2
import pytest

from tests.auth.conftest import MakeCtx
from tests.auth.helpers import (
    CIMD_URL,
    CLAUDE_REDIRECT,
    PUBLIC_URL,
    RESOURCE,
    Ctx,
    authorize_params,
    call_mcp,
    exchange,
    exists,
    full_flow,
    get_authorize,
    get_code,
    location_params,
    pair,
    pkce,
    post_authorize,
    post_form,
    register,
    register_client,
    token_json,
)
from voice_coding_mcp.auth import RateLimitSettings, pairing_path


def challenge() -> str:
    return pkce()[1]


def basic(client_id: str, secret: str) -> str:
    """`encodeURIComponent` on both parts, then base64."""
    raw = f"{quote(client_id, safe='')}:{quote(secret, safe='')}"
    return f"Basic {base64.b64encode(raw.encode()).decode()}"


# --- metadata -----------------------------------------------------------------------------------


async def test_serves_protected_resource_metadata_on_both_paths(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    for path in ["/.well-known/oauth-protected-resource/mcp", "/.well-known/oauth-protected-resource"]:
        res = await ctx.http.get(path)
        assert res.status_code == 200
        assert res.headers["cache-control"] == "public, max-age=300"
        assert res.json() == {
            "resource": RESOURCE,
            "authorization_servers": [PUBLIC_URL],
            "scopes_supported": ["mcp"],
            "bearer_methods_supported": ["header"],
        }


async def test_serves_authorization_server_metadata(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    meta = (await ctx.http.get("/.well-known/oauth-authorization-server")).json()
    assert meta["issuer"] == PUBLIC_URL
    assert meta["authorization_endpoint"] == f"{PUBLIC_URL}/authorize"
    assert meta["token_endpoint"] == f"{PUBLIC_URL}/token"
    assert meta["registration_endpoint"] == f"{PUBLIC_URL}/register"
    assert meta["revocation_endpoint"] == f"{PUBLIC_URL}/revoke"
    assert meta["code_challenge_methods_supported"] == ["S256"]
    assert meta["grant_types_supported"] == ["authorization_code", "refresh_token"]
    assert meta["response_types_supported"] == ["code"]
    assert meta["token_endpoint_auth_methods_supported"] == ["none", "client_secret_post", "client_secret_basic"]
    assert meta["scopes_supported"] == ["mcp"]
    assert meta["client_id_metadata_document_supported"] is True
    assert meta["authorization_response_iss_parameter_supported"] is True


# --- pairing gate -------------------------------------------------------------------------------


async def test_closes_register_and_authorize_without_a_pairing_window(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    reg = await register(ctx, {"redirect_uris": [CLAUDE_REDIRECT]})
    assert reg.status_code == 403
    assert "voice-coding-mcp pair" in reg.json()["error_description"]
    auth = await get_authorize(ctx, authorize_params("x", challenge()))
    assert auth.status_code == 403
    assert "voice-coding-mcp pair" in auth.text
    post = await post_authorize(ctx, {**authorize_params("x", challenge()), "pairing_code": "AAAAA-AAAAA"})
    assert post.status_code == 403


async def test_closes_again_when_the_window_expires(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    res = await register(ctx, {"redirect_uris": [CLAUDE_REDIRECT], "token_endpoint_auth_method": "none"})
    assert res.status_code == 201
    ctx.clock.t += 10 * 60_000
    assert (await register(ctx, {"redirect_uris": [CLAUDE_REDIRECT]})).status_code == 403


async def test_shows_a_consent_page_with_client_name_redirect_host_and_strict_csp(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx, client_name="Claude <b>")
    res = await get_authorize(ctx, authorize_params(client["client_id"], challenge()))
    assert res.status_code == 200
    html = res.text
    assert "Claude &lt;b&gt;" in html
    assert "claude.ai" in html
    assert 'name="pairing_code"' in html
    assert not re.search(r"<script", html, re.IGNORECASE)
    assert res.headers["content-type"] == "text/html; charset=utf-8"
    assert res.headers["content-security-policy"] == (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self' https://claude.ai; "
        "frame-ancestors 'none'; base-uri 'none'"
    )
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["cache-control"] == "no-store"
    assert f'<form method="post" action="{PUBLIC_URL}/authorize">' in html
    assert '<input type="hidden" name="state" value="st-123">' in html


async def test_rerenders_on_a_wrong_code_and_locks_the_address_out_after_five(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    pairing_code = await pair(ctx)
    client = await register_client(ctx)
    params = authorize_params(client["client_id"], challenge())
    for left in range(4, 0, -1):
        res = await post_authorize(ctx, {**params, "pairing_code": "WRONG-WRONG"})
        assert res.status_code == 400
        assert f"{left} attempt" in res.text
    last = await post_authorize(ctx, {**params, "pairing_code": "WRONG-WRONG"})
    assert last.status_code == 403
    assert "from this address" in last.text
    # The window stays open for other addresses; this one is locked even with the right code.
    assert exists(pairing_path(ctx.state_dir))
    assert (await get_authorize(ctx, params)).status_code == 200
    assert (await post_authorize(ctx, {**params, "pairing_code": pairing_code})).status_code == 403


async def test_locks_out_a_whole_ipv6_64_and_an_ipv4_mapped_address_as_its_ipv4(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    pairing_code = await pair(ctx)
    client = await register_client(ctx)
    params = {**authorize_params(client["client_id"], challenge()), "pairing_code": "WRONG-WRONG"}
    # Five wrong codes, each from another address of the same /64.
    for i in range(5):
        res = await ctx.http.post("/authorize", data=params, headers={"x-forwarded-for": f"2001:db8:1:2::{i + 1}"})
        assert res.status_code in (400, 403)
    good = {**params, "pairing_code": pairing_code}
    locked = await ctx.http.post("/authorize", data=good, headers={"x-forwarded-for": "2001:db8:1:2:ffff::9"})
    assert locked.status_code == 403
    assert "from this address" in locked.text
    # IPv4 and its IPv4-mapped IPv6 form share one counter.
    for _ in range(5):
        await ctx.http.post("/authorize", data=params, headers={"x-forwarded-for": "198.51.100.7"})
    mapped = await ctx.http.post("/authorize", data=good, headers={"x-forwarded-for": "::ffff:198.51.100.7"})
    assert "from this address" in mapped.text
    # Another /64 still pairs.
    other = await ctx.http.post("/authorize", data=good, headers={"x-forwarded-for": "2001:db8:1:3::1"})
    assert other.status_code == 302


async def test_sends_referrer_policy_same_origin_on_the_consent_page(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx)
    res = await get_authorize(ctx, authorize_params(client["client_id"], challenge()))
    assert res.status_code == 200
    assert res.headers["referrer-policy"] == "same-origin"


async def test_a_successful_authorization_closes_the_window(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx)
    await get_code(ctx, client["client_id"], challenge())
    assert not exists(pairing_path(ctx.state_dir))
    assert (await register(ctx, {"redirect_uris": [CLAUDE_REDIRECT]})).status_code == 403


# --- dynamic client registration -------------------------------------------------------------------


async def test_accepts_allowed_redirect_uris_and_returns_a_secret_for_confidential_clients(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    res = await register(ctx, {"client_name": "Claude", "redirect_uris": [CLAUDE_REDIRECT]})
    assert res.status_code == 201
    assert res.headers["cache-control"] == "no-store"
    body = res.json()
    assert body["client_id"].startswith("vcm-")
    assert body["token_endpoint_auth_method"] == "client_secret_basic"
    assert isinstance(body["client_secret"], str)
    ctx.secrets.append(body["client_secret"])
    assert list(body) == [
        "client_id",
        "client_id_issued_at",
        "client_secret",
        "client_secret_expires_at",
        "client_name",
        "redirect_uris",
        "token_endpoint_auth_method",
        "grant_types",
        "response_types",
        "scope",
    ]
    pub = (await register(ctx, {"redirect_uris": [CLAUDE_REDIRECT], "token_endpoint_auth_method": "none"})).json()
    assert "client_secret" not in pub
    assert pub["client_name"] == "Unnamed client"


REJECTED: list[tuple[str, dict[str, Any]]] = [
    ("other host", {"redirect_uris": ["https://evil.example/cb"]}),
    ("one bad among good", {"redirect_uris": [CLAUDE_REDIRECT, "https://evil.example/cb"]}),
    ("loopback when not enabled", {"redirect_uris": ["http://localhost:6274/oauth/callback"]}),
    ("fragment", {"redirect_uris": [f"{CLAUDE_REDIRECT}#x"]}),
    ("empty list", {"redirect_uris": []}),
    ("missing", {}),
]


@pytest.mark.parametrize("body", [b for _, b in REJECTED], ids=[n for n, _ in REJECTED])
async def test_rejects_redirect_uris(make_ctx: MakeCtx, body: dict[str, Any]) -> None:
    ctx = make_ctx()
    await pair(ctx)
    res = await register(ctx, body)
    assert res.status_code == 400
    assert res.json()["error"] == "invalid_redirect_uri"


async def test_strips_control_and_format_characters_from_the_client_name(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    res = await register(
        ctx, {"client_name": "\x1b]0;pwned\x07Cl\u202eaude\u200b\r\n", "redirect_uris": [CLAUDE_REDIRECT]}
    )
    assert res.status_code == 201
    assert res.json()["client_name"] == "]0;pwnedClaude"
    only_controls = await register(ctx, {"client_name": "\x1b\u202e", "redirect_uris": [CLAUDE_REDIRECT]})
    assert only_controls.json()["client_name"] == "Unnamed client"


async def test_keeps_a_zwj_emoji_in_the_client_name(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    res = await register(ctx, {"client_name": f"Claude {family}", "redirect_uris": [CLAUDE_REDIRECT]})
    assert res.status_code == 201
    assert res.json()["client_name"] == f"Claude {family}"


async def test_a_lone_surrogate_in_the_client_name_is_stripped_and_the_store_keeps_working(
    make_ctx: MakeCtx,
) -> None:
    ctx = make_ctx()
    await pair(ctx)
    # json.dumps escapes the lone surrogate as \\ud800 (ASCII), which json.loads on the server accepts.
    body = json.dumps({"client_name": "Cl\ud800aude", "redirect_uris": [CLAUDE_REDIRECT]}).encode()
    res = await ctx.http.post("/register", content=body, headers={"content-type": "application/json"})
    assert res.status_code == 201
    assert res.json()["client_name"] == "Claude"
    again = await register(ctx, {"client_name": "Second", "redirect_uris": [CLAUDE_REDIRECT]})
    assert again.status_code == 201


async def test_rejects_unsupported_metadata(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    bodies: list[dict[str, Any]] = [
        {"redirect_uris": [CLAUDE_REDIRECT], "token_endpoint_auth_method": "private_key_jwt"},
        {"redirect_uris": [CLAUDE_REDIRECT], "grant_types": ["client_credentials"]},
        {"redirect_uris": [CLAUDE_REDIRECT], "response_types": ["token"]},
    ]
    for body in bodies:
        res = await register(ctx, body)
        assert res.status_code == 400
        assert res.json()["error"] == "invalid_client_metadata"


async def test_accepts_loopback_redirects_when_allow_loopback_redirects_is_on(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(oauth={"allowLoopbackRedirects": True})
    await pair(ctx)
    for uri in ["http://localhost:6274/oauth/callback", "http://127.0.0.1:33418/callback"]:
        assert (await register(ctx, {"redirect_uris": [uri]})).status_code == 201, uri
    for uri in ["https://localhost:6274/cb", "http://localhost.evil.example/cb", "http://192.168.1.2:80/cb"]:
        assert (await register(ctx, {"redirect_uris": [uri]})).status_code == 400, uri


async def test_answers_malformed_json_with_an_oauth_error(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    res = await ctx.http.post("/register", headers={"content-type": "application/json"}, content=b"{nope")
    assert res.status_code == 400
    assert res.json()["error"] == "invalid_request"


async def test_answers_an_oversized_body_with_413(make_ctx: MakeCtx) -> None:
    """Python addition: the 16 KB body limits of the router's parsers."""
    ctx = make_ctx()
    await pair(ctx)
    big = json.dumps({"redirect_uris": [CLAUDE_REDIRECT], "client_name": "x" * 17_000}).encode()
    res = await ctx.http.post("/register", headers={"content-type": "application/json"}, content=big)
    assert res.status_code == 413
    assert res.json() == {"error": "invalid_request", "error_description": "request body too large"}
    res = await post_form(ctx, "/token", {"grant_type": "refresh_token", "refresh_token": "x" * 17_000})
    assert res.status_code == 413


# --- authorization request validation ----------------------------------------------------------------


async def test_renders_errors_before_client_and_redirect_uri_are_validated(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx)
    c = challenge()
    for params in [
        authorize_params("", c, client_id=None),
        authorize_params("vcm-unknown", c),
        authorize_params(client["client_id"], c, redirect_uri=None),
        authorize_params(client["client_id"], c, redirect_uri="https://claude.com/api/mcp/auth_callback"),
        authorize_params("https://evil.example/meta", c),
    ]:
        res = await get_authorize(ctx, params)
        assert res.status_code == 400, params
        assert "location" not in res.headers
        assert "text/html" in res.headers["content-type"]


async def test_redirects_errors_once_the_redirect_uri_is_trusted(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx)
    c = challenge()
    cases: list[tuple[dict[str, str | None], str]] = [
        ({"response_type": "token"}, "unsupported_response_type"),
        ({"code_challenge_method": "plain"}, "invalid_request"),
        ({"code_challenge_method": None}, "invalid_request"),
        ({"code_challenge": "short"}, "invalid_request"),
        ({"resource": "https://other.example/mcp"}, "invalid_target"),
    ]
    for extra, error in cases:
        res = await get_authorize(ctx, authorize_params(client["client_id"], c, **extra))
        assert res.status_code == 302, extra
        location = urlsplit(res.headers["location"])
        assert f"{location.scheme}://{location.netloc}{location.path}" == CLAUDE_REDIRECT
        params = location_params(res)
        assert params["error"] == error
        assert params["state"] == "st-123"
        assert params["iss"] == PUBLIC_URL


async def test_redirect_query_is_form_urlencoded(make_ctx: MakeCtx) -> None:
    """The Location header query is application/x-www-form-urlencoded (`urlencode`), byte for byte."""
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx)
    res = await get_authorize(
        ctx, authorize_params(client["client_id"], challenge(), response_type="x", state="a b~*'")
    )
    assert res.headers["location"] == (
        f"{CLAUDE_REDIRECT}?error=unsupported_response_type&error_description=response_type+must+be+code"
        "&state=a+b~%2A%27&iss=https%3A%2F%2Fvcm.example.com"
    )


async def test_accepts_a_request_without_resource_and_with_any_scope(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx)
    res = await get_authorize(
        ctx, authorize_params(client["client_id"], challenge(), resource=None, scope="openid admin")
    )
    assert res.status_code == 200


# --- authorization code flow --------------------------------------------------------------------


async def test_issues_a_code_with_state_and_iss_then_tokens_that_work_on_mcp(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx)
    verifier, c = pkce()
    pairing_code = await pair(ctx)
    res = await post_authorize(ctx, {**authorize_params(client["client_id"], c), "pairing_code": pairing_code.lower()})
    assert res.status_code == 302
    params = location_params(res)
    assert params["state"] == "st-123"
    assert params["iss"] == PUBLIC_URL
    code = params["code"]
    ctx.secrets.append(code)
    token_res = await exchange(ctx, client["client_id"], code, verifier)
    assert token_res.headers["cache-control"] == "no-store"
    assert token_res.headers["pragma"] == "no-cache"
    tokens = token_json(ctx, token_res)
    assert tokens["token_type"] == "Bearer"
    assert tokens["expires_in"] == 24 * 3600
    assert tokens["scope"] == "mcp"
    mcp = await call_mcp(ctx, tokens["access_token"])
    assert mcp.status_code == 200
    assert mcp.json() == {
        "clientId": client["client_id"],
        "scopes": ["mcp"],
        "resource": RESOURCE,
        "expiresAt": ctx.clock.t // 1000 + 24 * 3600,
    }


async def test_works_without_resource_in_the_token_request(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx)
    verifier, c = pkce()
    code = await get_code(ctx, client["client_id"], c, resource=None)
    res = await post_form(
        ctx,
        "/token",
        {
            "grant_type": "authorization_code",
            "client_id": client["client_id"],
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": CLAUDE_REDIRECT,
        },
    )
    token_json(ctx, res)


async def code_for(ctx: Ctx) -> tuple[str, str, str]:
    """(client id, verifier, code)."""
    await pair(ctx)
    client = await register_client(ctx)
    verifier, c = pkce()
    code = await get_code(ctx, client["client_id"], c)
    return client["client_id"], verifier, code


def assert_grant_error(res: httpx2.Response, error: str = "invalid_grant") -> None:
    assert res.status_code == 400
    assert res.headers["cache-control"] == "no-store"
    body = res.json()
    assert body["error"] == error
    assert isinstance(body["error_description"], str)


async def test_rejects_a_wrong_verifier_and_burns_the_code(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, verifier, code = await code_for(ctx)
    assert_grant_error(await exchange(ctx, client_id, code, pkce()[0]))
    assert_grant_error(await exchange(ctx, client_id, code, verifier))


async def test_rejects_a_malformed_or_missing_verifier(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, _, code = await code_for(ctx)
    assert_grant_error(await exchange(ctx, client_id, code, "short"))


async def test_rejects_code_reuse_and_revokes_the_tokens_issued_from_it(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, verifier, code = await code_for(ctx)
    tokens = token_json(ctx, await exchange(ctx, client_id, code, verifier))
    assert_grant_error(await exchange(ctx, client_id, code, verifier))
    assert (await call_mcp(ctx, tokens["access_token"])).status_code == 401


async def test_concurrent_redemptions_of_one_code_never_leave_a_live_grant(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, verifier, code = await code_for(ctx)
    results = await asyncio.gather(*(exchange(ctx, client_id, code, verifier) for _ in range(2)))
    assert sorted(r.status_code for r in results) == [200, 400]
    ok = next(r for r in results if r.status_code == 200)
    assert (await call_mcp(ctx, ok.json()["access_token"])).status_code == 401


async def test_rejects_an_expired_code(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, verifier, code = await code_for(ctx)
    ctx.clock.t += 61_000
    assert_grant_error(await exchange(ctx, client_id, code, verifier))


async def test_rejects_a_redirect_uri_mismatch(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, verifier, code = await code_for(ctx)
    res = await exchange(ctx, client_id, code, verifier, redirect_uri="https://claude.com/api/mcp/auth_callback")
    assert_grant_error(res)


async def test_rejects_a_client_mismatch(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    _, verifier, code = await code_for(ctx)
    await pair(ctx)
    other = await register_client(ctx)
    assert_grant_error(await exchange(ctx, other["client_id"], code, verifier))


async def test_rejects_a_resource_mismatch_with_invalid_target(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, verifier, code = await code_for(ctx)
    res = await exchange(ctx, client_id, code, verifier, resource="https://other.example/mcp")
    assert_grant_error(res, "invalid_target")


async def test_rejects_unknown_clients_unknown_codes_and_unsupported_grants(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, verifier, _ = await code_for(ctx)
    unknown = await exchange(ctx, "vcm-nope", "x", verifier)
    assert unknown.status_code == 401
    assert unknown.json()["error"] == "invalid_client"
    assert_grant_error(await exchange(ctx, client_id, "not-a-code", verifier))
    grant = await post_form(ctx, "/token", {"grant_type": "client_credentials", "client_id": client_id})
    assert grant.json()["error"] == "unsupported_grant_type"


async def test_authenticates_confidential_clients_with_basic_or_post(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx, token_endpoint_auth_method="client_secret_basic")
    secret = client["client_secret"]

    async def flow(extra: dict[str, str], headers: dict[str, str] | None = None) -> httpx2.Response:
        verifier, c = pkce()
        code = await get_code(ctx, client["client_id"], c)
        return await post_form(
            ctx,
            "/token",
            {
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": CLAUDE_REDIRECT,
                **extra,
            },
            headers,
        )

    token_json(ctx, await flow({}, {"authorization": basic(client["client_id"], secret)}))
    token_json(ctx, await flow({"client_id": client["client_id"], "client_secret": secret}))
    wrong = await flow({"client_id": client["client_id"], "client_secret": "wrong"})
    assert wrong.status_code == 401
    assert wrong.json()["error"] == "invalid_client"
    missing = await flow({"client_id": client["client_id"]})
    assert missing.status_code == 401


async def test_accepts_either_secret_method_regardless_of_the_registered_one(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    client = await register_client(ctx, token_endpoint_auth_method="client_secret_post")
    secret = client["client_secret"]
    verifier, c = pkce()
    code = await get_code(ctx, client["client_id"], c)
    tokens = token_json(
        ctx,
        await post_form(
            ctx,
            "/token",
            {
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": CLAUDE_REDIRECT,
            },
            {"authorization": basic(client["client_id"], secret)},
        ),
    )
    refused = await post_form(
        ctx,
        "/token",
        {"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]},
        {"authorization": basic(client["client_id"], "wrong")},
    )
    assert refused.status_code == 401
    assert refused.headers.get("www-authenticate", "").startswith("Basic")


async def test_rejects_a_malformed_basic_header(make_ctx: MakeCtx) -> None:
    """Python addition: decodeURIComponent failures give invalid_request."""
    ctx = make_ctx()
    header = f"Basic {base64.b64encode(b'%E0%A4%A:x').decode()}"
    res = await post_form(
        ctx, "/token", {"grant_type": "refresh_token", "refresh_token": "x"}, {"authorization": header}
    )
    assert res.status_code == 400
    assert res.json() == {"error": "invalid_request", "error_description": "malformed Basic authorization header"}


@pytest.mark.parametrize(
    "encoded",
    [
        "!!!not-base64",
        base64.b64encode(b"client:secret").decode().rstrip("="),  # missing padding
        base64.urlsafe_b64encode(b"\xfb\xff:secret").decode(),  # URL-safe alphabet
        "Y2xp ZW50OnNlY3JldA==",  # white space inside
        base64.b64encode(b"client:secret").decode() + "junk",
    ],
)
async def test_rejects_basic_credentials_that_are_not_strict_base64(make_ctx: MakeCtx, encoded: str) -> None:
    ctx = make_ctx()
    res = await post_form(
        ctx, "/token", {"grant_type": "refresh_token", "refresh_token": "x"}, {"authorization": f"Basic {encoded}"}
    )
    assert res.status_code == 401
    assert res.json() == {"error": "invalid_client", "error_description": "malformed Basic authorization header"}


# --- refresh tokens -------------------------------------------------------------------------------


async def test_rotates_refresh_tokens_and_keeps_the_grant_working(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, tokens = await full_flow(ctx)
    refreshed = token_json(
        ctx,
        await post_form(
            ctx,
            "/token",
            {
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": tokens["refresh_token"],
                "resource": RESOURCE,
            },
        ),
    )
    assert refreshed["refresh_token"] != tokens["refresh_token"]
    assert refreshed["access_token"] != tokens["access_token"]
    assert (await call_mcp(ctx, refreshed["access_token"])).status_code == 200
    assert (await call_mcp(ctx, tokens["access_token"])).status_code == 401


async def test_refresh_works_outside_the_pairing_window(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, tokens = await full_flow(ctx)
    assert not exists(pairing_path(ctx.state_dir))
    res = await post_form(
        ctx, "/token", {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": tokens["refresh_token"]}
    )
    assert res.status_code == 200
    assert token_json(ctx, res)["access_token"]


async def test_reuse_of_a_rotated_refresh_token_after_the_grace_revokes_the_family(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, tokens = await full_flow(ctx)

    async def refresh(token: str) -> httpx2.Response:
        return await post_form(
            ctx, "/token", {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": token}
        )

    second = token_json(ctx, await refresh(tokens["refresh_token"]))
    ctx.clock.t += 60_001
    reuse = await refresh(tokens["refresh_token"])
    assert reuse.status_code == 400
    assert reuse.json()["error"] == "invalid_grant"
    assert (await refresh(second["refresh_token"])).status_code == 400
    assert (await call_mcp(ctx, second["access_token"])).status_code == 401


async def test_answers_parallel_refreshes_with_the_same_token_with_the_same_response(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, tokens = await full_flow(ctx)

    async def refresh() -> httpx2.Response:
        return await post_form(
            ctx,
            "/token",
            {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": tokens["refresh_token"]},
        )

    a, b = await asyncio.gather(refresh(), refresh())
    first = token_json(ctx, a)
    second = token_json(ctx, b)
    assert second == first
    ctx.clock.t += 59_000
    assert token_json(ctx, await refresh()) == first
    assert (await call_mcp(ctx, first["access_token"])).status_code == 200


async def test_does_not_replay_a_cached_response_once_its_tokens_were_rotated_again(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, tokens = await full_flow(ctx)

    async def refresh(token: str) -> httpx2.Response:
        return await post_form(
            ctx, "/token", {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": token}
        )

    second = token_json(ctx, await refresh(tokens["refresh_token"]))
    third = token_json(ctx, await refresh(second["refresh_token"]))
    reuse = await refresh(tokens["refresh_token"])
    assert reuse.status_code == 400
    assert (await call_mcp(ctx, third["access_token"])).status_code == 401


async def test_rejects_refresh_by_another_client_and_after_expiry(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    client_id, tokens = await full_flow(ctx)
    await pair(ctx)
    other = await register_client(ctx)
    res = await post_form(
        ctx,
        "/token",
        {"grant_type": "refresh_token", "client_id": other["client_id"], "refresh_token": tokens["refresh_token"]},
    )
    assert res.json()["error"] == "invalid_grant"
    ctx.clock.t += 91 * 86_400_000
    expired = await post_form(
        ctx, "/token", {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": tokens["refresh_token"]}
    )
    assert expired.json()["error"] == "invalid_grant"


# --- revocation -----------------------------------------------------------------------------------


async def test_revokes_by_access_or_refresh_token_and_always_answers_200(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    a_client, a_tokens = await full_flow(ctx)
    res = await post_form(ctx, "/revoke", {"token": a_tokens["refresh_token"], "client_id": a_client})
    assert res.status_code == 200
    assert res.content == b""
    assert (await call_mcp(ctx, a_tokens["access_token"])).status_code == 401
    assert (await post_form(ctx, "/revoke", {"token": "unknown"})).status_code == 200

    b_client, b_tokens = await full_flow(ctx)
    assert (await post_form(ctx, "/revoke", {"token": b_tokens["access_token"]})).status_code == 200
    refresh = await post_form(
        ctx,
        "/token",
        {"grant_type": "refresh_token", "client_id": b_client, "refresh_token": b_tokens["refresh_token"]},
    )
    assert refresh.status_code == 400


async def test_does_not_revoke_another_clients_token(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    _, a_tokens = await full_flow(ctx)
    b_client, _ = await full_flow(ctx)
    res = await post_form(ctx, "/revoke", {"token": a_tokens["access_token"], "client_id": b_client})
    assert res.status_code == 200
    assert (await call_mcp(ctx, a_tokens["access_token"])).status_code == 200


# --- CIMD clients ---------------------------------------------------------------------------------

DOC: dict[str, Any] = {
    "client_id": CIMD_URL,
    "client_name": "Claude",
    "redirect_uris": [CLAUDE_REDIRECT],
    "token_endpoint_auth_method": "none",
}


class CimdStub:
    def __init__(self, body: Any = DOC) -> None:
        self.calls: list[str] = []
        self._body = body
        self.client = httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.calls.append(str(request.url))
        return httpx2.Response(
            200, content=json.dumps(self._body).encode(), headers={"content-type": "application/json"}
        )


async def test_runs_the_full_flow_with_claudes_published_identity(make_ctx: MakeCtx) -> None:
    stub = CimdStub()
    ctx = make_ctx(http_client=stub.client)
    await pair(ctx)
    page = await get_authorize(ctx, authorize_params(CIMD_URL, challenge()))
    assert page.status_code == 200
    assert "Claude" in page.text
    verifier, c = pkce()
    code = await get_code(ctx, CIMD_URL, c)
    tokens = token_json(ctx, await exchange(ctx, CIMD_URL, code, verifier))
    mcp = (await call_mcp(ctx, tokens["access_token"])).json()
    assert mcp["clientId"] == CIMD_URL
    assert stub.calls == [CIMD_URL]
    status = await ctx.store.status()
    assert status.clients[0].client_name == "Claude"


async def test_never_fetches_a_non_allowlisted_client_id_url(make_ctx: MakeCtx) -> None:
    stub = CimdStub()
    ctx = make_ctx(http_client=stub.client)
    await pair(ctx)
    res = await get_authorize(ctx, authorize_params("https://evil.example/meta", challenge()))
    assert res.status_code == 400
    assert stub.calls == []


async def test_rejects_a_redirect_uri_not_listed_in_the_document(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(http_client=CimdStub().client)
    await pair(ctx)
    params = authorize_params(CIMD_URL, challenge(), redirect_uri="https://claude.com/api/mcp/auth_callback")
    res = await get_authorize(ctx, params)
    assert res.status_code == 400
    assert "location" not in res.headers


async def test_rejects_a_document_redirect_uri_that_the_server_config_does_not_allow(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(http_client=CimdStub({**DOC, "redirect_uris": ["https://evil.example/cb"]}).client)
    await pair(ctx)
    res = await get_authorize(ctx, authorize_params(CIMD_URL, challenge(), redirect_uri="https://evil.example/cb"))
    assert res.status_code == 400
    assert "location" not in res.headers


async def test_rejects_an_invalid_document_on_the_authorize_page(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(http_client=CimdStub({**DOC, "client_id": "https://evil.example/meta"}).client)
    await pair(ctx)
    res = await get_authorize(ctx, authorize_params(CIMD_URL, challenge()))
    assert res.status_code == 400
    assert "client_id" in res.text


# --- rate limit -----------------------------------------------------------------------------------


async def test_limits_authorize_per_ip(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(rate_limit=RateLimitSettings(max=3, window_ms=60_000))
    statuses = [(await get_authorize(ctx, authorize_params("x", challenge()))).status_code for _ in range(4)]
    assert statuses == [403, 403, 403, 429]
    limited = await get_authorize(ctx, authorize_params("x", challenge()))
    assert limited.headers["retry-after"] == "60"
    assert limited.headers["cache-control"] == "no-store"
    assert limited.json() == {
        "error": "temporarily_unavailable",
        "error_description": "Too many requests, try again later.",
    }
    post = [
        (await post_authorize(ctx, {**authorize_params("x", challenge()), "pairing_code": "AAAAA-AAAAA"})).status_code
        for _ in range(4)
    ]
    assert post == [403, 403, 403, 429]
    ctx.clock.t += 60_001
    assert (await get_authorize(ctx, authorize_params("x", challenge()))).status_code == 403


async def test_limits_register_per_ip_with_its_own_counter(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(rate_limit=RateLimitSettings(max=3, window_ms=60_000))
    # /authorize uses up its own counters first; /register still gets its full allowance.
    for _ in range(4):
        await get_authorize(ctx, authorize_params("x", challenge()))
    register_statuses = [(await register(ctx, {"redirect_uris": [CLAUDE_REDIRECT]})).status_code for _ in range(5)]
    assert register_statuses == [403, 403, 403, 429, 429]
    limited = await register(ctx, {"redirect_uris": [CLAUDE_REDIRECT]})
    assert limited.headers["retry-after"] == "60"
    assert limited.json()["error"] == "temporarily_unavailable"
    entries = [json.loads(line) for line in ctx.log_lines]
    assert {"msg": "rate limited", "endpoint": "register", "ip": "127.0.0.1"}.items() <= next(
        e for e in entries if e.get("endpoint") == "register"
    ).items()
    ctx.clock.t += 60_001
    await pair(ctx)
    assert (await register(ctx, {"redirect_uris": [CLAUDE_REDIRECT]})).status_code == 201


async def test_limits_ipv6_clients_by_their_64(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(rate_limit=RateLimitSettings(max=3, window_ms=60_000))

    async def status(ip: str) -> int:
        res = await ctx.http.get(
            "/authorize", params=authorize_params("x", challenge()), headers={"x-forwarded-for": ip}
        )
        return res.status_code

    statuses = [await status(f"2001:db8:a:b:{i}::1") for i in range(4)]
    assert statuses == [403, 403, 403, 429]
    assert await status("[2001:db8:a:c::1]") == 403
    assert [await status(ip) for ip in ["203.0.113.9", "::ffff:203.0.113.9", "203.0.113.9"]] == [403, 403, 403]
    assert await status("::ffff:203.0.113.9") == 429
    entries = [json.loads(line) for line in ctx.log_lines]
    assert any(e.get("msg") == "rate limited" and e.get("ip") == "2001:db8:a:b:3::1" for e in entries)


async def test_never_limits_token(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(rate_limit=RateLimitSettings(max=3, window_ms=60_000))
    statuses = [(await post_form(ctx, "/token", {"grant_type": "refresh_token"})).status_code for _ in range(10)]
    assert statuses == [401] * 10


# --- logs -----------------------------------------------------------------------------------------


async def test_logs_never_contain_issued_tokens_codes_secrets_or_pairing_codes(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(oauth={"allowLoopbackRedirects": True})
    client_id, tokens = await full_flow(ctx)
    # wrong pairing code, failing exchange, refresh, reuse, revoke, bearer failure
    await pair(ctx)
    confidential = await register_client(ctx, token_endpoint_auth_method="client_secret_post")
    secret = confidential["client_secret"]
    wrong = {**authorize_params(confidential["client_id"], challenge()), "pairing_code": "WRONG-WRONG"}
    await post_authorize(ctx, wrong)
    verifier, c = pkce()
    code = await get_code(ctx, confidential["client_id"], c)
    await exchange(ctx, confidential["client_id"], code, pkce()[0], client_secret=secret)
    await exchange(ctx, confidential["client_id"], code, verifier, client_secret=secret)
    refresh_body = {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": tokens["refresh_token"]}
    refreshed = token_json(ctx, await post_form(ctx, "/token", refresh_body))
    await post_form(ctx, "/token", refresh_body)
    await call_mcp(ctx, refreshed["access_token"])
    await call_mcp(ctx, tokens["access_token"])
    await post_form(ctx, "/revoke", {"token": refreshed["refresh_token"]})

    assert len(ctx.log_lines) > 5
    assert len(ctx.secrets) >= 8
    text = "\n".join(ctx.log_lines)
    for s in ctx.secrets:
        assert s not in text, f"log leaked a secret (len {len(s)})"
    assert "sha256=" in text


# --- client cap -----------------------------------------------------------------------------------


async def test_register_evicts_the_oldest_client_but_keeps_one_with_a_pending_code(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    await pair(ctx)
    owner = await register_client(ctx)
    await get_code(ctx, owner["client_id"], challenge())  # the owner's pairing in progress
    await pair(ctx)
    flood: list[str] = []
    for _ in range(19):
        ctx.clock.t += 1000
        flood.append((await register_client(ctx))["client_id"])
    # The cap is full: a new registration pushes out the oldest client without a pending code,
    # never the owner's client that is still pairing.
    ctx.clock.t += 1000
    await register_client(ctx)
    assert await ctx.store.get_client(owner["client_id"])
    assert await ctx.store.get_client(flood[0]) is None
    assert await ctx.store.get_client(flood[1])
