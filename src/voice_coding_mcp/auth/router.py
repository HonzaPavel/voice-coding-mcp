"""The OAuth 2.1 authorization server endpoints, as a list of Starlette routes that
`http.create_app` mounts (and the auth tests mount alone):

- `GET /.well-known/oauth-protected-resource/mcp` and `/.well-known/oauth-protected-resource`
- `GET /.well-known/oauth-authorization-server`
- `POST /register` (JSON body, pairing window required, client cap; rate limited per IP)
- `GET` and `POST /authorize` (consent page, pairing code; rate limited per IP)
- `POST /token`, `POST /revoke` (form-urlencoded, never rate limited)

Body limits: JSON 16 KB, forms 16 KB (400, 413 or 415 as OAuth JSON errors). Client IPs come from
`web.client_ip`. Tests check the page HTML, CSP and headers exactly.
"""

from __future__ import annotations

import base64
import binascii
import html
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

import httpx2
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Route

from ..clock import MINUTE_MS, Clock, now_ms
from ..config import LOOPBACK_REDIRECT_HOSTS, Config, OAuthConfig
from ..log import Logger, fingerprint
from ..urls import host_of, parse_url
from ..web import BodyError, client_ip, content_type, json_response, read_capped, read_json_body
from .cimd import CimdResolver, create_cimd_resolver, is_cimd_client_id
from .crypto import safe_equal, sha256_base64url, sha256_hex
from .pairing import get_pairing, verify_pairing_code
from .rate_limit import RateLimiter, RateLimitOptions
from .store import ClientLimitError, NewClient, NewCode, OAuthStore, StoredCode, TokenEndpointAuthMethod
from .text import normalize_client_name

SCOPE = "mcp"
PAIR_HINT = "Run `voice-coding-mcp pair` on the server to open a pairing window, then try again."


@dataclass(frozen=True)
class RateLimitSettings:
    max: int
    window_ms: int


DEFAULT_AUTHORIZE_RATE_LIMIT = RateLimitSettings(max=30, window_ms=MINUTE_MS)
"""Per-IP limit of each rate-limited endpoint (GET /authorize, POST /authorize, POST /register)."""


@dataclass(frozen=True)
class AuthRouterOptions:
    config: Config
    store: OAuthStore
    state_dir: str
    logger: Logger
    now: Clock | None = None
    http_client: httpx2.AsyncClient | None = None
    """Used only for allowlisted CIMD URLs (tests inject a MockTransport client)."""
    rate_limit: RateLimitSettings | None = DEFAULT_AUTHORIZE_RATE_LIMIT
    """Per-IP limit for /authorize (GET and POST) and /register, each with its own counters; None
    disables it."""
    cimd: CimdResolver | None = None
    """Shared resolver (e.g. to keep its cache); created from config otherwise."""


def resource_metadata_url(config: Config) -> str:
    """The canonical URL of the protected resource metadata for /mcp."""
    return f"{config.issuer}/.well-known/oauth-protected-resource/mcp"


_CODE_CHALLENGE = re.compile(r"[A-Za-z0-9_-]{43}", re.ASCII)
_CODE_VERIFIER = re.compile(r"[A-Za-z0-9._~-]{43,128}", re.ASCII)
FORM_LIMIT_BYTES = 16 * 1024
"""Body limit of form-urlencoded requests (/authorize POST, /token, /revoke)."""
JSON_LIMIT_BYTES = 16 * 1024
"""Body limit of JSON requests (/register)."""
FORM_PARAMETER_LIMIT = 1000
"""At most this many form parameters; more is a 413."""


def is_allowed_redirect_uri(uri: object, oauth: OAuthConfig) -> bool:
    """True when a redirect URI may be registered or used under the current config: no fragment, no
    credentials, listed in `oauth.redirect_uris`, or loopback http when `allow_loopback_redirects`."""
    if not isinstance(uri, str) or "#" in uri:
        return False
    url = parse_url(uri)
    if url is None or url.username or url.password:
        return False
    if uri in oauth.redirect_uris:
        return True
    return oauth.allow_loopback_redirects and url.scheme == "http" and url.hostname in LOOPBACK_REDIRECT_HOSTS


# --- small helpers ------------------------------------------------------------------------------


def _str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _escape_html(text: str) -> str:
    return html.escape(text, quote=True)


def _host_of(uri: str) -> str:
    return host_of(uri, uri)


def _oauth_error(status: int, error: str, description: str, headers: Mapping[str, str] | None = None) -> Response:
    return json_response(
        status, {"error": error, "error_description": description}, {"Pragma": "no-cache", **(headers or {})}
    )


_PAGE_STYLE = (
    "body{font-family:system-ui,sans-serif;max-width:28rem;margin:3rem auto;padding:0 1rem;line-height:1.5;color:#222}"
    "h1{font-size:1.3rem}input{font-size:1.2rem;padding:.4rem;width:100%;box-sizing:border-box;letter-spacing:.1em}"
    "button{margin-top:1rem;font-size:1.1rem;padding:.5rem 1.2rem}.error{color:#a00}.muted{color:#666;font-size:.9rem}"
)


def _page(status: int, title: str, body_html: str, form_origin: str | None = None) -> Response:
    form_action = f"'self' {form_origin}" if form_origin else "'self'"
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_escape_html(title)}</title><style>{_PAGE_STYLE}</style></head><body>{body_html}</body></html>"
    )
    return Response(
        page,
        status_code=status,
        headers={
            "Content-Type": "text/html; charset=utf-8",
            "Content-Security-Policy": f"default-src 'none'; style-src 'unsafe-inline'; form-action {form_action}; frame-ancestors 'none'; base-uri 'none'",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            # Not no-referrer: with it, browsers send `Origin: null` on the consent form POST and the
            # Origin check refuses it. same-origin still sends no referrer to claude.ai.
            "Referrer-Policy": "same-origin",
            "Cache-Control": "no-store",
        },
    )


def _error_page(status: int, message: str) -> Response:
    return _page(status, "voice-coding-mcp", f'<h1>Cannot connect</h1><p class="error">{_escape_html(message)}</p>')


# --- query strings and form bodies ----------------------------------------------------------------


def _parse_simple(text: str, max_fields: int | None = None) -> dict[str, str | list[str]]:
    """Flat query or form parsing: a repeated key becomes a list, so it is not taken as a string
    parameter. More than `max_fields` parameters raise ValueError."""
    out: dict[str, str | list[str]] = {}
    for name, value in parse_qsl(text, keep_blank_values=True, max_num_fields=max_fields):
        existing = out.get(name)
        if existing is None:
            out[name] = value
        elif isinstance(existing, list):
            existing.append(value)
        else:
            out[name] = [existing, value]
    return out


def _with_query_params(redirect_uri: str, params: list[tuple[str, str]]) -> str:
    """`redirect_uri` with each pair set in its query: an existing key keeps its position and gets
    the new value (later duplicates are dropped), new keys are appended. The query is re-encoded as
    application/x-www-form-urlencoded."""
    parts = urlsplit(redirect_uri)
    updates = dict(params)
    pairs: list[tuple[str, str]] = []
    done: set[str] = set()
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key not in updates:
            pairs.append((key, value))
        elif key not in done:
            pairs.append((key, updates[key]))
            done.add(key)
    pairs += [(key, value) for key, value in updates.items() if key not in done]
    return parts._replace(query=urlencode(pairs)).geturl()


_MALFORMED_ESCAPE = re.compile("%(?![0-9A-Fa-f]{2})")


def _unquote_strict(text: str) -> str:
    """Percent-decodes `text` (`+` stays as is). Raises ValueError on a malformed `%` escape or when
    the escapes are not valid UTF-8."""
    if _MALFORMED_ESCAPE.search(text):
        raise ValueError("malformed percent escape")
    return unquote(text, errors="strict")


def _form_decode(text: str) -> str:
    """Decodes one form-urlencoded component (`+` is a space), strictly as `_unquote_strict`."""
    return _unquote_strict(text.replace("+", " "))


# --- request bodies ----------------------------------------------------------------------------


async def _form_body(request: Request) -> dict[str, str | list[str]]:
    """A form-urlencoded body (UTF-8, at most FORM_LIMIT_BYTES); other content types give an empty body."""
    media, charset = content_type(request.headers)
    if media != "application/x-www-form-urlencoded":
        return {}
    if charset is not None and charset != "utf-8":
        raise BodyError(415)
    if request.headers.get("content-encoding", "identity").strip().lower() != "identity":
        raise BodyError(415)
    raw = await read_capped(request.headers, request.stream(), FORM_LIMIT_BYTES)
    try:
        return _parse_simple(raw.decode("utf-8", "replace"), FORM_PARAMETER_LIMIT)
    except ValueError:
        raise BodyError(413) from None


async def _json_body(request: Request) -> object:
    """A JSON body (at most JSON_LIMIT_BYTES, objects and arrays only); other content types give an
    empty body."""
    if content_type(request.headers)[0] != "application/json":
        return {}
    _, body = await read_json_body(request.headers, request.stream(), JSON_LIMIT_BYTES)
    return body


def _as_dict(body: object) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


# --- authorization request ------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedClient:
    """A client as the authorization endpoint sees it, DCR or CIMD."""

    client_id: str
    client_name: str
    redirect_uris: tuple[str, ...]


@dataclass(frozen=True)
class _AuthorizeParams:
    client_id: str | None
    redirect_uri: str | None
    response_type: str | None
    code_challenge: str | None
    code_challenge_method: str | None
    state: str | None
    resource: str | None
    scope: str | None


def _authorize_params(source: Mapping[str, object]) -> _AuthorizeParams:
    return _AuthorizeParams(
        client_id=_str(source.get("client_id")),
        redirect_uri=_str(source.get("redirect_uri")),
        response_type=_str(source.get("response_type")),
        code_challenge=_str(source.get("code_challenge")),
        code_challenge_method=_str(source.get("code_challenge_method")),
        state=_str(source.get("state")),
        resource=_str(source.get("resource")),
        scope=_str(source.get("scope")),
    )


@dataclass(frozen=True)
class _CheckOk:
    client: _ResolvedClient
    redirect_uri: str
    code_challenge: str
    state: str | None


Endpoint = Callable[[Request], Awaitable[Response]]


def create_auth_routes(options: AuthRouterOptions) -> list[BaseRoute]:
    """The routes above, bound to `options`."""
    config = options.config
    store = options.store
    state_dir = options.state_dir
    logger = options.logger.child({"component": "oauth"})
    now: Clock = options.now or now_ms
    cimd = options.cimd or create_cimd_resolver(config.oauth.cimd_client_ids, client=options.http_client, now=now)
    issuer = config.issuer

    def limiter(endpoint: str) -> RateLimiter | None:
        if options.rate_limit is None:
            return None
        return RateLimiter(
            RateLimitOptions(
                max=options.rate_limit.max,
                window_ms=options.rate_limit.window_ms,
                now=now,
                on_limited=lambda ip: logger.warn("rate limited", {"endpoint": endpoint, "ip": ip}),
            )
        )

    # Separate counters: GET /authorize, POST /authorize and POST /register each have their own.
    get_authorize_limit = limiter("authorize")
    post_authorize_limit = limiter("authorize")
    register_limit = limiter("register")

    def rate_limited(limit: RateLimiter | None, ip: str) -> Response | None:
        """The 429 response when `ip` is over `limit`, else None (counts the request)."""
        return limit.check(ip) if limit is not None else None

    def guarded(endpoint: Endpoint) -> Endpoint:
        """The router's error handler: body errors as OAuth errors, anything else as 500."""

        async def wrapper(request: Request) -> Response:
            try:
                return await endpoint(request)
            except BodyError as err:
                return _oauth_error(
                    err.status,
                    "invalid_request",
                    "request body too large" if err.status == 413 else "malformed request body",
                )
            except Exception as err:
                logger.error("oauth endpoint failed", {"path": request.url.path, "err": err})
                return _oauth_error(500, "server_error", "internal error")

        return wrapper

    async def pairing_open() -> bool:
        return (await get_pairing(state_dir, now())) is not None

    async def resolve_client(client_id: str) -> _ResolvedClient | str:
        """Resolves a DCR client or fetches an allowlisted CIMD document (authorization endpoint).
        Returns the client or the message for the error page."""
        if is_cimd_client_id(client_id):
            if not cimd.is_allowed(client_id):
                return "This client_id URL is not on the allowlist (oauth.cimdClientIds)."
            try:
                doc = await cimd.resolve(client_id)
            except Exception as err:
                logger.warn("cimd rejected", {"clientId": client_id, "reason": str(err)})
                return f"The client metadata document was rejected: {err}."
            return _ResolvedClient(doc.client_id, doc.client_name, doc.redirect_uris)
        client = await store.get_client(client_id)
        if client is None:
            return "Unknown client. Remove the connector in the app and connect again."
        return _ResolvedClient(client.client_id, client.client_name, tuple(client.redirect_uris))

    async def check_authorize(p: _AuthorizeParams) -> _CheckOk | Response:
        """The checked request, or the error page (client or redirect URI unusable) or the error
        redirect (anything else)."""
        if not p.client_id:
            return _error_page(400, "The request has no client_id.")
        client = await resolve_client(p.client_id)
        if isinstance(client, str):
            return _error_page(400, client)
        if not p.redirect_uri:
            return _error_page(400, "The request has no redirect_uri.")
        if p.redirect_uri not in client.redirect_uris or not is_allowed_redirect_uri(p.redirect_uri, config.oauth):
            return _error_page(400, "The redirect_uri is not registered for this client or not allowed by the server.")
        redirect_uri = p.redirect_uri

        def redirect(error: str, description: str) -> Response:
            return redirect_to(redirect_uri, [("error", error), ("error_description", description), ("state", p.state)])

        if p.response_type != "code":
            return redirect("unsupported_response_type", "response_type must be code")
        if p.code_challenge_method != "S256":
            return redirect("invalid_request", "PKCE with code_challenge_method S256 is required")
        if not p.code_challenge or not _CODE_CHALLENGE.fullmatch(p.code_challenge):
            return redirect("invalid_request", "invalid code_challenge")
        if p.resource is not None and p.resource != config.resource_url:
            return redirect("invalid_target", f"resource must be {config.resource_url}")
        return _CheckOk(client, redirect_uri, p.code_challenge, p.state)

    def redirect_to(redirect_uri: str, params: list[tuple[str, str | None]]) -> Response:
        pairs = [(k, v) for k, v in params if v is not None]
        location = _with_query_params(redirect_uri, [*pairs, ("iss", issuer)])
        return Response(status_code=302, headers={"Cache-Control": "no-store", "Location": location})

    def render_consent(status: int, p: _AuthorizeParams, check: _CheckOk, error: str | None = None) -> Response:
        fields = [
            ("client_id", p.client_id),
            ("redirect_uri", p.redirect_uri),
            ("response_type", p.response_type),
            ("code_challenge", p.code_challenge),
            ("code_challenge_method", p.code_challenge_method),
            ("state", p.state),
            ("resource", p.resource),
            ("scope", p.scope),
        ]
        hidden = "".join(
            f'<input type="hidden" name="{name}" value="{_escape_html(value)}">'
            for name, value in fields
            if value is not None
        )
        body = (
            "<h1>Connect to voice-coding-mcp</h1>"
            f"<p><strong>{_escape_html(check.client.client_name)}</strong> wants access to Claude Code on this server.</p>"
            f'<p class="muted">After approval you return to <strong>{_escape_html(_host_of(check.redirect_uri))}</strong>.</p>'
            + (f'<p class="error">{_escape_html(error)}</p>' if error else "")
            + f'<form method="post" action="{_escape_html(issuer)}/authorize">{hidden}'
            '<label for="pairing_code">Pairing code shown by <code>voice-coding-mcp pair</code></label>'
            '<input id="pairing_code" name="pairing_code" autocomplete="one-time-code" autocapitalize="characters" spellcheck="false" required maxlength="32" placeholder="XXXXX-XXXXX">'
            '<button type="submit">Allow</button></form>'
        )
        parsed = parse_url(check.redirect_uri)
        return _page(status, "Connect to voice-coding-mcp", body, parsed.origin if parsed else None)

    # --- Metadata -------------------------------------------------------------------------------

    async def protected_resource(_request: Request) -> Response:
        return json_response(
            200,
            {
                "resource": config.resource_url,
                "authorization_servers": [issuer],
                "scopes_supported": [SCOPE],
                "bearer_methods_supported": ["header"],
            },
            {"Cache-Control": "public, max-age=300"},
        )

    async def authorization_server(_request: Request) -> Response:
        return json_response(
            200,
            {
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/token",
                "registration_endpoint": f"{issuer}/register",
                "revocation_endpoint": f"{issuer}/revoke",
                "response_types_supported": ["code"],
                "response_modes_supported": ["query"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["none", "client_secret_post", "client_secret_basic"],
                "revocation_endpoint_auth_methods_supported": ["none", "client_secret_post", "client_secret_basic"],
                "scopes_supported": [SCOPE],
                "client_id_metadata_document_supported": True,
                "authorization_response_iss_parameter_supported": True,
            },
            {"Cache-Control": "public, max-age=300"},
        )

    # --- Dynamic client registration ----------------------------------------------------------

    async def register(request: Request) -> Response:
        if limited := rate_limited(register_limit, client_ip(request)):
            return limited
        body = _as_dict(await _json_body(request))
        if not await pairing_open():
            logger.warn("register refused, pairing window closed", {"ip": client_ip(request)})
            return _oauth_error(403, "access_denied", f"Registration is closed. {PAIR_HINT}")
        uris = body.get("redirect_uris")
        if not isinstance(uris, list) or len(uris) == 0 or len(uris) > 10 or not all(isinstance(u, str) for u in uris):
            return _oauth_error(400, "invalid_redirect_uri", "redirect_uris must be a non-empty array of strings")
        redirect_uris: list[str] = list(uris)
        rejected = [u for u in redirect_uris if not is_allowed_redirect_uri(u, config.oauth)]
        if rejected:
            logger.warn("register rejected redirect uri", {"hosts": [_host_of(u) for u in rejected][:5]})
            return _oauth_error(
                400, "invalid_redirect_uri", "redirect URI not allowed by this server (oauth.redirectUris)"
            )
        method = body.get("token_endpoint_auth_method")
        auth_method: TokenEndpointAuthMethod
        if method is None or method == "client_secret_basic":
            auth_method = "client_secret_basic"
        elif method == "client_secret_post":
            auth_method = "client_secret_post"
        elif method == "none":
            auth_method = "none"
        else:
            return _oauth_error(400, "invalid_client_metadata", "unsupported token_endpoint_auth_method")
        grant_types = body.get("grant_types")
        if grant_types is None:
            grant_types = ["authorization_code", "refresh_token"]
        if not isinstance(grant_types, list) or not all(
            g in ("authorization_code", "refresh_token") for g in grant_types
        ):
            return _oauth_error(
                400, "invalid_client_metadata", "only authorization_code and refresh_token grants are supported"
            )
        response_types = body.get("response_types")
        if response_types is None:
            response_types = ["code"]
        if not isinstance(response_types, list) or not all(r == "code" for r in response_types):
            return _oauth_error(400, "invalid_client_metadata", "only the code response type is supported")
        raw_name = body.get("client_name")
        client_name = normalize_client_name(raw_name if isinstance(raw_name, str) else "")
        try:
            registered = await store.register_client(
                NewClient(client_name=client_name, redirect_uris=redirect_uris, token_endpoint_auth_method=auth_method)
            )
        except ClientLimitError:
            logger.warn("register refused, client limit reached", {"ip": client_ip(request)})
            return _oauth_error(
                403,
                "access_denied",
                "Too many connected clients. Run `voice-coding-mcp revoke-all --yes` on the server, then pair again.",
            )
        client = registered.client
        secret = registered.client_secret
        logger.info(
            "client registered",
            {
                "clientId": client.client_id,
                "clientName": client_name,
                "redirectHosts": list(dict.fromkeys(_host_of(u) for u in client.redirect_uris)),
                "authMethod": auth_method,
            },
        )
        payload: dict[str, Any] = {"client_id": client.client_id, "client_id_issued_at": client.created_at // 1000}
        if secret:
            payload["client_secret"] = secret
            payload["client_secret_expires_at"] = 0
        payload |= {
            "client_name": client.client_name,
            "redirect_uris": client.redirect_uris,
            "token_endpoint_auth_method": auth_method,
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": SCOPE,
        }
        return json_response(201, payload, {"Pragma": "no-cache"})

    # --- Authorization endpoint -----------------------------------------------------------------

    async def start_authorize(source: Mapping[str, object]) -> tuple[_AuthorizeParams, _CheckOk] | Response:
        """What GET and POST /authorize share: the pairing window must be open and the request
        valid. Returns the parameters and the checked request, or the error page or redirect."""
        if not await pairing_open():
            return _error_page(403, f"Pairing is closed. {PAIR_HINT}")
        p = _authorize_params(source)
        check = await check_authorize(p)
        if isinstance(check, Response):
            return check
        return p, check

    async def get_authorize(request: Request) -> Response:
        if limited := rate_limited(get_authorize_limit, client_ip(request)):
            return limited
        started = await start_authorize(_parse_simple(request.scope.get("query_string", b"").decode("latin-1")))
        if isinstance(started, Response):
            return started
        return render_consent(200, *started)

    async def post_authorize(request: Request) -> Response:
        ip = client_ip(request)
        if limited := rate_limited(post_authorize_limit, ip):
            return limited
        body = await _form_body(request)
        started = await start_authorize(body)
        if isinstance(started, Response):
            return started
        p, check = started
        result = await verify_pairing_code(state_dir, _str(body.get("pairing_code")) or "", now(), ip)
        if result.status == "invalid":
            logger.warn(
                "wrong pairing code",
                {"ip": ip, "clientId": check.client.client_id, "attemptsLeft": result.attempts_left},
            )
            return render_consent(400, p, check, f"Wrong pairing code. {result.attempts_left} attempt(s) left.")
        if result.status != "ok":
            logger.warn("pairing failed", {"ip": ip, "clientId": check.client.client_id, "reason": result.status})
            if result.status == "locked":
                message = f"Too many wrong codes from this address. {PAIR_HINT}"
            elif result.status == "exhausted":
                message = f"Too many wrong codes, pairing is closed. {PAIR_HINT}"
            else:
                message = f"Pairing is closed. {PAIR_HINT}"
            return _error_page(403, message)
        code = await store.create_authorization_code(
            NewCode(
                client_id=check.client.client_id,
                redirect_uri=check.redirect_uri,
                code_challenge=check.code_challenge,
                resource=config.resource_url,
                scope=SCOPE,
                client_name=check.client.client_name,
            )
        )
        logger.info(
            "authorization granted",
            {
                "clientId": check.client.client_id,
                "clientName": check.client.client_name,
                "redirectHost": _host_of(check.redirect_uri),
                "code": fingerprint(code),
            },
        )
        return redirect_to(check.redirect_uri, [("code", code), ("state", check.state)])

    # --- Token endpoint -------------------------------------------------------------------------

    async def authenticate_client(request: Request, body: Mapping[str, object]) -> str | Response:
        """Authenticates the client at /token and /revoke. Returns the client_id or an error response."""
        basic_id: str | None = None
        basic_secret: str | None = None
        header = request.headers.get("authorization")
        if header and header[:6].lower() == "basic ":
            # Strict base64 (RFC 7617): a malformed header authenticates nobody, like an unknown client.
            try:
                raw = base64.b64decode(header[6:].strip(" \t"), validate=True)
            except binascii.Error:
                return _oauth_error(401, "invalid_client", "malformed Basic authorization header")
            decoded = raw.decode("utf-8", "replace")
            sep = decoded.find(":")
            try:
                basic_id = _form_decode(decoded if sep == -1 else decoded[:sep])
                basic_secret = "" if sep == -1 else _form_decode(decoded[sep + 1 :])
            except ValueError:
                return _oauth_error(400, "invalid_request", "malformed Basic authorization header")
        body_id = _str(body.get("client_id"))
        body_secret = _str(body.get("client_secret"))
        if basic_id is not None and body_id is not None and basic_id != body_id:
            return _oauth_error(400, "invalid_request", "client_id in the body and in the Authorization header differ")
        client_id = basic_id if basic_id is not None else body_id
        secret = basic_secret if basic_id is not None else body_secret
        if not client_id:
            return _oauth_error(401, "invalid_client", "client authentication required")
        # A CIMD client is public: being on the allowlist is all there is to check.
        if is_cimd_client_id(client_id):
            return client_id if cimd.is_allowed(client_id) else _oauth_error(401, "invalid_client", "unknown client")
        client = await store.get_client(client_id)
        if client is None:
            return _oauth_error(401, "invalid_client", "unknown client")
        # A confidential client may use client_secret_basic or client_secret_post, whatever it
        # registered: some clients register one method and send the other.
        if client.client_secret_hash is not None and (
            not secret or not safe_equal(sha256_hex(secret), client.client_secret_hash)
        ):
            logger.warn("client authentication failed", {"clientId": client_id, "ip": client_ip(request)})
            headers = {"WWW-Authenticate": 'Basic realm="voice-coding-mcp"'} if basic_id is not None else None
            return _oauth_error(401, "invalid_client", "client authentication failed", headers)
        return client_id

    def send_tokens(access_token: str, refresh_token: str, expires_in: int, scope: str) -> Response:
        return json_response(
            200,
            {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": expires_in,
                "refresh_token": refresh_token,
                "scope": scope,
            },
            {"Pragma": "no-cache"},
        )

    async def token(request: Request) -> Response:
        body = await _form_body(request)
        grant_type = _str(body.get("grant_type"))
        if grant_type not in ("authorization_code", "refresh_token"):
            return _oauth_error(400, "unsupported_grant_type", "grant_type must be authorization_code or refresh_token")
        resource = _str(body.get("resource"))
        if resource is not None and resource != config.resource_url:
            return _oauth_error(400, "invalid_target", f"resource must be {config.resource_url}")
        client_id = await authenticate_client(request, body)
        if isinstance(client_id, Response):
            return client_id

        if grant_type == "authorization_code":
            code = _str(body.get("code"))
            verifier = _str(body.get("code_verifier"))
            redirect_uri = _str(body.get("redirect_uri"))
            if not code:
                return _oauth_error(400, "invalid_request", "code is required")

            def check_code(stored: StoredCode) -> str | None:
                if stored.client_id != client_id:
                    return "code was issued to another client"
                if redirect_uri != stored.redirect_uri:
                    return "redirect_uri does not match the authorization request"
                if not verifier or not _CODE_VERIFIER.fullmatch(verifier):
                    return "code_verifier missing or malformed"
                if not safe_equal(sha256_base64url(verifier), stored.code_challenge):
                    return "PKCE verification failed"
                if resource is not None and resource != stored.resource:
                    return "resource does not match the authorization request"
                return None

            # Redemption and grant creation are one store operation: a concurrent reuse of the code
            # always finds the grant and revokes it.
            exchanged = await store.exchange_authorization_code(code, check_code)
            if exchanged.status == "rejected":
                logger.warn(
                    "authorization code rejected",
                    {"clientId": client_id, "reason": exchanged.reason, "code": fingerprint(code)},
                )
                return _oauth_error(400, "invalid_grant", exchanged.reason)
            if exchanged.status != "ok":
                logger.warn(
                    "authorization code rejected",
                    {"clientId": client_id, "reason": exchanged.status, "code": fingerprint(code)},
                )
                reason = "already used" if exchanged.status == "reused" else exchanged.status
                return _oauth_error(400, "invalid_grant", f"authorization code is {reason}")
            tokens = exchanged.tokens
            logger.info(
                "tokens issued",
                {
                    "clientId": client_id,
                    "grant": tokens.grant_id,
                    "accessToken": fingerprint(tokens.access_token),
                },
            )
            return send_tokens(tokens.access_token, tokens.refresh_token, tokens.expires_in, tokens.scope)

        refresh_token = _str(body.get("refresh_token"))
        if not refresh_token:
            return _oauth_error(400, "invalid_request", "refresh_token is required")
        result = await store.refresh_grant(refresh_token, client_id, config.resource_url)
        if result.status == "reused":
            logger.warn(
                "refresh token reuse, grant revoked",
                {"clientId": client_id, "grant": result.grant_id, "refreshToken": fingerprint(refresh_token)},
            )
            return _oauth_error(400, "invalid_grant", "refresh token was already used; the authorization is revoked")
        if result.status != "ok":
            logger.warn(
                "refresh token rejected",
                {"clientId": client_id, "reason": result.status, "refreshToken": fingerprint(refresh_token)},
            )
            why = {
                "client_mismatch": "not valid for this client",
                "resource_mismatch": "issued for another resource",
            }.get(result.status, result.status)
            return _oauth_error(400, "invalid_grant", f"refresh token is {why}")
        logger.info(
            "tokens refreshed",
            {
                "clientId": client_id,
                "grant": result.tokens.grant_id,
                "accessToken": fingerprint(result.tokens.access_token),
            },
        )
        t = result.tokens
        return send_tokens(t.access_token, t.refresh_token, t.expires_in, t.scope)

    # --- Revocation (RFC 7009) ------------------------------------------------------------------

    async def revoke(request: Request) -> Response:
        body = await _form_body(request)
        revoke_token = _str(body.get("token"))
        if not revoke_token:
            return _oauth_error(400, "invalid_request", "token is required")
        has_credentials = "authorization" in request.headers or _str(body.get("client_id")) is not None
        client_id: str | None = None
        if has_credentials:
            authenticated = await authenticate_client(request, body)
            if isinstance(authenticated, Response):
                return authenticated
            client_id = authenticated
        if await store.revoke_token(revoke_token, client_id):
            logger.info("token revoked", {"clientId": client_id, "token": fingerprint(revoke_token)})
        return Response(status_code=200, headers={"Cache-Control": "no-store"})

    return [
        Route("/.well-known/oauth-protected-resource/mcp", guarded(protected_resource), methods=["GET"]),
        Route("/.well-known/oauth-protected-resource", guarded(protected_resource), methods=["GET"]),
        Route("/.well-known/oauth-authorization-server", guarded(authorization_server), methods=["GET"]),
        Route("/register", guarded(register), methods=["POST"]),
        Route("/authorize", guarded(get_authorize), methods=["GET"]),
        Route("/authorize", guarded(post_authorize), methods=["POST"]),
        Route("/token", guarded(token), methods=["POST"]),
        Route("/revoke", guarded(revoke), methods=["POST"]),
    ]
