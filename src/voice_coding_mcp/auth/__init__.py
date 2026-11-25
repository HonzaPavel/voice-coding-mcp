"""OAuth authorization server, pairing window, CIMD and bearer check."""

from .bearer import AuthInfo, BearerError, BearerMiddleware, request_auth, www_authenticate
from .cimd import CimdClient, CimdError, CimdResolver, cache_ttl_ms, create_cimd_resolver, is_cimd_client_id
from .pairing import (
    PAIRING_ATTEMPTS_PER_CLIENT,
    PAIRING_MAX_FAILURES,
    OpenedPairing,
    PairingWindow,
    VerifyPairingResult,
    close_pairing,
    format_pairing_code,
    get_pairing,
    normalize_pairing_code,
    open_pairing,
    pairing_path,
    verify_pairing_code,
)
from .rate_limit import RateLimiter, RateLimitOptions
from .router import (
    SCOPE,
    AuthRouterOptions,
    RateLimitSettings,
    create_auth_routes,
    is_allowed_redirect_uri,
    resource_metadata_url,
)
from .store import ClientLimitError, ClientStatus, OAuthStatus, OAuthStore, StoredClient, StoredGrant

__all__ = [
    "PAIRING_ATTEMPTS_PER_CLIENT",
    "PAIRING_MAX_FAILURES",
    "SCOPE",
    "AuthInfo",
    "AuthRouterOptions",
    "BearerError",
    "BearerMiddleware",
    "CimdClient",
    "CimdError",
    "CimdResolver",
    "ClientLimitError",
    "ClientStatus",
    "OAuthStatus",
    "OAuthStore",
    "OpenedPairing",
    "PairingWindow",
    "RateLimitOptions",
    "RateLimitSettings",
    "RateLimiter",
    "StoredClient",
    "StoredGrant",
    "VerifyPairingResult",
    "cache_ttl_ms",
    "close_pairing",
    "create_auth_routes",
    "create_cimd_resolver",
    "format_pairing_code",
    "get_pairing",
    "is_allowed_redirect_uri",
    "is_cimd_client_id",
    "normalize_pairing_code",
    "open_pairing",
    "pairing_path",
    "request_auth",
    "resource_metadata_url",
    "verify_pairing_code",
    "www_authenticate",
]
