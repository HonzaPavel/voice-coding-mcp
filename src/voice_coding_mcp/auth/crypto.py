"""Hashing and random tokens for the OAuth server."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets


def sha256_hex(value: str) -> str:
    """SHA-256 of a UTF-8 string, hex encoded."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def base64url(data: bytes) -> str:
    """Unpadded base64url."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def sha256_base64url(value: str) -> str:
    """SHA-256 of an ASCII string, base64url encoded (PKCE S256)."""
    return base64url(hashlib.sha256(value.encode("ascii", "replace")).digest())


def random_token(nbytes: int = 32) -> str:
    """32 random bytes, base64url (43 characters). Used for tokens, codes and secrets."""
    return base64url(secrets.token_bytes(nbytes))


def safe_equal(a: str, b: str) -> bool:
    """Constant-time comparison of two strings (compares their SHA-256 so lengths never leak)."""
    return hmac.compare_digest(hashlib.sha256(a.encode("utf-8")).digest(), hashlib.sha256(b.encode("utf-8")).digest())
