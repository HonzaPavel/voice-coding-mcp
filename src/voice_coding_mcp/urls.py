"""A subset of WHATWG URL parsing, used to validate URLs from the config and from clients.

Only what the server needs: validity, scheme, credentials, host (with a non-default port), path,
query, fragment and origin. Hostnames are lowercased; IPv6 hosts keep their brackets (`[::1]`). IDN hostnames are not converted to punycode (the config rejects a
non-ASCII `publicUrl` host).
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

SPECIAL_DEFAULT_PORTS: dict[str, int | None] = {
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
    "ftp": 21,
    "file": None,
}

_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")
_FORBIDDEN_HOST_CHARS = re.compile(r"[\s<>^|%\\/?#@\x00]")


@dataclass(frozen=True)
class ParsedUrl:
    scheme: str
    """Lowercase scheme without the colon, e.g. `https`."""
    username: str
    password: str
    hostname: str
    """Lowercase host name; IPv6 addresses in brackets."""
    port: int | None
    """None when absent or equal to the scheme's default port."""
    path: str
    """`/` for an empty path of a special scheme (http, https, ws, wss, ftp, file)."""
    query: str
    """Query without the leading `?`, empty when there is none or it is empty."""
    fragment: str
    """Fragment without the leading `#`, empty when there is none."""

    @property
    def host(self) -> str:
        """Host name plus `:port` when the port is not the default."""
        return self.hostname if self.port is None else f"{self.hostname}:{self.port}"

    @property
    def origin(self) -> str | None:
        """`scheme://host` for special schemes other than file; None (an opaque origin) otherwise."""
        if self.scheme in SPECIAL_DEFAULT_PORTS and self.scheme != "file":
            return f"{self.scheme}://{self.host}"
        return None


def parse_url(value: str) -> ParsedUrl | None:
    """Parses an absolute URL, or returns None when it is not a valid URL."""
    text = value.strip(" \t\n\r\f\v\x00")
    if "\n" in text or "\t" in text or "\r" in text:
        return None
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if not parts.scheme or not _SCHEME_RE.match(parts.scheme):
        return None
    scheme = parts.scheme.lower()
    special = scheme in SPECIAL_DEFAULT_PORTS
    raw_host = parts.hostname or ""
    if special and scheme != "file" and not raw_host:
        return None
    if raw_host and _FORBIDDEN_HOST_CHARS.search(raw_host):
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    if port is not None and port == SPECIAL_DEFAULT_PORTS.get(scheme):
        port = None
    hostname = raw_host.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    path = parts.path
    if special and not path:
        path = "/"
    return ParsedUrl(
        scheme=scheme,
        username=parts.username or "",
        password=parts.password or "",
        hostname=hostname,
        port=port,
        path=path,
        query=parts.query,
        fragment=parts.fragment,
    )


def can_parse(value: str) -> bool:
    """True when `parse_url` accepts `value`."""
    return parse_url(value) is not None


def host_of(uri: str, fallback: str = "?") -> str:
    """The host (with a non-default port) of `uri`, or `fallback` for an unparsable URI."""
    parsed = parse_url(uri)
    return parsed.host if parsed else fallback


def is_loopback_ip(address: str) -> bool:
    """True for an IP address in 127.0.0.0/8 or ::1, also IPv4-mapped (`::ffff:127.0.0.1`) and in
    brackets (`[::1]`). Host names, even `localhost`, are not IP addresses."""
    text = address.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback
