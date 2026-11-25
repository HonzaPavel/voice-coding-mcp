"""Tests for the shared helpers clock, urls and web."""

from __future__ import annotations

from voice_coding_mcp.auth.text import normalize_client_name, strip_control_chars
from voice_coding_mcp.clock import parse_iso, to_iso
from voice_coding_mcp.urls import can_parse, host_of, is_loopback_ip, parse_url
from voice_coding_mcp.web import client_ip_from, client_key


class TestClock:
    def test_to_iso_is_utc_with_milliseconds(self) -> None:
        assert to_iso(0) == "1970-01-01T00:00:00.000Z"
        assert to_iso(1763812800123) == "2025-11-22T12:00:00.123Z"

    def test_parse_iso(self) -> None:
        assert parse_iso("2025-11-22T12:00:00.123Z") == 1763812800123
        assert parse_iso("garbage") is None
        assert parse_iso(None) is None


class TestUrls:
    def test_can_parse(self) -> None:
        assert can_parse("https://claude.ai/api/mcp/auth_callback")
        assert can_parse("http://127.0.0.1:6274/oauth/callback")
        assert not can_parse("nope")
        assert not can_parse("not a url")
        assert not can_parse("https://")
        assert not can_parse("http://exa mple.com/")

    def test_parts(self) -> None:
        url = parse_url("HTTPS://U:P@Vcm.Example.com:443/a?b=1#c")
        assert url is not None
        assert url.scheme == "https"
        assert (url.username, url.password) == ("U", "P")
        assert url.host == "vcm.example.com"
        assert (url.path, url.query, url.fragment) == ("/a", "b=1", "c")
        assert url.origin == "https://vcm.example.com"
        opaque = parse_url("mailto:x@example.com")
        assert opaque is not None
        assert opaque.origin is None
        empty = parse_url("https://vcm.example.com")
        assert empty is not None
        assert empty.path == "/"

    def test_ipv6_and_ports(self) -> None:
        url = parse_url("http://[::1]:3000/x")
        assert url is not None
        assert url.hostname == "[::1]"
        assert url.host == "[::1]:3000"
        assert parse_url("http://h:99999/") is None

    def test_host_of(self) -> None:
        assert host_of("https://claude.ai/api/mcp/auth_callback") == "claude.ai"
        assert host_of("nope") == "?"
        assert host_of("nope", "nope") == "nope"


class TestClientIp:
    """Forwarding headers are trusted only through loopback hops."""

    def test_direct_peer(self) -> None:
        assert client_ip_from("203.0.113.5", None) == "203.0.113.5"
        assert client_ip_from("203.0.113.5", "198.51.100.1") == "203.0.113.5"

    def test_through_loopback_proxy(self) -> None:
        assert client_ip_from("127.0.0.1", "198.51.100.1") == "198.51.100.1"
        assert client_ip_from("::1", "10.0.0.1, 198.51.100.1") == "198.51.100.1"
        assert client_ip_from("127.0.0.1", "198.51.100.1, 127.0.0.1") == "198.51.100.1"
        assert client_ip_from("127.0.0.1", "127.0.0.2") == "127.0.0.2"
        assert client_ip_from("127.0.0.1", None) == "127.0.0.1"
        assert client_ip_from("::ffff:127.0.0.1", "198.51.100.7") == "198.51.100.7"


class TestClientKey:
    """Per-IP limits count IPv6 clients by their /64 and IPv4-mapped IPv6 as the IPv4 address."""

    def test_ipv4(self) -> None:
        assert client_key("203.0.113.5") == "203.0.113.5"
        assert client_key("::ffff:203.0.113.5") == "203.0.113.5"
        assert client_key("::ffff:cb00:7105") == "203.0.113.5"

    def test_ipv6_by_64(self) -> None:
        assert client_key("2001:db8:1:2:3:4:5:6") == "2001:db8:1:2::/64"
        assert client_key("2001:DB8:1:2::9") == "2001:db8:1:2::/64"
        assert client_key("[2001:db8:1:2::9]") == "2001:db8:1:2::/64"
        assert client_key("fe80::1%en0") == "fe80::/64"
        assert client_key("2001:db8:1:3::1") != client_key("2001:db8:1:2::1")

    def test_non_addresses_unchanged(self) -> None:
        assert client_key("unknown") == "unknown"
        assert client_key("2001:db8:1:2::/64") == "2001:db8:1:2::/64"


class TestStripControlChars:
    def test_strips_controls_and_ansi_escapes(self) -> None:
        assert strip_control_chars("\x1b[31mred\x07\r\n\t") == "[31mred"
        assert strip_control_chars("\x9b2J") == "2J"

    def test_strips_bidi_and_invisible_format_chars(self) -> None:
        assert strip_control_chars("a\u202eb\u200bc\ufeffd\u2066e\u200ff\u061cg\u2028h") == "abcdefgh"

    def test_strips_surrogates_private_use_and_unassigned(self) -> None:
        assert strip_control_chars("a\ud800b\udfffc\ue000d\U0010fffde\u0378f") == "abcdef"

    def test_keeps_joiners_needed_by_emoji_and_scripts(self) -> None:
        family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
        assert strip_control_chars(family) == family
        assert (
            strip_control_chars("\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645")
            == "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645"
        )
        assert strip_control_chars("soft\u00adhyphen") == "soft\u00adhyphen"

    def test_keeps_ordinary_text(self) -> None:
        assert strip_control_chars("Claude (Žluťoučký kůň) 🚗") == "Claude (Žluťoučký kůň) 🚗"


class TestNormalizeClientName:
    def test_strips_trims_and_caps(self) -> None:
        assert normalize_client_name("\u3000 \x1bCl\u202eaude\u00a0") == "Claude"
        assert normalize_client_name("x" * 300) == "x" * 200

    def test_falls_back_when_nothing_visible_is_left(self) -> None:
        for raw in ("", "  ", "\x1b\u202e", "\u200d", " \u200c\u200d\u00ad "):
            assert normalize_client_name(raw) == "Unnamed client"
        assert normalize_client_name("\U0001f468\u200d\U0001f469") == "\U0001f468\u200d\U0001f469"


def test_is_loopback_ip() -> None:
    for address in ("127.0.0.1", "127.1.2.3", "::1", "[::1]", "::ffff:127.0.0.1", " 127.0.0.1 "):
        assert is_loopback_ip(address), address
    for address in ("localhost", "128.0.0.1", "10.0.0.1", "::2", "127.999.0.1", "unknown"):
        assert not is_loopback_ip(address), address
