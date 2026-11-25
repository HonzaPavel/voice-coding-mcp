"""Cleaning client-controlled text before it is stored or printed to a terminal."""

from __future__ import annotations

import unicodedata

CLIENT_NAME_MAX = 200
UNNAMED_CLIENT = "Unnamed client"

_STRIPPED_CATEGORIES = frozenset({"Cc", "Cs", "Co", "Cn"})
"""Control characters (ESC, CR, newline, C1 such as CSI), lone surrogates (they cannot be encoded
to UTF-8, so the store could not save them), private-use and unassigned code points."""

_STRIPPED_CHARS = frozenset(
    chr(c)
    for c in [
        0x061C,  # Arabic letter mark
        0x200B,  # zero-width space
        0x200E,  # left-to-right mark
        0x200F,  # right-to-left mark
        *range(0x202A, 0x202F),  # bidi embeddings and overrides (LRE..RLO)
        0x2028,  # line separator
        0x2029,  # paragraph separator
        *range(0x2060, 0x2065),  # word joiner and invisible operators
        *range(0x2066, 0x206A),  # bidi isolates (LRI..PDI)
        0xFEFF,  # zero-width no-break space (BOM)
    ]
)
"""Invisible and bidi format characters that could hide text or reorder what the owner reads. Other
format characters stay: ZWJ (U+200D) builds emoji, ZWNJ (U+200C) is needed in Persian and Indic
names, the soft hyphen is harmless."""


def strip_control_chars(text: str) -> str:
    """Drops control, surrogate, private-use and unassigned characters plus the bidi and invisible
    format characters above, so a client name cannot inject terminal escape sequences, reorder what
    the owner reads, or break saving the store."""
    return "".join(
        ch for ch in text if ch not in _STRIPPED_CHARS and unicodedata.category(ch) not in _STRIPPED_CATEGORIES
    )


def normalize_client_name(raw: str) -> str:
    """The client name as stored and shown: control and invisible characters stripped, surrounding white
    space trimmed, at most CLIENT_NAME_MAX characters. UNNAMED_CLIENT when nothing visible
    is left (also when only format characters such as joiners remain)."""
    name = strip_control_chars(raw).strip()
    if all(unicodedata.category(ch) == "Cf" for ch in name):
        return UNNAMED_CLIENT
    return name[:CLIENT_NAME_MAX]
