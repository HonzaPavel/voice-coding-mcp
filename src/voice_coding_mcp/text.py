"""Small text helpers for spoken answers, log fields and notifications."""

ELLIPSIS = "…"


def clip(text: str, limit: int) -> str:
    """`text` unchanged when it fits in `limit` characters, else cut so that the result,
    including a trailing ellipsis, is at most `limit` characters long."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + ELLIPSIS


def plural(count: int, word: str) -> str:
    """`1 minute`, `2 minutes`: the count with an English plural `s` when it is not 1."""
    return f"{count} {word}{'' if count == 1 else 's'}"
