"""Allowed roots, project listing and resolving a spoken project name.

The name matching is shared by the task manager (projects) and the session lookup (session names).
Directory scans are small, so these functions are synchronous.
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

ResolveFailureReason = Literal["not_found", "ambiguous"]


@dataclass(frozen=True)
class Project:
    name: str
    """Directory name, unique across all roots."""
    path: str
    """realpath of the project directory, inside an allowed root."""


@dataclass(frozen=True)
class MatchOk[T]:
    item: T


@dataclass(frozen=True)
class MatchFailed:
    reason: ResolveFailureReason
    candidates: list[str] = field(default_factory=list)


type MatchResult[T] = MatchOk[T] | MatchFailed


def is_inside(child: str, parent: str) -> bool:
    """True when the path `child` lies strictly inside the directory `parent` (both normalized)."""
    prefix = parent if parent.endswith(os.sep) else parent + os.sep
    return child.startswith(prefix)


def list_projects(roots: Sequence[str]) -> list[Project]:
    """Immediate, non-hidden subdirectories of the allowed roots whose realpath stays inside the root's
    realpath. Sorted by name; when two roots hold the same name, the first root wins. Unreadable
    roots and entries are skipped."""
    by_name: dict[str, Project] = {}
    for root in roots:
        try:
            root_real = os.path.realpath(root, strict=True)
            with os.scandir(root_real) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if name.startswith(".") or name in by_name:
                continue
            try:
                if not entry.is_dir():  # follows symlinks; a broken one is not a directory
                    continue
                # Only a symlink can lead elsewhere: a plain entry of a realpath is its own realpath.
                real = os.path.realpath(entry.path, strict=True) if entry.is_symlink() else entry.path
            except OSError:
                continue
            if not is_inside(real, root_real):
                continue
            by_name[name] = Project(name=name, path=real)
    return sorted(by_name.values(), key=lambda project: project.name)


def normalize_name(value: str) -> str:
    """Lowercase, without diacritics, separators collapsed to single spaces.

    NFKD, drop Unicode marks, lowercase, every run of characters that are not letters or numbers
    (including `_`) becomes one space, trimmed."""
    decomposed = unicodedata.normalize("NFKD", value)
    out: list[str] = []
    gap = False
    for char in decomposed:
        category = unicodedata.category(char)
        if category.startswith("M"):
            continue
        if category.startswith(("L", "N")):
            if gap and out:
                out.append(" ")
            gap = False
            out.append(char)
        else:
            gap = True
    return "".join(out).lower().strip()


def compact_name(value: str) -> str:
    """Normalized form without spaces, so "voice coding MCP" equals "voice-coding-mcp"."""
    return normalize_name(value).replace(" ", "")


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i]
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[len(b)]


def is_path_like(value: str) -> bool:
    """True when the input looks like a path rather than a name (`/`, `\\`, `..` or NUL)."""
    return "/" in value or "\\" in value or ".." in value or "\0" in value


MIN_PARTIAL_QUERY = 3
"""Shorter queries match exactly only: "a" or "db" would otherwise hit a random project."""

MAX_QUERY_CHARS = 200
"""Only this much of a spoken name is matched, so long input stays cheap (names are far shorter)."""

MAX_QUERY_TOKENS = 12
"""Only runs of the first this many words are compared with whole names."""


def max_fuzzy_distance(query: str) -> int:
    """Largest Levenshtein distance accepted for a (compact) query: 1 up to 4 characters, else 2."""
    return 1 if len(query) <= 4 else 2


def match_by_name[T](items: Sequence[T], name_of: Callable[[T], str], spoken_name: str) -> MatchResult[T]:
    """Matches a spoken name against `items` (by `name_of`): exact compact form, then a unique prefix,
    then a unique "contains" (the query inside a name, or a name equal to a whole run of the query's
    words, so "the api server" finds api but "rapid" does not), then the unique best Levenshtein
    distance (at most 1 when the query or the name has up to 4 characters, else at most 2). Queries
    shorter than 3 characters match exactly only. Only the first MAX_QUERY_CHARS characters and, for
    word runs, the first MAX_QUERY_TOKENS words count. Path-like input never matches. Names should be
    unique; candidates are the names."""
    not_found = MatchFailed(reason="not_found", candidates=[name_of(item) for item in items])
    if is_path_like(spoken_name):
        return not_found
    spoken_name = spoken_name[:MAX_QUERY_CHARS]
    query = compact_name(spoken_name)
    if not query:
        return not_found

    keyed = [(item, compact_name(name_of(item))) for item in items]

    def pick(matches: list[tuple[T, str]]) -> MatchResult[T] | None:
        if len(matches) == 1:
            return MatchOk(item=matches[0][0])
        if len(matches) > 1:
            return MatchFailed(reason="ambiguous", candidates=[name_of(item) for item, _ in matches])
        return None

    exact = pick([k for k in keyed if k[1] == query])
    if exact:
        return exact
    if len(query) < MIN_PARTIAL_QUERY:
        return not_found
    prefix = pick([k for k in keyed if k[1].startswith(query)])
    if prefix:
        return prefix
    words = normalize_name(spoken_name).split(" ")[:MAX_QUERY_TOKENS]
    contains = pick(
        [k for k in keyed if query in k[1] or (len(k[1]) >= MIN_PARTIAL_QUERY and _is_word_run(k[1], words))]
    )
    if contains:
        return contains

    best = max_fuzzy_distance(query) + 1
    best_matches: list[tuple[T, str]] = []
    for k in keyed:
        # The shorter side sets the limit: two edits must not turn "happy" into "app".
        limit = max_fuzzy_distance(query if len(query) <= len(k[1]) else k[1])
        if abs(len(query) - len(k[1])) > limit:
            continue  # the distance is at least the length difference
        distance = levenshtein(query, k[1])
        if distance > limit:
            continue
        if distance < best:
            best = distance
            best_matches = [k]
        elif distance == best:
            best_matches.append(k)
    return pick(best_matches) or not_found


def _is_word_run(name: str, words: Sequence[str]) -> bool:
    """True when `name` (compact) equals some of `words` in a row, joined without spaces."""
    joined = "".join(words)
    bounds = {0}
    end = 0
    for word in words:
        end += len(word)
        bounds.add(end)
    start = joined.find(name)
    while start != -1:
        if start in bounds and start + len(name) in bounds:
            return True
        start = joined.find(name, start + 1)
    return False


def match_project(projects: Sequence[Project], spoken_name: str) -> MatchResult[Project]:
    """`match_by_name` over projects, by directory name."""
    return match_by_name(projects, lambda project: project.name, spoken_name)


def resolve_project(roots: Sequence[str], spoken_name: str) -> MatchResult[Project]:
    """Lists the projects under `roots` and resolves a spoken name among them."""
    return match_project(list_projects(roots), spoken_name)
