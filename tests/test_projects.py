"""Project listing and spoken project name matching."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from voice_coding_mcp.projects import (
    MatchFailed,
    MatchOk,
    Project,
    compact_name,
    levenshtein,
    list_projects,
    match_project,
    resolve_project,
)


@pytest.fixture
def roots(real_tmp: Path) -> tuple[str, str, str]:
    root_a = str(real_tmp / "a")
    root_b = str(real_tmp / "b")
    outside = str(real_tmp / "outside")
    for d in (root_a, root_b, outside):
        os.mkdir(d)
    for name in ["voice-coding-mcp", "web", "Žluťoučký_kůň", ".hidden", "shared"]:
        os.mkdir(os.path.join(root_a, name))
    Path(root_a, "notes.txt").write_text("file, not a project")
    os.mkdir(os.path.join(root_a, "web", "nested"))
    os.symlink(outside, os.path.join(root_a, "escape"))
    os.symlink(os.path.join(root_a, "web"), os.path.join(root_a, "web-link"))
    for name in ["shared", "api"]:
        os.mkdir(os.path.join(root_b, name))
    return root_a, root_b, str(real_tmp)


class TestListProjects:
    def test_lists_immediate_non_hidden_dirs_sorted_unique(self, roots: tuple[str, str, str]) -> None:
        root_a, root_b, _ = roots
        projects = list_projects([root_a, root_b])
        assert [p.name for p in projects] == ["api", "shared", "voice-coding-mcp", "web", "web-link", "Žluťoučký_kůň"]
        by_name = {p.name: p.path for p in projects}
        assert by_name["shared"] == os.path.join(root_a, "shared")
        assert by_name["web-link"] == os.path.join(root_a, "web")

    def test_drops_escaping_symlinks_and_missing_roots(self, roots: tuple[str, str, str]) -> None:
        root_a, _, base = roots
        projects = list_projects([root_a, os.path.join(base, "missing")])
        assert not any(p.name == "escape" for p in projects)


class TestCompactNameAndLevenshtein:
    def test_normalizes(self) -> None:
        assert compact_name("Žluťoučký_kůň") == "zlutouckykun"
        assert compact_name("voice coding MCP") == "voicecodingmcp"
        assert compact_name("voice-coding-mcp") == "voicecodingmcp"

    def test_edit_distance(self) -> None:
        assert levenshtein("kitten", "sitting") == 3
        assert levenshtein("", "abc") == 3
        assert levenshtein("same", "same") == 0


PROJECTS = [
    Project(name=n, path=f"/r/{n}")
    for n in ["voice-coding-mcp", "voice-notes", "web", "website", "Žluťoučký_kůň", "assistant"]
]


def matched(spoken: str) -> str | None:
    result = match_project(PROJECTS, spoken)
    return result.item.name if isinstance(result, MatchOk) else None


class TestMatchProject:
    def test_spoken_forms(self) -> None:
        assert matched("voice coding MCP") == "voice-coding-mcp"
        assert matched("Voice Coding") == "voice-coding-mcp"
        assert matched("zlutoucky kun") == "Žluťoučký_kůň"
        assert matched("web") == "web"
        assert matched("the assistant project") == "assistant"
        assert matched("asistant") == "assistant"
        assert matched("voice codin mcp") == "voice-coding-mcp"

    def test_ambiguity(self) -> None:
        assert match_project(PROJECTS, "voice") == MatchFailed(
            reason="ambiguous", candidates=["voice-coding-mcp", "voice-notes"]
        )

    def test_not_found(self) -> None:
        result = match_project(PROJECTS, "kubernetes")
        assert isinstance(result, MatchFailed)
        assert result.reason == "not_found"
        assert len(result.candidates) == len(PROJECTS)

    def test_rejects_path_like(self) -> None:
        for bad in ["../web", "web/..", "/r/web", "a\\web", ".."]:
            assert isinstance(match_project(PROJECTS, bad), MatchFailed), bad
        assert isinstance(match_project(PROJECTS, "  "), MatchFailed)


SMALL = [Project(name=n, path=f"/r/{n}") for n in ["web", "assistant", "voice-coding-mcp", "osvc"]]


class TestShortQueries:
    def resolved(self, spoken: str) -> str | None:
        result = match_project(SMALL, spoken)
        return result.item.name if isinstance(result, MatchOk) else None

    def test_short_queries_do_not_resolve_to_a_wrong_project(self) -> None:
        for spoken in ["a", "db", "dev", "ab", "s", "we", "vo"]:
            assert self.resolved(spoken) is None, spoken

    def test_short_exact_names_still_match(self) -> None:
        projects = [*SMALL, Project(name="db", path="/r/db")]
        result = match_project(projects, "DB")
        assert isinstance(result, MatchOk)
        assert result.item.name == "db"

    def test_prefix_and_contains_need_three_characters(self) -> None:
        assert self.resolved("ass") == "assistant"
        assert self.resolved("coding") == "voice-coding-mcp"
        assert self.resolved("as") is None

    def test_fuzzy_distance_depends_on_query_length(self) -> None:
        assert self.resolved("osvd") == "osvc"  # distance 1, 4 characters
        assert self.resolved("oxvd") is None  # distance 2, 4 characters
        assert self.resolved("asistnt") == "assistant"  # distance 2, 7 characters


WORDY = [Project(name=n, path=f"/r/{n}") for n in ["app", "web", "api", "voice-coding-mcp"]]


class TestNameInsideQuery:
    def resolved(self, spoken: str) -> str | None:
        result = match_project(WORDY, spoken)
        return result.item.name if isinstance(result, MatchOk) else None

    def test_a_name_counts_only_as_whole_words_of_the_query(self) -> None:
        assert self.resolved("the api server") == "api"
        assert self.resolved("voice coding mcp please") == "voice-coding-mcp"
        assert self.resolved("open the web_site of web-shop") == "web"

    def test_a_name_hidden_inside_a_word_does_not_match(self) -> None:
        for spoken in ["mapping tool", "happy", "website redesign", "rapid prototype"]:
            assert self.resolved(spoken) is None, spoken

    def test_a_very_long_query_stays_cheap(self) -> None:
        start = time.perf_counter()
        assert self.resolved("blah " * 20_000) is None
        assert self.resolved("please open the api server " + "and then some more words " * 5_000) == "api"
        assert time.perf_counter() - start < 1.0


def test_resolve_project(roots: tuple[str, str, str]) -> None:
    root_a, root_b, _ = roots
    result = resolve_project([root_a, root_b], "Voice Coding MCP")
    assert isinstance(result, MatchOk)
    assert result.item.path == os.path.join(root_a, "voice-coding-mcp")
    assert isinstance(resolve_project([root_a], "escape"), MatchFailed)
    assert isinstance(resolve_project([root_a], "web/nested"), MatchFailed)
