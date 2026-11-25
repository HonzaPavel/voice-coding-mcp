"""Tests of the JSON state file helpers."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from voice_coding_mcp.json_file import (
    FileIdentity,
    StateModel,
    dump_json,
    ensure_dir,
    file_identity,
    file_lock,
    loads_json,
    read_json,
    write_json_atomic,
    write_json_atomic_identity,
)


class TestReadJson:
    def test_fallback_for_missing_file(self, real_tmp: Path) -> None:
        assert read_json(str(real_tmp / "missing.json"), {"empty": True}) == {"empty": True}

    def test_invalid_json_names_file(self, real_tmp: Path) -> None:
        path = real_tmp / "bad.json"
        path.write_text("{nope")
        with pytest.raises(ValueError, match=str(path)):
            read_json(str(path), None)

    def test_rejects_nan_and_infinity(self, real_tmp: Path) -> None:
        path = real_tmp / "nan.json"
        path.write_text('{"a": NaN}')
        with pytest.raises(ValueError, match=str(path)):
            read_json(str(path), None)


class TestWriteJsonAtomic:
    def test_round_trip_mode_0600_trailing_newline(self, real_tmp: Path) -> None:
        path = str(real_tmp / "data.json")
        write_json_atomic(path, {"a": 1, "list": [1, 2]})
        assert read_json(path, None) == {"a": 1, "list": [1, 2]}
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert Path(path).read_text().endswith("\n")

    def test_same_text_as_json_stringify(self, real_tmp: Path) -> None:
        path = real_tmp / "fmt.json"
        write_json_atomic(str(path), {"a": 1, "b": [], "c": {}, "d": [1, {"e": None}], "s": "Přihlášení"})
        assert path.read_text(encoding="utf-8") == (
            '{\n  "a": 1,\n  "b": [],\n  "c": {},\n  "d": [\n    1,\n    {\n      "e": null\n    }\n  ],\n'
            '  "s": "Přihlášení"\n}\n'
        )
        assert dump_json({"x": 0.0123}) == '{\n  "x": 0.0123\n}\n'

    def test_replaces_and_leaves_no_temp_files(self, real_tmp: Path) -> None:
        path = str(real_tmp / "replace.json")
        write_json_atomic(path, {"v": 1})
        write_json_atomic(path, {"v": 2})
        assert read_json(path, None) == {"v": 2}
        assert [name for name in os.listdir(real_tmp) if name.endswith(".tmp")] == []

    def test_fails_when_directory_missing(self, real_tmp: Path) -> None:
        with pytest.raises(FileNotFoundError):
            write_json_atomic(str(real_tmp / "no" / "such" / "x.json"), {})

    def test_unserializable_leaves_no_temp_file(self, real_tmp: Path) -> None:
        with pytest.raises(TypeError):
            write_json_atomic(str(real_tmp / "x.json"), {"o": object()})
        assert os.listdir(real_tmp) == []


class TestEnsureDir:
    def test_creates_nested_with_0700(self, real_tmp: Path) -> None:
        path = str(real_tmp / "state" / "tasks")
        ensure_dir(path)
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o700
        ensure_dir(path)

    def test_tightens_existing(self, real_tmp: Path) -> None:
        path = real_tmp / "loose"
        path.mkdir()
        path.chmod(0o755)
        ensure_dir(str(path))
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o700


def test_loads_json_rejects_constants_and_optionally_non_containers() -> None:
    assert loads_json(' \n{"a": [1]}') == {"a": [1]}
    assert loads_json("7") == 7
    for text in ("NaN", "[Infinity]", '{"a": -Infinity}'):
        with pytest.raises(ValueError, match="invalid JSON constant"):
            loads_json(text)
    with pytest.raises(ValueError, match="Expecting property name"):
        loads_json("{nope")
    assert loads_json("\t[1]", containers_only=True) == [1]
    for text in ('"x"', "7", "null"):
        with pytest.raises(ValueError, match="not an object or array"):
            loads_json(text, containers_only=True)
    for text in ("", "\u00a0{}"):
        with pytest.raises(json.JSONDecodeError):
            loads_json(text, containers_only=True)


def test_file_lock_creates_the_directory_and_a_0600_lock_file(real_tmp: Path) -> None:
    path = str(real_tmp / "new" / "x.lock")
    with file_lock(path):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(real_tmp / "new").st_mode) == 0o700


class _Record(StateModel):
    id: int
    project_path: str
    target_session_id: str | None = None


def test_state_model_keys_and_unknown_fields() -> None:
    record = _Record.model_validate({"id": 1, "projectPath": "/p", "futureField": [1]})
    assert record.project_path == "/p"
    assert record.to_json() == {"id": 1, "projectPath": "/p", "targetSessionId": None, "futureField": [1]}
    assert record.to_json(exclude_none_fields=frozenset({"target_session_id"})) == {
        "id": 1,
        "projectPath": "/p",
        "futureField": [1],
    }
    assert _Record(id=2, project_path="/q").to_json()["projectPath"] == "/q"


def test_write_json_atomic_identity_matches_the_written_file(real_tmp: Path) -> None:
    path = str(real_tmp / "state.json")
    assert file_identity(path) is None
    first = write_json_atomic_identity(path, {"a": 1})
    info = os.stat(path)
    assert first == FileIdentity(info.st_ino, info.st_mtime_ns, info.st_size)
    assert file_identity(path) == first
    assert first.size == len(dump_json({"a": 1}).encode())
    # Every atomic write is a new inode, even with identical content.
    second = write_json_atomic_identity(path, {"a": 1})
    assert second.ino != first.ino
    assert file_identity(path) == second
