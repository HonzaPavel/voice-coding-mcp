"""Smoke tests for the fake claude fixture (the runner and manager tests depend on its behavior)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from tests.helpers import FAKE_CLAUDE, is_alive, read_records


def run(stdin: str, *args: str, record: Path | None = None) -> tuple[int, list[dict[str, Any]], str]:
    env = {"PATH": "/usr/bin:/bin", **({"FAKE_CLAUDE_RECORD": str(record)} if record else {})}
    proc = subprocess.run(
        [FAKE_CLAUDE, *args], input=stdin.encode(), capture_output=True, env=env, timeout=20, check=False
    )
    events = [json.loads(line) for line in proc.stdout.decode().splitlines() if line.startswith("{")]
    return proc.returncode, events, proc.stderr.decode()


def test_default_run_records_and_returns_ok(real_tmp: Path) -> None:
    record = real_tmp / "rec.jsonl"
    code, events, _ = run("hello\nPROGRESS 2", "-p", "--verbose", record=record)
    assert code == 0
    assert [e["type"] for e in events] == ["system", "assistant", "assistant", "result"]
    assert events[-1]["result"] == "OK"
    assert events[-1]["is_error"] is False
    assert events[1]["message"]["content"][1]["name"] == "Read"
    [rec] = read_records(str(record))
    assert rec["argv"] == ["-p", "--verbose"]
    assert rec["stdin"] == "hello\nPROGRESS 2"
    assert rec["env"]["FAKE_CLAUDE_RECORD"] == str(record)
    assert rec["childPid"] is None


def test_result_fail_exit_and_stderr() -> None:
    code, events, _ = run("RESULT Hotovo, testy prošly.")
    assert events[-1]["result"] == "Hotovo, testy prošly."
    code, events, _ = run("FAIL")
    assert code == 1
    assert events[-1]["is_error"] is True
    code, events, err = run("EXIT 3\nSTDERR boom")
    assert code == 3
    assert "boom" in err
    code, events, _ = run("NO_RESULT")
    assert code == 0
    assert [e["type"] for e in events] == ["system"]


def test_resume_and_fork_session() -> None:
    _, events, _ = run("x", "--resume", "abc")
    assert events[0]["session_id"] == "abc"
    _, events, _ = run("x", "--resume", "abc", "--fork-session")
    assert events[0]["session_id"] != "abc"
    code, events, err = run("RESUME_MISSING", "--resume", "gone")
    assert code == 1
    assert "No conversation found with session ID: gone" in err
    assert events[-1]["is_error"] is True
    assert "result" not in events[-1]


def test_split_output_still_parses() -> None:
    proc = subprocess.run([FAKE_CLAUDE], input=b"SPLIT\nRESULT r", capture_output=True, timeout=20, check=False)
    lines = proc.stdout.decode().splitlines()
    assert "this is not json" in lines
    assert json.loads(lines[-1])["result"] == "r"


def test_spawn_child_records_pid(real_tmp: Path) -> None:
    record = real_tmp / "rec.jsonl"
    run("SPAWN_CHILD", record=record)
    [rec] = read_records(str(record))
    child = rec["childPid"]
    assert isinstance(child, int)
    assert is_alive(child)
    subprocess.run(["kill", "-9", str(child)], check=False)
