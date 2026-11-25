"""Fake Claude home and transcript lines.

Lines are written as compact JSON (`separators=(",", ":")`) like Claude Code writes them:
the sessions module looks for substrings such as `"type":"user"` before parsing a line.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
from typing import Any

from voice_coding_mcp.clock import now_ms, to_iso


def claude_home(base: str) -> str:
    """Creates `<base>/claude-home` with `projects/` and `sessions/` and returns its path."""
    home = os.path.join(base, "claude-home")
    os.makedirs(os.path.join(home, "projects"), exist_ok=True)
    os.makedirs(os.path.join(home, "sessions"), exist_ok=True)
    return home


def prompt(cwd: str, text: str) -> dict[str, Any]:
    """A prompt the owner typed in the desktop app."""
    return {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": text},
        "timestamp": to_iso(now_ms()),
        "turnOrigin": "human",
        "entrypoint": "claude-desktop",
        "cwd": cwd,
    }


def assistant_text(cwd: str, text: str) -> dict[str, Any]:
    return {
        "isSidechain": False,
        "type": "assistant",
        "message": {"role": "assistant", "stop_reason": "end_turn", "content": [{"type": "text", "text": text}]},
        "entrypoint": "claude-desktop",
        "cwd": cwd,
    }


def peer_message(cwd: str, from_name: str, text: str) -> dict[str, Any]:
    """How a message from another session shows up in the target transcript."""
    return {
        "isSidechain": False,
        "type": "user",
        "turnOrigin": "peer",
        "message": {
            "role": "user",
            "content": (
                "Another Claude session sent a message:\n"
                f'<cross-session-message from="uds:/tmp/cc-socks/1.sock" from-name="{from_name}" from-mode="prompting">\n'
                f"{text}\n</cross-session-message>\n\nThis came from another Claude session."
            ),
        },
        "entrypoint": "claude-desktop",
        "cwd": cwd,
    }


def json_line(value: dict[str, Any]) -> str:
    """Compact JSON (no spaces), without the newline."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def transcript_line(session_id: str, line: dict[str, Any]) -> str:
    """One transcript line exactly as `write_transcript` writes it (sessionId first, compact JSON)."""
    return json_line({"sessionId": session_id, **line})


def write_transcript(home: str, session_id: str, lines: list[dict[str, Any]], mtime_ms: float | None = None) -> str:
    """Writes `<home>/projects/-encoded-cwd/<id>.jsonl`, sets its mtime and returns the path."""
    directory = os.path.join(home, "projects", "-encoded-cwd")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{session_id}.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(transcript_line(session_id, line) for line in lines) + "\n")
    stamp = (mtime_ms if mtime_ms is not None else time.time() * 1000) / 1000
    os.utime(path, (stamp, stamp))
    return path


def append_lines(path: str, lines: list[dict[str, Any]], session_id: str | None = None) -> None:
    """Appends compact JSON lines (for reply-watch tests); with `session_id`, prefixed like `write_transcript`."""
    with open(path, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write((transcript_line(session_id, line) if session_id else json_line(line)) + "\n")


def write_live_session(home: str, pid: int, data: dict[str, Any]) -> str:
    """Writes the live registry entry `<home>/sessions/<pid>.json` (keys as Claude Code writes them,
    e.g. `sessionId`, `cwd`, `name`, `status`, `messagingSocketPath`)."""
    path = os.path.join(home, "sessions", f"{pid}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"pid": pid, **data}, handle)
    return path


def stop_background_sessions(home: str) -> None:
    """SIGTERMs the `kind: "bg"` processes in the fake live registry (the fake claude's `--bg` sleepers)."""
    directory = os.path.join(home, "sessions")
    for name in os.listdir(directory) if os.path.isdir(directory) else []:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        pid = data.get("pid") if isinstance(data, dict) else None
        if isinstance(pid, int) and pid != os.getpid() and data.get("kind") == "bg":
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
