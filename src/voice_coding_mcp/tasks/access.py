"""What task runs can reach besides the built-in tools: MCP servers, claude.ai connectors and
settings (`claude.access`).

- `isolated` (default): `--strict-mcp-config --mcp-config <stateDir/mcp-config.json>`,
  `--setting-sources` from `claude.settingSources`, no claude.ai connectors and no nonessential
  traffic (ISOLATED_ENV).
- `desktop`: the same MCP servers, connectors and plugins as Claude Code on the owner's desktop.
  No strict MCP config (`claude.mcpServers`, when not empty, are added with a plain `--mcp-config`)
  and none of ISOLATED_ENV. One `--settings` JSON denies the voice server itself
  (`deniedMcpServers` with `<publicUrl>/*`, so a task cannot call the connector it came from) and
  disables the servers of the project's `.mcp.json` that the owner has not approved
  (`disabledMcpjsonServers`; `claude -p` would start them without asking). Setting sources are
  `user,project,local` when the folder is trusted in Claude Code, else `user` only.

The ask and relay runs always use `isolated_access` with an empty MCP config, whatever the mode.

`~/.claude.json` (or `$CLAUDE_CONFIG_DIR/.claude.json`) holds account data: it is only read, never
written, and nothing from it is logged. Every file here is read defensively: missing or invalid
means untrusted, not approved.

Verified with Claude Code 2.1.331: `disabledMcpjsonServers` from `--settings` keeps a `.mcp.json`
server from starting and wins over `enabledMcpjsonServers` from local settings; with setting
sources `user` only, `.mcp.json` servers do not load at all.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import ClaudeConfig, SettingSource
from ..json_file import read_json_object
from ..sessions import claude_home_dir

ISOLATED_ENV: Mapping[str, str] = {
    # The server itself is a claude.ai connector: loading connectors would recurse.
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
"""Set for isolated runs only."""

DESKTOP_SETTING_SOURCES: tuple[SettingSource, ...] = ("user", "project", "local")
"""Desktop runs in a trusted folder; in an untrusted one only `user`."""

MAX_CONFIG_BYTES = 64 * 1024 * 1024
"""Larger settings or `.claude.json` files are treated as missing."""


@dataclass(frozen=True)
class RunAccess:
    """The access-related part of a run's argv and environment."""

    isolated: bool
    mcp_config_path: str | None
    """`--mcp-config` (strict when isolated); None passes none."""
    setting_sources: tuple[str, ...]
    settings: Mapping[str, Any] = field(default_factory=dict)
    """`--settings` JSON; empty passes none."""

    def args(self) -> list[str]:
        args: list[str] = []
        if self.isolated:
            args.append("--strict-mcp-config")
        if self.mcp_config_path is not None:
            args += ["--mcp-config", self.mcp_config_path]
        args += ["--setting-sources", ",".join(self.setting_sources)]
        if self.settings:
            args += ["--settings", json.dumps(self.settings, separators=(",", ":"), ensure_ascii=False)]
        return args


def isolated_access(claude: ClaudeConfig, mcp_config_path: str) -> RunAccess:
    """Strict MCP config, `claude.settingSources`, no connectors."""
    return RunAccess(isolated=True, mcp_config_path=mcp_config_path, setting_sources=tuple(claude.setting_sources))


def voice_server_pattern(public_url: str) -> str:
    """`deniedMcpServers` URL pattern matching every endpoint of the voice server."""
    return f"{public_url.rstrip('/')}/*"


def global_config_path(claude: ClaudeConfig, home_dir: str | None = None) -> str:
    """`$CLAUDE_CONFIG_DIR/.claude.json` when `claude.configDir` is set (not None, like
    `claude_home_dir`), else `~/.claude.json`."""
    if claude.config_dir is not None:
        return os.path.join(claude.config_dir, ".claude.json")
    return os.path.join(home_dir if home_dir is not None else os.path.expanduser("~"), ".claude.json")


def user_settings_path(claude: ClaudeConfig, home_dir: str | None = None) -> str:
    """`settings.json` in the Claude home (`claude_home_dir`: `claude.configDir`, else `~/.claude`)."""
    return os.path.join(claude_home_dir(claude.config_dir, home_dir), "settings.json")


def _read_object(path: str) -> dict[str, Any]:
    """A JSON object file, or {} when missing, unreadable, too large or not an object."""
    return read_json_object(path, MAX_CONFIG_BYTES) or {}


_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}
"""Parsed files by path, with the (st_mtime_ns, st_size) they were read at."""


def _read_object_cached(path: str) -> dict[str, Any]:
    """`_read_object`, reused while the file's mtime and size stay the same (`~/.claude.json` can
    be large and is read for every task run). The result is shared: callers must not change it."""
    try:
        info = os.stat(path)
    except OSError:
        _cache.pop(path, None)
        return {}
    cached = _cache.get(path)
    if cached is not None and cached[:2] == (info.st_mtime_ns, info.st_size):
        return cached[2]
    data = _read_object(path)
    _cache[path] = (info.st_mtime_ns, info.st_size, data)
    return data


def _ancestors(path: str) -> Iterator[str]:
    """`path`, its parent, ..., up to the filesystem root."""
    current = os.path.abspath(path)
    while True:
        yield current
        parent = os.path.dirname(current)
        if parent == current:
            return
        current = parent


def _project_entry(projects: object, path: str) -> dict[str, Any]:
    entry = projects.get(path) if isinstance(projects, dict) else None
    return entry if isinstance(entry, dict) else {}


def is_trusted(global_config: Mapping[str, Any], project_path: str) -> bool:
    """The trust prompt was accepted for the folder or one of its parents, as Claude Code checks it."""
    projects = global_config.get("projects")
    return any(
        _project_entry(projects, path).get("hasTrustDialogAccepted") is True for path in _ancestors(project_path)
    )


def _names(value: object) -> set[str]:
    return {item for item in value if isinstance(item, str)} if isinstance(value, list) else set()


def mcpjson_server_names(project_path: str) -> set[str]:
    """Server names of the `.mcp.json` files in the folder and its parents (Claude Code also loads
    the parents' files; disabling a name that does not exist is harmless)."""
    names: set[str] = set()
    for path in _ancestors(project_path):
        servers = _read_object(os.path.join(path, ".mcp.json")).get("mcpServers")
        if isinstance(servers, dict):
            names.update(name for name in servers if isinstance(name, str))
    return names


def unapproved_mcpjson_servers(names: set[str], approvals: Sequence[Mapping[str, Any]]) -> list[str]:
    """The names desktop would not start: not enabled (by name or `enableAllProjectMcpServers`) in
    any approval source, or disabled in one (disabled wins)."""
    enable_all = any(source.get("enableAllProjectMcpServers") is True for source in approvals)
    enabled: set[str] = set()
    disabled: set[str] = set()
    for source in approvals:
        enabled |= _names(source.get("enabledMcpjsonServers"))
        disabled |= _names(source.get("disabledMcpjsonServers"))
    return sorted(name for name in names if name in disabled or not (enable_all or name in enabled))


def desktop_access(
    claude: ClaudeConfig,
    *,
    project_path: str,
    public_url: str,
    mcp_config_path: str | None,
    home_dir: str | None = None,
) -> RunAccess:
    """Access like Claude Code on the desktop, minus the voice server and unapproved `.mcp.json`
    servers. Reads files: call it off the event loop. `mcp_config_path` is passed (not strict) only
    when `claude.mcpServers` is not empty."""
    global_config = _read_object_cached(global_config_path(claude, home_dir))
    trusted = is_trusted(global_config, project_path)
    approvals: list[Mapping[str, Any]] = [
        _project_entry(global_config.get("projects"), project_path),
        _read_object_cached(user_settings_path(claude, home_dir)),
    ]
    if trusted:
        # Only a trusted folder's own settings count; an untrusted repo could commit approvals.
        approvals += [
            _read_object(os.path.join(project_path, ".claude", "settings.json")),
            _read_object(os.path.join(project_path, ".claude", "settings.local.json")),
        ]
    settings: dict[str, Any] = {"deniedMcpServers": [{"serverUrl": voice_server_pattern(public_url)}]}
    disabled = unapproved_mcpjson_servers(mcpjson_server_names(project_path), approvals)
    if disabled:
        settings["disabledMcpjsonServers"] = disabled
    return RunAccess(
        isolated=False,
        mcp_config_path=mcp_config_path if claude.mcp_servers else None,
        setting_sources=DESKTOP_SETTING_SOURCES if trusted else ("user",),
        settings=settings,
    )


def task_access(
    claude: ClaudeConfig,
    *,
    project_path: str,
    public_url: str | None,
    mcp_config_path: str,
    home_dir: str | None = None,
) -> RunAccess:
    """The access of a task run (run_task, run_tasks_parallel, Remote Control) by `claude.access`.
    Reads files in desktop mode: call it off the event loop."""
    if claude.access == "isolated":
        return isolated_access(claude, mcp_config_path)
    if not public_url:
        raise ValueError("claude.access desktop needs the public URL to exclude the voice server")
    return desktop_access(
        claude, project_path=project_path, public_url=public_url, mcp_config_path=mcp_config_path, home_dir=home_dir
    )
