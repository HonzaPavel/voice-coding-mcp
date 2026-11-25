"""`claude.access`: isolated and desktop argv, environment, trust and `.mcp.json` approvals."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import claude_config, read_records, tasks_config
from voice_coding_mcp.log import silent_logger
from voice_coding_mcp.tasks.access import (
    desktop_access,
    global_config_path,
    isolated_access,
    task_access,
    unapproved_mcpjson_servers,
    voice_server_pattern,
)
from voice_coding_mcp.tasks.manager import Started, TaskManager, TaskManagerOptions
from voice_coding_mcp.tasks.runner import RunOptions, build_args, build_env

PUBLIC_URL = "https://vcm.example.test"
DENY = {"serverUrl": "https://vcm.example.test/*"}

_counter = count(1)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@dataclass
class Dirs:
    home: Path
    project: Path

    def trust(self, path: Path | None = None, **entry: Any) -> None:
        """Writes `home/.claude.json` with one trusted project entry (plus `entry` keys)."""
        key = str(path or self.project)
        write_json(self.home / ".claude.json", {"projects": {key: {"hasTrustDialogAccepted": True, **entry}}})

    def mcpjson(self, *names: str, where: Path | None = None) -> None:
        write_json((where or self.project) / ".mcp.json", {"mcpServers": {n: {"command": "true"} for n in names}})

    def access(self, **claude: Any) -> Any:
        return desktop_access(
            claude_config("/r", access="desktop", **claude),
            project_path=str(self.project),
            public_url=PUBLIC_URL,
            mcp_config_path="/state/mcp-config.json",
            home_dir=str(self.home),
        )


@pytest.fixture
def dirs(real_tmp: Path) -> Dirs:
    home = real_tmp / "home"
    project = real_tmp / "projects" / "alpha"
    home.mkdir()
    project.mkdir(parents=True)
    return Dirs(home, project)


class TestIsolated:
    def test_keeps_the_strict_config_and_setting_sources(self) -> None:
        claude = claude_config("/r")
        access = task_access(claude, project_path="/p", public_url=PUBLIC_URL, mcp_config_path="/state/m.json")
        assert access == isolated_access(claude, "/state/m.json")
        assert access.args() == [
            "--strict-mcp-config",
            "--mcp-config",
            "/state/m.json",
            "--setting-sources",
            "user,local",
        ]

    def test_env_disables_connectors_and_nonessential_traffic(self) -> None:
        env = build_env(claude_config("/r", env={}), {})
        assert env == {
            "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
        }


class TestDesktop:
    def test_untrusted_folder_gets_user_settings_only_and_denies_the_voice_server(self, dirs: Dirs) -> None:
        access = dirs.access()
        assert access.isolated is False
        assert access.setting_sources == ("user",)
        assert access.mcp_config_path is None, "no --mcp-config without claude.mcpServers"
        assert access.args() == [
            "--setting-sources",
            "user",
            "--settings",
            json.dumps({"deniedMcpServers": [DENY]}, separators=(",", ":")),
        ]

    def test_trusted_folder_or_parent_gets_project_and_local_settings(self, dirs: Dirs) -> None:
        dirs.trust()
        assert dirs.access().setting_sources == ("user", "project", "local")
        dirs.trust(dirs.project.parent)
        assert dirs.access().setting_sources == ("user", "project", "local")
        write_json(dirs.home / ".claude.json", {"projects": {str(dirs.project): {"hasTrustDialogAccepted": "yes"}}})
        assert dirs.access().setting_sources == ("user",)

    def test_a_missing_or_invalid_claude_json_means_untrusted(self, dirs: Dirs) -> None:
        for text in ["{not json", "[]", '{"projects": []}', '{"projects": {"x": 1}}']:
            (dirs.home / ".claude.json").write_text(text, encoding="utf-8")
            assert dirs.access().setting_sources == ("user",)

    def test_reads_claude_json_from_the_config_dir_when_set(self, dirs: Dirs, real_tmp: Path) -> None:
        config_dir = real_tmp / "cfg"
        claude = claude_config("/r", config_dir=str(config_dir))
        assert global_config_path(claude, str(dirs.home)) == str(config_dir / ".claude.json")
        assert global_config_path(claude_config("/r"), str(dirs.home)) == str(dirs.home / ".claude.json")
        dirs.trust()
        assert dirs.access(config_dir=str(config_dir)).setting_sources == ("user",)
        write_json(config_dir / ".claude.json", {"projects": {str(dirs.project): {"hasTrustDialogAccepted": True}}})
        assert dirs.access(config_dir=str(config_dir)).setting_sources == ("user", "project", "local")

    def test_adds_claude_mcp_servers_without_strict(self, dirs: Dirs) -> None:
        access = dirs.access(mcp_servers={"x": {"command": "true"}})
        args = access.args()
        assert "--strict-mcp-config" not in args
        assert args[:2] == ["--mcp-config", "/state/mcp-config.json"]

    def test_disables_unapproved_mcpjson_servers(self, dirs: Dirs) -> None:
        dirs.mcpjson("approved", "local-ok", "user-ok", "denied", "unknown")
        dirs.mcpjson("parent-server", where=dirs.project.parent)
        dirs.trust(enabledMcpjsonServers=["approved", "denied"], disabledMcpjsonServers=["denied"])
        write_json(dirs.project / ".claude" / "settings.local.json", {"enabledMcpjsonServers": ["local-ok"]})
        write_json(dirs.home / ".claude" / "settings.json", {"enabledMcpjsonServers": ["user-ok"]})
        settings = dirs.access().settings
        assert settings == {
            "deniedMcpServers": [DENY],
            "disabledMcpjsonServers": ["denied", "parent-server", "unknown"],
        }

    def test_ignores_approvals_committed_in_an_untrusted_folder(self, dirs: Dirs) -> None:
        dirs.mcpjson("a", "b")
        write_json(dirs.project / ".claude" / "settings.json", {"enableAllProjectMcpServers": True})
        assert dirs.access().settings["disabledMcpjsonServers"] == ["a", "b"]
        dirs.trust()
        assert "disabledMcpjsonServers" not in dirs.access().settings

    def test_approval_rules(self) -> None:
        names = {"a", "b", "c"}
        assert unapproved_mcpjson_servers(names, []) == ["a", "b", "c"]
        assert unapproved_mcpjson_servers(names, [{"enableAllProjectMcpServers": True}]) == []
        assert unapproved_mcpjson_servers(
            names, [{"enableAllProjectMcpServers": True}, {"disabledMcpjsonServers": ["b"]}]
        ) == ["b"]
        assert unapproved_mcpjson_servers(names, [{"enabledMcpjsonServers": ["a", 3]}]) == ["b", "c"]
        assert unapproved_mcpjson_servers(names, [{"enabledMcpjsonServers": "a"}]) == ["a", "b", "c"]

    def test_deny_pattern_and_required_public_url(self) -> None:
        assert voice_server_pattern("https://h.example/") == "https://h.example/*"
        with pytest.raises(ValueError, match="public URL"):
            task_access(claude_config("/r", access="desktop"), project_path="/p", public_url=None, mcp_config_path="/m")

    def test_argv_and_env_of_a_desktop_run(self, dirs: Dirs) -> None:
        claude = claude_config("/r", access="desktop", env={})
        access = dirs.access()
        args = build_args(
            RunOptions(
                claude=claude, cwd="/", prompt="", mcp_config_path="/m", timeout_ms=1, base_env={}, access=access
            )
        )
        assert "--strict-mcp-config" not in args
        assert "--mcp-config" not in args
        assert args[args.index("--setting-sources") + 1] == "user"
        assert json.loads(args[args.index("--settings") + 1]) == {"deniedMcpServers": [DENY]}
        assert args.count("--settings") == 1
        assert build_env(claude, {}, isolated=False) == {"DISABLE_AUTOUPDATER": "1"}


@dataclass
class ManagerEnv:
    base: Path
    root: Path
    home: Path
    open: list[TaskManager] = field(default_factory=list)

    async def manager(self, **claude: Any) -> tuple[TaskManager, str]:
        n = next(_counter)
        record = str(self.base / f"record-{n}.jsonl")
        manager = await TaskManager.create(
            TaskManagerOptions(
                state_dir=str(self.base / f"state-{n}"),
                project_roots=[str(self.root)],
                claude=claude_config(record, **claude),
                tasks=tasks_config(),
                logger=silent_logger,
                kill_grace_ms=500,
                live_session_ids=set,
                public_url=PUBLIC_URL,
                home_dir=str(self.home),
            )
        )
        self.open.append(manager)
        return manager, record


@pytest.fixture
async def menv(real_tmp: Path) -> AsyncIterator[ManagerEnv]:
    root = real_tmp / "projects"
    (root / "alpha").mkdir(parents=True)
    home = real_tmp / "home"
    home.mkdir()
    e = ManagerEnv(real_tmp, root, home)
    yield e
    await asyncio.gather(*(m.shutdown("test end") for m in e.open))


async def run_one(manager: TaskManager, record: str) -> Any:
    started = await manager.start("alpha", "RESULT ok")
    assert isinstance(started, Started)
    done = await manager.wait(started.task.id, 5000)
    assert done is not None
    assert done.status == "done", done.error
    [rec] = read_records(record)
    return rec


async def test_run_task_in_isolated_mode_is_unchanged(menv: ManagerEnv) -> None:
    manager, record = await menv.manager()
    rec = await run_one(manager, record)
    argv = rec["argv"]
    assert argv[argv.index("--strict-mcp-config") + 1 : argv.index("--strict-mcp-config") + 5] == [
        "--mcp-config",
        manager.mcp_config_path,
        "--setting-sources",
        "user,local",
    ]
    assert "--settings" not in argv
    assert rec["env"]["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    assert rec["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


async def test_run_task_in_desktop_mode_gets_desktop_access(menv: ManagerEnv) -> None:
    project = menv.root / "alpha"
    write_json(menv.home / ".claude.json", {"projects": {str(project): {"hasTrustDialogAccepted": True}}})
    write_json(project / ".mcp.json", {"mcpServers": {"unapproved": {"command": "true"}}})
    manager, record = await menv.manager(access="desktop")
    rec = await run_one(manager, record)
    argv = rec["argv"]
    assert "--strict-mcp-config" not in argv
    assert "--mcp-config" not in argv
    assert argv[argv.index("--setting-sources") + 1] == "user,project,local"
    assert json.loads(argv[argv.index("--settings") + 1]) == {
        "deniedMcpServers": [DENY],
        "disabledMcpjsonServers": ["unapproved"],
    }
    assert "ENABLE_CLAUDEAI_MCP_SERVERS" not in rec["env"]
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in rec["env"]
    assert rec["env"]["DISABLE_AUTOUPDATER"] == "1"
    assert rec["stdin"] == "RESULT ok"
