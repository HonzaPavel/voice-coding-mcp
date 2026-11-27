"""Tests of config loading and validation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import config_data
from voice_coding_mcp.config import (
    Config,
    ConfigError,
    config_warnings,
    is_loopback_host,
    load_config,
    parse_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class Env:
    def __init__(self, base: Path) -> None:
        self.base = str(base)
        self.home = str(base / "home" / "vcm")
        self.root = os.path.join(self.home, "projects")
        self.state_dir = str(base / "state")
        os.makedirs(self.root)

    def minimal(self, **overrides: Any) -> dict[str, Any]:
        return config_data("https://vcm.example.com", self.state_dir, self.root, **overrides)

    def parse(self, data: object) -> Config:
        return parse_config(data, home_dir=self.home)

    def rejects(self, data: object, pattern: str) -> None:
        with pytest.raises(ConfigError) as info:
            self.parse(data)
        assert re.search(pattern, str(info.value)), str(info.value)


@pytest.fixture
def env(real_tmp: Path) -> Env:
    return Env(real_tmp)


class TestDefaultsAndDerived:
    def test_fills_every_optional_field(self, env: Env) -> None:
        config = env.parse(env.minimal())
        assert config.listen.model_dump(by_alias=True) == {"host": "127.0.0.1", "port": 8787, "allowNonLoopback": False}
        assert config.claude.model_dump(by_alias=True) == {
            "bin": "claude",
            "model": None,
            "effort": None,
            "maxTurns": None,
            "permissionMode": "auto",
            "access": "isolated",
            "settingSources": ["user"],
            "configDir": None,
            "mcpServers": {},
            "extraArgs": [],
            "env": {},
        }
        assert config.tasks.model_dump(by_alias=True) == {
            "syncWaitSeconds": 25,
            "timeoutMinutes": 15,
            "maxConcurrent": 2,
            "maxQueued": 10,
            "resumeWithinHours": 12,
            "keepDays": 14,
        }
        assert config.oauth.model_dump(by_alias=True) == {
            "accessTokenHours": 24,
            "refreshTokenDays": 90,
            "pairingMinutes": 10,
            "redirectUris": ["https://claude.ai/api/mcp/auth_callback", "https://claude.com/api/mcp/auth_callback"],
            "cimdClientIds": ["https://claude.ai/oauth/mcp-oauth-client-metadata"],
            "allowLoopbackRedirects": False,
        }
        assert config.notify.model_dump(by_alias=True) == {"url": None, "headers": {}, "includeResult": False}
        assert config.sessions.model_dump(by_alias=True) == {"days": 7, "replyWaitMinutes": 10}
        assert config.remote_control.model_dump(by_alias=True) == {
            "maxSessions": 3,
            "handoffToDesktop": False,
            "idleCloseMinutes": 0,
        }
        assert config.log_prompts is False
        assert config.idle_shutdown_minutes == 0

    def test_fills_defaults_inside_partial_sections(self, env: Env) -> None:
        config = env.parse(env.minimal(listen={"port": 9000}, tasks={"maxConcurrent": 1}))
        assert config.listen.host == "127.0.0.1"
        assert config.listen.port == 9000
        assert config.tasks.max_concurrent == 1
        assert config.tasks.sync_wait_seconds == 25

    def test_derives_issuer_resource_url_and_public_host(self, env: Env) -> None:
        config = env.parse(env.minimal(publicUrl="https://vcm.example.com/"))
        assert config.issuer == "https://vcm.example.com"
        assert config.public_url == "https://vcm.example.com"
        assert config.resource_url == "https://vcm.example.com/mcp"
        assert config.public_host == "vcm.example.com"

    def test_keeps_non_default_port_in_public_host(self, env: Env) -> None:
        config = env.parse(env.minimal(publicUrl="https://vcm.example.com:8443"))
        assert config.public_host == "vcm.example.com:8443"
        assert config.resource_url == "https://vcm.example.com:8443/mcp"

    def test_drops_default_port(self, env: Env) -> None:
        assert env.parse(env.minimal(publicUrl="https://vcm.example.com:443")).public_host == "vcm.example.com"

    def test_rejects_unknown_keys_with_field_name(self, env: Env) -> None:
        env.rejects(env.minimal(claude={"permisionMode": "auto"}), r"claude.*permisionMode")
        env.rejects(env.minimal(typo=1), r"typo")

    def test_rejects_snake_case_keys(self, env: Env) -> None:
        env.rejects(env.minimal(tasks={"sync_wait_seconds": 5}), r"tasks\.sync_wait_seconds")
        env.rejects({"public_url": "https://vcm.example.com", "projectRoots": [env.root]}, r"publicUrl")

    def test_requires_public_url_and_project_roots(self, env: Env) -> None:
        env.rejects({"stateDir": env.state_dir, "projectRoots": [env.root]}, r"publicUrl")
        env.rejects({"publicUrl": "https://vcm.example.com", "stateDir": env.state_dir}, r"projectRoots")
        env.rejects(env.minimal(projectRoots=[]), r"projectRoots")

    def test_rejects_non_object(self, env: Env) -> None:
        env.rejects([1], r"\(root\)")

    def test_accepts_integral_floats_and_rejects_bools(self, env: Env) -> None:
        assert env.parse(env.minimal(tasks={"syncWaitSeconds": 30.0})).tasks.sync_wait_seconds == 30
        env.rejects(env.minimal(tasks={"syncWaitSeconds": 2.5}), r"tasks\.syncWaitSeconds")
        env.rejects(env.minimal(tasks={"maxConcurrent": True}), r"tasks\.maxConcurrent")
        env.rejects(env.minimal(sessions={"days": True}), r"sessions\.days")

    def test_validates_remote_control(self, env: Env) -> None:
        assert env.parse(env.minimal(remoteControl={"maxSessions": 0})).remote_control.max_sessions == 0
        env.rejects(env.minimal(remoteControl={"maxSessions": -1}), r"remoteControl\.maxSessions")
        env.rejects(env.minimal(remoteControl={"maxSessions": 1.5}), r"remoteControl\.maxSessions")
        env.rejects(env.minimal(remoteControl={"limit": 2}), r"remoteControl\.limit")
        parsed = env.parse(env.minimal(remoteControl={"handoffToDesktop": True, "idleCloseMinutes": 0}))
        assert parsed.remote_control.handoff_to_desktop is True
        assert parsed.remote_control.idle_close_minutes == 0
        env.rejects(env.minimal(remoteControl={"idleCloseMinutes": -1}), r"remoteControl\.idleCloseMinutes")
        env.rejects(env.minimal(remoteControl={"idleCloseMinutes": 1.5}), r"remoteControl\.idleCloseMinutes")

    def test_validates_idle_shutdown_minutes(self, env: Env) -> None:
        assert env.parse(env.minimal(idleShutdownMinutes=180)).idle_shutdown_minutes == 180
        assert env.parse(env.minimal(idleShutdownMinutes=60.0)).idle_shutdown_minutes == 60
        assert env.parse(env.minimal(idleShutdownMinutes=0)).idle_shutdown_minutes == 0
        for bad in (-1, 1.5, True, "60"):
            env.rejects(env.minimal(idleShutdownMinutes=bad), r"idleShutdownMinutes")


class TestPublicUrl:
    def test_requires_https_for_non_loopback(self, env: Env) -> None:
        env.rejects(env.minimal(publicUrl="http://vcm.example.com"), r"publicUrl: must use https")

    def test_allows_http_for_loopback(self, env: Env) -> None:
        assert env.parse(env.minimal(publicUrl="http://127.0.0.1:8787")).issuer == "http://127.0.0.1:8787"
        assert env.parse(env.minimal(publicUrl="http://localhost:3000")).public_host == "localhost:3000"
        assert env.parse(env.minimal(publicUrl="http://[::1]:3000")).public_host == "[::1]:3000"

    def test_rejects_invalid_urls_paths_queries_credentials(self, env: Env) -> None:
        env.rejects(env.minimal(publicUrl="not a url"), r"publicUrl: not a valid URL")
        env.rejects(env.minimal(publicUrl="ftp://vcm.example.com"), r"publicUrl")
        env.rejects(env.minimal(publicUrl="https://vcm.example.com/base"), r"publicUrl: must be an origin")
        env.rejects(env.minimal(publicUrl="https://vcm.example.com/?a=1"), r"publicUrl: must be an origin")
        env.rejects(env.minimal(publicUrl="https://u:p@vcm.example.com"), r"publicUrl: must not contain credentials")

    def test_rejects_a_non_ascii_host_and_asks_for_punycode(self, env: Env) -> None:
        env.rejects(env.minimal(publicUrl="https://vcm.bücher.example"), r"publicUrl: .*not ASCII.*punycode")
        punycode = "https://vcm.xn--bcher-kva.example"
        assert env.parse(env.minimal(publicUrl=punycode)).issuer == punycode


class TestListenHost:
    def test_accepts_loopback(self, env: Env) -> None:
        for host in ["127.0.0.1", "127.0.0.2", "::1", "localhost"]:
            assert env.parse(env.minimal(listen={"host": host})).listen.host == host

    def test_rejects_non_loopback_unless_allowed(self, env: Env) -> None:
        env.rejects(env.minimal(listen={"host": "0.0.0.0"}), r"listen\.host: must be a loopback address")
        env.rejects(env.minimal(listen={"host": "10.0.0.5"}), r"listen\.host")
        config = env.parse(env.minimal(listen={"host": "0.0.0.0", "allowNonLoopback": True}))
        assert config.listen.host == "0.0.0.0"

    def test_validates_port(self, env: Env) -> None:
        env.rejects(env.minimal(listen={"port": 0}), r"listen\.port")
        env.rejects(env.minimal(listen={"port": 70000}), r"listen\.port")
        env.rejects(env.minimal(listen={"port": "8787"}), r"listen\.port")


def test_is_loopback_host() -> None:
    assert is_loopback_host("127.0.0.1") is True
    assert is_loopback_host("[::1]") is True
    assert is_loopback_host("LOCALHOST") is True
    assert is_loopback_host("128.0.0.1") is False
    assert is_loopback_host("localhost.example.com") is False
    assert is_loopback_host("127.0.0.1.example.com") is False
    assert is_loopback_host("0:0:0:0:0:0:0:1") is True
    assert is_loopback_host("127.1.2.3") is True


class TestProjectRoots:
    def test_must_be_absolute(self, env: Env) -> None:
        env.rejects(env.minimal(projectRoots=["projects"]), r"projectRoots\.0: must be an absolute path")

    def test_must_exist_and_be_directory(self, env: Env) -> None:
        env.rejects(
            env.minimal(projectRoots=[os.path.join(env.base, "missing")]), r"projectRoots\.0: directory does not exist"
        )
        file = os.path.join(env.base, "file.txt")
        Path(file).write_text("x")
        env.rejects(env.minimal(projectRoots=[file]), r"projectRoots\.0: not a directory")

    def test_resolved_with_realpath(self, env: Env) -> None:
        link = os.path.join(env.base, "link-to-projects")
        os.symlink(env.root, link)
        assert env.parse(env.minimal(projectRoots=[link])).project_roots == (env.root,)

    def test_rejects_filesystem_root(self, env: Env) -> None:
        env.rejects(
            env.minimal(projectRoots=["/"]), r"projectRoots\.0: (the filesystem root|must not be the home directory)"
        )

    def test_rejects_home_and_ancestors(self, env: Env) -> None:
        env.rejects(env.minimal(projectRoots=[env.home]), r"projectRoots\.0: must not be the home directory")
        env.rejects(
            env.minimal(projectRoots=[os.path.join(env.base, "home")]),
            r"projectRoots\.0: must not be the home directory",
        )
        env.rejects(env.minimal(projectRoots=[env.root, env.base]), r"projectRoots\.1: must not be the home directory")

    def test_allows_sibling_with_home_prefix(self, env: Env) -> None:
        sibling = env.home + "-other"
        os.makedirs(sibling)
        assert env.parse(env.minimal(projectRoots=[sibling])).project_roots == (sibling,)

    def test_rejects_root_containing_state_dir(self, env: Env) -> None:
        env.rejects(
            env.minimal(stateDir=os.path.join(env.root, ".state")), r"projectRoots\.0: must not contain stateDir"
        )
        env.rejects(env.minimal(stateDir=env.root), r"projectRoots\.0: must not contain stateDir")

    def test_rejects_duplicates(self, env: Env) -> None:
        env.rejects(env.minimal(projectRoots=[env.root, env.root + "/"]), r"projectRoots\.1: duplicate root")


def test_state_dir_must_be_absolute(env: Env) -> None:
    env.rejects(env.minimal(stateDir="state"), r"stateDir: must be an absolute path")


FORBIDDEN = [
    "--dangerously-skip-permissions",
    "--allow-dangerously-skip-permissions",
    "--permission-mode",
    "--mcp-config",
    "--strict-mcp-config",
    "--setting-sources",
    "--bare",
    "--settings",
    "--allowedTools",
    "--allowed-tools",
    "--tools",
    "--add-dir",
    "--permission-prompts",
    "--system-prompt",
    "--append-system-prompt",
    "--name",
    "-n",
]


class TestClaude:
    def test_accepts_both_access_modes_and_rejects_others(self, env: Env) -> None:
        for mode in ["isolated", "desktop"]:
            assert env.parse(env.minimal(claude={"access": mode})).claude.access == mode
        env.rejects(env.minimal(claude={"access": "full"}), r"claude\.access: Input should be 'isolated' or 'desktop'")
        env.rejects(env.minimal(claude={"access": None}), r"claude\.access")

    def test_warns_about_desktop_access(self, env: Env) -> None:
        assert config_warnings(env.parse(env.minimal(claude={"access": "isolated"}))) == []
        [warning] = config_warnings(env.parse(env.minimal(claude={"access": "desktop"})))
        assert "claude.access is desktop" in warning
        assert "connectors" in warning

    def test_accepts_every_allowed_permission_mode(self, env: Env) -> None:
        for mode in ["auto", "acceptEdits", "dontAsk", "plan", "manual"]:
            assert env.parse(env.minimal(claude={"permissionMode": mode})).claude.permission_mode == mode

    def test_rejects_bypass_and_unknown_modes(self, env: Env) -> None:
        env.rejects(
            env.minimal(claude={"permissionMode": "bypassPermissions"}),
            r"claude\.permissionMode: .*bypassPermissions is never allowed",
        )
        env.rejects(env.minimal(claude={"permissionMode": "default"}), r"claude\.permissionMode")

    @pytest.mark.parametrize("flag", FORBIDDEN)
    def test_rejects_forbidden_extra_args(self, env: Env, flag: str) -> None:
        env.rejects(
            env.minimal(claude={"extraArgs": ["--verbose", flag]}), rf"claude\.extraArgs\.1: {flag} is not allowed"
        )
        env.rejects(env.minimal(claude={"extraArgs": [f"{flag}=x"]}), rf"claude\.extraArgs\.0: {flag} is not allowed")

    def test_rejects_flag_value_form(self, env: Env) -> None:
        env.rejects(env.minimal(claude={"extraArgs": ["--permission-mode=bypassPermissions"]}), r"claude\.extraArgs\.0")

    def test_accepts_harmless_extra_args(self, env: Env) -> None:
        config = env.parse(env.minimal(claude={"extraArgs": ["--fallback-model", "sonnet"]}))
        assert config.claude.extra_args == ["--fallback-model", "sonnet"]

    def test_setting_sources_default_and_explicit(self, env: Env) -> None:
        assert env.parse(env.minimal()).claude.setting_sources == ["user"]
        config = env.parse(env.minimal(claude={"settingSources": ["user", "project", "local"]}))
        assert config.claude.setting_sources == ["user", "project", "local"]

    def test_rejects_unknown_or_duplicate_setting_sources(self, env: Env) -> None:
        env.rejects(env.minimal(claude={"settingSources": ["user", "policy"]}), r"claude\.settingSources\.1")
        env.rejects(
            env.minimal(claude={"settingSources": ["user", "user"]}),
            r"claude\.settingSources: must not contain duplicates",
        )

    def test_validates_other_fields(self, env: Env) -> None:
        env.rejects(env.minimal(claude={"effort": "huge"}), r"claude\.effort")
        env.rejects(env.minimal(claude={"maxTurns": 0}), r"claude\.maxTurns")
        env.rejects(env.minimal(claude={"configDir": "relative"}), r"claude\.configDir: must be an absolute path")
        env.rejects(env.minimal(claude={"env": {"A": 1}}), r"claude\.env\.A")
        env.rejects(env.minimal(claude={"bin": ""}), r"claude\.bin")
        config = env.parse(env.minimal(claude={"model": "opus", "effort": "high", "maxTurns": 30, "env": {"A": "b"}}))
        assert config.claude.model == "opus"
        assert config.claude.effort == "high"
        assert config.claude.max_turns == 30


class TestTasksOAuthNotify:
    def test_validates_numeric_ranges(self, env: Env) -> None:
        env.rejects(env.minimal(tasks={"syncWaitSeconds": 0}), r"tasks\.syncWaitSeconds")
        env.rejects(env.minimal(tasks={"maxConcurrent": 0}), r"tasks\.maxConcurrent")
        env.rejects(env.minimal(tasks={"maxQueued": -1}), r"tasks\.maxQueued")
        env.rejects(env.minimal(oauth={"accessTokenHours": 0}), r"oauth\.accessTokenHours")

    def test_validates_urls(self, env: Env) -> None:
        env.rejects(env.minimal(oauth={"redirectUris": ["nope"]}), r"oauth\.redirectUris\.0")
        env.rejects(
            env.minimal(oauth={"cimdClientIds": ["http://claude.ai/x"]}),
            r"oauth\.cimdClientIds\.0: must be a valid https URL",
        )
        env.rejects(env.minimal(notify={"url": "ftp://ntfy.sh/x"}), r"notify\.url")

    def test_redirect_uris_must_be_https_except_on_loopback(self, env: Env) -> None:
        env.rejects(
            env.minimal(oauth={"redirectUris": ["http://claude.ai/api/mcp/auth_callback"]}),
            r"oauth\.redirectUris\.0: must use https \(plain http is allowed only for 127\.0\.0\.1, \[::1\] and localhost\)",
        )
        env.rejects(
            env.minimal(oauth={"redirectUris": ["http://127.0.0.2/cb"]}), r"oauth\.redirectUris\.0: must use https"
        )
        env.rejects(env.minimal(oauth={"redirectUris": ["ftp://claude.ai/cb"]}), r"oauth\.redirectUris\.0")
        loopback = ["http://127.0.0.1:6274/cb", "http://[::1]/cb", "http://localhost:3000/cb", "https://x.example/cb"]
        assert env.parse(env.minimal(oauth={"redirectUris": loopback})).oauth.redirect_uris == loopback

    def test_warns_about_allow_loopback_redirects(self, env: Env) -> None:
        assert config_warnings(env.parse(env.minimal())) == []
        warnings = config_warnings(env.parse(env.minimal(oauth={"allowLoopbackRedirects": True})))
        assert len(warnings) == 1
        assert "oauth.allowLoopbackRedirects" in warnings[0]
        config = env.parse(env.minimal(notify={"url": "https://ntfy.sh/topic", "headers": {"Priority": "high"}}))
        assert config.notify.url == "https://ntfy.sh/topic"

    def test_warns_that_the_desktop_handoff_is_macos_only(self, env: Env) -> None:
        config = env.parse(env.minimal(remoteControl={"handoffToDesktop": True}))
        assert config_warnings(config, platform="darwin") == []
        [warning] = config_warnings(config, platform="linux")
        assert "remoteControl.handoffToDesktop" in warning


class TestLoadConfig:
    def test_loads_and_validates_file(self, env: Env) -> None:
        path = os.path.join(env.base, "config.json")
        Path(path).write_text(json.dumps(env.minimal()))
        assert load_config(path, home_dir=env.home).resource_url == "https://vcm.example.com/mcp"

    def test_names_file_in_errors(self, env: Env) -> None:
        missing = os.path.join(env.base, "missing.json")
        with pytest.raises(ConfigError) as info:
            load_config(missing)
        assert missing in str(info.value)
        bad = os.path.join(env.base, "bad.json")
        Path(bad).write_text("{")
        with pytest.raises(ConfigError, match="is not valid JSON"):
            load_config(bad)
        Path(bad).write_text('{"tasks": {"syncWaitSeconds": Infinity}}')
        with pytest.raises(ConfigError, match="is not valid JSON"):
            load_config(bad)
        invalid = os.path.join(env.base, "invalid.json")
        Path(invalid).write_text(json.dumps(env.minimal(publicUrl="http://vcm.example.com")))
        with pytest.raises(ConfigError) as info:
            load_config(invalid, home_dir=env.home)
        assert str(info.value).startswith(f"{invalid}: publicUrl")

    def test_accepts_config_example_json(self, env: Env) -> None:
        example = json.loads((REPO_ROOT / "config.example.json").read_text())
        config = env.parse({**example, "stateDir": env.state_dir, "projectRoots": [env.root]})
        assert config.issuer == "https://vcm.example.com"
        assert config.claude.setting_sources == ["user"]
