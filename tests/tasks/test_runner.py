"""One `claude -p` run against the fake claude: argv, environment, stream-json, timeout and cancel."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import claude_config, is_alive, read_records, until
from voice_coding_mcp.config import ClaudeConfig
from voice_coding_mcp.tasks import runner
from voice_coding_mcp.tasks.prompt import SPOKEN_ANSWER_PROMPT
from voice_coding_mcp.tasks.runner import (
    RunOptions,
    ToolUse,
    build_args,
    build_env,
    run_claude,
    summarize_tool_use,
)

_counter = count(1)


@dataclass
class Opts:
    options: RunOptions
    record: str


@pytest.fixture
def make(real_tmp: Path) -> Any:
    def options(prompt: str, **overrides: Any) -> Opts:
        record = str(real_tmp / f"record-{next(_counter)}.jsonl")
        base_env = {
            **os.environ,
            "ANTHROPIC_API_KEY": "sk-should-not-leak",
            "ANTHROPIC_AUTH_TOKEN": "nope",
            "SECRET_THING": "x",
            "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
        }
        fields: dict[str, Any] = {
            "claude": claude_config(record),
            "cwd": str(real_tmp),
            "prompt": prompt,
            "mcp_config_path": str(real_tmp / "mcp-config.json"),
            "timeout_ms": 10_000,
            "base_env": base_env,
            **overrides,
        }
        return Opts(RunOptions(**fields), record)

    return options


def argv(claude: ClaudeConfig, **options: Any) -> list[str]:
    return build_args(RunOptions(claude=claude, cwd="/", prompt="", timeout_ms=1000, base_env={}, **options))


class TestBuildArgs:
    def test_builds_the_design_argv_with_optional_flags(self) -> None:
        args = argv(
            claude_config("/x", model="opus", effort="high", max_turns=30, extra_args=["--foo"]),
            mcp_config_path="/state/mcp-config.json",
            resume_session_id="abc",
        )
        assert args == [
            "-p", "--output-format", "stream-json", "--verbose",
            "--permission-mode", "auto", "--permission-prompts", "none",
            "--strict-mcp-config", "--mcp-config", "/state/mcp-config.json",
            "--setting-sources", "user,local",
            "--append-system-prompt", SPOKEN_ANSWER_PROMPT,
            "--model", "opus", "--effort", "high", "--max-turns", "30", "--resume", "abc", "--foo",
        ]  # fmt: skip

    def test_adds_the_session_options(self) -> None:
        fork = argv(
            claude_config("/x"),
            mcp_config_path="/m",
            resume_session_id="abc",
            fork_session=True,
            permission_mode="plan",
            system_prompt="P",
        )
        assert fork[fork.index("--permission-mode") + 1] == "plan"
        assert fork[-3:] == ["--resume", "abc", "--fork-session"]
        assert fork[fork.index("--append-system-prompt") + 1] == "P"
        relay = argv(
            claude_config("/x", max_turns=30),
            mcp_config_path="/m",
            tools=["SendMessage"],
            session_name="vcm-relay-3",
            max_turns=3,
            fork_session=True,
        )
        assert relay[relay.index("--tools") + 1] == "SendMessage"
        assert relay[relay.index("--name") + 1] == "vcm-relay-3"
        assert relay[relay.index("--max-turns") + 1] == "3"
        assert "--fork-session" not in relay, "no fork without resume"
        none = argv(claude_config("/x"), mcp_config_path="/m", tools=[])
        assert none[none.index("--tools") + 1] == ""


class TestBuildEnv:
    def test_copies_the_allowlist_never_api_keys_and_lets_claude_env_override(self) -> None:
        env = build_env(
            claude_config(
                "/r", config_dir="/cfg", env={"ANTHROPIC_API_KEY": "from-config", "FAKE_CLAUDE_RECORD": "/r"}
            ),
            {
                "HOME": "/home/vcm",
                "PATH": "/usr/bin",
                "ANTHROPIC_API_KEY": "from-server",
                "ANTHROPIC_AUTH_TOKEN": "from-server",
                "CLAUDE_CODE_OAUTH_TOKEN": "tok",
                "OTHER": "x",
            },
        )
        assert env == {
            "HOME": "/home/vcm",
            "PATH": "/usr/bin",
            "CLAUDE_CODE_OAUTH_TOKEN": "tok",
            "CLAUDE_CONFIG_DIR": "/cfg",
            "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "ANTHROPIC_API_KEY": "from-config",
            "FAKE_CLAUDE_RECORD": "/r",
        }
        assert list(env) == [
            "HOME",
            "PATH",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CONFIG_DIR",
            "ENABLE_CLAUDEAI_MCP_SERVERS",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            "DISABLE_AUTOUPDATER",
            "ANTHROPIC_API_KEY",
            "FAKE_CLAUDE_RECORD",
        ]


class TestSummarizeToolUse:
    def test_uses_a_short_target(self) -> None:
        assert summarize_tool_use("Edit", {"file_path": "/a/b/config.py"}) == "Edit config.py"
        assert summarize_tool_use("Bash", {"command": "npm   test"}) == "Bash npm test"
        assert summarize_tool_use("WebFetch", {"url": "https://example.com/x?y=1"}) == "WebFetch example.com"
        assert summarize_tool_use("TodoWrite", {"todos": []}) == "TodoWrite"
        assert len(summarize_tool_use("Bash", {"command": "x" * 100})) < 50


class TestRunClaude:
    async def test_passes_the_prompt_on_stdin_only_with_a_clean_environment(self, make: Any, real_tmp: Path) -> None:
        prompt = "Secret plan: refactor the parser\nRESULT All good."
        opts = make(prompt)
        result = await run_claude(opts.options)
        assert result.ok is True
        assert result.outcome == "success"
        assert result.result == "All good."
        assert result.exit_code == 0
        assert re.match(r"^[0-9a-f-]{36}$", result.session_id or "")
        assert result.cost_usd == 0.0123
        assert result.num_turns == 2
        [rec] = read_records(opts.record)
        assert rec["stdin"] == prompt
        assert not any("Secret plan" in a for a in rec["argv"])
        assert rec["cwd"] == str(real_tmp)
        assert rec["env"]["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
        assert rec["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"
        assert "ANTHROPIC_API_KEY" not in rec["env"]
        assert "ANTHROPIC_AUTH_TOKEN" not in rec["env"]
        assert "SECRET_THING" not in rec["env"]

    async def test_passes_resume_and_keeps_the_session_id(self, make: Any) -> None:
        opts = make("hi", resume_session_id="33333333-0000-4000-8000-000000000001")
        result = await run_claude(opts.options)
        assert result.ok is True
        assert result.session_id == "33333333-0000-4000-8000-000000000001"
        [rec] = read_records(opts.record)
        i = rec["argv"].index("--resume")
        assert rec["argv"][i : i + 2] == ["--resume", "33333333-0000-4000-8000-000000000001"]

    async def test_parses_split_output_with_non_json_lines_and_reports_progress(self, make: Any) -> None:
        seen: list[list[str]] = []
        result = await run_claude(
            make("SPLIT\nPROGRESS 4\nRESULT Příliš žluťoučký kůň.", on_progress=seen.append).options
        )
        assert result.ok is True
        assert result.result == "Příliš žluťoučký kůň."
        assert result.progress == ["Bash npm test -- --reporter dot", "Edit README.md", "Grep TODO"]
        assert len(seen) == 4
        assert seen[0] == ["Read config.py"]
        assert [use.name for use in result.tool_uses] == ["Read", "Bash", "Edit", "Grep"]
        assert result.tool_uses[0] == ToolUse(name="Read", input={"file_path": "/srv/project/src/config.py"})

    async def test_records_the_send_message_calls_of_a_relay_run(self, make: Any) -> None:
        prompt = "Send it.\n----- MESSAGE abc123 -----\nZkontroluj testy.\n----- MESSAGE abc123 -----"
        result = await run_claude(
            make(prompt, tools=["SendMessage"], system_prompt='with "to" set to "uds:/tmp/a.sock", and').options
        )
        assert result.ok is True
        [use] = result.tool_uses
        assert use.name == "SendMessage"
        assert use.input["to"] == "uds:/tmp/a.sock"
        assert use.input["message"] == "Zkontroluj testy."

    async def test_marks_tool_uses_whose_result_is_an_error(self, make: Any) -> None:
        prompt = (
            "SEND_MESSAGE denied_then_sent\n----- MESSAGE abc123 -----\nZkontroluj testy.\n----- MESSAGE abc123 -----"
        )
        result = await run_claude(
            make(prompt, tools=["SendMessage"], system_prompt='with "to" set to "uds:/tmp/a.sock", and').options
        )
        assert [(use.id, use.is_error) for use in result.tool_uses] == [("toolu_send_0", True), ("toolu_send_1", False)]
        assert [d["tool_use_id"] for d in result.permission_denials] == ["toolu_send_0"]

    async def test_treats_is_error_true_as_failure_even_with_subtype_success(self, make: Any) -> None:
        result = await run_claude(make("FAIL").options)
        assert result.ok is False
        assert result.outcome == "error"
        assert result.is_error is True
        assert "API Error" in (result.error or "")

    async def test_fails_without_a_result_event_or_with_a_non_zero_exit_code(self, make: Any) -> None:
        none = await run_claude(make("NO_RESULT").options)
        assert none.ok is False
        assert "no result" in (none.error or "")
        exit_ = await run_claude(make("EXIT 3").options)
        assert exit_.ok is False
        assert exit_.exit_code == 3
        assert exit_.result == "OK"

    async def test_keeps_only_the_last_4_kb_of_stderr(self, make: Any) -> None:
        result = await run_claude(make(f"STDERR {'e' * 10_000}end\nFAIL").options)
        assert len(result.stderr_tail.encode("utf-8")) <= 4096
        assert result.stderr_tail.endswith("end\n")

    async def test_flags_a_resume_of_a_missing_session_and_describes_the_error(self, make: Any) -> None:
        result = await run_claude(make("RESUME_MISSING", resume_session_id="gone").options)
        assert result.ok is False
        assert result.resume_missing is True
        assert result.is_error is True
        assert result.num_turns == 0
        assert result.exit_code == 1
        assert result.result is None
        assert result.error == "error during execution: No conversation found with session ID: gone"
        fresh = await run_claude(make("FAIL", resume_session_id="there").options)
        assert fresh.resume_missing is False

    async def test_does_not_flag_a_resumed_run_that_did_work_and_then_failed(self, make: Any) -> None:
        result = await run_claude(
            make(
                "PROGRESS 1\nSTDERR No conversation found with session ID: there\nFAIL", resume_session_id="there"
            ).options
        )
        assert result.ok is False
        assert result.resume_missing is False
        other = await run_claude(make("STDERR session there not found\nNO_RESULT", resume_session_id="there").options)
        assert other.resume_missing is False
        assert other.error == "no result (exit 0): session there not found"

    async def test_keeps_a_successful_result_when_an_abort_arrives_while_stdout_drains(self, make: Any) -> None:
        abort = asyncio.Event()
        opts = make("SPAWN_HOLDER\nRESULT finished", abort=abort)
        running = asyncio.create_task(run_claude(opts.options))

        # The holder keeps stdout open, so the runner waits in the drain window after claude exits.
        def claude_exited() -> bool:
            records = read_records(opts.record)
            return bool(records) and not is_alive(records[0]["pid"])

        await until(claude_exited)
        abort.set()
        result = await running
        assert result.outcome == "success"
        assert result.ok is True
        assert result.result == "finished"
        assert result.error is None
        [rec] = read_records(opts.record)
        child = rec["childPid"]
        assert child is not None
        await until(lambda: not is_alive(child))

    async def test_kills_the_whole_process_group_on_timeout(self, make: Any) -> None:
        opts = make("SPAWN_CHILD\nSLEEP 60000", timeout_ms=300)
        start = time.monotonic()
        result = await run_claude(opts.options)
        assert result.outcome == "timeout"
        assert result.ok is False
        assert time.monotonic() - start < 5
        [rec] = read_records(opts.record)
        child = rec["childPid"]
        assert child
        await until(lambda: not is_alive(child))
        assert is_alive(rec["pid"]) is False

    async def test_settles_when_claude_exits_even_if_a_child_holds_stdout(self, make: Any) -> None:
        opts = make("SPAWN_HOLDER\nRESULT held")
        start = time.monotonic()
        result = await run_claude(opts.options)
        assert result.ok is True
        assert result.result == "held"
        elapsed = time.monotonic() - start
        assert elapsed < 4, f"took {elapsed:.1f} s"
        [rec] = read_records(opts.record)
        child = rec["childPid"]
        assert child
        await until(lambda: not is_alive(child))

    async def test_kills_leftover_background_processes_after_a_normal_run(self, make: Any) -> None:
        opts = make("SPAWN_CHILD\nRESULT done")
        result = await run_claude(opts.options)
        assert result.ok is True
        [rec] = read_records(opts.record)
        child = rec["childPid"]
        assert child
        await until(lambda: not is_alive(child))

    async def test_escalates_to_sigkill_when_sigterm_is_ignored(self, make: Any) -> None:
        result = await run_claude(make("IGNORE_TERM\nSLEEP 60000", timeout_ms=200, kill_grace_ms=200).options)
        assert result.outcome == "timeout"
        assert result.exit_signal == "SIGKILL"

    async def test_stops_on_abort(self, make: Any) -> None:
        abort = asyncio.Event()
        running = asyncio.create_task(run_claude(make("SLEEP 60000", abort=abort).options))
        asyncio.get_running_loop().call_later(0.2, abort.set)
        result = await running
        assert result.outcome == "aborted"
        assert result.ok is False

        pre = asyncio.Event()
        pre.set()
        assert (await run_claude(make("hi", abort=pre).options)).outcome == "aborted"

    async def test_reports_a_missing_executable(self, make: Any, real_tmp: Path) -> None:
        opts = make("hi")
        opts.options.claude = opts.options.claude.model_copy(update={"bin": str(real_tmp / "no-such-claude")})
        result = await run_claude(opts.options)
        assert result.ok is False
        assert "spawn failed" in (result.error or "")


class TestStdoutLines:
    """The line splitting of `_Run.on_stdout`, fed directly."""

    @staticmethod
    def run(make: Any) -> Any:
        return runner._Run(make("unused").options, asyncio.get_running_loop())

    @staticmethod
    def event(text: str) -> bytes:
        return ('{"type":"result","subtype":"success","is_error":false,"result":"' + text + '"}').encode()

    async def test_joins_lines_split_across_chunks(self, make: Any) -> None:
        run = self.run(make)
        line = self.event("joined")
        run.on_stdout(line[:10])
        run.on_stdout(line[10:20])
        run.on_stdout(line[20:] + b"\n")
        assert run.state.result == "joined"
        assert not run.buffer

    async def test_drops_an_oversized_line_without_its_content_and_keeps_going(
        self, make: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner, "MAX_LINE_BYTES", 100)
        run = self.run(make)
        run.on_stdout(self.event("x" * 60))  # unfinished and over the limit: dropped
        run.on_stdout(self.event("y" * 60))  # same line, still no newline
        assert len(run.buffer) == 0
        assert run.state.dropped_lines == 1
        run.on_stdout(b"z" * 500)
        assert len(run.buffer) == 0, "the rest of the dropped line is skipped"
        run.on_stdout(b"tail\n" + self.event("ok") + b"\n")
        assert run.state.result == "ok"
        assert run.state.dropped_lines == 1
        long_line = self.event("w" * 200)
        run.on_stdout(long_line + b"\n")  # complete in one chunk but too long
        assert run.state.result == "ok"
        assert run.state.dropped_lines == 2

    async def test_cuts_a_very_long_result(self, make: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "RESULT_MAX_CHARS", 10)
        run = self.run(make)
        run.on_stdout(self.event("abcdefghijklmnop") + b"\n")
        assert run.state.result == "abcdefghi…"

    async def test_reads_tool_results_of_large_user_lines_without_a_full_parse(
        self, make: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner, "LARGE_USER_LINE_BYTES", 100)
        run = self.run(make)
        uses = [{"type": "tool_use", "id": f"t{i}", "name": "Read", "input": {"file_path": "/a"}} for i in (1, 2, 3)]
        run.on_stdout(json.dumps({"type": "assistant", "message": {"content": uses}}).encode() + b"\n")

        def user(*blocks: dict[str, Any]) -> bytes:
            return json.dumps({"type": "user", "message": {"content": list(blocks)}}).encode() + b"\n"

        big = "x" * 200 + ' "is_error": true, "tool_use_id": "t3" '
        parsed: list[bytes] = []
        real_loads = json.loads

        def loads(raw: bytes) -> Any:
            parsed.append(raw)
            return real_loads(raw)

        monkeypatch.setattr(json, "loads", loads)
        run.on_stdout(
            user(
                {"tool_use_id": "t1", "type": "tool_result", "content": big, "is_error": False},
                {"tool_use_id": "t2", "type": "tool_result", "content": "ok", "is_error": True},
            )
        )
        assert parsed == [], "a large user line with the usual layout is not parsed"
        assert [use.is_error for use in run.state.tool_uses] == [False, True, None]
        run.on_stdout(user({"type": "tool_result", "is_error": True, "content": big, "tool_use_id": "t3"}))
        assert len(parsed) == 1, "an unusual layout falls back to a full parse"
        assert [use.is_error for use in run.state.tool_uses] == [False, True, True]
