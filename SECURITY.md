# Security policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Report them privately through GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**. Include what you found, how to reproduce it and the impact you expect. You will get an answer in the advisory thread.

## Supported versions

Only the latest `main` is supported. There are no maintained release branches; fixes land on `main`.

## Scope

voice-coding-mcp gives a remote client the ability to run Claude Code on the owner's machine, so the areas that matter most are:

- **Authentication and OAuth**: pairing window and one-time codes, dynamic client registration and Client ID Metadata Documents, authorization codes, access and refresh tokens, redirect URI and Host/Origin checks.
- **Task execution**: how `claude -p` is started (project selection limited to configured roots, prompt via stdin, `--strict-mcp-config`, permission mode, setting sources, rejected extra arguments), and anything that lets a task or a voice prompt escape those limits or keep access after revocation.
- **Session relay**: listing your own Claude Code sessions, read-only forks (`ask_session_readonly`), passing messages into running sessions (`send_message_to_session`) and reopening tasks with Remote Control.
- **Secrets in logs and state**: tokens, codes or prompts that end up in logs, notifications or state files in clear text.
- **Deployment files**: `deploy/setup.sh`, the systemd unit, the Caddyfile and `deploy/mac/vcm-mac` (for example a Funnel left open after `vcm-mac off`).

Known limits that are already documented (for example that Claude Code runs as the same user as the server, see [Security](README.md#security) in the README and [Security model](docs/DESIGN.md#security-model)) are not new findings, but ideas for reducing them are welcome.

Out of scope: vulnerabilities in Claude Code, the Claude apps, Tailscale or other third-party software (report those to their vendors), and prompt injection that stays within what Claude Code's own permission mode allows.

If you think your own installation was compromised, see [If something is compromised](README.md#if-something-is-compromised) in the README.
