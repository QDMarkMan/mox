# Repository Guidelines

## Project Structure & Module Organization

Python sources live in `molx_agent`, with CLI entrypoints in `__main__.py`, reusable agents under `agents/`, stateful helpers in `memory/`, and tool shims in `tools/`. Shared logic for other runtimes sits in `molx_core/`, the API server in `molx_server/`, and the web client in `molx_client/`. Reference assets (including coverage badges) stay in `assets/`, configuration defaults in `config/`, and container files in `docker/`. Place runnable walkthroughs inside `examples/` and mirrored explanations in `docs/`. All tests live in `tests/`; mirror the source layout when adding suites to keep discovery predictable.

## Build, Test, and Development Commands

- `uv sync --extra dev` — create the virtualenv and install app + dev tooling.
- `make install && make pre-commit-install` — sync Poetry/conda workflows and register hooks for formatting and linting.
- `uv run molx-agent --help` — smoke-test the CLI entrypoints locally.
- `make formatting` / `make check-codestyle` — auto-format or lint with Ruff without touching files.
- `make lint` — run style, safety, and test suites in the same pass (equivalent to `make check-codestyle && make test && make check-safety`).
- `make test` — execute `pytest` with coverage, regenerating `assets/images/coverage.svg` for the badge.
- `make docker-build VERSION=latest` — produce the container image; pair with `make docker-remove` for cleanup.

## Coding Style & Naming Conventions

Target Python 3.12, 4-space indentation, and explicit type hints for all public APIs. Keep imports sorted by Ruff’s defaults and prefer descriptive names like `run_sar_agent` over abbreviations. Write module docstrings summarizing intent and include Markdown-friendly help text for CLI commands. Any new configuration knobs should land in `molx_agent/config.py` with snake_case keys and mirrored documentation in `docs/`.

## Testing Guidelines

Use `pytest` with the standard `test_*.py` naming under folders that mirror their source counterparts (e.g., tests for `molx_agent/agents/run.py` live in `tests/agents/test_run.py`). Cover new branches before opening a PR and keep overall coverage at or above the current badge level (visible in `assets/images/coverage.svg`). For focused checks, run `uv run pytest -k <pattern>` locally, but always finish with `make test` to match CI.

## Commit & Pull Request Guidelines

Follow Conventional Commits (`feat:`, `fix:`, `chore:`) as shown in recent history, grouping related changes into a single commit where possible. PRs should describe intent, list key commands run (formatting, lint, tests), and link the relevant issue. UI-facing updates (e.g., under `molx_client/`) should attach screenshots or short clips. Keep branch diffs small, note schema or API impacts explicitly, and ensure documentation/examples stay in sync with code changes.

## Security & Configuration Tips

Never commit secrets; load agent credentials via environment variables and reference them through the helpers in `config/mcp_servers.json`. Validate third-party tool access with `make check-safety` before merging, and prefer scoped tokens when testing against remote services. When sharing datasets or checkpoints, store them outside the repo and document retrieval steps in `docs/`.
