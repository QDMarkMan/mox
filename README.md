# molx-agent

<div align="center">

[![Build status](https://github.com/xtalpi.com/molx-agent/workflows/build/badge.svg?branch=main&event=push)](https://github.com/xtalpi.com/molx-agent/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/molx-agent.svg)](https://pypi.org/project/molx-agent/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/xtalpi.com/molx-agent/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/xtalpi.com/molx-agent/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/xtalpi.com/molx-agent/releases)
[![License](https://img.shields.io/github/license/xtalpi.com/molx-agent)](https://github.com/xtalpi.com/molx-agent/blob/main/LICENSE)
![Coverage Report](assets/images/coverage.svg)

Drug design agent

</div>

## Quick start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager

### Installation

Install uv (if not already installed):

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone the repository and install dependencies:

```bash
git clone https://github.com/xtalpi.com/molx-agent.git
cd molx-agent
uv sync
```

To install with development dependencies:

```bash
uv sync --extra dev
```

### Usage

Run the client using:

```bash
uv run molx-agent --help
```

Or activate the virtual environment first:

```bash
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

molx-agent --help
```

## Run the full-stack MVP

The FastAPI server in `molx_server/` now exposes the chat + session APIs consumed by the Vite
client in `molx_client/`. To exercise the complete flow:

1. Start the backend
   ```bash
   uv sync --extra server
   uv run molx-server run --reload
   ```
2. In a new terminal, install and launch the web client
   ```bash
   cd molx_client
   npm install
   echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local  # optional when using the dev proxy
   npm run dev
   ```
3. Open http://localhost:5173 to chat with the agent. Session creation, history loading, and the
   SSE stream all traverse the running API server. For production builds configure
   `VITE_API_BASE_URL` to point at the deployed FastAPI instance.

## Session files and artifact previews

The FastAPI server now exposes dedicated endpoints so that user uploads are persisted under the
agent's `artifacts_root/uploads/<session_id>` directory (configure the `uploads_subdir` field in
`molx_agent/config.py` if you need a different layout). Uploaded files are reflected in the agent
state so the DataCleaner worker can immediately consume them, and every generated report or
`output_files` entry is mirrored back into session metadata for previewing.

Available endpoints:

- `POST /api/v1/session/{session_id}/files` — accepts `multipart/form-data` with an
  `uploaded_file` field and optional `description`, stores the file on disk, and registers it in the
  session memory.
- `GET /api/v1/session/{session_id}/files` — lists both user uploads and generated artifacts so the
  client can present previews or download links.
- `GET /api/v1/session/{session_id}/files/{file_id}` — streams the binary contents of a tracked
  file with the correct MIME type for quick HTML/JSON previews.

The session metadata returned by `/session/{id}/data` and `/session/{id}/files` now carries
`uploaded_files`, `artifacts`, and per-turn artifact summaries so the UI can highlight which report
belongs to which agent turn.

## Makefile usage

[`Makefile`](https://github.com/xtalpi.com/molx-agent/blob/main/Makefile) contains a lot of functions for faster development.

<details>
<summary>Install all dependencies and pre-commit hooks</summary>
<p>

Install requirements:

```bash
make install
```

Pre-commit hooks could be installed after `git init` via

```bash
make pre-commit-install
```

</p>
</details>

<details>
<summary>Codestyle and type checks</summary>
<p>

Automatic formatting uses `ruff`.

```bash
make polish-codestyle

# or use synonym
make formatting
```

Codestyle checks only, without rewriting files:

```bash
make check-codestyle
```

> Note: `check-codestyle` uses `ruff` library

</p>
</details>

<details>
<summary>Code security</summary>
<p>

> If this command is not selected during installation, it cannot be used.

```bash
make check-safety
```

This command identifies security issues with `Safety` and `Bandit`.

</p>
</details>

<details>
<summary>Tests with coverage badges</summary>
<p>

Run `pytest`

```bash
make test
```

</p>
</details>

<details>
<summary>All linters</summary>
<p>

Of course there is a command to run all linters in one:

```bash
make lint
```

the same as:

```bash
make check-codestyle && make test && make check-safety
```

</p>
</details>

<details>
<summary>Docker</summary>
<p>

```bash
make docker-build
```

which is equivalent to:

```bash
make docker-build VERSION=latest
```

Build with development dependencies:

```bash
docker build -t molx_agent:dev . -f ./docker/Dockerfile --build-arg INSTALL_DEV=true
```

Remove docker image with

```bash
make docker-remove
```

More information [about docker](https://github.com/Undertone0809/python-package-template/tree/main/%7B%7B%20cookiecutter.project_name%20%7D%7D/docker).

</p>
</details>

<details>
<summary>Cleanup</summary>
<p>
Delete pycache files

```bash
make cleanup
```

</p>
</details>

## Running the API server

MolX 的 FastAPI 后端打包在 `molx_server`，需要安装 server extra：

```bash
uv sync --extra dev --extra server
make serve            # 等价于 uv run molx-server run --reload
```

默认监听 `http://127.0.0.1:8000`，OpenAPI 文档位于 `/docs`。若要自定义主机/端口，可在 `make serve` 前导出 `MOLX_SERVER_HOST/MOLX_SERVER_PORT` 或直接运行 `uv run molx-server run --host 0.0.0.0 --port 9000`。

## Web client preview

`molx_client` 提供基于 React + Vite 的最小聊天界面：

```bash
cd molx_client
pnpm install
echo "VITE_API_BASE=http://localhost:8000/api/v1" > .env.local
pnpm dev
```

浏览器访问 `http://localhost:5173` 即可连到本地 API。修改 `VITE_API_BASE` 可指向远程部署。

## Architecture & Docs

- [SAR Agent POC 报告](docs/sar_agent_poc.md)：当前能力、演示流程与下一步计划。
- [review.md](review.md)：针对 Agent/Mem/Server/Client 的缺陷列表与风险提示。

## 🛡 License

[![License](https://img.shields.io/github/license/xtalpi.com/molx-agent)](https://github.com/xtalpi.com/molx_agent/blob/main/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/xtalpi.com/molx-agent/blob/main/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{molx-agent,
  author = {tongfu.e},
  title = {Drug design agent},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/QDMarkman/molx-agent}}
}
```
