# Mox

<div align="center">

[![Build status](https://github.com/xtalpi.com/molx-agent/workflows/build/badge.svg?branch=main&event=push)](https://github.com/xtalpi.com/molx-agent/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/molx-agent.svg)](https://pypi.org/project/molx-agent/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/xtalpi.com/molx-agent/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/xtalpi.com/molx-agent/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/xtalpi.com/molx-agent/releases)
[![License](https://img.shields.io/github/license/xtalpi.com/molx-agent)](https://github.com/xtalpi.com/molx-agent/blob/main/LICENSE)
![Coverage Report](assets/images/coverage.svg)

A **mol agent** used for drug design from **X**talPi, you can refer to it simply as **mox**.

</div>

## Quick start

### Prerequisites

- Python 3.12+
- Node.js 22+ (for client development, optional)
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
uv run molx --help
```

Or activate the virtual environment first:

```bash
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

molx --help
```

## Run the full-stack app

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

## Architecture & Docs

- [POC Report](docs/sar_agent_poc.md): Current capabilities, demo flow, and next steps.
- [review.md](review.md): Known issues and risks for Agent/Mem/Server/Client.
- [todo.md](todo.md): TODOs for Agent/Mem/Server/Client.

- [Agent Design](molx_agent/README.md): Detail document for agent part
- [Memory Design](molx_core/README.md): Detail document for memory part
- [Server Design](molx_server/README.md): Detail document for server part
- [Client Design](molx_client/README.md): Detail document for client part


## Makefile usage

[`Makefile`](https://github.com/xtalpi.com/molx-agent/blob/main/Makefile) contains a lot of functions for faster development.

<details>
<summary>Install all dependencies and pre-commit hooks</summary>
<p>

Install requirements:

```bash
# install dependencies by uv
make install

# install dependencies by npm for client
make install-client
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

The MolX FastAPI backend is packaged in `molx_server`; install the server extra before running:

```bash
uv sync --extra dev --extra server
make serve-api            # equivalent to uv run molx-server run --reload
```

The server listens on `http://127.0.0.1:8000` by default with OpenAPI docs at `/docs`. To customize the host/port, export `MOLX_SERVER_HOST`/`MOLX_SERVER_PORT` before `make serve` or run `uv run molx-server run --host 0.0.0.0 --port 9000` directly.

## Web client preview

`molx_client` ships a React + Vite chat UI:

1. Install dependencies
```bash

cd molx_client
pnpm install
echo "VITE_API_BASE=http://localhost:8000/api/v1" > .env.local
pnpm dev
```
2. Start the service
```bash
make serve-client
```

Visit `http://localhost:5173` to connect to the local API. Update `VITE_API_BASE` to point at a remote deployment.

## 🛡 License

[![License](https://img.shields.io/github/license/xtalpi.com/molx-agent)](https://github.com/xtalpi.com/molx_agent/blob/main/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/xtalpi.com/molx-agent/blob/main/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{molx-agent,
  author = {tongfu.e},
  title = {Mox Agent},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```
