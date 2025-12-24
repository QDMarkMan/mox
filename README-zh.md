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

来自 **X**talPi 的用于药物设计的 **mol agent**，简称 **mox**。

</div>

## 快速开始

### 前置条件

- Python 3.12+
- Node.js 22+（用于客户端开发，可选）
- [uv](https://docs.astral.sh/uv/) - 快速的 Python 包管理器

### 安装

安装 uv（如果尚未安装）：

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

克隆仓库并安装依赖：

```bash
git clone https://github.com/xtalpi.com/molx-agent.git
cd molx-agent
uv sync
```

如果需要安装开发依赖：

```bash
uv sync --extra dev
```

### 使用方法

运行客户端：

```bash
uv run molx --help
```

或先激活虚拟环境：

```bash
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

molx --help
```

## 运行全栈应用

`molx_server/` 中的 FastAPI 服务器暴露了被 `molx_client/` 中的 Vite 客户端消费的聊天与会话 API。完整体验流程：

1. 启动后端
   ```bash
   uv sync --extra server
   uv run molx-server run --reload
   ```
2. 在新终端中安装并启动 Web 客户端
   ```bash
   cd molx_client
   npm install
   echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local  # 使用开发代理时可选
   npm run dev
   ```
3. 打开 http://localhost:5173 与 Agent 聊天。会话创建、历史加载以及 SSE 流都会经过运行中的 API 服务器。生产环境构建时请将 `VITE_API_BASE_URL` 配置为已部署的 FastAPI 地址。

## 架构与文档

- [POC Report](docs/sar_agent_poc.md)：当前能力、演示流程与后续计划。
- [review.md](review.md)：Agent/Mem/Server/Client 的已知问题与风险。
- [todo.md](todo.md)：Agent/Mem/Server/Client 的 TODO 列表。

- [Agent 设计](molx_agent/README.md)：Agent 部分的详细文档
- [Memory 设计](molx_core/README.md)：Memory 部分的详细文档
- [Server 设计](molx_server/README.md)：Server 部分的详细文档
- [Client 设计](molx_client/README.md)：Client 部分的详细文档

## Makefile 用法

[`Makefile`](https://github.com/xtalpi.com/molx-agent/blob/main/Makefile) 提供了许多加速开发的功能。

<details>
<summary>安装全部依赖与 pre-commit 钩子</summary>
<p>

安装依赖：

```bash
# 通过 uv 安装依赖
make install

# 安装客户端 npm 依赖
make install-client
```

在 `git init` 之后可以安装 pre-commit 钩子：

```bash
make pre-commit-install
```

</p>
</details>

<details>
<summary>代码风格与类型检查</summary>
<p>

自动格式化使用 `ruff`：

```bash
make polish-codestyle

# 或使用同义命令
make formatting
```

仅进行代码风格检查，不会改写文件：

```bash
make check-codestyle
```

> 注意：`check-codestyle` 使用 `ruff` 库

</p>
</details>

<details>
<summary>代码安全</summary>
<p>

> 如果安装时未选择该命令，将无法使用。

```bash
make check-safety
```

该命令使用 `Safety` 与 `Bandit` 识别安全问题。

</p>
</details>

<details>
<summary>带覆盖率徽章的测试</summary>
<p>

运行 `pytest`

```bash
make test
```

</p>
</details>

<details>
<summary>所有代码检查</summary>
<p>

当然也有一次性运行全部检查的命令：

```bash
make lint
```

等价于：

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

等同于：

```bash
make docker-build VERSION=latest
```

包含开发依赖的构建：

```bash
docker build -t molx_agent:dev . -f ./docker/Dockerfile --build-arg INSTALL_DEV=true
```

删除镜像：

```bash
make docker-remove
```

更多 [关于 Docker 的信息](https://github.com/Undertone0809/python-package-template/tree/main/%7B%7B%20cookiecutter.project_name%20%7D%7D/docker)。

</p>
</details>

## 运行 API 服务器

MolX 的 FastAPI 后端打包在 `molx_server` 中；运行前先安装 server 额外依赖：

```bash
uv sync --extra dev --extra server
make serve-api            # 等同于 uv run molx-server run --reload
```

服务器默认监听 `http://127.0.0.1:8000`，OpenAPI 文档位于 `/docs`。如需自定义 host/port，可在 `make serve` 前导出 `MOLX_SERVER_HOST`/`MOLX_SERVER_PORT`，或直接运行 `uv run molx-server run --host 0.0.0.0 --port 9000`。

## Web 客户端预览

`molx_client` 提供了一个 React + Vite 的聊天界面：

1. 安装依赖
```bash

cd molx_client
pnpm install
echo "VITE_API_BASE=http://localhost:8000/api/v1" > .env.local
pnpm dev
```
2. 启动服务
```bash
make serve-client
```

访问 `http://localhost:5173` 连接到本地 API。将 `VITE_API_BASE` 更新为远程部署地址即可连接远程服务。

## 🛡 许可证

[![License](https://img.shields.io/github/license/xtalpi.com/molx-agent)](https://github.com/xtalpi.com/molx_agent/blob/main/LICENSE)

本项目基于 `MIT` 许可证发布。详见 [LICENSE](https://github.com/xtalpi.com/molx-agent/blob/main/LICENSE)。

## 📃 引用

```bibtex
@misc{molx-agent,
  author = {tongfu.e},
  title = {Mox Agent},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```
