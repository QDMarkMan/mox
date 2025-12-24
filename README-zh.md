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

来自 **X**talPi 的用于药物设计的 **mol agent**，可简称 **mox**。

</div>

## 快速开始

### 前置条件

- Python 3.12+
- Node.js 22+(客户端开发环境，可选)
- [uv](https://docs.astral.sh/uv/) - 快速的 Python 包管理器

### 安装

安装 uv（如尚未安装）：

```bash
# 在 macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在 Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

克隆仓库并安装依赖：

```bash
git clone https://github.com/xtalpi.com/molx-agent.git
cd molx-agent
uv sync
```

安装开发依赖：

```bash
uv sync --extra dev
```

### 用法

运行客户端：

```bash
uv run molx-agent --help
```

或先激活虚拟环境：

```bash
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

molx-agent --help
```

## 运行全栈 MVP

`molx_server/` 中的 FastAPI 服务器现在暴露了被 `molx_client/` 消费的聊天与会话 API。要体验完整流程：

1. 启动后端
   ```bash
   uv sync --extra server
   uv run molx-server run --reload
   ```
2. 在新终端安装并启动 Web 客户端
   ```bash
   cd molx_client
   npm install
   echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local  # 使用 dev 代理时可选
   npm run dev
   ```
3. 打开 http://localhost:5173 与智能体聊天。会话创建、历史加载和 SSE 流都会经过正在运行的 API 服务器。生产环境下请将 `VITE_API_BASE_URL` 指向部署好的 FastAPI 实例。

## 会话文件与制品预览

FastAPI 服务器提供了专用端点，将用户上传保存到智能体的 `artifacts_root/uploads/<session_id>` 目录（如需调整目录结构，可在 `molx_agent/config.py` 中设置 `uploads_subdir`）。上传文件会同步到智能体状态，DataCleaner worker 可立即消费，而每份生成的报告或 `output_files` 也会写回会话元数据用于预览。

可用端点：

- `POST /api/v1/session/{session_id}/files` — 接收带 `uploaded_file` 字段和可选 `description` 的 `multipart/form-data`，将文件存盘并注册到会话记忆。
- `GET /api/v1/session/{session_id}/files` — 列出用户上传与生成制品，便于客户端展示预览或下载链接。
- `GET /api/v1/session/{session_id}/files/{file_id}` — 以正确的 MIME 类型流式传输跟踪文件，方便快速预览 HTML/JSON。

`/session/{id}/data` 与 `/session/{id}/files` 返回的会话元数据包含 `uploaded_files`、`artifacts` 以及每轮的制品摘要，方便 UI 显示每份报告的来源轮次。

## Makefile 用法

[`Makefile`](https://github.com/xtalpi.com/molx-agent/blob/main/Makefile) 提供了多种快速开发的命令。

<details>
<summary>安装全部依赖与 pre-commit 钩子</summary>
<p>

安装依赖：

```bash
make install
```

在 `git init` 后可安装 pre-commit 钩子：

```bash
make pre-commit-install
```

</p>
</details>

<details>
<summary>代码格式与类型检查</summary>
<p>

自动格式化使用 `ruff`：

```bash
make polish-codestyle

# 或使用同义命令
make formatting
```

仅做格式检查，不改写文件：

```bash
make check-codestyle
```

> 说明：`check-codestyle` 使用 `ruff` 库

</p>
</details>

<details>
<summary>代码安全</summary>
<p>

> 若安装时未选择该命令，则无法使用。

```bash
make check-safety
```

该命令使用 `Safety` 与 `Bandit` 识别安全问题。

</p>
</details>

<details>
<summary>带覆盖率徽章的测试</summary>
<p>

运行 `pytest`：

```bash
make test
```

</p>
</details>

<details>
<summary>全部 linters</summary>
<p>

当然也可以一条命令跑完所有检查：

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

携带开发依赖构建：

```bash
docker build -t molx_agent:dev . -f ./docker/Dockerfile --build-arg INSTALL_DEV=true
```

移除镜像：

```bash
make docker-remove
```

更多信息参见 [docker](https://github.com/Undertone0809/python-package-template/tree/main/%7B%7B%20cookiecutter.project_name%20%7D%7D/docker)。

</p>
</details>

<details>
<summary>清理</summary>
<p>
删除 pycache 文件：

```bash
make cleanup
```

</p>
</details>

## 运行 API 服务器

MolX 的 FastAPI 后端位于 `molx_server`，运行前需要安装 server extra：

```bash
uv sync --extra dev --extra server
make serve-api            # 等价于 uv run molx-server run --reload
```

默认监听 `http://127.0.0.1:8000`，OpenAPI 文档位于 `/docs`。如需自定义主机或端口，可在执行 `make serve` 前导出 `MOLX_SERVER_HOST`/`MOLX_SERVER_PORT`，或直接运行 `uv run molx-server run --host 0.0.0.0 --port 9000`。

## Web 客户端预览

`molx_client` 提供基于 React + Vite 的聊天界面：

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

在浏览器访问 `http://localhost:5173` 即可连接本地 API。修改 `VITE_API_BASE` 可指向远程部署。

## 架构与文档

- [POC 报告](docs/sar_agent_poc.md)：当前能力、演示流程与下一步计划。
- [review.md](review.md)：Agent/Mem/Server/Client 的缺陷列表与风险提示。
- [todo.md](todo.md)：Agent/Mem/Server/Client 的待办事项列表。

- [Agent 设计](molx_agent/README.md)
- [Memory 设计](molx_core/README.md)
- [Server 设计](molx_server/README.md)
- [Client 设计](molx_client/README.md)

## 🛡 许可证

[![License](https://img.shields.io/github/license/xtalpi.com/molx-agent)](https://github.com/xtalpi.com/molx_agent/blob/main/LICENSE)

本项目基于 `MIT` 许可证发布，详见 [LICENSE](https://github.com/xtalpi.com/molx-agent/blob/main/LICENSE)。

## 📃 引用

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
