# molx_server

> MolX Server - Application Service Layer for Molx Agent

应用服务层 (Application Layer) - 为 Molx Agent 提供 HTTP API 网关，使用 FastAPI + LangServe 实现高性能异步处理和流式输出支持。

## 特性

- **高性能异步处理**: 基于 FastAPI 的异步架构
- **流式输出支持**: Server-Sent Events (SSE) 实现实时响应流
- **多轮对话**: 会话管理支持上下文保持
- **持久化存储**: 支持 PostgreSQL 后端 (通过 molx_core)
- **API 文档**: 自动生成 OpenAPI/Swagger 文档
- **可配置中间件**: 日志、认证、CORS 等

## 快速开始

### 安装依赖

```bash
cd /data/worksapce/molx_agent
uv sync --extra server
```

### 启动服务器

```bash
# 使用 CLI
uv run molx-server run

# 或使用 Makefile
make serve

# 开发模式 (热重载)
uv run molx-server run --reload --verbose
```

### 访问 API

- API 文档: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

## 存储后端配置

默认使用内存存储，可切换到 PostgreSQL 持久化：

```bash
# 环境变量配置
export MOLX_MEMORY_BACKEND=postgres
export MOLX_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/molx

# 初始化数据库 (可选，自动创建表)
psql -d molx -f molx_core/memory/schema.sql

# 启动服务器
uv run molx-server run
```

### 配置项

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MOLX_MEMORY_BACKEND` | `memory` | 存储后端: `memory` / `postgres` |
| `MOLX_DATABASE_URL` | - | PostgreSQL 连接 URL |
| `MOLX_SESSION_TTL` | `3600` | 会话过期时间 (秒) |

## API 端点

### Agent API (`/api/v1/agent`)

| 端点 | 方法 | 描述 |
|------|------|------|
| `/invoke` | POST | 同步调用 Agent |
| `/stream` | POST | 流式调用 (SSE) |
| `/batch` | POST | 批量处理 |

### SAR 分析 (`/api/v1/sar`)

| 端点 | 方法 | 描述 |
|------|------|------|
| `/analyze` | POST | 执行 SAR 分析 |
| `/stream` | POST | 流式 SAR 分析 |
| `/report` | POST | 生成报告 |

### 会话管理 (`/api/v1/session`)

| 端点 | 方法 | 描述 |
|------|------|------|
| `/create` | POST | 创建会话 |
| `/{id}` | GET | 获取会话信息 |
| `/{id}` | DELETE | 删除会话 |
| `/{id}/history` | GET | 获取历史 |

## 架构

```
molx_core/           # 共享模块 (存储抽象)
├── memory/
│   ├── store.py          # 抽象接口
│   ├── memory_store.py   # 内存实现
│   └── postgres_store.py # PostgreSQL 实现

molx_server/         # HTTP 服务层
├── app.py               # FastAPI 应用
├── session.py           # 会话管理 (使用 molx_core)
├── routes/              # API 路由
└── middleware/          # 中间件
```

## 使用示例

### Python

```python
import httpx

# 创建会话
response = httpx.post("http://localhost:8000/api/v1/session/create")
session_id = response.json()["session_id"]

# 调用 Agent
response = httpx.post(
    "http://localhost:8000/api/v1/agent/invoke",
    json={"query": "Analyze SAR", "session_id": session_id}
)
print(response.json()["result"])
```

### cURL

```bash
# Agent 调用
curl -X POST http://localhost:8000/api/v1/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, what can you do?"}'
```