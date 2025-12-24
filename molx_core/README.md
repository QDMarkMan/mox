# molx_core

MolX 生态的共享核心库，提供配置、会话存储与元数据工具，供 CLI 智能体（`molx_agent/`）和 FastAPI 服务端（`molx_server/`）共用。

## 注意

**当前的版本只是雏形，对于记忆数据模型，需要进一步完成更加可靠的数据存储方案**

## 包结构

- `config.py`：基于 Pydantic 的 `CoreSettings`（环境变量驱动），内置内存/数据库后端选择与连接池默认值。
- `memory/`：存储抽象与实现。
  - `store.py`：`ConversationStore` 接口与 `SessionData` 容器。
  - `memory_store.py`：线程安全的内存后端（默认）。
  - `postgres_store.py`：基于 `asyncpg` 的异步 PostgreSQL 后端，会创建 `conversations` 表（见 `schema.sql`）。
  - `recorder.py`：`SessionRecorder` 及元数据结构（`SessionMetadata`、`TurnRecord`、`ReportRecord`、`FileRecord`），用于记录对话轮次、制品与报告。
  - `factory.py`：`get_conversation_store`、`initialize_store`、`close_store` 助手。
  - `DESIGN.md`：记忆层端到端设计说明。

## 配置

`CoreSettings` 读取环境变量（前缀 `MOLX_`，若存在 `.env.local` 优先于 `.env`）：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MOLX_MEMORY_BACKEND` | `memory` | 存储后端：`memory` 或 `postgres`。 |
| `MOLX_DATABASE_URL` | _None_ | PostgreSQL 连接串（`postgres` 模式必填），支持 `postgresql+asyncpg://...`，会去掉 `+asyncpg` 以供 `asyncpg` 使用。 |
| `MOLX_DATABASE_POOL_SIZE` | `5` | Postgres 连接池大小。 |
| `MOLX_DATABASE_MAX_OVERFLOW` | `10` | 连接池额外容量（当前仅从配置读取）。 |
| `MOLX_DATABASE_POOL_TIMEOUT` | `30.0` | 连接池获取超时时间（秒）。 |
| `MOLX_SESSION_TTL` | `3600` | 会话存活时长，供清理任务使用。 |
| `MOLX_SESSION_CLEANUP_INTERVAL` | `300` | 清理任务周期（秒）。 |

## 选择存储后端

```bash
# 内存（默认）
export MOLX_MEMORY_BACKEND=memory

# PostgreSQL
export MOLX_MEMORY_BACKEND=postgres
export MOLX_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/molx
```

在 Postgres 模式下，`initialize_store()` 会使用 `postgres_store.py` 中的 SQL（`schema.sql` 同步）自动创建 `conversations` 表。

## 用法示例

应用启动/关闭的典型流程：

```python
from molx_core.memory.factory import get_conversation_store, initialize_store, close_store

async def startup():
    await initialize_store()

async def shutdown():
    await close_store()

async def handle_message(session_id: str, role: str, content: str):
    store = get_conversation_store()
    session = await store.get(session_id) or await store.create(session_id)
    session.add_message(role, content)
    await store.save(session)
```

记录轮次与制品：

```python
from molx_core.memory.recorder import SessionRecorder

async def record_turn(session, query: str, response: str, state: dict):
    recorder = SessionRecorder(session)
    recorder.record_turn(query=query, response=response, state=state)
    await get_conversation_store().save(session)
```

`SessionRecorder` 会限制存储的轮次/报告/文件数量（`MAX_TURNS_STORED`、`MAX_REPORTS_STORED`、`MAX_FILE_RECORDS_STORED`），同时保留 `latest`、`reports` 及结构化结果摘要，避免载荷膨胀。

## 开发与测试

```bash
uv sync --extra dev
make test          # 运行 pytest 并更新覆盖率徽章
make lint          # 格式、静态检查与安全扫描
```

更深入的记忆层设计可参见 `memory/DESIGN.md`，项目级工作流参考仓库根目录的 `README.md`。
