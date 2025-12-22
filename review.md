# MolX 全栈项目架构 Review

## 总览

- **Agent 层**：`MolxAgent` 通过意图识别 → 规划 → DataCleaner/SAR/Reporter 工人 → 反思优化的 ReAct 流程，但缺少 LangGraph 图编排实现，仍以顺序流程执行。
- **Memory 层**：`molx_core.memory` 定义了可插拔的会话存储、录制器与 PostgreSQL 适配，但服务器端仅异步封装，没有在 CLI 中暴露。
- **Server 层**：FastAPI 服务 `molx_server` 暴露 /agent 与 /sar API，并通过 `SessionManager` 复用对话。然而路由内部仍同步调用阻塞式 LLM，且 SSE 仅在任务结束后回放文本。
- **Client 层**：`molx_client` 提供 React/Tailwind 聊天 UI，默认连接 `/api/v1/agent/stream`，但缺少与服务端真实流式机制和会话列表的打通。
- **Tools/Config**：RDKit/标准化/MCP 工具序列已就绪，但输出路径、配置项和测试覆盖仍需补足。

## 主要缺陷

### Agents & Tools

1. **LangGraph DAG 仍未真正实现**：`molx_agent/agents/molx.py:83-331` 里的 `build_sar_graph/get_sar_graph/run_sar_agent` 虽已集中在同一模块，但内部依然是手写的顺序流程与自定义状态机，没有利用 LangGraph 的异步/容错能力，导致导出名与实际能力不符。
2. **数据清洗仅支持文件系统路径**：`molx_agent/agents/data_cleaner.py:103-144` 直接匹配字符串中的路径并用 `os.path.exists` 检查，无法处理通过 API 上传的文件内容或对象存储路径，也没有安全白名单，部署在服务器后几乎不可用。
3. **输出目录硬编码**：`molx_agent/tools/standardize.py:20-116` 将所有清洗结果写入 `os.getcwd()/output`。这会在 CLI/Server/多租户环境下产生权限和冲突问题，也无法依据 session_id 隔离产物。
4. **工具加载日志与实际不符**：`molx_agent/agents/modules/tools.py:18-75` 多处日志写死“Loaded ... (3)”但实际载入 5 个工具，同时缺少对工具失败的上报。虽然问题轻量，但会误导排障。

### Memory / Session

1. **过期清理比较不正确**：`molx_server/session.py:257-270` 把 `datetime.timestamp()`（epoch 秒）与 `asyncio.get_event_loop().time()`（monotonic 秒）相减，量纲不一致，会把所有历史 session 视为已过期或永不过期，导致缓存频繁抖动或泄露。
2. **CLI 未接入持久化**：虽然有 `SessionRecorder`（`molx_core/memory/recorder.py:114-205`），但 CLI/agents 仅在 server 里被绑定，终端模式无法复用历史记录或导出报告列表，违背了“全栈记忆”初衷。

### Server API

1. **异步路由内执行阻塞 LLM**：`molx_server/routes/agent.py:45-167`、`molx_server/routes/sar.py:42-86` 直接调用 `chat_session.send` 和 `run_sar_agent`。二者内部会访问 OpenAI/本地 LLM 与 RDKit，阻塞事件循环，任何一次请求都会挂住整个 Uvicorn worker。需要 `run_in_executor` 或专门的任务队列。
2. **SSE 仅回放结果**：`molx_server/streaming.py:157-250` 把完整回答生成完成后再拆词发送 token，并未把 `StreamingCallbackHandler` 注册到 `chat_session`。前端虽然显示“流式”，但用户要等待所有计算结束才看到第一个 token。
3. **SAR API 未复用 session**：`molx_server/routes/sar.py:42-74` 调用 `run_sar_agent` 新建 `MolxAgent`，与现有 Session/MCP 配置完全割裂，也不会写入 `session_data`，导致 /sar 与 /agent 的记忆、速率限制无法统一。

### Client

1. **会话列表仅本地模拟**：`molx_client/src/App.tsx:5-34` 生成的 sessionId 与服务器无关联，刷新即丢失，也无法列出服务器中真实的历史记录，破坏“云端对话”体验。
2. **未处理 SSE 错误/心跳**：`molx_client/src/hooks/use-streaming-chat.ts:80-166` 未处理 204/超时/心跳，也没有在中断后回收 `AbortController`，连续失败将导致悬挂请求。

### 测试与工程质量

1. **关键路径无测试**：测试目录只验证 MCP Loader（`tests/test_agents/test_mcp.py`）和少量 session route，缺少对 Planner、DataCleaner、SAR、Reporter、FastAPI endpoints、React hooks 的覆盖，意味着 CI 很难捕获上述缺陷。
2. **缺少 README 对 Server/Client 的指引**：当前 README 只记录 CLI/Make 命令，没有提及 `uv sync --extra server`、`make serve`、前端构建/部署，也没有指向新的 POC/Review 文档。

## 建议

- 补充缺失的 LangGraph 编排或删除相关导出，避免误导使用者。
- 将文件处理重构为「上传 → 临时存储 → DataCleaner」，并把输出挂钩 session 特定目录或对象存储。
- 修复 session TTL 计算、在 FastAPI 路由中使用线程池/后台 worker，并真正把 LangChain callback 接入 SSE。
- 前端需要通过 API 拉取 session 元数据，并处理 SSE 的心跳、重连。
- 逐步补齐单元/集成测试，至少覆盖 Planner/SAR Agent 主流程与关键 API。
- 在 README 中同步这些状态，指向本文与 POC 文档，降低新成员上手成本。
