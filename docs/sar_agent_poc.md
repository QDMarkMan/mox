# SAR Agent POC 报告

> 状态：2025-12-21 · Owner: MolX Core Team

## 1. 目标 & 范围
- 证明 LangChain/LLM + RDKit 工具链可自动完成 SAR（Structure-Activity Relationship）分析、报告撰写与多轮对话。
- 搭建“Agent → Memory → Server → Client”全栈骨架，支撑后续产品化。
- 不包含：真实数据库、权限体系、自动化部署、GPU 调度。

## 2. 架构概览
```
CLI / React UI ─┬─> FastAPI (molx_server)
                │       ├─ SessionManager / molx_core.memory
                │       └─ LangGraph orchestration
                └─> Direct CLI (uv run molx-agent)

LangGraph Graph
  ├─ IntentClassifier (LLM)
  ├─ Planner (LLM ReAct)
  └─ Workers
      ├─ DataCleaner → extractor/standardize 工具 + RDKit
      ├─ SARAgent   → R group / OCAT / LLM insight
      └─ Reporter   → RunFullSARAnalysis/GenerateHTML
```
- **配置**：`molx_agent/config.py`（LLM/MCP）、`molx_core/config.py`（存储）、`molx_server/config.py`（API）。
- **工具**：RDKit / pandas / langchain-mcp-adapters，全部通过 `molx_agent.tools.*` 管理。
- **记忆**：`SessionRecorder` 将任务规划、反思、报告元数据写入 `SessionData`，支持内存或 PostgreSQL。

## 3. 关键能力
| 能力 | 描述 | 入口 |
| --- | --- | --- |
| SAR 分析 CLI | `uv run molx-agent sar "Analyze ..."` 输出 Markdown 报告 | CLI |
| 交互式 Chat | `uv run molx-agent chat`，支持历史、清空、打印 JSON | CLI |
| FastAPI API | `/api/v1/agent/invoke`（同步）、`/api/v1/agent/stream`（SSE）、`/api/v1/agent/batch` | Server |
| 专用 SAR API | `/api/v1/sar/analyze`、`/api/v1/sar/stream` | Server |
| React Web UI | `molx_client` Vite + Tailwind，调用 `/api/v1/agent/stream`，支持多 Session 列表（本地） | Client |
| Session 存储 | `molx_core.memory` 提供内存/数据库实现、转写报告、清理任务 | Memory |

## 4. 演示流程
1. `uv sync --extra dev --extra server` 初始化虚拟环境和 server 依赖。
2. `make serve` -> `uv run molx-server run --reload` 启动 API，默认 `http://localhost:8000/docs`。
3. 另起终端 `cd molx_client && pnpm install && pnpm dev`，在 `.env` 中配置 `VITE_API_BASE=http://localhost:8000/api/v1` 以体验 UI。
4. CLI 演示：
   ```bash
   uv run molx-agent sar "Analyze SAR of celecoxib analogs for COX-2 selectivity"
   ```
   观察 DataCleaner -> SARAgent -> Reporter 的控制台输出和 `output/cleaned_*.json`。

## 5. 现状评估
- ✅ Agent 工作流已经完成：意图识别、规划、DataCleaner/SAR/Reporter 的串联，以及 RDKit/报告工具。
- ✅ Memory/SessionManager 可把 CLI 状态录入 store，FastAPI 端具备异步 CRUD 与定时清理。
- ✅ Server/Client 骨架跑通 SSE/批量接口，React UI 具备最小体验。
- ⚠️ 真实 LangGraph DAG 尚未落地（`molx_agent/agents/molx.py` 仍使用手写 ReAct），需要补齐 LangGraph 图编排。
- ⚠️ FastAPI 路由同步执行 LLM，缺少真正的流式 token 推送，前端体验与后端实现不一致。
- ⚠️ DataCleaner 仅支持本地路径输入，Server 场景无法上传文件，且输出目录固定为 repo 下 `output/`。
- ⚠️ 测试覆盖率不足，目前只覆盖 MCP Loader 与少量 Session API。

## 6. 风险与依赖
1. **LLM**：依赖本地 `ChatOpenAI` 兼容接口（`LOCAL_OPENAI_*`），需保证模型可访问，且具备工具调用权限。
2. **化学栈**：RDKit、pandas、safety/bandit 在 `uv` 环境中安装时间较长，CI/CD 需准备缓存镜像。
3. **存储**：默认内存实现，PostgreSQL 版本尚未在 README 中指引用户启用，需要补充迁移脚本和连接配置。
4. **安全**：缺少 API Key / CORS 白名单默认值，只能在受控环境运行。

## 7. POC 完成度与下一步
| 模块 | 完成度 | 下一步 |
| --- | --- | --- |
| Agent & Tools | 70% | 实装 LangGraph DAG、完善 DataCleaner 输入渠道、增加工具异常遥测 |
| Memory | 60% | 修复 TTL 算法、在 CLI 中可选持久化、暴露历史查询 API |
| Server | 50% | 将 `chat_session.send` 放入 executor/worker，接入 LangChain streaming callbacks，补齐鉴权/限速 |
| Client | 40% | 与服务器 SessionManager 同步、处理 SSE 断线/心跳、增加报告下载入口 |
| Testing & Docs | 30% | 建立端到端 smoke test、覆盖 Planner/SAR/Reporter、在 README/Docs 中列出 Review & POC 状态 |

---
**结论**：POC 已验证“多智能体 SAR 分析 + 报告”核心价值，下一阶段应集中在工程化（异步执行、文件流、鉴权）和产品化（Session UI、报告管理、稳定性测试），以支撑真实药化团队落地。
