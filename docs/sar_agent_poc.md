# SAR Agent POC 报告

> 状态：2025-12-22 · Owner: MolX Core Team

## 1. 目标 & 范围
- 证明 LangChain/LLM + RDKit 工具链可自动完成 SAR（Structure-Activity Relationship）分析、报告撰写与多轮对话。
- 搭建“Agent → Memory → Server → Client”全栈骨架，支撑后续产品化。
- 不包含：真实数据库、权限体系、自动化部署、GPU 调度。

## 2. 架构概览
```
CLI / React UI ─┬─> FastAPI (molx_server)
                │       ├─ SessionManager / molx_core.memory
                │       └─ LangGraph orchestration
                └─> Direct CLI (uv run molx)

LangGraph Graph
  ├─ IntentClassifier (LLM)
  ├─ Planner (LLM ReAct)
  └─ Workers
      ├─ DataCleaner → extractor/standardize 工具 + RDKit
      ├─ SARAgent   → R group / OCAT / LLM insight
      └─ Reporter   → RunFullSARAnalysis/GenerateHTML/BuildReportSummary
```
- **LangGraph 流程**：`molx_agent.agents.modules.graph` 将 Intent → Plan → Act → Reflect → Optimize → Finalize 的状态机编译为 `StateGraph`，并在 CLI / Server / Client 中复用。
- **配置**：`molx_agent/config.py`（LLM/MCP）、`molx_core/config.py`（存储）、`molx_server/config.py`（API）。
- **工具**：RDKit / pandas / langchain-mcp-adapters，全部通过 `molx_agent.tools.*` 管理，产物统一写入 `artifacts/`（`get_tool_output_dir` 管理子目录）。
- **记忆**：`SessionRecorder` 将任务规划、反思、报告元数据写入 `SessionData`，支持内存或 PostgreSQL。
- **流控**：`molx_server.streaming.stream_agent_response` 捕获 LangGraph 状态，推送 `start/status/thinking/token/complete` SSE 事件给 React UI 与 CLI。

## 3. 关键能力
| 能力 | 描述 | 入口 |
| --- | --- | --- |
| SAR 分析 CLI | `uv run molx sar "Analyze ..."` 输出 Markdown 报告 | CLI |
| 交互式 Chat | `uv run molx chat`，支持历史、清空、打印 JSON | CLI |
| FastAPI API | `/api/v1/agent/invoke`（同步）、`/api/v1/agent/stream`（SSE）、`/api/v1/agent/batch` | Server |
| 专用 SAR API | `/api/v1/sar/analyze`、`/api/v1/sar/stream` | Server |
| React Web UI | `molx_client` Vite + Tailwind，调用 `/api/v1/agent/stream`，支持多 Session 列表（本地） | Client |
| Session 存储 | `molx_core.memory` 提供内存/数据库实现、转写报告、清理任务 | Memory |
| SSE Streaming | `stream_agent_response` 将 LangGraph 状态与最终回答切分为 token 级 SSE 事件，含 thinking/status 元数据 | Server |

## 4. 演示流程
1. `uv sync --extra dev --extra server` 初始化虚拟环境和 server 依赖。
2. `make serve` -> `uv run molx-server run --reload` 启动 API，默认 `http://localhost:8000/docs`。
3. 另起终端 `cd molx_client && pnpm install && pnpm dev`，在 `.env` 中配置 `VITE_API_BASE=http://localhost:8000/api/v1` 以体验 UI。
4. CLI 演示：
   ```bash
   uv run molx sar "Analyze SAR of celecoxib analogs for COX-2 selectivity"
   ```
   观察 DataCleaner -> SARAgent -> Reporter 的控制台输出以及 `artifacts/tools/data_cleaner/cleaned_*.{json,csv}`。

## 5. 现状评估
### ✅ 已完成
- Agent 工作流已转向 LangGraph：`MolxAgent` 复用 `build_molx_graph`，Planner 支持 Think/Reflect/Optimize，多任务循环可迭代 3 次。
- DataCleaner 现可自动解析内联 CSV、智能识别列名、保存多活动列并写入 artifacts 目录；Reporter 增加 single-site / subset 模式。
- Memory/SessionManager 与 `molx_core` store 打通，TTL 清理、Session 恢复、Recorder 绑定已在 CLI/Server 生效。
- Server 已提供同步/批量/SSE 路由，`stream_agent_response` 注入 thinking/status 事件，React UI 支持多 Session + 自动滚动体验。

### ⚠️ 遗留风险
- SSE 仍由阻塞执行+结果切词伪流式，尚未接入 LangChain 原生 streaming callback，也未传回工具中间产物。
- Server 端缺上传通道：DataCleaner 虽支持 inline CSV，但 `/sar` / `/agent` 依旧只能消化文本路径，文件/Blob 上传与云存储落地缺位。
- LangGraph 仅覆盖 SAR 主线，缺少对非 SAR 意图（数据库查询、文献总结等）的 fallback worker 与自定义节点，intent 不支持插件扩展。
- 测试覆盖度仍只有 26%（`assets/images/coverage.svg`），缺少端到端 CLI/Server 测试及化学工具回归，报告模板也无人校验。

## 6. 风险与依赖
1. **LLM**：依赖本地 `ChatOpenAI` 兼容接口（`LOCAL_OPENAI_*`），需保证模型可访问，且具备工具调用权限。
2. **化学栈**：RDKit、pandas、safety/bandit 在 `uv` 环境中安装时间较长，CI/CD 需准备缓存镜像。
3. **存储**：默认内存实现，PostgreSQL 版本尚未在 README 中指引用户启用，需要补充迁移脚本和连接配置。
4. **安全**：缺少 API Key / CORS 白名单默认值，只能在受控环境运行。

## 7. POC 完成度与下一步
| 模块 | 完成度 | 下一步 |
| --- | --- | --- |
| Agent & Tools | 80% | 扩展 LangGraph worker 注册表、记录工具指标/异常、为 Reporter 增加多目标分析与摘要评测 |
| Memory | 70% | 提供 Postgres 部署指引、Session 查询 API、CLI 开关持久化/TTL、补充多租户隔离 |
| Server | 65% | 接入 LangChain streaming callback、实现文件上传+对象存储、补齐鉴权/限速/审计日志 |
| Client | 50% | Session 与服务器实时同步、SSE 重连/心跳提示、报告下载/预览、输入数据向导 |
| Testing & Docs | 40% | 增加 agent/langgraph 回归、化学工具 fixtures、Server/Client e2e、更新 README/Docs（配置、部署、POC 里程碑） |

---
**结论**：POC 已从“手写 ReAct”升级到 LangGraph + SSE 流程，并补齐数据清洗/Reporter 的多模式能力；下一阶段需攻克真实流式推理、文件入库/权限链路与高质量测试矩阵，以支撑药化团队在生产环境中可回溯、可扩展地运行 SAR 智能体。
