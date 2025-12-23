# Agents Module

`molx_agent/agents` 汇集了面向药物设计/SAR 分析的多智能体实现。模块基于 LangGraph 构建 ReAct 流程：先识别用户意图，再规划任务 DAG，随后派发到具备领域能力的 worker，最后对结果进行反思与优化，生成可交付的报告。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `base.py` | 所有 agent 的抽象基类，约束 `run(state: AgentState)` 接口与自描述属性。 |
| `molx.py` | 主 orchestrator (`MolxAgent`) 以及面向 CLI/服务端的 `run_sar_agent`、`ChatSession` 包装。 |
| `planner.py` | 采用 THINK → ACT → REFLECT → OPTIMIZE 流程的任务规划/评估器，生成任务 DAG 并负责复盘与重规划。 |
| `intent_classifier.py` | 基于 LLM 的意图分类器，区分 SAR 分析/数据处理/分子查询等场景，并输出置信度与解释。 |
| `data_cleaner.py` | 数据抽取与标准化 worker，调用 CSV/Excel/SDF 提取工具与清洗、持久化工具。 |
| `sar.py` | SAR worker，负责骨架选择、R-group 拆分、OCAT 分析、模式识别与 LLM 洞察生成。 |
| `reporter.py` | 报告 worker，汇总分析结果并生成 HTML 报告，支持 Full/Site/Sub-set 等模式。 |
| `tool_agent.py` | 通用工具式 agent，利用 LangGraph 的 prebuilt ReAct Agent 与 `get_all_tools()` 交互。 |
| `modules/graph.py` | LangGraph 构建辅助，定义状态机节点、路由及缓存。 |
| `modules/state.py` | `AgentState`/`Task`/`Message` TypedDict，统一跨节点的上下文结构。 |
| `modules/llm.py` | LLM 工厂与 JSON 解析助手，封装 `ChatOpenAI` 初始化与对话调用。 |
| `modules/tools.py` | 工具注册中心，聚合 RDKit/converter/SAR/report/MCP 工具集合。 |
| `modules/mcp.py` | MCP 工具加载器，支持 stdio/HTTP transport，并与 `langchain-mcp-adapters` 集成。 |
| `modules/prompts.py` | Planner/ReAct worker 的系统 prompt 模板，可用于自定义提示词。 |

## LangGraph 执行模型

1. **Classify**：`IntentClassifierAgent` 读取 `state['user_query']`，调用 LLM 输出 `Intent`、置信度及推理步骤。
2. **Plan (THINK)**：`PlannerAgent` 使用 `PLANNER_SYSTEM_PROMPT` 生成任务 DAG，仅允许 `data_cleaner`、`sar`、`reporter` 三类 worker。任务保存在 `state['tasks']`，状态初始为 `pending`。
3. **Act**：`MolxGraphNodes.act` 根据依赖关系选取下一个 `pending` 任务，调用对应 worker (`DataCleanerAgent`/`SARAgent`/`ReporterAgent`，或自定义 worker)。运行结果写入 `state['results'][task_id]`。
4. **Reflect / Optimize**：Planner 通过 `_derive_reflection_from_state` 或 LLM 评估产出，必要时触发 `OPTIMIZE_SYSTEM_PROMPT` 生成新计划，迭代不超过 `MAX_ITERATIONS`。
5. **Finalize**：`MolxGraphNodes.finalize` 负责拼接总结文本，附上报告路径，并写入 `state['final_response']` 与 `state['final_answer']`。

所有节点共享 `AgentState`（TypedDict），其中包含 `messages`、`tasks`、`results`、`reflection`、`intent_*` 等字段，确保 orchestrator 与 worker 间无缝传递信息。

## 核心 Agent 角色

### MolxAgent
- 封装 LangGraph pipeline，持有意图分类器、Planner 与 worker map，可通过构造函数注入自定义组件。
- 提供 `run(state)` 与 `invoke(user_query, state=None)`，以及 `ChatSession` 以实现多轮对话、状态持久化与 `SessionRecorder` 接入。

### IntentClassifierAgent
- 依赖 `invoke_llm` 与 `INTENT_CLASSIFIER_PROMPT`，生成结构化 JSON 结果。
- 结果写入 `intent`、`intent_confidence`、`intent_reasoning(_steps)`，并提供 `is_supported()` 与友好回复映射 `INTENT_RESPONSES`。

### PlannerAgent
- THINK 阶段使用 LLM 生成任务 DAG，并在 state 中设置 `current_task_id`、`iteration`。
- REFLECT 阶段优先基于 state 推断成功/失败，否则调用 LLM (`REFLECT_SYSTEM_PROMPT`)。
- OPTIMIZE 阶段读取 reflection（`should_replan`/`issues`）后重跑 LLM (`OPTIMIZE_SYSTEM_PROMPT`)，最多尝试 `MAX_ITERATIONS` 次。
- 内部 `_pick_next_task` 负责拓扑调度，仅当依赖任务均 `done` 时才返回可执行任务。

### 数据/分析/报告 Worker
- **DataCleanerAgent**：读取任务描述/输入自适应选择提取工具 (`ExtractFromCSV/Excel/SDF`)，随后串行调用 `CleanCompoundDataTool` 与 `SaveCleanedDataTool`，并在结果中保留 `activity_columns` 方便 Reporter。
- **SARAgent**：封装规则与工具组合，选择 scaffold (`find_mcs_scaffold`/`find_common_murcko_scaffold`)，执行 R-group decomposition、OCAT pairing (`identify_ocat_series`)，并可生成单点扫描洞察。支持外部注入规则或 prompt，所有分析 metadata 写入结果字典中。
- **ReporterAgent**：根据 Planner 传入的 `report_intent` 或从用户查询提取关键字，决定运行 Full/SINGLE_SITE/Molecule subset 分析。聚合 `RunFullSARAnalysisTool`、`GenerateHTMLReportTool`、`BuildReportSummaryTool` 输出报告与摘要。
- **ToolAgent**（可选）：基于 `create_react_agent` 自动处理工具调用任务，适合单次问答型任务或调试工具链。

## 共享基础模块

- **`modules/state.py`**：集中维护 `AgentState`/`Task` 结构，LangGraph nodes 使用 `Annotated[list[BaseMessage], add_messages]` 追踪上下文。
- **`modules/graph.py`**：构建 `StateGraph`，并在 LangGraph 未安装时提供本地降级实现；包含 `MolxGraphNodes`，定义节点函数及路由逻辑。
- **`modules/llm.py`**：读取配置 (`get_settings`) 初始化本地或远端 OpenAI 兼容模型，附带 markdown JSON 清洗逻辑。
- **`modules/tools.py`**：集中加载工具，按功能分组（RDKit、标准化、SAR、报告、MCP），并导出 `get_tool_names()` 供调试。
- **`modules/mcp.py`**：`MCPToolLoader` 支持从文件/环境/配置对象中读取 server 列表，懒加载 `langchain-mcp-adapters`，暴露同步与异步 `get_mcp_tools` 接口。
- **`modules/prompts.py`**：包含 Planner/Reflect/Optimize 及其它 worker prompt，可直接 import 覆盖或在运行时修改。

## 工具与 MCP 配置

1. 本地工具：确保 `molx_agent/tools/` 下的依赖可被导入（RDKit、标准化工具等），`get_all_tools()` 会自动捕获 ImportError 并记录日志。
2. MCP：
   - 在 `config/mcp_servers.json` 或环境变量 `MCP_SERVERS_CONFIG` 中声明 server：
     ```json
     {
       "chemistry": {
         "command": "python",
         "args": ["./examples/example_mcp_server.py"],
         "transport": "stdio"
       }
     }
     ```
   - 或者配置 HTTP 端点：`{"search": {"url": "http://localhost:8000/mcp", "transport": "http"}}`
   - 在 `.env`/环境中设置 `MCP_ENABLED=true` 以启用；若需禁用，置 `false` 或不配置 server。
3. Reporter/SAR 等工具会在结果中写入 `output_files`/`report_path`，LangGraph `finalize` 节点据此拼接总结。

## 扩展指南

- **新增 worker**：继承 `BaseAgent`，在 `MolxAgent` 初始化时将其注入 `workers` 映射，同时更新 Planner prompt 以允许新的 `type`。
- **自定义计划策略**：覆写 `PlannerAgent._invoke_planner_llm` 或在 `modules/prompts.py` 中调整 PROMPT；如需更严格的任务模板，可在 `_extract_tasks` 中添加校验。
- **强化模式识别**：向 `SARAgent` 注册新的 `Rule`（`add_rule`），或扩展 `SARAgentConfig` 控制阈值。
- **工具扩展**：在 `molx_agent/tools/` 内实现新的 `BaseTool`，并在 `modules/tools.get_all_tools()` 中加载；MCP 工具可通过更新 server 配置即时生效。
- **多轮对话**：使用 `ChatSession`，在每次 `send()` 后检查 `state['final_response']`，必要时调用 `clear()` 重置上下文。

## 使用与测试

- 安装依赖：`uv sync --extra dev` 或 `make install && make pre-commit-install`。
- 本地运行 SAR agent：
  ```bash
  uv run molx-agent run --query "分析这些分子的 R1 SAR" --data ./data/example.csv
  ```
  或启动交互模式：`uv run molx-agent chat`（内部使用 `ChatSession`）。
- 运行完整测试/覆盖：`make test`（会触发 `pytest` 并更新覆盖率徽章）。如只需验证 Agent 逻辑，可使用 `uv run pytest tests/agents -k sar`。

上述文档旨在帮助贡献者快速理解 Agent 模块的协作方式，对接新的工具链或扩展工作流时请同步更新本 README 与相关 `docs/` 条目。
