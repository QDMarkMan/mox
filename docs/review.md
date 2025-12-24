# MolX 全栈项目 Review（2025-12-24）

## 现状快照
- **LangGraph 编排已落地**：`build_molx_graph` 负责 classify → plan → act → reflect → optimize → finalize 的状态机，默认 worker 映射 DataCleaner/SAR/Reporter，CLI 与 Server 共用同一编译好的 graph。
- **记忆与会话链路**：`SessionManager` 将 `SessionRecorder` 绑定到 `ChatSession`，持久化到 molx_core store，TTL 清理使用 wall-clock；`/session/{id}/files|history|data` 路由可上传/列出/下载文件并复原对话。
- **流式与前端整合**：`stream_agent_response` 在 executor 中运行聊天，捕获 stdout/status 后按 start/thinking/token/complete SSE 事件推送；React hook (`use-streaming-chat.ts`) 消费这些事件并从 session API 复原历史与 artifacts。

## 主要问题（按优先级）
1. **伪流式实现**：SSE 的 token 由最终文本事后按单词切分，没有接入 LangChain streaming callbacks，也没有工具/计划节点事件与心跳，长耗时步骤期间 UI 仍然空窗。
2. **工件隔离与生命周期缺失**：工具输出目录 `artifacts/tools/<tool>` 未按 session 分桶且无清理逻辑；`SessionFile` 下载依赖这些绝对路径，多 session 共享时存在路径泄露与磁盘膨胀风险。
3. **LangGraph 覆盖范围有限**：worker map 仅含 data_cleaner/sar/reporter，IntentClassifier 以 SAR 为中心，非 SAR 请求直接进入 unsupported 分支，缺少可扩展的 fallback handler/插件注册。
4. **测试空洞**：当前用例覆盖 planner/recorder/file routes，但缺少 /agent stream SSE 事件顺序、LangGraph 迭代/优化路径、上传 → DataCleaner → Reporter 的端到端回归，流式与多 session 场景缺少防回归保障。
