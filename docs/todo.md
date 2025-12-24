# TODO（按优先级）

## P0
- [ ] 深度完成的完成Memory模块的设计，包括了回话，文件，制品，轮次，报告等所有内容的持久化存储。
- [ ] 接入 LangChain streaming callbacks 到 `stream_agent_response`，实时推送 token、工具/计划节点状态与心跳，并支持超时/取消，避免长任务期间 UI 空窗。
- [ ] 为工件引入 session 隔离与清理：`get_tool_output_dir` 支持 session_id 分桶，Recorder/SessionMetadata 记录相对路径，SessionManager 清理时同步删除输出，并确保 `/session/{id}/files/*` 仅暴露本 session 文件。

## P1
- [ ] 扩展 LangGraph worker 注册表与 IntentClassifier，提供可配置的非 SAR fallback（文献总结、数据库问答等），避免 unsupported 硬编码。
- [ ] 增补端到端测试：覆盖 `/agent` invoke/stream SSE 事件顺序、上传 → DataCleaner → Reporter 的 artifacts 注册、Session 恢复后继续流式推理的场景。
- [ ] 文档化持久化与部署：在 README/docs 中补充 Postgres store 启用方法、API 鉴权/限速默认值、前端 SSE 心跳/重连示例。
