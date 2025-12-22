# Shared Memory Design

## Overview

The memory layer unifies agent runtime state and server APIs via the `molx_core.memory` package. Every chat session owns a `SessionData` record composed of streaming-friendly `messages` plus structured `metadata`. The metadata is managed by `SessionRecorder`, which ingests each agent turn, extracts reasoning artifacts, structured results, and report info, then normalizes them into an append-only log. Both in-memory and Postgres stores persist the same JSON-friendly representation so backend choice does not impact schema.

## Metadata Schema

- `turns[]`: Ordered `TurnRecord` entries capturing the query, response, intent classification (value/reasoning/confidence), planner reasoning, reflection payload, task summary (id/type/status/depends_on), and structured result summaries for each executed worker. Optional `report` nodes snapshot `report_path`, summary text, and previews.
- `latest`: Copy of the newest turn for quick reads without traversing the full log.
- `structured_data`: Rolling index keyed by task id, storing lightweight summaries (compound counts, activity columns, output file paths, etc.) culled from raw results to avoid bloating the store.
- `reports[]`: Bounded list (20) containing `ReportRecord` metadata for quick gallery-style access.

## Recorder Flow

1. `ChatSession` attaches a `SessionRecorder` when bound to a `SessionData` instance via `SessionManager`.
2. After each `send`, the recorder ingests the updated `AgentState`, normalizes intent/planner reasoning, summarizes tasks, and sanitizes results (counts, selected keys, truncated previews).
3. Recorder updates `SessionMetadata` in-place; `SessionManager.save_session` copies chat history and writes the serialized metadata to the configured store.

## Server Integration

- `SessionManager` now exposes async `create_session_async` / `get_or_create_session_async` so every API path can round-trip through the shared store.
- Agent (`/agent/*`) and SAR (`/sar/*`) routes invoke/save sessions per request and stream via wrappers that flush metadata once generators complete.
- Session management routes (`/session/...`) read/update cached sessions but fall back to the persistent store when needed. Clearing history wipes both `messages` and `SessionMetadata` before persisting.

## Retrieval API

`GET /session/{session_id}/data` surfaces the structured metadata graph using the new `SessionMetadataResponse` schema. Clients receive the latest snapshot, structured-data index, report list, and the bounded turn log without rehydrating every chat message. This API underpins UI sidebars and debugging tools while keeping the persisted schema consistent across backends.
