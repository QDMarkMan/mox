"""Streaming support for LangChain agent responses."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import re
from typing import Any, AsyncGenerator, Callable, Optional

from molx_server.config import get_server_settings
from molx_agent.memory import SessionMetadata

logger = logging.getLogger(__name__)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _clean_console_line(text: str) -> str:
    """Strip ANSI sequences and whitespace from console output."""
    if not text:
        return ""
    cleaned = ANSI_ESCAPE_RE.sub("", text)
    return cleaned.strip()


class _QueueStream(io.TextIOBase):
    """File-like object that forwards writes into the SSE event queue."""

    def __init__(self, emit: Callable[[str, dict[str, Any]], None]) -> None:
        self._emit = emit
        self._buffer = ""

    def write(self, data: str) -> int:  # pragma: no cover - exercised indirectly
        if not data:
            return 0
        text = data.replace("\r", "\n")
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            cleaned = _clean_console_line(line)
            if cleaned:
                self._emit("status", {"message": cleaned})
        return len(data)

    def flush(self) -> None:  # pragma: no cover - exercised indirectly
        if self._buffer:
            cleaned = _clean_console_line(self._buffer)
            if cleaned:
                self._emit("status", {"message": cleaned})
            self._buffer = ""


def format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format data as an SSE event string."""
    json_data = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {json_data}\n\n"


async def stream_agent_response(
    session_data: Any,
    query: str,
    on_token: Optional[Callable[[str], None]] = None,
) -> AsyncGenerator[str, None]:
    """Stream agent response as SSE events."""
    settings = get_server_settings()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
    thinking_events: list[dict[str, Any]] = []

    def emit(event_type: str, payload: dict[str, Any]) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (event_type, payload))
        except RuntimeError:  # pragma: no cover - defensive during shutdown
            logger.debug("Event loop closed before emitting stream event")

    def run_agent() -> str:
        """Execute the blocking chat session and capture console output."""
        stream = _QueueStream(emit)
        result: str = ""
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                result = session_data.chat_session.send(query)
        finally:
            stream.flush()

        # Capture thinking metadata from the last state
        state = getattr(session_data.chat_session, "_last_state", None)
        if state:
            intent = state.get("intent")
            reasoning = state.get("intent_reasoning")
            confidence = state.get("intent_confidence")
            if intent or reasoning:
                thinking_events.append(
                    {
                        "intent": getattr(intent, "value", intent),
                        "reasoning": reasoning,
                        "confidence": confidence,
                    }
                )

        return result

    yield format_sse_event("start", {"query": query})
    yield format_sse_event(
        "thinking",
        {"status": "analyzing", "message": "🎯 Analyzing query intent..."},
    )

    executor_task = asyncio.ensure_future(loop.run_in_executor(None, run_agent))
    start_time = loop.time()

    while True:
        if executor_task.done():
            # Drain any remaining status events before final payload
            while not queue.empty():
                event_type, payload = queue.get_nowait()
                yield format_sse_event(event_type, payload)

            try:
                result_text = executor_task.result()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Streaming executor error: %s", exc)
                yield format_sse_event("error", {"message": str(exc)})
                break

            latest_payload = _latest_turn_payload(session_data)

            for thinking in thinking_events:
                yield format_sse_event(
                    "thinking",
                    {
                        "status": "complete",
                        "intent": thinking.get("intent"),
                        "reasoning": thinking.get("reasoning"),
                        "confidence": thinking.get("confidence"),
                    },
                )
                await asyncio.sleep(0.05)

            words = result_text.split(" ") if result_text else []
            for idx, word in enumerate(words):
                token = (" " + word) if idx > 0 else word
                if not token:
                    continue
                yield format_sse_event("token", {"content": token})
                if on_token:
                    on_token(token)
                await asyncio.sleep(0.02)

            if latest_payload:
                yield format_sse_event("artifacts", latest_payload)

            yield format_sse_event("complete", {"result": result_text})
            break

        # Timeout handling
        elapsed = loop.time() - start_time
        if elapsed > settings.stream_timeout:
            logger.warning("Streaming timeout hit after %.2fs", elapsed)
            executor_task.cancel()
            yield format_sse_event("error", {"message": "Timeout exceeded"})
            break

        try:
            event_type, payload = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield format_sse_event(event_type, payload)
        except asyncio.TimeoutError:
            continue

    if not executor_task.done():
        executor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await executor_task


def _latest_turn_payload(session_data: Any) -> Optional[dict[str, Any]]:
    """Return sanitized metadata for the most recent turn."""

    metadata = getattr(session_data, "metadata", None)
    if metadata is None:
        return None

    if not isinstance(metadata, SessionMetadata):
        try:
            metadata = SessionMetadata.from_dict(metadata)
            session_data.metadata = metadata
        except Exception:  # pragma: no cover - defensive conversion
            return None

    latest = metadata.latest or {}
    artifacts = []
    raw_artifacts = latest.get("artifacts")
    if isinstance(raw_artifacts, list):
        for record in raw_artifacts:
            if not isinstance(record, dict):
                continue
            file_id = record.get("file_id")
            file_name = record.get("file_name")
            if not file_id or not file_name:
                continue
            artifacts.append(
                {
                    "file_id": file_id,
                    "file_name": file_name,
                    "content_type": record.get("content_type"),
                    "size_bytes": record.get("size_bytes"),
                    "description": record.get("description"),
                    "created_at": record.get("created_at"),
                }
            )

    payload: dict[str, Any] = {}
    if artifacts:
        payload["artifacts"] = artifacts

    report = latest.get("report")
    if report:
        payload["report"] = report

    structured = latest.get("structured_data")
    if structured:
        payload["structured_data"] = structured

    return payload or None
