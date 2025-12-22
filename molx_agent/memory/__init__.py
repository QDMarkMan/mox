"""Memory utilities for bridging molx-agent with shared stores."""

from __future__ import annotations

from typing import Any

from molx_core.memory import SessionData, SessionMetadata, SessionRecorder, TurnRecord

__all__ = [
    "SessionData",
    "SessionMetadata",
    "SessionRecorder",
    "TurnRecord",
    "bind_chat_session",
]


def bind_chat_session(chat_session: Any, session_data: SessionData) -> SessionRecorder:
    """Attach a recorder to a chat session and return it."""
    recorder = SessionRecorder(session_data)
    attach = getattr(chat_session, "attach_recorder", None)
    if callable(attach):
        attach(recorder)
    return recorder
