"""Memory utilities for bridging molx-agent with shared stores."""

from __future__ import annotations

import mimetypes
import os
from typing import Any, Optional
from uuid import uuid4

from molx_core.memory import (
    SessionData,
    SessionMetadata,
    SessionRecorder,
    TurnRecord,
)
from molx_core.memory.recorder import FileRecord

__all__ = [
    "SessionData",
    "SessionMetadata",
    "SessionRecorder",
    "TurnRecord",
    "FileRecord",
    "bind_chat_session",
    "register_uploaded_file",
]


def bind_chat_session(chat_session: Any, session_data: SessionData) -> SessionRecorder:
    """Attach a recorder to a chat session and return it."""
    recorder = SessionRecorder(session_data)
    attach = getattr(chat_session, "attach_recorder", None)
    if callable(attach):
        attach(recorder)
    return recorder


def register_uploaded_file(
    session_data: SessionData,
    *,
    file_name: str,
    file_path: str,
    content_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    description: Optional[str] = None,
) -> FileRecord:
    """Register a user uploaded file in session metadata."""
    metadata = session_data.metadata
    if not isinstance(metadata, SessionMetadata):
        metadata = SessionMetadata.from_dict(metadata)
        session_data.metadata = metadata

    if content_type is None:
        content_type = mimetypes.guess_type(file_name)[0]

    if size_bytes is None:
        try:
            size_bytes = os.stat(file_path).st_size
        except OSError:
            size_bytes = None

    record = FileRecord(
        file_id=str(uuid4()),
        file_name=file_name,
        file_path=file_path,
        content_type=content_type,
        size_bytes=size_bytes,
        description=description,
    )
    metadata.add_uploaded_file(record)
    return record
