"""
Session management API endpoints.

Provides endpoints for managing conversation sessions.
"""

import logging
from datetime import datetime
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException

from molx_agent.memory import SessionMetadata
from molx_server.schemas.models import (
    SessionCreateResponse,
    SessionHistoryResponse,
    SessionListResponse,
    SessionInfo,
    SessionMetadataResponse,
    SessionStatus,
)
from molx_server.session_utils import metadata_to_response
from molx_server.session import get_session_manager


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/session", tags=["Session"])


def _latest_query(metadata: SessionMetadata | dict[str, Any] | None) -> str | None:
    """Extract the last user query from stored metadata."""
    if isinstance(metadata, SessionMetadata):
        latest = metadata.latest
    elif isinstance(metadata, dict):
        latest = metadata.get("latest", {})
    else:
        latest = {}

    if isinstance(latest, dict):
        query = latest.get("query")
        if isinstance(query, str):
            return query.strip()
    return None


def _session_info_from_record(record: Any) -> SessionInfo:
    """Convert a SessionData record into API-facing SessionInfo."""
    metadata = record.metadata
    if not isinstance(metadata, SessionMetadata):
        metadata = SessionMetadata.from_dict(metadata)
        record.metadata = metadata

    return SessionInfo(
        session_id=record.session_id,
        status=SessionStatus.ACTIVE,
        created_at=record.created_at,
        last_activity=record.last_activity,
        message_count=record.message_count,
        preview=_latest_query(metadata),
    )


def _sort_sessions(records: Iterable[SessionInfo]) -> list[SessionInfo]:
    return sorted(records, key=lambda info: info.last_activity, reverse=True)


@router.get("", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """Return summary information for all active sessions."""
    session_manager = get_session_manager()
    records = await session_manager.list_sessions()
    infos = [_session_info_from_record(record) for record in records]
    return SessionListResponse(sessions=_sort_sessions(infos))


@router.post("/create", response_model=SessionCreateResponse)
async def create_session() -> SessionCreateResponse:
    """
    Create a new conversation session.

    Returns a session ID for use in subsequent requests.
    """
    session_manager = get_session_manager()

    try:
        session = await session_manager.create_session_async()

        return SessionCreateResponse(
            session_id=session.session_id,
            created_at=session.created_at,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    """
    Get session information.

    Returns session status and metadata.
    """
    session_manager = get_session_manager()
    session_data = await session_manager.get_session_async(session_id)

    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return _session_info_from_record(session_data.session_data)


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    """
    Delete a session.

    Removes the session and all associated conversation history.
    """
    session_manager = get_session_manager()

    if await session_manager.delete_session_async(session_id):
        return {"deleted": True, "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    """
    Get conversation history for a session.

    Returns all messages in the session.
    """
    session_manager = get_session_manager()
    session_data = await session_manager.get_session_async(session_id)

    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    history = session_data.chat_session.get_history()

    return SessionHistoryResponse(
        session_id=session_id,
        messages=history if history else [],
    )


@router.post("/{session_id}/clear")
async def clear_session_history(session_id: str) -> dict:
    """
    Clear conversation history for a session.

    Keeps the session but removes all messages.
    """
    session_manager = get_session_manager()
    session_data = await session_manager.get_session_async(session_id)

    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data.chat_session.clear()
    session_data.session_data.messages = []
    session_data.session_data.metadata = SessionMetadata()
    await session_manager.save_session(session_data)

    return {"cleared": True, "session_id": session_id}


@router.get("/{session_id}/data", response_model=SessionMetadataResponse)
async def get_session_metadata(session_id: str) -> SessionMetadataResponse:
    """Return structured metadata for a session."""
    session_manager = get_session_manager()
    session_data = await session_manager.get_session_async(session_id)

    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = session_data.session_data.metadata
    if not isinstance(metadata, SessionMetadata):
        metadata = SessionMetadata.from_dict(metadata)
        session_data.session_data.metadata = metadata

    return metadata_to_response(session_id, metadata)
