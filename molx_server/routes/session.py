"""
Session management API endpoints.

Provides endpoints for managing conversation sessions.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from molx_server.schemas.models import (
    SessionCreateResponse,
    SessionHistoryResponse,
    SessionInfo,
    SessionStatus,
)
from molx_server.session import get_session_manager


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/session", tags=["Session"])


@router.post("/create", response_model=SessionCreateResponse)
async def create_session() -> SessionCreateResponse:
    """
    Create a new conversation session.

    Returns a session ID for use in subsequent requests.
    """
    session_manager = get_session_manager()

    try:
        session_id = session_manager.create_session()
        session_data = session_manager.get_session(session_id)

        return SessionCreateResponse(
            session_id=session_id,
            created_at=session_data.created_at if session_data else datetime.utcnow(),
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
    session_data = session_manager.get_session(session_id)

    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionInfo(
        session_id=session_data.session_id,
        status=SessionStatus.ACTIVE,
        created_at=session_data.created_at,
        last_activity=session_data.last_activity,
        message_count=session_data.message_count,
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    """
    Delete a session.

    Removes the session and all associated conversation history.
    """
    session_manager = get_session_manager()

    if session_manager.delete_session(session_id):
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
    session_data = session_manager.get_session(session_id)

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
    session_data = session_manager.get_session(session_id)

    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data.chat_session.clear()

    return {"cleared": True, "session_id": session_id}
