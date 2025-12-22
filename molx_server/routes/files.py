"""Routes for uploading and retrieving session-bound files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from molx_agent.memory import SessionMetadata, register_uploaded_file
from molx_agent.utils.paths import get_uploads_dir
from molx_server.schemas.models import (
    SessionFileInfo,
    SessionFileListResponse,
    SessionFileUploadResponse,
)
from molx_server.session import get_session_manager


router = APIRouter(prefix="/session", tags=["Session Files"])


def _safe_filename(filename: Optional[str]) -> str:
    """Return a filesystem-safe name for persisted uploads."""
    if not filename:
        return "upload.bin"
    name = os.path.basename(filename)
    sanitized = "".join(ch for ch in name if ch.isalnum() or ch in {".", "-", "_"})
    return sanitized or "upload.bin"


def _ensure_unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        alt = directory / f"{stem}_{counter}{suffix}"
        if not alt.exists():
            return alt
        counter += 1


def _to_response(record) -> SessionFileInfo:
    return SessionFileInfo(**record.to_dict())


@router.post("/{session_id}/files", response_model=SessionFileUploadResponse)
async def upload_session_file(
    session_id: str,
    uploaded_file: UploadFile = File(...),
    description: Optional[str] = Form(None),
) -> SessionFileUploadResponse:
    """Persist a user provided file and register it in session memory."""
    session_manager = get_session_manager()
    managed_session = await session_manager.get_session_async(session_id)
    if managed_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    upload_dir = get_uploads_dir(session_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(uploaded_file.filename)
    destination = _ensure_unique_path(upload_dir, safe_name)

    contents = await uploaded_file.read()
    destination.write_bytes(contents)

    size_bytes = destination.stat().st_size
    record = register_uploaded_file(
        managed_session.session_data,
        file_name=destination.name,
        file_path=str(destination),
        content_type=uploaded_file.content_type,
        size_bytes=size_bytes,
        description=description,
    )
    managed_session.chat_session.register_uploaded_file(record)

    await session_manager.save_session(managed_session)

    return SessionFileUploadResponse(
        session_id=session_id,
        file=_to_response(record),
    )


@router.get("/{session_id}/files", response_model=SessionFileListResponse)
async def list_session_files(session_id: str) -> SessionFileListResponse:
    """List uploaded files and generated artifacts for a session."""
    session_manager = get_session_manager()
    managed_session = await session_manager.get_session_async(session_id)
    if managed_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    metadata: SessionMetadata = managed_session.session_data.metadata  # type: ignore[assignment]
    return SessionFileListResponse(
        session_id=session_id,
        uploaded_files=[record.to_dict() for record in metadata.uploaded_files],
        artifacts=[record.to_dict() for record in metadata.artifacts],
    )


@router.get("/{session_id}/files/{file_id}")
async def download_session_file(session_id: str, file_id: str) -> FileResponse:
    """Return the binary contents of a stored session file."""
    session_manager = get_session_manager()
    managed_session = await session_manager.get_session_async(session_id)
    if managed_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    metadata: SessionMetadata = managed_session.session_data.metadata  # type: ignore[assignment]
    record = metadata.find_file(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")

    if not os.path.exists(record.file_path):
        raise HTTPException(status_code=404, detail="File is missing from storage")

    media_type = record.content_type or "application/octet-stream"
    return FileResponse(
        path=record.file_path,
        media_type=media_type,
        filename=record.file_name,
    )
