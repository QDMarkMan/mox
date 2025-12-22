"""Unit tests for the asynchronous session manager."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from molx_core.config import get_core_settings
from molx_core.memory.factory import reset_store
from molx_server.session import SessionManager


@pytest.fixture(autouse=True)
def _use_memory_backend(monkeypatch):
    """Force each test to run against the in-memory backend."""
    monkeypatch.setenv("MOLX_MEMORY_BACKEND", "memory")
    get_core_settings.cache_clear()
    reset_store()
    yield
    reset_store()
    get_core_settings.cache_clear()


@pytest.mark.asyncio
async def test_list_sessions_returns_persisted_records() -> None:
    """The manager should proxy the store when listing sessions."""
    reset_store()
    manager = SessionManager()
    await manager.initialize()

    try:
        session = await manager.create_session_async()
        session.session_data.add_message("user", "hello")
        await manager.save_session(session)

        records = await manager.list_sessions()
        assert any(record.session_id == session.session_id for record in records)
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cleanup_expired_drops_stale_cache_entries() -> None:
    """Expired sessions should be removed from both store and cache."""
    reset_store()
    manager = SessionManager()
    await manager.initialize()

    try:
        session = await manager.create_session_async()
        await manager.save_session(session)
        session.session_data.last_activity = datetime.utcnow() - timedelta(hours=2)
        manager._settings.session_ttl = 60  # seconds

        removed = await manager.cleanup_expired()

        assert removed >= 1
        assert session.session_id not in {
            cached.session_id for cached in manager.list_cached_sessions()
        }
    finally:
        await manager.close()
