"""
In-memory conversation storage implementation.

Default storage backend using Python dictionaries.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional

from molx_core.memory.store import ConversationStore, SessionData


logger = logging.getLogger(__name__)


class MemoryStore(ConversationStore):
    """
    In-memory conversation storage.
    
    Features:
    - Fast dictionary-based storage
    - Thread-safe operations
    - TTL-based expiration
    
    Note: Data is lost on restart. Use PostgresStore for persistence.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._lock = Lock()

    async def create(self, session_id: str) -> SessionData:
        """Create a new session."""
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id} already exists")
            
            session = SessionData(session_id=session_id)
            self._sessions[session_id] = session
            logger.debug(f"Created session: {session_id}")
            return session

    async def get(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    async def save(self, session: SessionData) -> None:
        """Save session state."""
        with self._lock:
            session.touch()
            self._sessions[session.session_id] = session
            logger.debug(f"Saved session: {session.session_id}")

    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug(f"Deleted session: {session_id}")
                return True
            return False

    async def update_activity(self, session_id: str) -> bool:
        """Update last activity timestamp."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
                return True
            return False

    async def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove expired sessions."""
        with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=ttl_seconds)
            
            expired = [
                sid for sid, session in self._sessions.items()
                if session.last_activity < cutoff
            ]
            
            for sid in expired:
                del self._sessions[sid]
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
            
            return len(expired)

    async def list_sessions(self) -> list[SessionData]:
        """List all active sessions."""
        with self._lock:
            return list(self._sessions.values())

    @property
    def session_count(self) -> int:
        """Get current number of sessions."""
        with self._lock:
            return len(self._sessions)
