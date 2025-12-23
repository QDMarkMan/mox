"""
Session management for multi-turn conversations.

Refactored to use molx_core memory module for storage.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from molx_agent.agents.molx import ChatSession
from molx_agent.memory import SessionMetadata, bind_chat_session
from molx_core.config import get_core_settings
from molx_core.memory import (
    ConversationStore,
    SessionData,
    get_conversation_store,
)
from molx_core.memory.factory import initialize_store, close_store


logger = logging.getLogger(__name__)


class ManagedSession:
    """
    Wrapper combining SessionData with ChatSession.
    
    Bridges molx_core storage with molx_agent ChatSession.
    """

    def __init__(self, session_data: SessionData) -> None:
        self.session_data = session_data
        if not isinstance(self.session_data.metadata, SessionMetadata):
            self.session_data.metadata = SessionMetadata.from_dict(self.session_data.metadata)
        self.chat_session = ChatSession()
        
        # Restore messages from storage
        self._restore_messages()
        bind_chat_session(self.chat_session, self.session_data)
        self.chat_session.load_uploaded_files(self.session_data.metadata)

    def _restore_messages(self) -> None:
        """Restore messages from stored session data."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        normalized = self._normalize_messages(self.session_data.messages)
        self.session_data.messages = normalized

        for msg in normalized:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                self.chat_session.state["messages"].append(HumanMessage(content=content))
            elif role == "agent":
                self.chat_session.state["messages"].append(AIMessage(content=content))
            elif role == "system":
                self.chat_session.state["messages"].append(SystemMessage(content=content))

    @staticmethod
    def _normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
        """Convert serialized messages from storage into dict form."""
        messages: list[dict[str, str]] = []

        if isinstance(raw_messages, str):
            try:
                parsed = json.loads(raw_messages)
            except json.JSONDecodeError:
                parsed = [{"role": "agent", "content": raw_messages}]
            raw_messages = parsed

        if not isinstance(raw_messages, list):
            return messages

        for entry in raw_messages:
            if isinstance(entry, dict):
                role = entry.get("role", "user")
                content = entry.get("content", "")
            else:
                role, content = "agent", str(entry)
            messages.append({"role": role, "content": content})

        return messages

    @property
    def session_id(self) -> str:
        return self.session_data.session_id

    @property
    def created_at(self):
        return self.session_data.created_at

    @property
    def last_activity(self):
        return self.session_data.last_activity

    @property
    def message_count(self) -> int:
        return self.session_data.message_count

    def touch(self) -> None:
        """Update last activity."""
        self.session_data.touch()


class SessionManager:
    """
    Manages chat sessions using molx_core storage backend.
    
    Features:
    - Configurable storage backend (memory/postgres)
    - Session persistence
    - TTL-based expiration
    - Background cleanup task
    """

    def __init__(self) -> None:
        self._store: Optional[ConversationStore] = None
        self._sessions_cache: dict[str, ManagedSession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._settings = get_core_settings()
        self._max_sessions = 1000

    @property
    def store(self) -> ConversationStore:
        """Get the conversation store, initializing if needed."""
        if self._store is None:
            self._store = get_conversation_store()
        return self._store

    async def initialize(self) -> None:
        """Initialize the session manager and storage."""
        await initialize_store()
        self._store = get_conversation_store()
        logger.info("SessionManager initialized")

    async def close(self) -> None:
        """Close the session manager and storage."""
        await close_store()
        self._sessions_cache.clear()
        logger.info("SessionManager closed")

    def create_session(self) -> str:
        """
        Create a new session synchronously.
        
        Returns:
            ManagedSession backed by the configured store.
        """
        session_id = str(uuid.uuid4())
        session_data = SessionData(session_id=session_id)
        managed = ManagedSession(session_data)
        self._sessions_cache[session_id] = managed
        logger.debug(f"Created session: {session_id}")
        return session_id

    async def create_session_async(self) -> ManagedSession:
        """
        Create a new session with persistence.
        
        Returns:
            ManagedSession backed by the configured store.
        """
        session_id = str(uuid.uuid4())
        session_data = await self.store.create(session_id)
        managed = ManagedSession(session_data)
        self._sessions_cache[session_id] = managed
        logger.debug(f"Created persistent session: {session_id}")
        return managed

    def get_session(self, session_id: str) -> Optional[ManagedSession]:
        """
        Get session by ID from cache.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            ManagedSession if found, None otherwise.
        """
        managed = self._sessions_cache.get(session_id)
        if managed:
            managed.touch()
        return managed

    async def get_session_async(self, session_id: str) -> Optional[ManagedSession]:
        """
        Get session by ID, loading from storage if needed.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            ManagedSession if found, None otherwise.
        """
        # Check cache first
        if session_id in self._sessions_cache:
            managed = self._sessions_cache[session_id]
            managed.touch()
            return managed
        
        # Load from storage
        session_data = await self.store.get(session_id)
        if session_data is None:
            return None
        
        managed = ManagedSession(session_data)
        self._sessions_cache[session_id] = managed
        return managed

    def get_or_create_session(self, session_id: Optional[str]) -> ManagedSession:
        """
        Get existing session or create new one synchronously.
        
        Args:
            session_id: Optional session ID.
            
        Returns:
            ManagedSession instance.
        """
        if session_id:
            managed = self.get_session(session_id)
            if managed:
                return managed
        
        new_id = self.create_session()
        return self._sessions_cache[new_id]

    async def get_or_create_session_async(self, session_id: Optional[str]) -> ManagedSession:
        """Get or create session with persistence."""
        if session_id:
            managed = await self.get_session_async(session_id)
            if managed:
                return managed

        managed = await self.create_session_async()
        return managed

    async def save_session(self, session: ManagedSession) -> None:
        """
        Save session state to storage.
        
        Args:
            session: Session to persist.
        """
        # Sync messages from ChatSession to SessionData
        history = session.chat_session.get_history()
        session.session_data.messages = history
        
        # Debug: Log metadata state before save
        metadata = session.session_data.metadata
        if hasattr(metadata, 'latest'):
            logger.debug(f"Saving session {session.session_id}, latest query: {metadata.latest.get('query', 'N/A')[:50] if metadata.latest else 'EMPTY'}")
        
        await self.store.save(session.session_data)
        logger.debug(f"Saved session: {session.session_id}")

    async def list_sessions(self) -> list[SessionData]:
        """Return all active sessions from the backing store."""
        return await self.store.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from cache.
        
        Args:
            session_id: Session to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        if session_id in self._sessions_cache:
            del self._sessions_cache[session_id]
            logger.debug(f"Deleted session: {session_id}")
            return True
        return False

    async def delete_session_async(self, session_id: str) -> bool:
        """
        Delete a session from cache and storage.
        
        Args:
            session_id: Session to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        self._sessions_cache.pop(session_id, None)
        return await self.store.delete(session_id)

    def list_cached_sessions(self) -> list[ManagedSession]:
        """Get all cached sessions without hitting the backing store."""
        return list(self._sessions_cache.values())

    async def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        ttl_seconds = self._settings.session_ttl
        count = await self.store.cleanup_expired(ttl_seconds)

        # Also clean cache using wall-clock timestamps
        ttl_delta = timedelta(seconds=ttl_seconds)
        from datetime import timezone
        now = datetime.now(timezone.utc)
        expired: list[str] = []
        for sid, session in list(self._sessions_cache.items()):
            last_activity = session.session_data.last_activity
            
            # Ensure last_activity is aware for comparison
            if last_activity.tzinfo is None:
                last_activity = last_activity.replace(tzinfo=timezone.utc)
                
            if now - last_activity > ttl_delta:
                expired.append(sid)

        for sid in expired:
            del self._sessions_cache[sid]

        return count + len(expired)

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is not None:
            return

        async def _cleanup_loop() -> None:
            while True:
                await asyncio.sleep(self._settings.session_cleanup_interval)
                try:
                    await self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")

        self._cleanup_task = asyncio.create_task(_cleanup_loop())
        logger.info("Started session cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped session cleanup task")

    @property
    def session_count(self) -> int:
        """Get current number of cached sessions."""
        return len(self._sessions_cache)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
