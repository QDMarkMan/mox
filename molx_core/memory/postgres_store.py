"""
PostgreSQL conversation storage implementation.

Persistent storage backend using asyncpg for async database operations.
"""

import json
import logging
import uuid as uuid_module
from datetime import datetime, timedelta
from typing import Any, Optional

from molx_core.memory.store import ConversationStore, SessionData


logger = logging.getLogger(__name__)


class PostgresStore(ConversationStore):
    """
    PostgreSQL-based conversation storage.
    
    Features:
    - Persistent storage across restarts
    - Async operations with connection pooling
    - JSONB storage for messages and metadata
    
    Requires:
    - asyncpg library
    - PostgreSQL database with conversations table
    """

    # SQL statements
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS conversations (
        session_id UUID PRIMARY KEY,
        messages JSONB NOT NULL DEFAULT '[]',
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_conversations_activity 
        ON conversations(last_activity);
    """

    INSERT_SQL = """
    INSERT INTO conversations (session_id, messages, metadata, created_at, last_activity)
    VALUES ($1, $2, $3, $4, $5)
    """

    SELECT_SQL = """
    SELECT session_id, messages, metadata, created_at, last_activity
    FROM conversations WHERE session_id = $1
    """

    UPDATE_SQL = """
    UPDATE conversations 
    SET messages = $2, metadata = $3, last_activity = $4
    WHERE session_id = $1
    """

    DELETE_SQL = """
    DELETE FROM conversations WHERE session_id = $1
    """

    UPDATE_ACTIVITY_SQL = """
    UPDATE conversations SET last_activity = $2 WHERE session_id = $1
    """

    CLEANUP_SQL = """
    DELETE FROM conversations WHERE last_activity < $1
    """

    LIST_SQL = """
    SELECT session_id, messages, metadata, created_at, last_activity
    FROM conversations ORDER BY last_activity DESC
    """

    def __init__(self, database_url: str, pool_size: int = 5) -> None:
        """
        Initialize PostgreSQL store.
        
        Args:
            database_url: PostgreSQL connection URL.
            pool_size: Connection pool size.
        """
        self._database_url = database_url
        self._pool_size = pool_size
        self._pool = None

    async def initialize(self) -> None:
        """Create connection pool and ensure table exists."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgreSQL storage. "
                "Install with: pip install asyncpg"
            )

        # Parse URL - asyncpg doesn't support +asyncpg suffix
        db_url = self._database_url
        if "+asyncpg" in db_url:
            db_url = db_url.replace("postgresql+asyncpg", "postgresql")

        self._pool = await asyncpg.create_pool(
            db_url,
            min_size=1,
            max_size=self._pool_size,
        )
        
        # Create table if not exists
        async with self._pool.acquire() as conn:
            await conn.execute(self.CREATE_TABLE_SQL)
        
        logger.info("PostgreSQL store initialized")

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL store closed")

    def _ensure_pool(self) -> None:
        """Ensure connection pool is available."""
        if self._pool is None:
            raise RuntimeError("PostgresStore not initialized. Call initialize() first.")

    def _to_uuid(self, session_id: str) -> uuid_module.UUID:
        """Convert session_id string to UUID object for asyncpg."""
        try:
            return uuid_module.UUID(session_id)
        except ValueError:
            # If not a valid UUID string, generate a deterministic UUID from the string
            return uuid_module.uuid5(uuid_module.NAMESPACE_OID, session_id)

    async def create(self, session_id: str) -> SessionData:
        """Create a new session."""
        self._ensure_pool()
        
        session = SessionData(session_id=session_id)
        uuid_id = self._to_uuid(session_id)
        
        async with self._pool.acquire() as conn:
            await conn.execute(
                self.INSERT_SQL,
                uuid_id,
                json.dumps(session.messages),
                json.dumps(session.metadata),
                session.created_at,
                session.last_activity,
            )
        
        logger.debug(f"Created session in DB: {session_id}")
        return session

    async def get(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        self._ensure_pool()
        uuid_id = self._to_uuid(session_id)
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(self.SELECT_SQL, uuid_id)
            
            if row is None:
                return None
            
            session = SessionData(
                session_id=str(row["session_id"]),
                messages=row["messages"] if row["messages"] else [],
                metadata=row["metadata"] if row["metadata"] else {},
                created_at=row["created_at"],
                last_activity=row["last_activity"],
            )
            
            # Update activity on read
            await self.update_activity(session_id)
            
            return session

    async def save(self, session: SessionData) -> None:
        """Save session state."""
        self._ensure_pool()
        
        session.touch()
        
        uuid_id = self._to_uuid(session.session_id)
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                self.UPDATE_SQL,
                uuid_id,
                json.dumps(session.messages),
                json.dumps(session.metadata),
                session.last_activity,
            )
            
            # If no rows updated, insert new
            if result == "UPDATE 0":
                await conn.execute(
                    self.INSERT_SQL,
                    uuid_id,
                    json.dumps(session.messages),
                    json.dumps(session.metadata),
                    session.created_at,
                    session.last_activity,
                )
        
        logger.debug(f"Saved session to DB: {session.session_id}")

    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        self._ensure_pool()
        uuid_id = self._to_uuid(session_id)
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(self.DELETE_SQL, uuid_id)
            deleted = result != "DELETE 0"
            
            if deleted:
                logger.debug(f"Deleted session from DB: {session_id}")
            
            return deleted

    async def update_activity(self, session_id: str) -> bool:
        """Update last activity timestamp."""
        self._ensure_pool()
        uuid_id = self._to_uuid(session_id)
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                self.UPDATE_ACTIVITY_SQL,
                uuid_id,
                datetime.utcnow(),
            )
            return result != "UPDATE 0"

    async def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove expired sessions."""
        self._ensure_pool()
        
        cutoff = datetime.utcnow() - timedelta(seconds=ttl_seconds)
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(self.CLEANUP_SQL, cutoff)
            
            # Parse DELETE count from result
            count = int(result.split()[-1]) if result else 0
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired sessions from DB")
            
            return count

    async def list_sessions(self) -> list[SessionData]:
        """List all active sessions."""
        self._ensure_pool()
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(self.LIST_SQL)
            
            return [
                SessionData(
                    session_id=str(row["session_id"]),
                    messages=row["messages"] if row["messages"] else [],
                    metadata=row["metadata"] if row["metadata"] else {},
                    created_at=row["created_at"],
                    last_activity=row["last_activity"],
                )
                for row in rows
            ]
