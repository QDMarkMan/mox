"""
Factory for creating conversation store instances.

Selects storage backend based on configuration.
"""

import logging
from typing import Optional

from molx_core.config import get_core_settings
from molx_core.memory.store import ConversationStore
from molx_core.memory.memory_store import MemoryStore


logger = logging.getLogger(__name__)

# Global store instance
_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """
    Get the configured conversation store instance.
    
    Uses MOLX_MEMORY_BACKEND environment variable to select backend:
    - "memory" (default): In-memory storage
    - "postgres": PostgreSQL storage
    
    Returns:
        ConversationStore instance.
        
    Raises:
        ValueError: If unknown backend specified.
        ImportError: If postgres backend requested but asyncpg not installed.
    """
    global _store
    
    if _store is not None:
        return _store
    
    settings = get_core_settings()
    backend = settings.memory_backend.lower()
    
    if backend == "memory":
        logger.info("Using in-memory conversation store")
        _store = MemoryStore()
        
    elif backend == "postgres":
        if not settings.database_url:
            raise ValueError(
                "MOLX_DATABASE_URL must be set for postgres backend"
            )
        
        from molx_core.memory.postgres_store import PostgresStore
        
        logger.info("Using PostgreSQL conversation store")
        _store = PostgresStore(
            database_url=settings.database_url,
            pool_size=settings.database_pool_size,
        )
        
    else:
        raise ValueError(
            f"Unknown memory backend: {backend}. "
            "Valid options: 'memory', 'postgres'"
        )
    
    return _store


async def initialize_store() -> None:
    """
    Initialize the conversation store.
    
    Call this during application startup.
    """
    store = get_conversation_store()
    await store.initialize()
    logger.info("Conversation store initialized")


async def close_store() -> None:
    """
    Close the conversation store.
    
    Call this during application shutdown.
    """
    global _store
    
    if _store is not None:
        await _store.close()
        _store = None
        logger.info("Conversation store closed")


def reset_store() -> None:
    """
    Reset the global store instance.
    
    Useful for testing.
    """
    global _store
    _store = None
