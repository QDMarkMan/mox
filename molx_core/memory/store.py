"""
Abstract conversation storage interface.

Defines the contract for all storage backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class SessionData:
    """
    Container for session state.
    
    Attributes:
        session_id: Unique session identifier.
        messages: List of conversation messages.
        metadata: Additional session metadata.
        created_at: Session creation timestamp.
        last_activity: Last activity timestamp.
    """
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.touch()

    @property
    def message_count(self) -> int:
        """Get number of messages in session."""
        return len(self.messages)


class ConversationStore(ABC):
    """
    Abstract base class for conversation storage backends.
    
    All storage implementations must implement these methods.
    """

    @abstractmethod
    async def create(self, session_id: str) -> SessionData:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier.
            
        Returns:
            New SessionData instance.
        """
        ...

    @abstractmethod
    async def get(self, session_id: str) -> Optional[SessionData]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            SessionData if found, None otherwise.
        """
        ...

    @abstractmethod
    async def save(self, session: SessionData) -> None:
        """
        Save session state.
        
        Args:
            session: SessionData to persist.
        """
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def update_activity(self, session_id: str) -> bool:
        """
        Update last activity timestamp.
        
        Args:
            session_id: Session to update.
            
        Returns:
            True if updated, False if not found.
        """
        ...

    @abstractmethod
    async def cleanup_expired(self, ttl_seconds: int) -> int:
        """
        Remove expired sessions.
        
        Args:
            ttl_seconds: Session time-to-live in seconds.
            
        Returns:
            Number of sessions removed.
        """
        ...

    @abstractmethod
    async def list_sessions(self) -> list[SessionData]:
        """
        List all active sessions.
        
        Returns:
            List of active SessionData instances.
        """
        ...

    async def initialize(self) -> None:
        """
        Initialize the store (e.g., create tables, connection pool).
        
        Override in subclasses if needed.
        """
        pass

    async def close(self) -> None:
        """
        Close the store and release resources.
        
        Override in subclasses if needed.
        """
        pass
