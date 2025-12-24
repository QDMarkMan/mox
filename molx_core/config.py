"""
Shared configuration for molx_core module.

Uses pydantic-settings for environment-based configuration.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = ".env.local" if os.path.exists(".env.local") else ".env"


class CoreSettings(BaseSettings):
    """Core configuration loaded from environment variables.
    
    Session settings here are the authoritative source for all modules.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="MOLX_",
    )

    # Memory/Session storage backend
    memory_backend: str = "memory"  # "memory" | "postgres"

    # Database settings (for postgres backend)
    database_url: Optional[str] = None
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout: float = 30.0

    # Session settings (authoritative for all modules)
    session_ttl: int = 315360000  # 10 years - keep sessions permanently by default
    session_cleanup_enabled: bool = False  # Disable auto cleanup by default
    session_cleanup_interval: int = 86400  # Check daily when enabled
    session_max_count: int = 5000  # Maximum number of sessions


@lru_cache
def get_core_settings() -> CoreSettings:
    """Get cached core settings instance."""
    return CoreSettings()

