"""
Server configuration management for molx-server.

Uses pydantic-settings for environment-based configuration.
Session settings are managed by molx_core.config.CoreSettings.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = ".env.local" if os.path.exists(".env.local") else ".env"


class ServerSettings(BaseSettings):
    """Server configuration loaded from environment variables.
    
    Note: Session TTL, cleanup interval, and cleanup enabled settings
    are managed by molx_core.config.CoreSettings (MOLX_ prefix).
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="MOLX_SERVER_",
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    reload: bool = False

    # API settings
    api_version: str = "v1"
    api_prefix: str = f"/api/{api_version}"
    api_title: str = "Molx Agent Server API"
    api_description: str = "Drug design agent API named MolX"

    # CORS settings
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # Rate limiting
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Authentication
    api_key_enabled: bool = False
    api_keys: list[str] = []

    # Streaming settings
    stream_chunk_size: int = 1
    stream_timeout: float = 360.0  # Timeout for AI summary generation


@lru_cache
def get_server_settings() -> ServerSettings:
    """Get cached server settings instance."""
    return ServerSettings()

