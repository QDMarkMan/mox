"""Configuration management for molx-agent."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = ".env.local" if os.path.exists(".env.local") else ".env"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_artifacts_dir() -> str:
    return str(PROJECT_ROOT / "artifacts")

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    LOCAL_OPENAI_API_KEY: str = ""
    LOCAL_OPENAI_MODEL: str = "gpt-4o"
    LOCAL_OPENAI_BASE_URL: Optional[str] = None

    # Agent Configuration
    agent_verbose: bool = False
    agent_max_iterations: int = 10

    # Artifact & output configuration
    artifacts_root: str = Field(default_factory=_default_artifacts_dir)
    standardize_subdir: str = "standardized"
    reports_subdir: str = "reports"
    visualizations_subdir: str = "visualizations"
    uploads_subdir: str = "uploads"

    # MCP (Model Context Protocol) Configuration
    MCP_ENABLED: bool = True
    MCP_SERVERS_CONFIG: Optional[str] = None  # JSON string or path to config file


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
