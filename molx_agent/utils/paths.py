"""Filesystem helpers for agent artifacts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from molx_agent.config import get_settings


@lru_cache
def _artifacts_root() -> Path:
    settings = get_settings()
    return Path(settings.artifacts_root).expanduser()


def ensure_artifact_dir(*parts: str) -> Path:
    """Resolve and create a sub-directory under the artifacts root."""
    base = _artifacts_root()
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_standardize_output_dir() -> Path:
    """Directory dedicated to molecule standardization outputs."""
    settings = get_settings()
    return ensure_artifact_dir(settings.standardize_subdir)


def get_reports_dir() -> Path:
    """Directory containing generated SAR reports."""
    settings = get_settings()
    return ensure_artifact_dir(settings.reports_subdir)


def get_visualizations_dir() -> Path:
    """Directory used for visualization assets."""
    settings = get_settings()
    return ensure_artifact_dir(settings.visualizations_subdir)


def get_tool_output_dir(tool_name: str) -> Path:
    """Directory bucket for tool specific artifacts."""
    safe_name = tool_name.replace(' ', '_')
    return ensure_artifact_dir('tools', safe_name)


def get_uploads_dir(session_id: str) -> Path:
    """Directory reserved for user uploaded files for a session."""
    settings = get_settings()
    return ensure_artifact_dir(settings.uploads_subdir, session_id)
