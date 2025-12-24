"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-15].
*  @Description Tools package - exports get_all_tools and get_tool_names.
**************************************************************************
"""

from __future__ import annotations

from typing import List


def get_all_tools():
    """Lazily import heavy tool registry when required."""
    from molx_agent.agents.modules.tools import get_all_tools as _get_all_tools

    return _get_all_tools()


def get_tool_names() -> List[str]:
    """Lazily list tool names."""
    from molx_agent.agents.modules.tools import get_tool_names as _get_tool_names

    return _get_tool_names()


def get_registry():
    """Get the singleton ToolRegistry instance."""
    from molx_agent.agents.modules.tools import get_registry as _get_registry

    return _get_registry()


__all__ = ["get_all_tools", "get_tool_names", "get_registry"]
