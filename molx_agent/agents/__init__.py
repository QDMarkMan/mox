"""Lazy exports for agent classes."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseAgent",
    "PlannerAgent",
    "DataCleanerAgent",
    "SARAgent",
    "ReporterAgent",
    "ToolAgent",
    "AgentState",
    "Task",
    "MolxAgent",
    "run_sar_agent",
    "get_all_tools",
    "get_tool_names",
]

_EXPORTS = {
    "BaseAgent": "molx_agent.agents.base",
    "PlannerAgent": "molx_agent.agents.planner",
    "DataCleanerAgent": "molx_agent.agents.data_cleaner",
    "SARAgent": "molx_agent.agents.sar",
    "ReporterAgent": "molx_agent.agents.reporter",
    "ToolAgent": "molx_agent.agents.tool_agent",
    "AgentState": "molx_agent.agents.modules.state",
    "Task": "molx_agent.agents.modules.state",
    "MolxAgent": "molx_agent.agents.molx",
    "run_sar_agent": "molx_agent.agents.molx",
    "get_all_tools": "molx_agent.agents.modules.tools",
    "get_tool_names": "molx_agent.agents.modules.tools",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
