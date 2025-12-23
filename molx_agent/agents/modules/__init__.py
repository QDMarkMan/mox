"""Agents module for SAR analysis."""

from .state import AgentState, Task


# Lazy imports for heavy modules
def _lazy_imports():
    from .graph import MAX_ITERATIONS, build_molx_graph, get_molx_graph, reset_molx_graph
    from molx_agent.agents.molx import run_sar_agent
    from .mcp import MCPToolLoader, get_mcp_tools, get_mcp_tools_async, get_mcp_loader

    return {
        "build_molx_graph": build_molx_graph,
        "get_molx_graph": get_molx_graph,
        "reset_molx_graph": reset_molx_graph,
        "MAX_ITERATIONS": MAX_ITERATIONS,
        "run_sar_agent": run_sar_agent,
        "MCPToolLoader": MCPToolLoader,
        "get_mcp_tools": get_mcp_tools,
        "get_mcp_tools_async": get_mcp_tools_async,
        "get_mcp_loader": get_mcp_loader,
    }

__all__ = [
    "AgentState",
    "Task",
    "build_molx_graph",
    "get_molx_graph",
    "reset_molx_graph",
    "MAX_ITERATIONS",
    "run_sar_agent",
    # MCP
    "MCPToolLoader",
    "get_mcp_tools",
    "get_mcp_tools_async",
    "get_mcp_loader",
]

def __getattr__(name: str):
    """Lazy load heavy modules on first access."""
    lazy_modules = _lazy_imports()
    if name in lazy_modules:
        return lazy_modules[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
