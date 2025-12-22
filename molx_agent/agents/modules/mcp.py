"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-18].
*  @Description MCP (Model Context Protocol) integration module.
*               Provides tools loading from MCP servers.
**************************************************************************
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool

from molx_agent.config import get_settings
logger = logging.getLogger(__name__)


class MCPToolLoader:
    """MCP Tool Loader - Load tools from MCP servers.
    
    Supports both stdio and HTTP transport modes.
    Uses langchain-mcp-adapters to convert MCP tools to LangChain tools.
    
    Example config:
        {
            "math": {
                "command": "python",
                "args": ["./mcp_servers/math_server.py"],
                "transport": "stdio"
            },
            "search": {
                "url": "http://localhost:8001/mcp",
                "transport": "http"
            }
        }
    """

    def __init__(self, servers_config: Optional[dict] = None) -> None:
        """Initialize the MCP Tool Loader.
        
        Args:
            servers_config: Dictionary of server configurations.
                           If None, loads from environment or config file.
                           If empty dict {}, no servers will be loaded.
        """
        if servers_config is None:
            self._servers_config = self._load_config()
        else:
            self._servers_config = servers_config
        self._client = None
        self._tools: list[BaseTool] = []
        self._loaded = False

    def _load_config(self) -> dict:
        """Load MCP server configuration from environment or file."""
        settings = get_settings()
        config_source = settings.MCP_SERVERS_CONFIG or os.getenv("MCP_SERVERS_CONFIG")
        if config_source:
            path_candidate = Path(config_source)
            if path_candidate.exists():
                try:
                    with open(path_candidate, "r", encoding="utf-8") as handle:
                        return json.load(handle)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(f"Failed to load MCP config from {path_candidate}: {exc}")
            else:
                try:
                    return json.loads(config_source)
                except json.JSONDecodeError as exc:
                    logger.warning(f"Failed to parse MCP_SERVERS_CONFIG payload: {exc}")

        # Try config file
        config_paths = [
            Path("config/mcp_servers.json"),
            Path("mcp_servers.json"),
            Path.home() / ".molx" / "mcp_servers.json",
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                        logger.info(f"Loaded MCP config from {config_path}")
                        return config
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load {config_path}: {e}")

        return {}

    @property
    def is_configured(self) -> bool:
        """Check if MCP servers are configured."""
        return bool(self._servers_config)

    async def _ensure_client(self) -> None:
        """Ensure MCP client is initialized."""
        if self._client is None and self._servers_config:
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                self._client = MultiServerMCPClient(self._servers_config)
                logger.info(f"MCP client initialized with {len(self._servers_config)} servers")
            except ImportError:
                logger.warning("langchain-mcp-adapters not installed, MCP tools disabled")
            except Exception as e:
                logger.error(f"Failed to initialize MCP client: {e}")

    async def get_tools_async(self) -> list[BaseTool]:
        """Get all tools from configured MCP servers (async).
        
        Returns:
            List of LangChain-compatible tools from MCP servers.
        """
        if self._loaded:
            return self._tools

        if not self._servers_config:
            logger.debug("No MCP servers configured")
            return []

        await self._ensure_client()

        if self._client is None:
            return []

        try:
            self._tools = await self._client.get_tools()
            self._loaded = True
            logger.info(f"Loaded {len(self._tools)} tools from MCP servers")
            
            # Log tool names for debugging
            if self._tools:
                tool_names = [t.name for t in self._tools]
                logger.debug(f"MCP tools: {tool_names}")
                
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            self._tools = []

        return self._tools

    def get_tools(self) -> list[BaseTool]:
        """Get all tools from configured MCP servers (sync wrapper).
        
        This is a convenience method that runs the async version
        in a new event loop if needed.
        
        Returns:
            List of LangChain-compatible tools from MCP servers.
        """
        if self._loaded:
            return self._tools

        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # If we get here, we're in an async context
            # Create a new thread to run the async code
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.get_tools_async())
                )
                return future.result(timeout=30)
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(self.get_tools_async())

    def get_server_names(self) -> list[str]:
        """Get names of configured MCP servers."""
        return list(self._servers_config.keys())


# Global instance for convenience
_mcp_loader: Optional[MCPToolLoader] = None


def get_mcp_loader() -> MCPToolLoader:
    """Get the global MCP tool loader instance."""
    global _mcp_loader
    if _mcp_loader is None:
        _mcp_loader = MCPToolLoader()
    return _mcp_loader


def get_mcp_tools() -> list[BaseTool]:
    """Get MCP tools using the global loader.
    
    This is a convenience function for synchronous contexts.
    
    Returns:
        List of LangChain-compatible tools from MCP servers.
    """
    loader = get_mcp_loader()
    return loader.get_tools()


async def get_mcp_tools_async() -> list[BaseTool]:
    """Get MCP tools using the global loader (async version).
    
    Returns:
        List of LangChain-compatible tools from MCP servers.
    """
    loader = get_mcp_loader()
    return await loader.get_tools_async()
