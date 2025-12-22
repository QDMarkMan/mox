"""Tests for MCP integration module."""

import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock

from molx_agent.agents.modules.mcp import MCPToolLoader, get_mcp_loader
import molx_agent.config as agent_config


class TestMCPToolLoader:
    """Tests for MCPToolLoader class."""

    def test_loader_explicit_empty_config(self) -> None:
        """Test loader with explicitly empty configuration."""
        # When explicitly passing empty dict, is_configured should be False
        # Note: if config/mcp_servers.json exists, a loader with None config
        # will load from file. This test uses explicit empty dict.
        loader = MCPToolLoader(servers_config={})
        # Empty dict means no servers configured = not configured? No.
        # Actually is_configured just checks bool(dict), and {} is falsy
        # Let's check the behavior is consistent.
        assert loader.get_server_names() == []

    def test_loader_with_config(self) -> None:
        """Test loader with configuration."""
        config = {
            "test_server": {
                "command": "python",
                "args": ["./test.py"],
                "transport": "stdio",
            }
        }
        loader = MCPToolLoader(servers_config=config)
        assert loader.is_configured
        assert loader.get_server_names() == ["test_server"]

    def test_loader_reads_config_from_settings_path(self, tmp_path, monkeypatch) -> None:
        """Loader should respect MCP_SERVERS_CONFIG path configured via settings."""
        config_path = tmp_path / "mcp_servers.json"
        config_payload = {
            "file_server": {
                "command": "python",
                "args": ["./file_server.py"],
                "transport": "stdio",
            }
        }
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")
        monkeypatch.setenv("MCP_SERVERS_CONFIG", str(config_path))
        agent_config.get_settings.cache_clear()

        loader = MCPToolLoader()
        assert loader.is_configured
        assert loader.get_server_names() == ["file_server"]

        agent_config.get_settings.cache_clear()

    def test_loader_config_from_dict(self) -> None:
        """Test loader initialized with dict config."""
        config = {
            "server1": {"url": "http://localhost:8000/mcp", "transport": "http"},
            "server2": {"command": "node", "args": ["server.js"], "transport": "stdio"},
        }
        loader = MCPToolLoader(servers_config=config)
        assert loader.is_configured
        assert set(loader.get_server_names()) == {"server1", "server2"}

    def test_get_tools_sync_no_config(self) -> None:
        """Test sync tool loading with no config returns empty list."""
        loader = MCPToolLoader(servers_config={})
        tools = loader.get_tools()
        assert tools == []


class TestMCPConfiguration:
    """Tests for MCP configuration loading."""

    def test_default_mcp_settings(self) -> None:
        """Test default MCP settings."""
        from molx_agent.config import Settings
        
        settings = Settings()
        assert settings.MCP_ENABLED is True
        assert settings.MCP_SERVERS_CONFIG is None

    def test_mcp_disabled(self) -> None:
        """Test MCP can be disabled via settings."""
        from molx_agent.config import Settings
        
        settings = Settings(MCP_ENABLED=False)
        assert settings.MCP_ENABLED is False


class TestMCPGlobalLoader:
    """Tests for global MCP loader functions."""

    def test_get_mcp_loader_singleton(self) -> None:
        """Test that get_mcp_loader returns the same instance."""
        # Reset global loader for test
        import molx_agent.agents.modules.mcp as mcp_module
        mcp_module._mcp_loader = None
        
        loader1 = get_mcp_loader()
        loader2 = get_mcp_loader()
        assert loader1 is loader2


class TestMCPServerConfig:
    """Tests for MCP server configuration file."""

    def test_config_file_valid_json(self) -> None:
        """Test that example config file is valid JSON."""
        import os
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "config",
            "mcp_servers.json"
        )
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            assert isinstance(config, dict)
