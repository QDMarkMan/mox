# Agents Module

This module contains AI agent implementations for drug design.

## Structure

- `base.py` - Base agent class
- `molx.py` - Main orchestrator agent using ReAct pattern
- `planner.py` - Task planning and optimization
- `intent_classifier.py` - Query intent classification
- `data_cleaner.py` - Molecular data extraction
- `sar.py` - SAR analysis (R-group decomposition)
- `reporter.py` - HTML report generation
- `tool_agent.py` - Generic tool-based agent

## Modules

- `modules/state.py` - Agent state definitions
- `modules/llm.py` - LLM utilities
- `modules/tools.py` - Tool registration
- `modules/mcp.py` - **MCP (Model Context Protocol) integration**

## MCP Integration

The agents support external tools via MCP (Model Context Protocol).

### Configuration

1. Create `config/mcp_servers.json`:
```json
{
    "chemistry": {
        "command": "python",
        "args": ["./examples/example_mcp_server.py"],
        "transport": "stdio"
    }
}
```

2. Or set environment variable:
```bash
export MCP_SERVERS_CONFIG='{"server": {"url": "http://localhost:8000/mcp", "transport": "http"}}'
```

### Disabling MCP

Set `MCP_ENABLED=false` in your `.env` file or environment.

### Creating MCP Servers

See `examples/example_mcp_server.py` for a template.

