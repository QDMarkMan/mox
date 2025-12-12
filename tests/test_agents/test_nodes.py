"""Tests for SAR agent utilities."""

import pytest

from molx_agent.agents.modules.llm import parse_json_response
from molx_agent.agents.modules.state import AgentState, Task


class TestParseJsonResponse:
    """Tests for JSON parsing utility."""

    def test_parse_plain_json(self) -> None:
        """Test parsing plain JSON."""
        content = '{"key": "value"}'
        result = parse_json_response(content)
        assert result == {"key": "value"}

    def test_parse_json_with_code_block(self) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        content = '```json\n{"key": "value"}\n```'
        result = parse_json_response(content)
        assert result == {"key": "value"}

    def test_parse_json_with_plain_code_block(self) -> None:
        """Test parsing JSON wrapped in plain code block."""
        content = '```\n{"nested": {"a": 1}}\n```'
        result = parse_json_response(content)
        assert result == {"nested": {"a": 1}}


class TestAgentState:
    """Tests for AgentState type."""

    def test_create_state(self) -> None:
        """Test creating agent state."""
        state: AgentState = {
            "user_query": "Test query",
            "tasks": {},
            "current_task_id": None,
            "results": {},
        }
        assert state["user_query"] == "Test query"

    def test_state_with_tasks(self) -> None:
        """Test state with tasks."""
        task: Task = {
            "id": "t1",
            "type": "data_cleaner",
            "description": "Clean data",
            "inputs": {},
            "expected_outputs": [],
            "depends_on": [],
            "status": "pending",
        }

        state: AgentState = {
            "user_query": "Test",
            "tasks": {"t1": task},
            "results": {},
        }

        assert "t1" in state["tasks"]
        assert state["tasks"]["t1"]["type"] == "data_cleaner"
