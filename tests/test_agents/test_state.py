"""Tests for SAR agent state definitions."""

import pytest

from molx_agent.agents.modules.state import AgentState, Task


class TestTask:
    """Tests for Task TypedDict."""

    def test_task_creation(self) -> None:
        """Test creating a valid Task."""
        task: Task = {
            "id": "task_1",
            "type": "literature",
            "description": "Search for SAR data",
            "inputs": {"target": "COX-2"},
            "expected_outputs": ["summary", "compounds"],
            "depends_on": [],
            "status": "pending",
        }

        assert task["id"] == "task_1"
        assert task["type"] == "literature"
        assert task["status"] == "pending"

    def test_task_with_dependencies(self) -> None:
        """Test creating a Task with dependencies."""
        task: Task = {
            "id": "task_2",
            "type": "chemo",
            "description": "Analyze compounds",
            "inputs": {},
            "expected_outputs": ["sar_table"],
            "depends_on": ["task_1"],
            "status": "pending",
        }

        assert task["depends_on"] == ["task_1"]


class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_state_creation(self) -> None:
        """Test creating a valid AgentState."""
        state: AgentState = {
            "user_query": "Analyze aspirin SAR",
            "tasks": {},
            "current_task_id": None,
            "results": {},
            "final_answer": None,
        }

        assert state["user_query"] == "Analyze aspirin SAR"
        assert state["tasks"] == {}

    def test_state_with_tasks(self) -> None:
        """Test AgentState with tasks."""
        task: Task = {
            "id": "t1",
            "type": "literature",
            "description": "Search",
            "inputs": {},
            "expected_outputs": [],
            "depends_on": [],
            "status": "done",
        }

        state: AgentState = {
            "user_query": "Test query",
            "tasks": {"t1": task},
            "current_task_id": None,
            "results": {"t1": {"summary": "Found data"}},
        }

        assert "t1" in state["tasks"]
        assert state["tasks"]["t1"]["status"] == "done"
        assert state["results"]["t1"]["summary"] == "Found data"
