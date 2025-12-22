"""Tests for planner utilities."""

from __future__ import annotations

from molx_agent.agents.planner import PlanResult, PlannerAgent
from molx_agent.agents.modules.state import AgentState, Task
from molx_agent.memory import FileRecord, SessionMetadata


def test_planner_seeds_uploaded_file_into_tasks(monkeypatch):
    """Planner should automatically hint uploaded file paths to data cleaner tasks."""
    planner = PlannerAgent()

    def _fake_invoke(user_query: str, uploads):
        tasks: dict[str, Task] = {
            "clean": {
                "id": "clean",
                "type": "data_cleaner",
                "description": "Process user uploaded dataset",
                "inputs": {},
                "depends_on": [],
                "status": "pending",
            }
        }
        return PlanResult(reasoning="ok", tasks=tasks)

    monkeypatch.setattr(planner, "_invoke_planner_llm", _fake_invoke)

    state: AgentState = {
        "user_query": "Analyze my upload",
        "messages": [],
        "results": {},
        "uploaded_files": [
            {"file_path": "/tmp/dataset.csv", "file_name": "dataset.csv"},
        ],
    }

    planner.think(state)

    inputs = state["tasks"]["clean"]["inputs"]
    assert inputs["file_path"] == "/tmp/dataset.csv"
    assert inputs["file_name"] == "dataset.csv"


def test_planner_reads_metadata_when_state_is_empty(monkeypatch):
    planner = PlannerAgent()

    def _fake_invoke(user_query: str, uploads):
        tasks: dict[str, Task] = {
            "clean": {
                "id": "clean",
                "type": "data_cleaner",
                "description": "Process user uploaded dataset",
                "inputs": {},
                "depends_on": [],
                "status": "pending",
            }
        }
        return PlanResult(reasoning="ok", tasks=tasks)

    monkeypatch.setattr(planner, "_invoke_planner_llm", _fake_invoke)

    metadata = SessionMetadata()
    metadata.add_uploaded_file(
        FileRecord(
            file_id="file-1",
            file_name="dataset.csv",
            file_path="/tmp/dataset.csv",
        )
    )

    state: AgentState = {
        "user_query": "Analyze my upload",
        "messages": [],
        "results": {},
        "uploaded_files": [],
        "_memory_metadata": metadata,
    }

    planner.think(state)

    inputs = state["tasks"]["clean"]["inputs"]
    assert inputs["file_path"] == "/tmp/dataset.csv"
