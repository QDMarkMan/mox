"""Tests for the LangGraph-enabled MolxAgent."""

from __future__ import annotations

import pytest

from molx_agent.agents.intent_classifier import Intent
from molx_agent.agents.molx import MolxAgent, reset_molx_graph
from molx_agent.agents.modules.state import AgentState


class _StubClassifier:
    def __init__(self, supported: bool = True) -> None:
        self.supported = supported

    def run(self, state):
        state["intent"] = Intent.SAR_ANALYSIS if self.supported else Intent.UNSUPPORTED
        return state

    def is_supported(self, intent):
        return self.supported and intent == Intent.SAR_ANALYSIS

    def get_response(self, intent):
        return "friendly message"


class _StubPlanner:
    def think(self, state):
        state["tasks"] = {
            "task-1": {
                "id": "task-1",
                "type": "data_cleaner",
                "description": "stub",
                "inputs": {},
                "status": "pending",
            }
        }
        state["current_task_id"] = "task-1"
        state["iteration"] = state.get("iteration", 0) + 1
        return state

    def _pick_next_task(self, state):
        for tid, task in state.get("tasks", {}).items():
            if task.get("status") in (None, "pending"):
                task["status"] = "pending"
                return tid
        return None

    def reflect(self, state):
        state["reflection"] = {"success": True, "summary": "All good"}
        return state

    def optimize(self, state):
        return state


class _StubWorker:
    def run(self, state):
        tid = state.get("current_task_id")
        state.setdefault("results", {})
        state["results"][tid] = {"output": "done"}
        return state


@pytest.fixture(autouse=True)
def _reset_graph():
    reset_molx_graph()
    yield
    reset_molx_graph()


def test_molx_agent_handles_unsupported_intent():
    agent = MolxAgent(intent_classifier=_StubClassifier(supported=False), workers={})
    final_state = agent.run(AgentState(user_query="hello", messages=[], tasks={}, results={}))
    assert "friendly message" in final_state["final_response"]


def test_molx_agent_executes_worker():
    agent = MolxAgent(
        intent_classifier=_StubClassifier(supported=True),
        planner=_StubPlanner(),
        workers={"data_cleaner": _StubWorker()},
    )

    final_state = agent.run(AgentState(user_query="analyze", messages=[], tasks={}, results={}))

    assert final_state["results"]["task-1"]["output"] == "done"
    assert "All good" in final_state["final_response"]
