"""LangGraph-enabled MolxAgent orchestrator."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.data_cleaner import DataCleanerAgent
from molx_agent.agents.intent_classifier import IntentClassifierAgent
from molx_agent.agents.modules.graph import (
    MAX_ITERATIONS,
    build_sar_graph,
    get_sar_graph,
    reset_sar_graph,
)
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.planner import PlannerAgent
from molx_agent.agents.reporter import ReporterAgent
from molx_agent.agents.sar import SARAgent
from molx_agent.memory import FileRecord, SessionMetadata, SessionRecorder

logger = logging.getLogger(__name__)


class MolxAgent(BaseAgent):
    """Main orchestrator that executes the LangGraph pipeline."""

    def __init__(
        self,
        *,
        intent_classifier: Optional[IntentClassifierAgent] = None,
        planner: Optional[PlannerAgent] = None,
        workers: Optional[Dict[str, BaseAgent]] = None,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        super().__init__(
            name="molx",
            description="Main orchestrator for SAR analysis using LangGraph",
        )
        self.intent_classifier = intent_classifier or IntentClassifierAgent()
        self.planner = planner or PlannerAgent()
        self.workers = workers or {
            "data_cleaner": DataCleanerAgent(),
            "sar": SARAgent(),
            "reporter": ReporterAgent(),
        }
        self.max_iterations = max_iterations
        self._graph = build_sar_graph(
            intent_classifier=self.intent_classifier,
            planner=self.planner,
            workers=self.workers,
            max_iterations=self.max_iterations,
        )

    @property
    def graph(self):  # pragma: no cover - simple proxy
        return self._graph

    def run(self, state: AgentState) -> AgentState:
        """Execute the graph with the provided state."""
        state.setdefault("messages", [])
        state.setdefault("results", {})
        state.setdefault("tasks", {})
        state.setdefault("uploaded_files", [])
        return self._graph.invoke(state, config={"recursion_limit": 100})

    def invoke(self, user_query: str, *, state: Optional[AgentState] = None) -> AgentState:
        """Convenience helper to run a query with optional persistent state."""
        state = state or AgentState(messages=[], tasks={}, results={}, uploaded_files=[])
        state["user_query"] = user_query
        return self.run(state)


def run_sar_agent(user_query: str) -> AgentState:
    """Helper for CLI/server callers that don't need custom wiring."""
    graph = get_sar_graph()
    initial_state: AgentState = {
        "user_query": user_query,
        "messages": [],
        "results": {},
        "tasks": {},
    }
    return graph.invoke(initial_state, config={"recursion_limit": 100})


class ChatSession:
    """Interactive chat session wrapper for MolxAgent."""

    def __init__(self, recorder: Optional[SessionRecorder] = None):
        self.agent = MolxAgent()
        self.state = AgentState(messages=[], tasks={}, results={}, uploaded_files=[])
        self.state["_memory_metadata"] = None
        self._recorder = recorder
        self._last_state: Optional[AgentState] = None
        self._metadata: Optional[SessionMetadata] = None

    def attach_recorder(self, recorder: SessionRecorder) -> None:
        """Attach a session recorder for persistence."""
        self._recorder = recorder
        metadata = recorder.metadata
        if isinstance(metadata, SessionMetadata):
            self._metadata = metadata
            self.state["_memory_metadata"] = metadata

    def send(self, user_input: str) -> str:
        """Send user input to the agent and get a response."""
        self.state.setdefault("messages", []).append(HumanMessage(content=user_input))
        self.state = self.agent.invoke(user_input, state=self.state)
        self._last_state = self.state

        final_response = self.state.get("final_response", "")

        if self._recorder:
            try:
                self._recorder.record_turn(
                    query=user_input,
                    response=final_response,
                    state=self.state,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to record session turn: %s", exc)

        self.state.setdefault("messages", []).append(AIMessage(content=final_response))
        return final_response

    def clear(self) -> None:
        """Clear conversation history."""
        self.state = AgentState(messages=[], tasks={}, results={}, uploaded_files=[])
        self.state["_memory_metadata"] = self._metadata

    def get_history(self) -> list[dict[str, str]]:
        """Get conversation history."""
        history: list[dict[str, str]] = []
        for msg in self.state.get("messages", []):
            role = "user"
            if hasattr(msg, "type"):
                if msg.type == "ai":
                    role = "agent"
                elif msg.type == "system":
                    role = "system"
                elif msg.type == "human":
                    role = "user"
            content = msg.content if hasattr(msg, "content") else str(msg)
            history.append({"role": role, "content": content})
        return history

    def load_uploaded_files(self, metadata: SessionMetadata) -> None:
        """Hydrate the current state with uploaded file metadata."""
        self._metadata = metadata
        self.state["_memory_metadata"] = metadata
        self.state["uploaded_files"] = [record.to_dict() for record in metadata.uploaded_files]

    def register_uploaded_file(self, record: FileRecord) -> None:
        """Record a new uploaded file inside the conversation state."""
        uploads = self.state.setdefault("uploaded_files", [])
        uploads.append(record.to_dict())
