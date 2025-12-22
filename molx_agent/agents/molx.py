"""LangGraph-enabled MolxAgent orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage

try:  # pragma: no cover - prefer LangGraph when installed
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - fallback used in tests
    class _CompiledGraph:
        def __init__(
            self,
            entry: str,
            nodes: Dict[str, Callable[[AgentState], AgentState]],
            edges: Dict[str, str],
            conditionals: Dict[
                str, tuple[Callable[[AgentState], str], Dict[str, str]]
            ],
        ) -> None:
            self._entry = entry
            self._nodes = nodes
            self._edges = edges
            self._conditionals = conditionals

        def invoke(self, state: AgentState) -> AgentState:
            current = self._entry
            while current != "__end__":
                node = self._nodes[current]
                state = node(state)
                if current in self._conditionals:
                    fn, mapping = self._conditionals[current]
                    route = fn(state)
                    current = mapping[route]
                elif current in self._edges:
                    current = self._edges[current]
                else:
                    break
            return state

    class StateGraph:  # type: ignore[override]
        def __init__(self, _state_type: Any) -> None:
            self._nodes: Dict[str, Callable[[AgentState], AgentState]] = {}
            self._edges: Dict[str, str] = {}
            self._conditionals: Dict[
                str, tuple[Callable[[AgentState], str], Dict[str, str]]
            ] = {}
            self._entry: Optional[str] = None

        def add_node(self, name: str, fn: Callable[[AgentState], AgentState]) -> None:
            self._nodes[name] = fn

        def set_entry_point(self, name: str) -> None:
            self._entry = name

        def add_conditional_edges(
            self,
            name: str,
            fn: Callable[[AgentState], str],
            mapping: Dict[str, str],
        ) -> None:
            self._conditionals[name] = (fn, mapping)

        def add_edge(self, source: str, target: str) -> None:
            self._edges[source] = target

        def compile(self) -> _CompiledGraph:
            if self._entry is None:
                raise RuntimeError("Entry point not set")
            return _CompiledGraph(
                self._entry,
                self._nodes,
                self._edges,
                self._conditionals,
            )

    END = "__end__"

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.data_cleaner import DataCleanerAgent
from molx_agent.agents.intent_classifier import Intent, IntentClassifierAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.planner import PlannerAgent
from molx_agent.agents.reporter import ReporterAgent
from molx_agent.agents.sar import SARAgent
from molx_core.memory import SessionRecorder

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3
_DEFAULT_GRAPH = None


def _ensure_results(state: AgentState) -> None:
    state.setdefault("results", {})


def _has_pending_tasks(state: AgentState) -> bool:
    tasks = state.get("tasks", {})
    return any(task.get("status") in (None, "pending", "running") for task in tasks.values())


def _pick_next_task(state: AgentState, planner: PlannerAgent) -> Optional[str]:
    task_id = state.get("current_task_id")
    if task_id:
        return task_id
    pick_fn = getattr(planner, "_pick_next_task", None)
    if pick_fn is None:
        return None
    return pick_fn(state)


def _finalize_response(state: AgentState) -> AgentState:
    results = state.get("results", {})
    reflection = state.get("reflection", {})
    report_path = None

    for task_result in results.values():
        if isinstance(task_result, dict):
            files = task_result.get("output_files") or {}
            if "html" in files:
                report_path = files["html"]
                break
            if task_result.get("report_path"):
                report_path = task_result["report_path"]
                break

    summary = reflection.get("summary", "Analysis complete.")
    response = f"✅ {summary}"
    if report_path:
        response += f"\n\n📊 Report generated: {report_path}"

    state["final_response"] = response
    state.setdefault("messages", [])
    return state


def build_sar_graph(
    *,
    intent_classifier: Optional[IntentClassifierAgent] = None,
    planner: Optional[PlannerAgent] = None,
    workers: Optional[Dict[str, BaseAgent]] = None,
    max_iterations: int = MAX_ITERATIONS,
):
    """Compile the LangGraph responsible for SAR orchestration."""

    classifier = intent_classifier or IntentClassifierAgent()
    plan_agent = planner or PlannerAgent()
    worker_map = workers or {
        "data_cleaner": DataCleanerAgent(),
        "sar": SARAgent(),
        "reporter": ReporterAgent(),
    }

    def classify_node(state: AgentState) -> AgentState:
        return classifier.run(state)

    def unsupported_node(state: AgentState) -> AgentState:
        intent = state.get("intent", Intent.UNSUPPORTED)
        message = classifier.get_response(intent) or "Unsupported request."
        state["final_response"] = message
        state["final_answer"] = message
        return state

    def planner_think_node(state: AgentState) -> AgentState:
        return plan_agent.think(state)

    def planner_reflect_node(state: AgentState) -> AgentState:
        return plan_agent.reflect(state)

    def planner_optimize_node(state: AgentState) -> AgentState:
        return plan_agent.optimize(state)

    def act_node(state: AgentState) -> AgentState:
        _ensure_results(state)
        task_id = _pick_next_task(state, plan_agent)
        if not task_id:
            return state

        tasks = state.get("tasks", {})
        task = tasks.get(task_id)
        if not task:
            state["current_task_id"] = None
            return state

        task_type = task.get("type")
        worker = worker_map.get(task_type)

        if not worker:
            task["status"] = "error"
            state["results"][task_id] = {"error": f"Unknown worker: {task_type}"}
            state["current_task_id"] = None
            return state

        try:
            state["current_task_id"] = task_id
            task["status"] = "running"
            state = worker.run(state)
            task_result = state.get("results", {}).get(task_id, {})
            if task_result.get("error"):
                task["status"] = "error"
            else:
                task["status"] = "done"
        except Exception as exc:  # pragma: no cover - safety net
            task["status"] = "error"
            state["results"][task_id] = {"error": str(exc)}
            state["error"] = str(exc)
        finally:
            state["current_task_id"] = None

        return state

    def finalize_node(state: AgentState) -> AgentState:
        return _finalize_response(state)

    def route_after_classify(state: AgentState) -> str:
        intent = state.get("intent", Intent.SAR_ANALYSIS)
        if not classifier.is_supported(intent):
            return "unsupported"
        return "plan"

    def route_after_plan(state: AgentState) -> str:
        return "act" if _has_pending_tasks(state) else "reflect"

    def route_after_act(state: AgentState) -> str:
        return "act" if _has_pending_tasks(state) else "reflect"

    def route_after_reflect(state: AgentState) -> str:
        reflection = state.get("reflection", {})
        if reflection.get("success"):
            return "finalize"
        if reflection.get("should_replan") and state.get("iteration", 0) < max_iterations:
            return "optimize"
        return "finalize"

    def route_after_optimize(state: AgentState) -> str:
        if _has_pending_tasks(state):
            return "act"
        return "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("plan", planner_think_node)
    graph.add_node("act", act_node)
    graph.add_node("reflect", planner_reflect_node)
    graph.add_node("optimize", planner_optimize_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("unsupported", unsupported_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", route_after_classify, {"plan": "plan", "unsupported": "unsupported"}
    )
    graph.add_conditional_edges("plan", route_after_plan, {"act": "act", "reflect": "reflect"})
    graph.add_conditional_edges("act", route_after_act, {"act": "act", "reflect": "reflect"})
    graph.add_conditional_edges(
        "reflect", route_after_reflect, {"finalize": "finalize", "optimize": "optimize"}
    )
    graph.add_conditional_edges(
        "optimize", route_after_optimize, {"act": "act", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    graph.add_edge("unsupported", END)

    return graph.compile()


def get_sar_graph():
    """Return the shared SAR graph compiled with default components."""
    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        _DEFAULT_GRAPH = build_sar_graph()
    return _DEFAULT_GRAPH


def reset_sar_graph() -> None:
    """Clear the cached SAR graph."""
    global _DEFAULT_GRAPH
    _DEFAULT_GRAPH = None


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
        return self._graph.invoke(state)

    def invoke(self, user_query: str, *, state: Optional[AgentState] = None) -> AgentState:
        """Convenience helper to run a query with optional persistent state."""
        state = state or AgentState(messages=[], tasks={}, results={})
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
    return graph.invoke(initial_state)


class ChatSession:
    """Interactive chat session wrapper for MolxAgent."""

    def __init__(self, recorder: Optional[SessionRecorder] = None):
        self.agent = MolxAgent()
        self.state = AgentState(messages=[], tasks={}, results={})
        self._recorder = recorder
        self._last_state: Optional[AgentState] = None

    def attach_recorder(self, recorder: SessionRecorder) -> None:
        """Attach a session recorder for persistence."""
        self._recorder = recorder

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
        self.state = AgentState(messages=[], tasks={}, results={})

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
