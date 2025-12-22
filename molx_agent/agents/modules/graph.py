"""Shared LangGraph helpers for Molx SAR orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.data_cleaner import DataCleanerAgent
from molx_agent.agents.intent_classifier import Intent, IntentClassifierAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.planner import PlannerAgent
from molx_agent.agents.reporter import ReporterAgent
from molx_agent.agents.sar import SARAgent

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

        def invoke(self, state: AgentState, **kwargs: Any) -> AgentState:
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

MAX_ITERATIONS = 3
_DEFAULT_GRAPH = None


@dataclass
class SarGraphNodes:
    """Encapsulates all LangGraph node handlers and routing logic."""

    classifier: IntentClassifierAgent
    planner: PlannerAgent
    worker_map: Dict[str, BaseAgent]
    max_iterations: int

    def classify(self, state: AgentState) -> AgentState:
        return self.classifier.run(state)

    def unsupported(self, state: AgentState) -> AgentState:
        intent = state.get("intent", Intent.UNSUPPORTED)
        message = self.classifier.get_response(intent) or "Unsupported request."
        state["final_response"] = message
        state["final_answer"] = message
        return state

    def plan(self, state: AgentState) -> AgentState:
        return self.planner.think(state)

    def reflect(self, state: AgentState) -> AgentState:
        return self.planner.reflect(state)

    def optimize(self, state: AgentState) -> AgentState:
        return self.planner.optimize(state)

    def act(self, state: AgentState) -> AgentState:
        self._ensure_results(state)
        task_id = self._pick_next_task(state)
        if not task_id:
            return state

        tasks = state.get("tasks", {})
        task = tasks.get(task_id)
        if not task:
            state["current_task_id"] = None
            return state

        task_type = task.get("type")
        worker = self.worker_map.get(task_type or "")
        if not worker:
            task["status"] = "error"
            state["results"][task_id] = {"error": f"Unknown worker: {task_type}"}
            state["current_task_id"] = None
            return state

        try:
            state["current_task_id"] = task_id
            task["status"] = "running"
            state = worker.run(state)
            
            # Re-fetch task from updated state to avoid stale references if worker replaced the state
            updated_tasks = state.get("tasks", {})
            updated_task = updated_tasks.get(task_id, task)
            
            task_result = state.get("results", {}).get(task_id, {})
            updated_task["status"] = "error" if task_result.get("error") else "done"
        except Exception as exc:  # pragma: no cover - safety net
            # Re-fetch task again for error handling
            err_tasks = state.get("tasks", {})
            err_task = err_tasks.get(task_id, task)
            err_task["status"] = "error"
            state.setdefault("results", {})[task_id] = {"error": str(exc)}
            state["error"] = str(exc)
        finally:
            state["current_task_id"] = None

        return state

    def finalize(self, state: AgentState) -> AgentState:
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

    def route_after_classify(self, state: AgentState) -> str:
        intent = state.get("intent", Intent.SAR_ANALYSIS)
        return "plan" if self.classifier.is_supported(intent) else "unsupported"

    def route_after_plan(self, state: AgentState) -> str:
        return "act" if self._has_pending_tasks(state) else "reflect"

    def route_after_act(self, state: AgentState) -> str:
        return "act" if self._has_pending_tasks(state) else "reflect"

    def route_after_reflect(self, state: AgentState) -> str:
        reflection = state.get("reflection", {})
        if reflection.get("success"):
            return "finalize"
        if reflection.get("should_replan") and state.get("iteration", 0) < self.max_iterations:
            return "optimize"
        return "finalize"

    def route_after_optimize(self, state: AgentState) -> str:
        return "act" if self._has_pending_tasks(state) else "finalize"

    @staticmethod
    def _ensure_results(state: AgentState) -> None:
        state.setdefault("results", {})

    @staticmethod
    def _has_pending_tasks(state: AgentState) -> bool:
        tasks = state.get("tasks", {})
        return any(task.get("status") in (None, "pending", "running") for task in tasks.values())

    def _pick_next_task(self, state: AgentState) -> Optional[str]:
        task_id = state.get("current_task_id")
        if task_id:
            return task_id
        pick_fn = getattr(self.planner, "_pick_next_task", None)
        if pick_fn is None:
            return None
        return pick_fn(state)


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

    nodes = SarGraphNodes(
        classifier=classifier,
        planner=plan_agent,
        worker_map=worker_map,
        max_iterations=max_iterations,
    )

    graph = StateGraph(AgentState)
    graph.add_node("classify", nodes.classify)
    graph.add_node("plan", nodes.plan)
    graph.add_node("act", nodes.act)
    graph.add_node("reflect", nodes.reflect)
    graph.add_node("optimize", nodes.optimize)
    graph.add_node("finalize", nodes.finalize)
    graph.add_node("unsupported", nodes.unsupported)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", nodes.route_after_classify, {"plan": "plan", "unsupported": "unsupported"}
    )
    graph.add_conditional_edges(
        "plan", nodes.route_after_plan, {"act": "act", "reflect": "reflect"}
    )
    graph.add_conditional_edges(
        "act", nodes.route_after_act, {"act": "act", "reflect": "reflect"}
    )
    graph.add_conditional_edges(
        "reflect", nodes.route_after_reflect, {"finalize": "finalize", "optimize": "optimize"}
    )
    graph.add_conditional_edges(
        "optimize", nodes.route_after_optimize, {"act": "act", "finalize": "finalize"}
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


__all__ = [
    "MAX_ITERATIONS",
    "SarGraphNodes",
    "build_sar_graph",
    "get_sar_graph",
    "reset_sar_graph",
]
