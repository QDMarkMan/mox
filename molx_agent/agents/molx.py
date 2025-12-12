"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-12].
*  @Description Molx agent, the main agent orchestrator for molx-agent.
**************************************************************************
"""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.data_cleaner import DataCleanerAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.planner import PlannerAgent
from molx_agent.agents.reporter import ReporterAgent

logger = logging.getLogger(__name__)


class MolxAgent(BaseAgent):
    """Main orchestrator agent that coordinates all sub-agents."""

    def __init__(self) -> None:
        super().__init__(
            name="molx",
            description="Main orchestrator for SAR analysis workflow",
        )
        self.planner = PlannerAgent()
        self.data_cleaner = DataCleanerAgent()
        self.reporter = ReporterAgent()
        self._graph = None

    def _pick_next_task(self, state: AgentState) -> Optional[str]:
        """Pick the next executable task."""
        tasks = state.get("tasks", {})
        for tid, task in tasks.items():
            if task.get("status") != "pending":
                continue
            depends_on = task.get("depends_on", [])
            if all(tasks.get(dep, {}).get("status") == "done" for dep in depends_on):
                return tid
        return None

    def _route_after_planner(self, state: AgentState) -> str:
        """Route to appropriate agent after planner."""
        current_task_id = state.get("current_task_id")

        if current_task_id is None:
            return "reporter"

        task = state.get("tasks", {}).get(current_task_id)
        if not task:
            return "reporter"

        task_type = task.get("type", "")

        if task_type == "data_cleaner":
            return "data_cleaner"
        elif task_type == "reporter":
            return "reporter"
        else:
            # Unknown task type: mark as done and pick next
            state["tasks"][current_task_id]["status"] = "done"
            state["current_task_id"] = self._pick_next_task(state)
            return self._route_after_planner(state)

    def _route_after_worker(self, state: AgentState) -> str:
        """Route after worker completion."""
        state["current_task_id"] = self._pick_next_task(state)
        return self._route_after_planner(state)

    def _data_cleaner_node(self, state: AgentState) -> AgentState:
        """Execute data cleaner and update task scheduling."""
        state = self.data_cleaner.run(state)
        state["current_task_id"] = self._pick_next_task(state)
        return state

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        sg = StateGraph(AgentState)

        # Add nodes
        sg.add_node("planner", self.planner.run)
        sg.add_node("data_cleaner", self._data_cleaner_node)
        sg.add_node("reporter", self.reporter.run)

        # Set entry point
        sg.set_entry_point("planner")

        # Add conditional edges from planner
        sg.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "data_cleaner": "data_cleaner",
                "reporter": "reporter",
            },
        )

        # Add conditional edges from data_cleaner
        sg.add_conditional_edges(
            "data_cleaner",
            self._route_after_worker,
            {
                "data_cleaner": "data_cleaner",
                "reporter": "reporter",
            },
        )

        # Reporter leads to END
        sg.add_edge("reporter", END)

        return sg.compile()

    def get_graph(self) -> StateGraph:
        """Get or create the graph instance."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def run(self, state: AgentState) -> AgentState:
        """Execute the full workflow.

        Args:
            state: Initial state with user_query.

        Returns:
            Final state with results.
        """
        graph = self.get_graph()
        return graph.invoke(state)

    def analyze(self, query: str) -> tuple[str, dict]:
        """Run SAR analysis with a user query.

        Args:
            query: The SAR analysis query from the user.

        Returns:
            A tuple of (text_report, structured_results).
        """
        initial_state: AgentState = {"user_query": query}
        final_state = self.run(initial_state)

        text_report = final_state.get("final_answer", "No report generated.")
        structured = final_state.get("results", {}).get("final_structured", {})

        return text_report, structured


# Convenience function for backward compatibility
def run_sar_agent(user_query: str) -> tuple[str, dict]:
    """Run the SAR agent with a user query.

    Args:
        user_query: The SAR analysis query from the user.

    Returns:
        A tuple of (text_report, structured_results).
    """
    agent = MolxAgent()
    return agent.analyze(user_query)
