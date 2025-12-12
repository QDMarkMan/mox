"""SAR Agent Graph construction and execution."""

from typing import Any

from langgraph.graph import END, StateGraph

from .nodes import (
    bio_worker_node,
    chemo_worker_node,
    literature_worker_node,
    planner_node,
    reviewer_node,
    route_after_planner,
    route_after_worker,
)
from .state import AgentState


def build_sar_graph() -> StateGraph:
    """Build and compile the SAR agent graph."""
    sg = StateGraph(AgentState)

    # Add nodes
    sg.add_node("planner", planner_node)
    sg.add_node("literature_worker", literature_worker_node)
    sg.add_node("chemo_worker", chemo_worker_node)
    sg.add_node("bio_worker", bio_worker_node)
    sg.add_node("reviewer", reviewer_node)

    # Set entry point
    sg.set_entry_point("planner")

    # Add conditional edges from planner
    sg.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "literature_worker": "literature_worker",
            "chemo_worker": "chemo_worker",
            "bio_worker": "bio_worker",
            "reviewer": "reviewer",
        },
    )

    # Add conditional edges from each worker
    for worker in ["literature_worker", "chemo_worker", "bio_worker"]:
        sg.add_conditional_edges(
            worker,
            route_after_worker,
            {
                "literature_worker": "literature_worker",
                "chemo_worker": "chemo_worker",
                "bio_worker": "bio_worker",
                "reviewer": "reviewer",
            },
        )

    # Reviewer leads to END
    sg.add_edge("reviewer", END)

    return sg.compile()


# Global graph instance
_sar_graph = None


def get_sar_graph() -> Any:
    """Get or create the SAR graph instance."""
    global _sar_graph
    if _sar_graph is None:
        _sar_graph = build_sar_graph()
    return _sar_graph


def run_sar_agent(user_query: str) -> tuple[str, dict]:
    """
    Run the SAR agent with a user query.

    Args:
        user_query: The SAR analysis query from the user.

    Returns:
        A tuple of (text_report, structured_results).
    """
    graph = get_sar_graph()
    initial_state: AgentState = {"user_query": user_query}

    final_state = graph.invoke(initial_state)

    text_report = final_state.get("final_answer", "No report generated.")
    structured = final_state.get("results", {}).get("final_structured", {})

    return text_report, structured
