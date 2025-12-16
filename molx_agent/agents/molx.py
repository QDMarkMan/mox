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
from molx_agent.agents.modules.state import AgentState, Message
from molx_agent.agents.planner import PlannerAgent
from molx_agent.agents.reporter import ReporterAgent
from molx_agent.agents.sar import SARAgent
from molx_agent.agents.tool_agent import ToolAgent

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
        self.tool_agent = ToolAgent()
        self.sar_agent = SARAgent()
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

        if task_type == "tool":
            return "tool_agent"
        elif task_type == "data_cleaner":
            return "data_cleaner"
        elif task_type == "sar":
            return "sar_agent"
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

    def _tool_agent_node(self, state: AgentState) -> AgentState:
        """Execute tool agent and update task scheduling."""
        state = self.tool_agent.run(state)
        state["current_task_id"] = self._pick_next_task(state)
        return state

    def _sar_agent_node(self, state: AgentState) -> AgentState:
        """Execute SAR agent and update task scheduling."""
        state = self.sar_agent.run(state)
        state["current_task_id"] = self._pick_next_task(state)
        return state

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        sg = StateGraph(AgentState)

        # Add nodes
        sg.add_node("planner", self.planner.run)
        sg.add_node("tool_agent", self._tool_agent_node)
        sg.add_node("data_cleaner", self._data_cleaner_node)
        sg.add_node("sar_agent", self._sar_agent_node)
        sg.add_node("reporter", self.reporter.run)

        # Set entry point
        sg.set_entry_point("planner")

        # Route mapping for all workers
        worker_routes = {
            "tool_agent": "tool_agent",
            "data_cleaner": "data_cleaner",
            "sar_agent": "sar_agent",
            "reporter": "reporter",
        }

        # Add conditional edges from planner
        sg.add_conditional_edges("planner", self._route_after_planner, worker_routes)

        # Add conditional edges from each worker
        for worker in ["tool_agent", "data_cleaner", "sar_agent"]:
            sg.add_conditional_edges(worker, self._route_after_worker, worker_routes)

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

    def chat(self, message: str, history: list[Message] | None = None) -> tuple[str, list[Message]]:
        """Run a chat turn with conversation history.

        Args:
            message: User message.
            history: Optional previous conversation history.

        Returns:
            A tuple of (response, updated_history).
        """
        from rich.console import Console

        from molx_agent.agents.intent_classifier import (
            classify_intent,
            get_intent_response,
            is_supported_intent,
        )

        console = Console()

        # Initialize history if not provided
        if history is None:
            history = []

        # Add user message to history
        history.append({"role": "user", "content": message})

        # Classify user intent
        console.print("[cyan]🤔 Classifying intent...[/]")
        intent, confidence = classify_intent(message)
        console.print(f"[dim]   Intent: {intent.value} ({confidence:.0%})[/]")

        # Check if intent is supported
        if not is_supported_intent(intent):
            # Return friendly message for unsupported intents
            response = get_intent_response(intent)
            if response is None:
                response = "抱歉，我不太理解您的需求。请尝试询问分子分析相关的问题。"

            history.append({"role": "assistant", "content": response})
            console.print("[yellow]⚠ Non-SAR intent detected, providing guidance[/]")
            return response, history

        # Build context from history for the planner
        context = self._build_context(history)

        # Run the agent with context
        initial_state: AgentState = {
            "user_query": context,
            "messages": history,
        }
        final_state = self.run(initial_state)

        # Get the response
        response = final_state.get("final_answer", "I couldn't generate a response.")

        # Add assistant response to history
        history.append({"role": "assistant", "content": response})

        return response, history

    def _build_context(self, history: list[Message]) -> str:
        """Build context string from conversation history.

        Args:
            history: Conversation history.

        Returns:
            Context string for the planner.
        """
        if len(history) <= 1:
            # Only current message, no context needed
            return history[-1]["content"] if history else ""

        # Build context with recent history
        context_parts = ["Previous conversation:"]
        for msg in history[:-1]:  # All except current message
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_parts.append(f"- {role.upper()}: {content}")

        context_parts.append(f"\nCurrent query: {history[-1]['content']}")
        return "\n".join(context_parts)


class ChatSession:
    """Interactive chat session with conversation history."""

    def __init__(self, agent: MolxAgent | None = None) -> None:
        """Initialize chat session.

        Args:
            agent: Optional MolxAgent instance. Creates new one if not provided.
        """
        self.agent = agent or MolxAgent()
        self.history: list[Message] = []

    def send(self, message: str) -> str:
        """Send a message and get response.

        Args:
            message: User message.

        Returns:
            Agent response.
        """
        response, self.history = self.agent.chat(message, self.history)
        return response

    def get_history(self) -> list[Message]:
        """Get conversation history.

        Returns:
            List of messages.
        """
        return self.history.copy()

    def clear(self) -> None:
        """Clear conversation history."""
        self.history = []


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

