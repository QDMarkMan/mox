"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-12].
*  @Description Tool agent using LangGraph prebuilt ReAct agent.
**************************************************************************
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.llm import get_llm
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.modules.tools import get_registry

logger = logging.getLogger(__name__)


class ToolAgent(BaseAgent):
    """Agent that uses tools to answer chemistry-related questions."""

    def __init__(self) -> None:
        super().__init__(
            name="tool_agent",
            description="Uses different tools to analyze molecules",
        )
        self._agent = None
        self._tools = None

    def _get_agent(self) -> Any:
        """Get or create the ReAct agent."""
        if self._agent is None:
            llm = get_llm()
            registry = get_registry()
            self._tools = registry.get_tools(agent=self.name)
            if self._tools:
                self._agent = create_react_agent(llm, self._tools)
        return self._agent

    def run(self, state: AgentState) -> AgentState:
        """Execute tool-based task.

        Args:
            state: Current agent state with task info.

        Returns:
            Updated state with tool results.
        """
        import json

        from rich.console import Console

        console = Console()

        tid = state.get("current_task_id")
        if not tid:
            return state

        task = state.get("tasks", {}).get(tid)
        if not task:
            return state

        console.print(f"[cyan]🔧 ToolAgent: Processing task {tid}...[/]")

        agent = self._get_agent()
        if agent is None:
            console.print("[yellow]⚠ ToolAgent: No tools available[/]")
            state["results"][tid] = {"error": "No tools available"}
            state["tasks"][tid]["status"] = "done"
            return state

        try:
            # Build query from task description and inputs
            query = task.get("description", "")
            inputs = task.get("inputs", {})
            if inputs:
                query += f"\n\nInputs: {json.dumps(inputs)}"

            # Run the ReAct agent
            messages = [HumanMessage(content=query)]
            result = agent.invoke({"messages": messages})

            # Extract final response
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                output = final_message.content
            else:
                output = str(final_message)

            state["results"][tid] = {
                "output": output,
                "tool_calls": len(result.get("messages", [])) - 1,
            }
            state["tasks"][tid]["status"] = "done"

            console.print(f"[green]✓ ToolAgent: Completed task {tid}[/]")
            logger.info(f"ToolAgent completed task {tid}")

        except Exception as e:
            console.print(f"[red]✗ ToolAgent error: {e}[/]")
            logger.error(f"ToolAgent error: {e}")
            state["results"][tid] = {"error": str(e)}
            state["tasks"][tid]["status"] = "done"

        return state
