"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-12].
*  @Description Planner agent, plan the analysis process.
**************************************************************************
"""

import logging
from typing import Optional

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.llm import invoke_llm
from molx_agent.agents.modules.state import AgentState, Task

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """
You are the Planner of a multi-agent SAR (Structure-Activity Relationship)
analysis system.

Your role is to:
1. Understand the user's drug design query
2. Decompose it into a DAG (Directed Acyclic Graph) of subtasks
3. Assign each subtask to the appropriate worker type

Available worker types:
- "tool": Use chemistry tools (SMILES conversion, MW, similarity, safety)
- "data_cleaner": Data file preprocessing and cleaning
- "reporter": Report generation and summarization

For each task, specify:
- id: Unique task identifier (e.g., "task_1", "analyze_mol")
- type: One of "tool", "data_cleaner", "reporter"
- description: What this task should accomplish
- inputs: Required input data
- expected_outputs: List of expected output keys
- depends_on: List of task IDs that must complete before this task

Return ONLY a valid JSON object with this structure:
{
  "tasks": [
    {
      "id": "task_id",
      "type": "tool|data_cleaner|reporter",
      "description": "Task description",
      "inputs": {},
      "expected_outputs": ["output1"],
      "depends_on": []
    }
  ]
}

Keep the task graph simple. For MVP, prefer 2-3 tasks maximum.
"""


class PlannerAgent(BaseAgent):
    """Planner agent that decomposes user queries into task DAGs."""

    def __init__(self) -> None:
        super().__init__(
            name="planner",
            description="Plans and decomposes user queries into executable tasks",
        )

    def run(self, state: AgentState) -> AgentState:
        """Plan tasks based on user query.

        Args:
            state: Current agent state with user_query.

        Returns:
            Updated state with tasks DAG.
        """
        from rich.console import Console

        console = Console()
        user_query = state.get("user_query", "")

        console.print("[cyan]🔍 Planner: Analyzing query...[/]")

        try:
            user_message = (
                f"User query:\n{user_query}\n\nReturn only the JSON DAG of tasks."
            )
            dag = invoke_llm(PLANNER_SYSTEM_PROMPT, user_message, parse_json=True)

            console.print("[green]✓ Planner: Received response[/]")

            tasks: dict[str, Task] = {}
            for t in dag.get("tasks", []):
                t["status"] = "pending"
                tasks[t["id"]] = t
                console.print(f"  • Task: [bold]{t['id']}[/] ({t['type']})")

            state["tasks"] = tasks
            state["results"] = {}
            state["current_task_id"] = self._pick_next_task(state)

            console.print(f"[green]✓ Planner: Created {len(tasks)} tasks[/]")
            logger.info(f"Planner created {len(tasks)} tasks")

        except Exception as e:
            console.print(f"[red]✗ Planner error: {e}[/]")
            logger.error(f"Planner error: {e}")
            state["error"] = f"Planner error: {e}"
            state["tasks"] = {}
            state["current_task_id"] = None

        return state

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
