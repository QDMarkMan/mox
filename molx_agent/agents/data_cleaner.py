"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-12].
*  @Description Data cleaner agent, clean and preprocess data for analysis.
**************************************************************************
"""

import logging

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.llm import invoke_llm
from molx_agent.agents.modules.state import AgentState

logger = logging.getLogger(__name__)

DATA_CLEANER_PROMPT = """
You are the Data Cleaner agent in a SAR analysis system.

Your responsibilities:
1. Clean and preprocess input data
2. Extract relevant information for analysis
3. Validate and normalize data formats
4. Extract SMILES and activity data from a excel/csv file path

Given a task description and inputs, return a JSON object with:
{
  "cleaned_data": {
    "compounds": [...],
    "activities": [...],
    "metadata": {}
  },
  "validation_notes": ["Note 1", "Note 2"],
  "summary": "Brief summary of cleaning operations"
}

Be thorough but concise in your cleaning operations.
"""


class DataCleanerAgent(BaseAgent):
    """Data cleaner agent that preprocesses data for analysis."""

    def __init__(self) -> None:
        super().__init__(
            name="data_cleaner",
            description="Cleans and preprocesses data for SAR analysis",
        )

    def run(self, state: AgentState) -> AgentState:
        """Execute data cleaning task.

        Args:
            state: Current agent state with task info.

        Returns:
            Updated state with cleaned data results.
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

        console.print(f"[cyan]🧹 DataCleaner: Processing task {tid}...[/]")

        try:
            user_message = (
                f"Task:\n{json.dumps(task, indent=2)}\n\n"
                "Return JSON with keys: cleaned_data, validation_notes, summary."
            )
            result = invoke_llm(DATA_CLEANER_PROMPT, user_message, parse_json=True)

            state["results"][tid] = result
            state["tasks"][tid]["status"] = "done"
            console.print(f"[green]✓ DataCleaner: Completed task {tid}[/]")
            logger.info(f"DataCleaner completed task {tid}")

        except Exception as e:
            console.print(f"[red]✗ DataCleaner error: {e}[/]")
            logger.error(f"DataCleaner error: {e}")
            state["results"][tid] = {"error": str(e)}
            state["tasks"][tid]["status"] = "done"

        return state
