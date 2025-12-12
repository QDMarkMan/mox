"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-12].
*  @Description Reporter agent, generate the report for analysis.
**************************************************************************
"""

import json
import logging

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.llm import invoke_llm
from molx_agent.agents.modules.state import AgentState

logger = logging.getLogger(__name__)

REPORTER_PROMPT = """
You are the Reporter agent in a SAR analysis system.

Your role is to:
1. Synthesize all task results into a coherent report
2. Validate consistency across analyses
3. Highlight key findings and recommendations

Given the user query and all task results, produce both:
1. A human-readable text report in Markdown format
2. A machine-readable structured summary

Return a JSON object with:
{
  "text_report": "# SAR Analysis Report\\n\\n## Summary\\n...",
  "structured": {
    "key_findings": ["Finding 1", "Finding 2"],
    "recommendations": ["Rec 1"],
    "next_steps": ["Step 1"]
  }
}

Write clear, actionable reports.
"""


class ReporterAgent(BaseAgent):
    """Reporter agent that generates final analysis reports."""

    def __init__(self) -> None:
        super().__init__(
            name="reporter",
            description="Generates final analysis reports from task results",
        )

    def run(self, state: AgentState) -> AgentState:
        """Generate final report from all task results.

        Args:
            state: Current agent state with all task results.

        Returns:
            Updated state with final report.
        """
        from rich.console import Console

        console = Console()

        user_query = state.get("user_query", "")
        tasks = state.get("tasks", {})
        results = state.get("results", {})

        context = {"tasks": tasks, "results": results}

        console.print("[cyan]📝 Reporter: Generating report...[/]")

        try:
            user_message = (
                f"User query:\n{user_query}\n\n"
                f"All tasks and results:\n{json.dumps(context, indent=2)}\n\n"
                "Return JSON with keys 'text_report' and 'structured'."
            )
            out = invoke_llm(REPORTER_PROMPT, user_message, parse_json=True)

            state["final_answer"] = out.get("text_report", "")
            state["results"]["final_structured"] = out.get("structured", {})
            console.print("[green]✓ Reporter: Report generated[/]")
            logger.info("Reporter completed final report")

        except Exception as e:
            console.print(f"[red]✗ Reporter error: {e}[/]")
            logger.error(f"Reporter error: {e}")
            state["final_answer"] = f"Error generating report: {e}"
            state["error"] = str(e)

        return state
