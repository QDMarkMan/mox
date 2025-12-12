"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-12].
*  @Description Data cleaner agent with Python sandbox for data extraction.
**************************************************************************
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.llm import get_llm
from molx_agent.agents.modules.state import AgentState

logger = logging.getLogger(__name__)

DATA_CLEANER_SYSTEM_PROMPT = """You are a Data Cleaner agent specialized in extracting and cleaning molecular data.

You have access to a Python sandbox to execute code for data processing.

When given a file path or data extraction task:
1. Use the python_repl tool to read and process files
2. Extract SMILES, activities, and relevant molecular data
3. Clean and validate the extracted data
4. Return structured results

Common operations:
- Read CSV/Excel: pd.read_csv(path) or pd.read_excel(path)
- Extract SMILES column: df['SMILES'] or df['smiles']
- Extract activity data: df['activity'] or df['IC50']

Always return your final results as a JSON string with:
- compounds: list of SMILES strings
- activities: list of (name, value) tuples
- metadata: any additional info
"""


def _get_python_repl_tool():
    """Create Python REPL tool for code execution."""

    @tool
    def python_repl(code: str) -> str:
        """Execute Python code in a sandboxed environment.

        Use this to read files, process data, and extract molecular information.
        Common imports available: pandas (pd), numpy (np), json, os.

        Args:
            code: Python code to execute.

        Returns:
            Output of code execution or error message.
        """
        import io
        import sys

        # Pre-import common libraries
        exec_globals = {
            "__builtins__": __builtins__,
        }

        # Try to import common data science libraries
        try:
            import pandas as pd

            exec_globals["pd"] = pd
        except ImportError:
            pass

        try:
            import numpy as np

            exec_globals["np"] = np
        except ImportError:
            pass

        try:
            import json

            exec_globals["json"] = json
        except ImportError:
            pass

        try:
            import os

            exec_globals["os"] = os
        except ImportError:
            pass

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            exec(code, exec_globals)
            output = buffer.getvalue()
            return output if output else "Code executed successfully (no output)"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout

    return python_repl


class DataCleanerAgent(BaseAgent):
    """Data cleaner agent with Python sandbox for data extraction."""

    def __init__(self) -> None:
        super().__init__(
            name="data_cleaner",
            description="Cleans and extracts data using Python code execution",
        )
        self._agent = None

    def _get_agent(self) -> Any:
        """Get or create the ReAct agent with Python REPL."""
        if self._agent is None:
            llm = get_llm()
            tools = [_get_python_repl_tool()]
            self._agent = create_react_agent(
                llm,
                tools,
                prompt=DATA_CLEANER_SYSTEM_PROMPT,
            )
        return self._agent

    def run(self, state: AgentState) -> AgentState:
        """Execute data cleaning task using Python sandbox.

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

        agent = self._get_agent()

        try:
            # Build query from task
            query = task.get("description", "")
            inputs = task.get("inputs", {})
            if inputs:
                query += f"\n\nInputs: {json.dumps(inputs)}"

            query += "\n\nExtract and clean the data, then return JSON results."

            # Run the agent
            messages = [HumanMessage(content=query)]
            result = agent.invoke({"messages": messages})

            # Extract final response
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                output = final_message.content
            else:
                output = str(final_message)

            # Try to parse as JSON
            try:
                parsed = json.loads(output)
                state["results"][tid] = parsed
            except json.JSONDecodeError:
                state["results"][tid] = {"output": output, "raw": True}

            state["tasks"][tid]["status"] = "done"
            console.print(f"[green]✓ DataCleaner: Completed task {tid}[/]")
            logger.info(f"DataCleaner completed task {tid}")

        except Exception as e:
            console.print(f"[red]✗ DataCleaner error: {e}[/]")
            logger.error(f"DataCleaner error: {e}")
            state["results"][tid] = {"error": str(e)}
            state["tasks"][tid]["status"] = "done"

        return state
