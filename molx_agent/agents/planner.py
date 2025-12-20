"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description PlannerAgent with ReAct pattern support.
*               Implements: Think → Act → Reflect → Optimize
**************************************************************************
"""

import json
import logging
from typing import Optional, Any

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.llm import invoke_llm
from molx_agent.agents.modules.state import AgentState, Task

logger = logging.getLogger(__name__)


# =============================================================================
# System Prompts
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are the Planner of a multi-agent SAR (Structure-Activity Relationship) analysis system.

Your role is to analyze user queries and create a plan of tasks (DAG).

## Available Workers (ONLY THESE THREE TYPES ARE ALLOWED):
- "data_cleaner": Extract and clean molecular data from files (CSV, Excel, SDF)
- "sar": SAR analysis (R-group decomposition, scaffold selection)  
- "reporter": Generate HTML reports

IMPORTANT CONSTRAINTS:
- You MUST ONLY use these exact worker types: "data_cleaner", "sar", "reporter"
- Do NOT invent new worker types like "dependency_checker", "dependency_installer", "validator", etc.
- All data extraction, cleaning, and validation should be handled by "data_cleaner"
- All molecular analysis should be handled by "sar"
- All report generation should be handled by "reporter"

## Report Intent Detection:
The reporter supports different analysis modes. Detect user intent and pass it to reporter:

1. **Single-Site Analysis** - User wants to focus on one R-group position:
   - Keywords: "R1位点", "只看R2", "单一位点", "R1的SAR", "position R1"
   - Pass: `"inputs": {"report_intent": "single_site", "target_position": "R1"}`

2. **Molecule Subset Analysis** - User wants to analyze specific molecules:
   - Keywords: "只分析这几个", "Cpd-1, Cpd-2", "比较 X 和 Y"
   - Pass: `"inputs": {"report_intent": "molecule_subset", "target_molecules": ["Cpd-1", "Cpd-2"]}`

3. **Full Report (default)** - Standard full SAR analysis:
   - No special keywords, or explicit "完整SAR报告", "full report"
   - No special inputs needed

## Task Format:
Return a JSON object with this structure:
{
  "reasoning": "Your step-by-step thinking about how to approach this query",
  "tasks": [
    {
      "id": "task_id",
      "type": "data_cleaner|sar|reporter",
      "description": "What this task should accomplish",
      "inputs": {},
      "expected_outputs": ["output_key"],
      "depends_on": []
    }
  ]
}

## Guidelines:
- Keep plans simple (2-4 tasks maximum)
- Typical SAR flow: data_cleaner → sar → reporter
- Each task should have clear inputs and expected outputs
- Dependencies define execution order
- For single-site or subset analysis, include report_intent in reporter task inputs
"""

REFLECT_SYSTEM_PROMPT = """You are evaluating the results of executed tasks.

Given the original query, the planned tasks, and their results, assess:
1. Did all tasks complete successfully?
2. Were the expected outputs produced?
3. Is the quality of results satisfactory?
4. Should any tasks be retried or replanned?

Return a JSON object:
{
  "success": true/false,
  "summary": "Brief summary of what was accomplished",
  "issues": ["list of any issues found"],
  "should_replan": true/false,
  "replan_reason": "If should_replan is true, explain why"
}
"""

OPTIMIZE_SYSTEM_PROMPT = """You are optimizing a failed or incomplete plan.

Given the original query, previous plan, and the issues encountered, create an improved plan.

IMPORTANT: You can ONLY use these three worker types:
- "data_cleaner": Extract and clean molecular data from files
- "sar": SAR analysis (R-group decomposition, scaffold selection)
- "reporter": Generate HTML reports

Do NOT create tasks with any other type (e.g., "dependency_checker", "validator", etc.).
If you see errors about "Unknown worker type", remove those invalid tasks and use only valid types.

Consider:
- What went wrong in the previous attempt?
- How can the tasks be restructured using ONLY the three valid worker types?
- Are there alternative approaches within the available workers?

Return a JSON object with the same format as the original plan:
{
  "reasoning": "Your revised thinking",
  "tasks": [...]
}
"""


class PlannerAgent(BaseAgent):
    """Planner agent implementing ReAct pattern.
    
    Phases:
    - THINK: Analyze query and create task plan
    - ACT: (Handled by MolxAgent dispatching to workers)
    - REFLECT: Evaluate results and check for issues
    - OPTIMIZE: Replan if needed
    """

    MAX_ITERATIONS = 3  # Prevent infinite loops

    def __init__(self) -> None:
        super().__init__(
            name="planner",
            description="Plans and orchestrates SAR analysis using ReAct pattern",
        )

    def think(self, state: AgentState) -> AgentState:
        """THINK phase: Analyze query and create task plan.
        
        Args:
            state: Current agent state with user_query.
            
        Returns:
            Updated state with tasks DAG.
        """
        from rich.console import Console
        console = Console()
        
        user_query = state.get("user_query", "")
        console.print("\n[bold cyan]🧠 THINK: Analyzing query and creating plan...[/]")

        try:
            user_message = f"User query:\n{user_query}\n\nCreate a task plan."
            result = invoke_llm(PLANNER_SYSTEM_PROMPT, user_message, parse_json=True)

            # Show reasoning
            reasoning = result.get("reasoning", "")
            if reasoning:
                console.print(f"[dim]   Reasoning: {reasoning[:200]}...[/]" if len(reasoning) > 200 else f"[dim]   Reasoning: {reasoning}[/]")

            # Parse tasks
            tasks: dict[str, Task] = {}
            for t in result.get("tasks", []):
                t["status"] = "pending"
                tasks[t["id"]] = t
                console.print(f"   • Task: [bold]{t['id']}[/] ({t['type']}) - {t['description'][:50]}...")

            state["tasks"] = tasks
            state["results"] = state.get("results", {})
            state["current_task_id"] = self._pick_next_task(state)
            state["iteration"] = state.get("iteration", 0) + 1

            console.print(f"[green]✓ THINK: Created {len(tasks)} tasks (iteration {state['iteration']})[/]")
            logger.info(f"Planner created {len(tasks)} tasks")

        except Exception as e:
            console.print(f"[red]✗ THINK error: {e}[/]")
            logger.error(f"Planner THINK error: {e}")
            state["error"] = f"Planner error: {e}"
            state["tasks"] = {}
            state["current_task_id"] = None

        return state

    def reflect(self, state: AgentState) -> AgentState:
        """REFLECT phase: Evaluate execution results.
        
        Args:
            state: State with executed task results.
            
        Returns:
            Updated state with reflection.
        """
        from rich.console import Console
        console = Console()
        
        console.print("\n[bold yellow]🔍 REFLECT: Evaluating results...[/]")

        try:
            tasks = state.get("tasks", {})
            results = state.get("results", {})
            
            # First, check task completion status directly (don't rely on LLM)
            all_done = all(t.get("status") == "done" for t in tasks.values())
            has_errors = any(
                isinstance(results.get(tid), dict) and "error" in results.get(tid, {})
                for tid in tasks.keys()
            )
            
            # Check if report was generated (key success indicator)
            report_generated = False
            for tid, result in results.items():
                if isinstance(result, dict):
                    if result.get("report_path") or result.get("output_files", {}).get("html"):
                        report_generated = True
                        break
            
            # If all tasks done and report generated, mark as success without LLM
            if all_done and report_generated and not has_errors:
                summary = f"All {len(tasks)} tasks completed successfully. Report generated."
                state["reflection"] = {
                    "success": True,
                    "summary": summary,
                    "issues": [],
                    "should_replan": False,
                    "replan_reason": ""
                }
                console.print(f"[green]✓ REFLECT: Success - {summary}[/]")
                return state
            
            # If there are actual errors, report them but don't replan
            if has_errors:
                error_tasks = [
                    tid for tid, r in results.items() 
                    if isinstance(r, dict) and "error" in r
                ]
                state["reflection"] = {
                    "success": False,
                    "summary": f"Tasks completed with errors in: {error_tasks}",
                    "issues": [f"Error in {tid}: {results[tid].get('error', 'unknown')}" for tid in error_tasks],
                    "should_replan": False,  # Don't replan on errors, just report
                    "replan_reason": ""
                }
                console.print(f"[yellow]⚠ REFLECT: Completed with errors in {error_tasks}[/]")
                return state
            
            # If all tasks done but no report, that's still success (maybe no reporter task)
            if all_done:
                summary = f"All {len(tasks)} tasks completed."
                state["reflection"] = {
                    "success": True,
                    "summary": summary,
                    "issues": [],
                    "should_replan": False,
                    "replan_reason": ""
                }
                console.print(f"[green]✓ REFLECT: Success - {summary}[/]")
                return state
            
            # Only use LLM if tasks are not all done (unusual case)
            user_query = state.get("user_query", "")
            context = f"""
Original Query: {user_query}

Planned Tasks:
{json.dumps(list(tasks.values()), indent=2, default=str)[:2000]}

Task Results:
{json.dumps(results, indent=2, default=str)[:2000]}
"""
            
            result = invoke_llm(REFLECT_SYSTEM_PROMPT, context, parse_json=True)
            
            success = result.get("success", False)
            summary = result.get("summary", "")
            issues = result.get("issues", [])
            should_replan = result.get("should_replan", False)
            
            # Store reflection
            state["reflection"] = {
                "success": success,
                "summary": summary,
                "issues": issues,
                "should_replan": should_replan,
                "replan_reason": result.get("replan_reason", "")
            }
            
            if success:
                console.print(f"[green]✓ REFLECT: Success - {summary}[/]")
            else:
                console.print(f"[yellow]⚠ REFLECT: Issues found - {', '.join(issues)}[/]")
                
            if should_replan:
                console.print(f"[yellow]   → Will attempt to optimize plan[/]")

        except Exception as e:
            console.print(f"[red]✗ REFLECT error: {e}[/]")
            logger.error(f"Planner REFLECT error: {e}")
            state["reflection"] = {"success": False, "issues": [str(e)], "should_replan": False}

        return state

    def optimize(self, state: AgentState) -> AgentState:
        """OPTIMIZE phase: Replan if needed.
        
        Args:
            state: State with reflection indicating issues.
            
        Returns:
            Updated state with new plan or same state if optimization not needed.
        """
        from rich.console import Console
        console = Console()
        
        reflection = state.get("reflection", {})
        if not reflection.get("should_replan", False):
            return state
            
        iteration = state.get("iteration", 0)
        if iteration >= self.MAX_ITERATIONS:
            console.print(f"[red]✗ OPTIMIZE: Max iterations ({self.MAX_ITERATIONS}) reached, stopping[/]")
            return state
            
        console.print("\n[bold magenta]🔧 OPTIMIZE: Replanning...[/]")

        try:
            user_query = state.get("user_query", "")
            tasks = state.get("tasks", {})
            issues = reflection.get("issues", [])
            reason = reflection.get("replan_reason", "")
            
            context = f"""
Original Query: {user_query}

Previous Plan:
{json.dumps(list(tasks.values()), indent=2, default=str)[:1500]}

Issues Encountered:
{json.dumps(issues, indent=2)}

Reason for Replanning: {reason}

Create an improved plan that addresses these issues.
"""
            
            result = invoke_llm(OPTIMIZE_SYSTEM_PROMPT, context, parse_json=True)
            
            # Parse new tasks
            new_tasks: dict[str, Task] = {}
            for t in result.get("tasks", []):
                t["status"] = "pending"
                new_tasks[t["id"]] = t
                
            if new_tasks:
                state["tasks"] = new_tasks
                state["current_task_id"] = self._pick_next_task(state)
                console.print(f"[green]✓ OPTIMIZE: Created new plan with {len(new_tasks)} tasks[/]")
            else:
                console.print("[yellow]⚠ OPTIMIZE: No new tasks generated[/]")

        except Exception as e:
            console.print(f"[red]✗ OPTIMIZE error: {e}[/]")
            logger.error(f"Planner OPTIMIZE error: {e}")

        return state

    def should_continue(self, state: AgentState) -> bool:
        """Check if ReAct loop should continue.
        
        Returns True if:
        - There are pending tasks, OR
        - Reflection says should_replan AND iteration < MAX
        """
        # Check for pending tasks
        tasks = state.get("tasks", {})
        has_pending = any(t.get("status") == "pending" for t in tasks.values())
        if has_pending:
            return True
            
        # Check if should replan
        reflection = state.get("reflection", {})
        should_replan = reflection.get("should_replan", False)
        iteration = state.get("iteration", 0)
        
        return should_replan and iteration < self.MAX_ITERATIONS

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

    def run(self, state: AgentState) -> AgentState:
        """Run the planner (THINK phase only for backward compatibility)."""
        return self.think(state)
