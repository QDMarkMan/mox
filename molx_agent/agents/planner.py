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
from dataclasses import dataclass
from typing import Optional, Any, Dict

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


@dataclass
class PlanResult:
    reasoning: str
    tasks: Dict[str, Task]


@dataclass
class ReflectionResult:
    success: bool
    summary: str
    issues: list[str]
    should_replan: bool
    replan_reason: str = ""


class PlannerAgent(BaseAgent):
    """Planner agent implementing ReAct pattern.

    Phases:
    - THINK: Analyze query and create task plan
    - ACT: (Handled by MolxAgent dispatching to workers)
    - REFLECT: Evaluate results and check for issues
    - OPTIMIZE: Replan if needed
    """

    MAX_ITERATIONS = 5  # Prevent infinite loops

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
            plan = self._invoke_planner_llm(user_query)
            state["plan_reasoning"] = plan.reasoning
            self._log_plan(console, plan)

            state["tasks"] = plan.tasks
            state["results"] = state.get("results", {})
            state["current_task_id"] = self._pick_next_task(state)
            state["iteration"] = state.get("iteration", 0) + 1

            console.print(
                f"[green]✓ THINK: Created {len(plan.tasks)} tasks (iteration {state['iteration']})[/]"
            )
            logger.info("Planner created %d tasks", len(plan.tasks))

        except Exception as exc:
            console.print(f"[red]✗ THINK error: {exc}[/]")
            logger.error("Planner THINK error", exc_info=exc)
            state["error"] = f"Planner error: {exc}"
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
            reflection = self._derive_reflection_from_state(state)
            if reflection is None:
                reflection = self._invoke_reflection_llm(state)

            self._apply_reflection(console, state, reflection)

        except Exception as exc:
            console.print(f"[red]✗ REFLECT error: {exc}[/]")
            logger.error("Planner REFLECT error", exc_info=exc)
            state["reflection"] = {"success": False, "issues": [str(exc)], "should_replan": False}

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
            improved_plan = self._invoke_optimize_llm(state, reflection)
            if improved_plan.tasks:
                state["tasks"] = improved_plan.tasks
                state["current_task_id"] = self._pick_next_task(state)
                console.print(
                    f"[green]✓ OPTIMIZE: Created new plan with {len(improved_plan.tasks)} tasks"
                )
            else:
                console.print("[yellow]⚠ OPTIMIZE: No new tasks generated[/]")

        except Exception as exc:
            console.print(f"[red]✗ OPTIMIZE error: {exc}[/]")
            logger.error("Planner OPTIMIZE error", exc_info=exc)

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invoke_planner_llm(self, user_query: str) -> PlanResult:
        user_message = f"User query:\n{user_query}\n\nCreate a task plan."
        payload = invoke_llm(PLANNER_SYSTEM_PROMPT, user_message, parse_json=True)
        tasks = self._extract_tasks(payload)
        reasoning = payload.get("reasoning", "")
        return PlanResult(reasoning=reasoning, tasks=tasks)

    def _invoke_reflection_llm(self, state: AgentState) -> ReflectionResult:
        tasks = state.get("tasks", {})
        results = state.get("results", {})
        user_query = state.get("user_query", "")
        context = f"""
Original Query: {user_query}

Planned Tasks:
{json.dumps(list(tasks.values()), indent=2, default=str)[:2000]}

Task Results:
{json.dumps(results, indent=2, default=str)[:2000]}
"""
        payload = invoke_llm(REFLECT_SYSTEM_PROMPT, context, parse_json=True)
        return ReflectionResult(
            success=payload.get("success", False),
            summary=payload.get("summary", ""),
            issues=list(payload.get("issues", [])),
            should_replan=payload.get("should_replan", False),
            replan_reason=payload.get("replan_reason", ""),
        )

    def _invoke_optimize_llm(
        self, state: AgentState, reflection: dict[str, Any]
    ) -> PlanResult:
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
        payload = invoke_llm(OPTIMIZE_SYSTEM_PROMPT, context, parse_json=True)
        return PlanResult(reasoning=payload.get("reasoning", ""), tasks=self._extract_tasks(payload))

    def _extract_tasks(self, payload: dict[str, Any]) -> dict[str, Task]:
        tasks: dict[str, Task] = {}
        for raw in payload.get("tasks", []):
            raw["status"] = "pending"
            tasks[raw["id"]] = raw
        return tasks

    def _derive_reflection_from_state(self, state: AgentState) -> Optional[ReflectionResult]:
        tasks = state.get("tasks", {})
        results = state.get("results", {})
        if not tasks:
            return None

        all_done = all(t.get("status") == "done" for t in tasks.values())
        has_errors = any(
            isinstance(results.get(tid), dict) and "error" in results.get(tid, {})
            for tid in tasks
        )

        report_generated = any(
            isinstance(result, dict)
            and (
                result.get("report_path")
                or result.get("output_files", {}).get("html")
            )
            for result in results.values()
        )

        if all_done and report_generated and not has_errors:
            summary = f"All {len(tasks)} tasks completed successfully. Report generated."
            return ReflectionResult(True, summary, [], False)

        if has_errors:
            error_tasks = [
                tid for tid, res in results.items() if isinstance(res, dict) and "error" in res
            ]
            issues = [f"Error in {tid}: {results[tid].get('error', 'unknown')}" for tid in error_tasks]
            summary = f"Tasks completed with errors in: {error_tasks}"
            return ReflectionResult(False, summary, issues, False)

        if all_done:
            summary = f"All {len(tasks)} tasks completed."
            return ReflectionResult(True, summary, [], False)

        return None

    def _apply_reflection(
        self, console, state: AgentState, reflection: ReflectionResult
    ) -> None:
        state["reflection"] = {
            "success": reflection.success,
            "summary": reflection.summary,
            "issues": reflection.issues,
            "should_replan": reflection.should_replan,
            "replan_reason": reflection.replan_reason,
        }

        if reflection.success:
            console.print(f"[green]✓ REFLECT: Success - {reflection.summary}[/]")
        else:
            issue_text = ", ".join(reflection.issues) or reflection.summary
            console.print(f"[yellow]⚠ REFLECT: Issues found - {issue_text}[/]")

        if reflection.should_replan:
            console.print("[yellow]   → Will attempt to optimize plan[/]")

    def _log_plan(self, console, plan: PlanResult) -> None:
        if plan.reasoning:
            snippet = plan.reasoning[:200]
            suffix = "..." if len(plan.reasoning) > 200 else ""
            console.print(f"[dim]   Reasoning: {snippet}{suffix}[/]")
        for task in plan.tasks.values():
            description = task.get("description", "")
            snippet = description[:50] + ("..." if len(description) > 50 else "")
            short_name = self._task_short_name(task)
            detail = snippet if snippet and snippet != short_name else ""
            detail_suffix = f" - {detail}" if detail else ""
            console.print(f"   • Task: [bold]{short_name}[/] ({task['type']}){detail_suffix}")

    @staticmethod
    def _task_short_name(task: Task) -> str:
        """Return a concise display name for a task based on its metadata."""
        if name := task.get("name"):
            base = name.strip()
        else:
            description = (task.get("description", "") or "").strip()
            base = description
            for stop in (". ", ".", "!", "?"):
                idx = base.find(stop)
                if idx > 0:
                    base = base[:idx]
                    break
        if not base:
            base = task.get("id", "task")
        base = base.strip()
        return base[:60] + ("..." if len(base) > 60 else "")
