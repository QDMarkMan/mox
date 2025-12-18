"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description MolxAgent - Main orchestrator using ReAct pattern.
*               Integrates PlannerAgent for Think/Reflect/Optimize.
**************************************************************************
"""

import logging
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from rich.console import Console

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.planner import PlannerAgent
from molx_agent.agents.data_cleaner import DataCleanerAgent
from molx_agent.agents.sar import SARAgent
from molx_agent.agents.reporter import ReporterAgent
from molx_agent.agents.intent_classifier import IntentClassifierAgent, Intent

logger = logging.getLogger(__name__)
console = Console()


class MolxAgent(BaseAgent):
    """Main orchestrator agent using ReAct pattern.
    
    ReAct Loop:
    1. CLASSIFY: IntentClassifierAgent determines query type
    2. THINK: PlannerAgent creates task DAG
    3. ACT: Execute tasks via worker agents
    4. REFLECT: PlannerAgent evaluates results
    5. OPTIMIZE: Replan if needed (loop back to THINK)
    """

    def __init__(self) -> None:
        super().__init__(
            name="molx",
            description="Main orchestrator for SAR analysis using ReAct pattern",
        )
        
        # Initialize sub-agents
        self.intent_classifier = IntentClassifierAgent()
        self.planner = PlannerAgent()
        self.workers = {
            "data_cleaner": DataCleanerAgent(),
            "sar": SARAgent(),
            "reporter": ReporterAgent(),
        }

    def run(self, state: AgentState) -> AgentState:
        """Execute the ReAct workflow.
        
        Args:
            state: Initial agent state with user_query.
            
        Returns:
            Final agent state with results.
        """
        console.print("\n[bold blue]═════════════════════════════[/]")
        console.print("[bold blue]       🧪 MolX Agent       [/]")
        console.print("[bold blue]═════════════════════════════[/]\n")
        
        user_query = state.get("user_query", "")
        
        # Step 0: Classify intent using IntentClassifierAgent
        state = self.intent_classifier.run(state)
        
        intent = state.get("intent", Intent.SAR_ANALYSIS)
        
        if not self.intent_classifier.is_supported(intent):
            response = self.intent_classifier.get_response(intent)
            state["final_response"] = response
            state["messages"] = state.get("messages", []) + [AIMessage(content=response)]
            return state
        
        # Initialize state
        state["iteration"] = 0
        state["tasks"] = {}
        state["results"] = {}
        
        # ReAct Loop
        max_loops = 3  # Safety limit
        loop_count = 0
        
        while loop_count < max_loops:
            loop_count += 1
            
            # Check if we need to THINK (no pending tasks from OPTIMIZE)
            tasks = state.get("tasks", {})
            has_pending = any(t.get("status") == "pending" for t in tasks.values())
            
            if not has_pending:
                # THINK: Create new plan
                state = self.planner.think(state)
                
                if state.get("error"):
                    console.print(f"[red]Planning failed: {state['error']}[/]")
                    break
            
            # ACT: Execute all pending tasks
            state = self._execute_tasks(state)
            
            # REFLECT: Evaluate results
            state = self.planner.reflect(state)
            
            # Check if should continue
            reflection = state.get("reflection", {})
            if reflection.get("success", False):
                # Success - stop loop
                break
            
            if not reflection.get("should_replan", False):
                # No replan needed - stop loop
                break
            
            # OPTIMIZE: Replan if needed
            state = self.planner.optimize(state)
            
            # If optimize created new tasks, continue to ACT
            # Otherwise, stop
            tasks = state.get("tasks", {})
            has_new_pending = any(t.get("status") == "pending" for t in tasks.values())
            if not has_new_pending:
                console.print("[yellow]⚠ No new tasks created, stopping[/]")
                break
        
        if loop_count >= max_loops:
            console.print(f"[yellow]⚠ Reached max iterations ({max_loops}), stopping[/]")
        
        # Generate final response
        state = self._generate_final_response(state)
        
        console.print("\n[bold blue]═══════════════════════════════════════════[/]")
        console.print("[bold blue]              ✅ Task Complete          [/]")
        console.print("[bold blue]═══════════════════════════════════════════[/]\n")
        
        return state

    def _execute_tasks(self, state: AgentState) -> AgentState:
        """Execute pending tasks via worker agents.
        
        Args:
            state: State with tasks to execute.
            
        Returns:
            Updated state with results.
        """
        console.print("\n[bold green]⚡ ACT: Executing tasks...[/]")
        
        tasks = state.get("tasks", {})
        
        while True:
            # Find next executable task
            task_id = state.get("current_task_id")
            if not task_id:
                task_id = self.planner._pick_next_task(state)
                
            if not task_id:
                break  # No more tasks to execute
                
            task = tasks.get(task_id)
            if not task:
                break
                
            task_type = task.get("type", "")
            console.print(f"\n   [cyan]→ Executing: {task_id} ({task_type})[/]")
            
            # Get worker agent
            worker = self.workers.get(task_type)
            if not worker:
                console.print(f"   [red]Unknown worker type: {task_type}[/]")
                task["status"] = "error"
                state["results"][task_id] = {"error": f"Unknown worker: {task_type}"}
                state["current_task_id"] = None
                continue
            
            # Inject inputs from previous results
            self._inject_task_inputs(state, task)
            
            # Execute worker
            try:
                state["current_task_id"] = task_id
                state = worker.run(state)
                
                # Check result
                result = state.get("results", {}).get(task_id, {})
                if result.get("error"):
                    task["status"] = "error"
                    console.print(f"   [red]✗ Task {task_id} failed: {result['error']}[/]")
                else:
                    task["status"] = "done"
                    console.print(f"   [green]✓ Task {task_id} completed[/]")
                    
            except Exception as e:
                task["status"] = "error"
                state["results"][task_id] = {"error": str(e)}
                console.print(f"   [red]✗ Task {task_id} exception: {e}[/]")
                logger.error(f"Task {task_id} failed: {e}")
            
            state["current_task_id"] = None
            
        return state

    def _inject_task_inputs(self, state: AgentState, task: dict) -> None:
        """Inject outputs from previous tasks as inputs to current task."""
        results = state.get("results", {})
        depends_on = task.get("depends_on", [])
        
        for dep_id in depends_on:
            dep_result = results.get(dep_id, {})
            if dep_result:
                # Merge dependency outputs into task inputs
                if "inputs" not in task:
                    task["inputs"] = {}
                task["inputs"][f"{dep_id}_output"] = dep_result

    def _generate_final_response(self, state: AgentState) -> AgentState:
        """Generate final response based on results."""
        results = state.get("results", {})
        reflection = state.get("reflection", {})
        
        # Find report path if available
        report_path = None
        for tid, result in results.items():
            if isinstance(result, dict):
                # Check for report output
                output_files = result.get("output_files", {})
                if "html" in output_files:
                    report_path = output_files["html"]
                    break
                # Check for direct path
                if result.get("report_path"):
                    report_path = result["report_path"]
                    break
        
        # Build response
        summary = reflection.get("summary", "Analysis complete.")
        
        if report_path:
            response = f"✅ {summary}\n\n📊 Report generated: {report_path}"
        else:
            response = f"✅ {summary}"
            
        state["final_response"] = response
        state["messages"] = state.get("messages", []) + [AIMessage(content=response)]
        
        return state


def run_sar_agent(user_query: str) -> dict:
    """Run the SAR agent workflow.
    
    Args:
        user_query: The user's query string.
        
    Returns:
        Final agent state.
    """
    agent = MolxAgent()
    state = AgentState(user_query=user_query, messages=[], tasks={}, results={})
    final_state = agent.run(state)
    return final_state


class ChatSession:
    """Interactive chat session wrapper for MolxAgent."""
    
    def __init__(self):
        self.agent = MolxAgent()
        self.state = AgentState(messages=[], tasks={}, results={})

    def send(self, user_input: str) -> str:
        """Send user input to the agent and get response."""
        self.state["user_query"] = user_input
        self.state["messages"].append(HumanMessage(content=user_input))
        
        self.state = self.agent.run(self.state)
        
        return self.state.get("final_response", "")

    def clear(self):
        """Clear conversation history."""
        self.state = AgentState(messages=[], tasks={}, results={})

    def get_history(self) -> list[dict]:
        """Get conversation history."""
        history = []
        for msg in self.state.get("messages", []):
            role = "user"
            if hasattr(msg, 'type'):
                if msg.type == "ai":
                    role = "agent"
                elif msg.type == "system":
                    role = "system"
                elif msg.type == "human":
                    role = "user"
            
            content = msg.content if hasattr(msg, 'content') else str(msg)
            history.append({"role": role, "content": content})
        return history
