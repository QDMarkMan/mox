"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-16].
*  @Description Molx agent, the main agent orchestrator for molx-agent.
*               Refactored to use ReAct pattern with LangGraph.
**************************************************************************
"""

import logging
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from rich.console import Console

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.modules.llm import get_llm
from molx_agent.agents.react_tools import (
    clean_data_tool,
    sar_analysis_tool,
    generate_report_tool,
)

logger = logging.getLogger(__name__)
console = Console()

SYSTEM_PROMPT = """You are MolX, an expert AI drug design assistant specializing in SAR (Structure-Activity Relationship) analysis.

Your goal is to help users analyze chemical data, identify SAR trends, and generate reports.

You have access to the following tools:
1. `clean_data_tool`: Extract and clean chemical data from text or files. ALWAYS use this first to get data.
2. `sar_analysis_tool`: Perform SAR analysis (R-group decomposition, etc.) on the cleaned data.
3. `generate_report_tool`: Generate a comprehensive HTML report from the analysis results.

**Workflow:**
1.  **Understand the Goal**: Read the user's query.
2.  **Get Data**: Use `clean_data_tool` to parse input data. This tool returns a file path to the cleaned data.
3.  **Analyze**: Use `sar_analysis_tool` with the file path from step 2.
4.  **Report**: Use `generate_report_tool` with the results from step 3 to create a final report.

**Important:**
- Always show your reasoning ("Thinking") before taking an action.
- When calling `sar_analysis_tool`, prefer passing the `file_path` returned by `clean_data_tool` to avoid token limits.
- If the user provides a file path, pass it to `clean_data_tool`.
- If the user provides raw data in the prompt, pass it to `clean_data_tool`.
"""


class MolxAgent(BaseAgent):
    """Main orchestrator agent using ReAct pattern."""

    def __init__(self) -> None:
        super().__init__(
            name="molx",
            description="Main orchestrator for SAR analysis workflow",
        )
        
        # Define tools
        self.tools = [clean_data_tool, sar_analysis_tool, generate_report_tool]
        
        # Initialize LLM with tools
        self.llm = get_llm()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build graph
        self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state graph."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        # Define edges
        workflow.add_edge(START, "agent")
        
        # Conditional edge: agent -> tools OR end
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
        )
        
        # Tools always go back to agent
        workflow.add_edge("tools", "agent")

        self._graph = workflow.compile()

    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM and print thinking."""
        messages = state["messages"]
        
        # Trim messages to avoid context length exceeded
        # Keep System Prompt (first message) and last 10 messages
        if len(messages) > 11:
            trimmed_messages = [messages[0]] + messages[-10:]
        else:
            trimmed_messages = messages
        
        # Invoke LLM
        response = self.llm_with_tools.invoke(trimmed_messages)
        
        # Print Thinking (content before tool calls)
        if response.content:
            console.print("\n[bold cyan]🧠 Thinking:[/]")
            console.print(f"[cyan]{response.content}[/]\n")
            
        return {"messages": [response]}

    def run(self, state: AgentState) -> AgentState:
        """Execute the agent workflow.
        
        Args:
            state: Initial agent state.
            
        Returns:
            Final agent state.
        """
        # Initialize messages if empty
        if not state.get("messages"):
            state["messages"] = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=state.get("user_query", "")),
            ]
            
        # Run the graph
        # We need to stream events to show tool calls if desired, 
        # but for now we just run invoke and let _call_model handle thinking print.
        # ToolNode automatically logs tool calls if configured, but we can add custom logging if needed.
        
        final_state = self._graph.invoke(state)
        return final_state


def run_sar_agent(user_query: str) -> dict:
    """Run the SAR agent workflow.
    
    Args:
        user_query: The user's query string.
        
    Returns:
        Final agent state.
    """
    agent = MolxAgent()
    state = AgentState(user_query=user_query)
    final_state = agent.run(state)
    return final_state


class ChatSession:
    """Interactive chat session wrapper for MolxAgent."""
    
    def __init__(self):
        self.agent = MolxAgent()
        self.state = AgentState(messages=[], tasks={}, results={})

    def send(self, user_input: str) -> str:
        """Send user input to the agent and get response."""
        if not self.state.get("messages"):
            # First turn
            self.state["user_query"] = user_input
        else:
            # Subsequent turns
            self.state["messages"].append(HumanMessage(content=user_input))
            
        self.state = self.agent.run(self.state)
        
        # Return last message content
        messages = self.state["messages"]
        if messages:
            return messages[-1].content
        return ""

    def clear(self):
        """Clear conversation history."""
        self.state = AgentState(messages=[], tasks={}, results={})

    def get_history(self) -> list[dict]:
        """Get conversation history."""
        history = []
        for msg in self.state.get("messages", []):
            role = "user"
            if msg.type == "ai":
                role = "agent"
            elif msg.type == "system":
                role = "system"
            elif msg.type == "human":
                role = "user"
            
            history.append({"role": role, "content": msg.content})
        return history
