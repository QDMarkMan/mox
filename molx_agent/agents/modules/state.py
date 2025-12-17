"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com 
*  @Date [2025-12-17 09:47:09].
*  @Description State definitions for SAR agent.
**************************************************************************
"""

from typing import Any, Literal, Optional, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class Message(TypedDict, total=False):
    """A single message in the conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str


class Task(TypedDict, total=False):
    """A single task in the SAR analysis workflow."""

    id: str
    type: Literal["tool", "data_cleaner", "reporter"]
    description: str
    inputs: dict[str, Any]
    expected_outputs: list[str]
    depends_on: list[str]
    status: Literal["pending", "running", "done", "skipped"]


class AgentState(TypedDict, total=False):
    """State for the SAR agent graph."""

    # Input
    user_query: str

    # Conversation history (for multi-turn)
    messages: Annotated[list[BaseMessage], add_messages]

    # Task management
    tasks: dict[str, Task]
    current_task_id: Optional[str]

    # Results
    results: dict[str, Any]
    final_answer: Optional[str]

    # Error handling
    error: Optional[str]

