"""State definitions for SAR agent."""

from typing import Any, Literal, Optional

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
    messages: list[Message]

    # Task management
    tasks: dict[str, Task]
    current_task_id: Optional[str]

    # Results
    results: dict[str, Any]
    final_answer: Optional[str]

    # Error handling
    error: Optional[str]

