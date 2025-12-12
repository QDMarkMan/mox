"""State definitions for SAR agent."""

from typing import Any, Literal, Optional

from typing_extensions import TypedDict


class Task(TypedDict, total=False):
    """A single task in the SAR analysis workflow."""

    id: str
    type: Literal["literature", "chemo", "bio", "meta"]
    description: str
    inputs: dict[str, Any]
    expected_outputs: list[str]
    depends_on: list[str]
    status: Literal["pending", "running", "done", "skipped"]


class AgentState(TypedDict, total=False):
    """State for the SAR agent graph."""

    # Input
    user_query: str

    # Task management
    tasks: dict[str, Task]
    current_task_id: Optional[str]

    # Results
    results: dict[str, Any]
    final_answer: Optional[str]

    # Error handling
    error: Optional[str]
