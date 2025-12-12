"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-12].
*  @Description Base class for all agents.
**************************************************************************
"""

from abc import ABC, abstractmethod
from typing import Any

from molx_agent.agents.modules.state import AgentState


class BaseAgent(ABC):
    """Base class for all agents in the molx-agent system."""

    def __init__(self, name: str, description: str) -> None:
        """Initialize the agent.

        Args:
            name: The name of the agent.
            description: A brief description of the agent's purpose.
        """
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        """Execute the agent's main logic.

        Args:
            state: The current agent state.

        Returns:
            The updated agent state.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
