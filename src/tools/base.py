"""NRAG-027: A2A Tool Agent Framework - Base classes and models."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class AgentCard:
    """A2A Agent Card describing a tool agent's capabilities and interface."""

    name: str
    description: str
    version: str
    capabilities: list[str]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    endpoint: str  # Logical identifier, not HTTP URL

    def matches_capability(self, capability: str) -> bool:
        """Check if this agent supports a given capability."""
        return capability in self.capabilities


class ToolResult(BaseModel):
    """Standardized result from a tool agent execution."""

    success: bool
    data: Any
    error: str | None = None
    metadata: dict = {}
    agent_name: str = ""


class ToolAgent(ABC):
    """Abstract base class for all tool agents following A2A protocol."""

    @abstractmethod
    def get_agent_card(self) -> AgentCard:
        """Return the agent card describing this tool's capabilities."""
        ...

    @abstractmethod
    async def execute(self, task: dict[str, Any]) -> ToolResult:
        """Execute the tool agent's task and return a result."""
        ...

    def can_handle(self, task_description: str) -> float:
        """
        Return a confidence score (0.0 - 1.0) for handling the given task.

        Subclasses should override this with keyword/capability matching logic.
        Default returns 0.0 (cannot handle).
        """
        return 0.0

    def health_check(self) -> bool:
        """Check if the tool agent is operational. Default returns True."""
        return True

    @property
    def name(self) -> str:
        return self.get_agent_card().name
