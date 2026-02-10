"""NRAG-033: A2A Discovery & Invocation Client."""

import asyncio
import logging
from typing import Any

from src.tools.base import AgentCard, ToolResult
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class A2AToolClient:
    """Client for core agents to discover and invoke tool agents via A2A protocol."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._card_cache: dict[str, AgentCard] = {}

    async def discover_agents(self, capability: str | None = None) -> list[AgentCard]:
        """Discover available tool agents, optionally filtered by capability."""
        cards = self._registry.discover(capability)
        # Update cache
        for card in cards:
            self._card_cache[card.name] = card
        return cards

    async def invoke_tool(
        self, agent_name: str, task: dict[str, Any], timeout: float = 30.0
    ) -> ToolResult:
        """Invoke a tool agent by name with timeout handling."""
        agent = self._registry.get_agent(agent_name)
        if agent is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"Agent '{agent_name}' not found in registry",
                agent_name=agent_name,
            )

        try:
            result = await asyncio.wait_for(agent.execute(task), timeout=timeout)
            result.agent_name = agent_name
            return result
        except TimeoutError:
            logger.warning(f"Tool agent '{agent_name}' timed out after {timeout}s")
            return ToolResult(
                success=False,
                data=None,
                error=f"Agent '{agent_name}' timed out after {timeout}s",
                agent_name=agent_name,
            )
        except Exception as e:
            logger.exception(f"Tool agent '{agent_name}' failed")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                agent_name=agent_name,
            )

    def select_best_agent(
        self, task_description: str, available_agents: list[AgentCard]
    ) -> AgentCard | None:
        """Select the best agent for a task based on can_handle confidence scores."""
        best_card: AgentCard | None = None
        best_score = 0.0

        for card in available_agents:
            agent = self._registry.get_agent(card.name)
            if agent is None:
                continue
            score = agent.can_handle(task_description)
            if score > best_score:
                best_score = score
                best_card = card

        if best_card and best_score > 0.0:
            logger.debug(
                f"Selected agent '{best_card.name}' with confidence {best_score:.2f}"
            )
        return best_card
