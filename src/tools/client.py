"""NRAG-033: A2A Discovery & Invocation Client."""

import asyncio
import logging
import time
from typing import Any

from src.tools.base import AgentCard, ToolResult
from src.tools.registry import ToolRegistry
from src.utils.debugging import log_trace_event

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
        log_trace_event(
            "tool_discovery",
            {
                "capability": capability,
                "agents": [card.name for card in cards],
                "count": len(cards),
            },
        )
        return cards

    async def invoke_tool(
        self, agent_name: str, task: dict[str, Any], timeout: float = 30.0
    ) -> ToolResult:
        """Invoke a tool agent by name with timeout handling."""
        agent = self._registry.get_agent(agent_name)
        started = time.perf_counter()
        if agent is None:
            result = ToolResult(
                success=False,
                data=None,
                error=f"Agent '{agent_name}' not found in registry",
                agent_name=agent_name,
            )
            log_trace_event(
                "tool_invoke_end",
                {
                    "agent_name": agent_name,
                    "success": result.success,
                    "error": result.error,
                    "duration_ms": 0.0,
                    "task": task,
                },
            )
            return result

        log_trace_event(
            "tool_invoke_start",
            {
                "agent_name": agent_name,
                "timeout": timeout,
                "task": task,
            },
        )
        try:
            result = await asyncio.wait_for(agent.execute(task), timeout=timeout)
            result.agent_name = agent_name
            log_trace_event(
                "tool_invoke_end",
                {
                    "agent_name": agent_name,
                    "success": result.success,
                    "error": result.error,
                    "metadata": result.metadata,
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        3,
                    ),
                    "task": task,
                },
            )
            return result
        except TimeoutError:
            logger.warning(f"Tool agent '{agent_name}' timed out after {timeout}s")
            result = ToolResult(
                success=False,
                data=None,
                error=f"Agent '{agent_name}' timed out after {timeout}s",
                agent_name=agent_name,
            )
            log_trace_event(
                "tool_invoke_end",
                {
                    "agent_name": agent_name,
                    "success": result.success,
                    "error": result.error,
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        3,
                    ),
                    "task": task,
                },
            )
            return result
        except Exception as e:
            logger.exception(f"Tool agent '{agent_name}' failed")
            result = ToolResult(
                success=False,
                data=None,
                error=str(e),
                agent_name=agent_name,
            )
            log_trace_event(
                "tool_invoke_end",
                {
                    "agent_name": agent_name,
                    "success": result.success,
                    "error": result.error,
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        3,
                    ),
                    "task": task,
                },
            )
            return result

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
        log_trace_event(
            "tool_selection",
            {
                "task_description": task_description,
                "selected_agent": best_card.name if best_card else None,
                "score": best_score,
                "candidate_count": len(available_agents),
            },
        )
        return best_card
