"""NRAG-027: Tool Agent Registry - Singleton registry for discovering and managing tool agents."""

import logging
from typing import TYPE_CHECKING

from src.tools.base import AgentCard, ToolAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Singleton registry for tool agents, following the same pattern as vectorstore/session_manager."""

    _instance: "ToolRegistry | None" = None
    _agents: dict[str, ToolAgent]

    def __init__(self) -> None:
        self._agents = {}

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    def register(self, agent: ToolAgent) -> None:
        """Register a tool agent."""
        card = agent.get_agent_card()
        self._agents[card.name] = agent
        logger.info(f"Registered tool agent: {card.name} (v{card.version})")

    def unregister(self, name: str) -> None:
        """Remove a tool agent from the registry."""
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Unregistered tool agent: {name}")

    def discover(self, capability: str | None = None) -> list[AgentCard]:
        """Discover registered agents, optionally filtered by capability."""
        cards = []
        for agent in self._agents.values():
            card = agent.get_agent_card()
            if capability is None or card.matches_capability(capability):
                cards.append(card)
        return cards

    def get_agent(self, name: str) -> ToolAgent | None:
        """Get a registered tool agent by name."""
        return self._agents.get(name)

    def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all registered agents."""
        results = {}
        for name, agent in self._agents.items():
            try:
                results[name] = agent.health_check()
            except Exception:
                logger.exception(f"Health check failed for {name}")
                results[name] = False
        return results


def get_tool_registry() -> ToolRegistry:
    """Get the singleton ToolRegistry instance."""
    return ToolRegistry.get_instance()


def register_default_agents(registry: ToolRegistry) -> None:
    """Register all built-in tool agents based on settings."""
    from config.settings import settings

    tool_config = settings.tool_agents

    if not tool_config.enabled:
        logger.info("Tool agents are disabled in settings.")
        return

    if tool_config.web_searcher_enabled:
        from src.tools.web_searcher import WebSearcherAgent

        registry.register(WebSearcherAgent())

    if tool_config.code_executor_enabled:
        from src.tools.code_executor import CodeExecutorAgent

        registry.register(CodeExecutorAgent())

    if tool_config.citation_validator_enabled:
        from src.tools.citation_validator import CitationValidatorAgent

        registry.register(CitationValidatorAgent())

    if tool_config.math_solver_enabled:
        from src.tools.math_solver import MathSolverAgent

        registry.register(MathSolverAgent())

    if tool_config.diagram_generator_enabled:
        from src.tools.diagram_generator import DiagramGeneratorAgent

        registry.register(DiagramGeneratorAgent())

    logger.info(f"Registered {len(registry.discover())} default tool agents.")
