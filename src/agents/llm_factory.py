import logging
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import AgentModelProfile, settings
from src.agents.providers import get_chat_provider
from src.agents.providers.base import ChatModelRequest

logger = logging.getLogger(__name__)

AgentType = Literal[
    "planner",
    "researcher",
    "reasoner",
    "synthesiser",
    "tool_agent",
]


def _get_agent_configs() -> dict[AgentType, dict]:
    """Get dynamic agent configurations based on current settings."""
    return {
        "planner": {
            "profile": settings.models.planner,
            "description": "Task decomposition - fast, focused",
        },
        "researcher": {
            "profile": settings.models.researcher,
            "description": "Query formulation - precise, systematic",
        },
        "reasoner": {
            "profile": settings.models.reasoner,
            "description": "Complex analysis - nuanced synthesis",
        },
        "synthesiser": {
            "profile": settings.models.synthesiser,
            "description": "Response generation - coherent and clear",
        },
        "tool_agent": {
            "profile": settings.models.tool_agent,
            "description": "Tool agent tasks - utility, focused",
        },
    }


def _build_chat_model(profile: AgentModelProfile) -> BaseChatModel:
    provider = get_chat_provider(profile.provider)
    return provider.create_chat_model(
        ChatModelRequest(
            model=profile.model,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
        )
    )


def get_agent_llm(agent_type: AgentType) -> BaseChatModel:
    """
    Get a configured LLM for the specified agent type.
    """
    configs = _get_agent_configs()
    if agent_type not in configs:
        raise ValueError(f"Unknown agent type: {agent_type}")

    config = configs[agent_type]
    profile: AgentModelProfile = config["profile"]
    llm = _build_chat_model(profile)

    logger.debug(
        "Created LLM for %s via provider=%s model=%s (%s)",
        agent_type,
        profile.provider,
        profile.model,
        config["description"],
    )
    return llm


def get_model_info(agent_type: AgentType) -> dict:
    """Get information about the model used for an agent."""
    configs = _get_agent_configs()
    config = configs.get(agent_type)
    if config is None:
        return {}

    profile: AgentModelProfile = config["profile"]
    return {
        "provider": profile.provider,
        "model": profile.model,
        "temperature": profile.temperature,
        "max_tokens": profile.max_tokens,
        "description": config["description"],
    }
