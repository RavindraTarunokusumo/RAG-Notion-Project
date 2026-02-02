import logging
from typing import Literal

from langchain_cohere import ChatCohere

from config.settings import settings

logger = logging.getLogger(__name__)

AgentType = Literal["planner", "researcher", "reasoner", "synthesiser"]

# Model assignments based on task complexity
AGENT_CONFIGS = {
    "planner": {
        "model": settings.models.planner_model,
        "temperature": settings.models.planner_temperature,
        "max_tokens": 1024,
        "description": "Task decomposition - fast, focused"
    },
    "researcher": {
        "model": settings.models.researcher_model,
        "temperature": settings.models.researcher_temperature,
        "max_tokens": 2048,
        "description": "Query formulation - precise, systematic"
    },
    "reasoner": {
        "model": settings.models.reasoner_model,
        "temperature": settings.models.reasoner_temperature,
        "max_tokens": 4096,
        "description": "Complex analysis - powerful, nuanced"
    },
    "synthesiser": {
        "model": settings.models.synthesiser_model,
        "temperature": settings.models.synthesiser_temperature,
        "max_tokens": 4096,
        "description": "Response generation - creative, coherent"
    }
}

def get_agent_llm(agent_type: AgentType) -> ChatCohere:
    """
    Get a configured LLM for the specified agent type.
    
    Model sizes:
    - Planner: command-r-08-2024 (35B) - Fast task decomposition
    - Researcher: command-r-08-2024 (35B) - Efficient query handling
    - Reasoner: command-a-reasoning-08-2025 (111B) - Deep analysis (Flagship)
    - Synthesiser: command-r-plus-08-2024 (104B) - Quality generation
    """
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    config = AGENT_CONFIGS[agent_type]
    
    # Note: langchain-cohere uses 'model' parameter for model name
    llm = ChatCohere(
        model=config["model"],
        temperature=config["temperature"],
        cohere_api_key=settings.cohere_api_key,
        max_tokens=config["max_tokens"]
    )
    
    logger.debug(f"Created LLM for {agent_type}: {config['model']} ({config['description']})")
    
    return llm

def get_model_info(agent_type: AgentType) -> dict:
    """Get information about the model used for an agent."""
    return AGENT_CONFIGS.get(agent_type, {})
