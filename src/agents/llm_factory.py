import functools
import logging
from typing import Literal

from langchain_cohere import ChatCohere

from config.settings import settings

logger = logging.getLogger(__name__)

# --- PATCH: Handle Cohere Reasoning Models in V2 API ---
# The langchain-cohere library (v0.4.6) crashes when encountering 'Thinking' content blocks
# from the V2 API because it expects all content items to be text.
# We patch the cohere client to merge 'Thinking' + 'Text' blocks into a single 'Text' block
# so that LangChain consumes the full response without error.

_PATCH_APPLIED = False

def _apply_cohere_patch(llm_instance):
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    try:
        client = llm_instance.client
        
        # Langchain-cohere calls client.v2.chat for V2 models
        target_client = client.v2 if hasattr(client, "v2") else client
            
        ClientClass = type(target_client)
        
        # Only patch if we haven't already patched this class
        if hasattr(ClientClass, "_is_patched_for_reasoning"):
            _PATCH_APPLIED = True
            return

        original_chat = ClientClass.chat
        
        @functools.wraps(original_chat)
        def patched_chat(self, *args, **kwargs):
            response = original_chat(self, *args, **kwargs)
            
            # Check for V2 structured content
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                content = response.message.content
                if isinstance(content, list) and len(content) > 1:
                    
                    merged_text = ""
                    has_thinking = False
                    
                    for item in content:
                        if hasattr(item, 'thinking'):
                            has_thinking = True
                            merged_text += f"<THINKING>\n{item.thinking}\n</THINKING>\n\n"
                        elif hasattr(item, 'text'):
                            merged_text += item.text
                    
                    if has_thinking:
                        logger.debug("Merging Reasoning trace into response content")
                        from cohere import (
                            TextAssistantMessageResponseContentItem,
                        )
                        
                        new_item = TextAssistantMessageResponseContentItem(
                            type="text",
                            text=merged_text
                        )
                        response.message.content = [new_item]
                        
            return response

        ClientClass.chat = patched_chat
        ClientClass._is_patched_for_reasoning = True
        _PATCH_APPLIED = True
        logger.info(f"Applied compatibility patch for Cohere Reasoning models on {ClientClass.__name__}.")
        
    except Exception as e:
        logger.warning(f"Failed to apply Cohere Reasoning patch: {e}")

# -------------------------------------------------------

AgentType = Literal["planner", "researcher", "reasoner", "synthesiser"]

# Model assignments based on task complexity
def _get_agent_configs() -> dict:
    """Get dynamic agent configurations based on current settings."""
    return {
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
    configs = _get_agent_configs()
    if agent_type not in configs:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    config = configs[agent_type]
    
    # Note: langchain-cohere uses 'model' parameter for model name
    llm = ChatCohere(
        model=config["model"],
        temperature=config["temperature"],
        cohere_api_key=settings.cohere_api_key,
        max_tokens=config["max_tokens"]
    )
    
    # Apply patch if using a reasoning model (or globally)
    _apply_cohere_patch(llm)
    
    logger.debug(f"Created LLM for {agent_type}: {config['model']} ({config['description']})")
    
    return llm

def get_model_info(agent_type: AgentType) -> dict:
    """Get information about the model used for an agent."""
    return _get_agent_configs().get(agent_type, {})
