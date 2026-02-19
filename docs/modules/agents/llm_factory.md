# src/agents/llm_factory.py

## Purpose
Factory for per-agent Cohere chat models, including compatibility patching for reasoning responses.

## Main responsibilities
- Define model/temperature/token settings per agent role.
- Instantiate `ChatCohere` for planner/researcher/reasoner/synthesiser/tool-agent.
- Apply runtime patch that merges Cohere V2 Thinking+Text blocks for LangChain compatibility.

## Key symbols
- `_apply_cohere_patch(llm_instance)`
- `_get_agent_configs()`
- `get_agent_llm(agent_type)`
- `get_model_info(agent_type)`

## Supported agent types
- `planner`, `researcher`, `reasoner`, `synthesiser`, `tool_agent`
