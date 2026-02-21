# src/agents/llm_factory.py

## Purpose
Factory for per-agent chat models with provider abstraction.

## Main responsibilities
- Resolve per-agent provider/model config.
- Build provider-specific chat models through a registry.
- Expose provider-agnostic model info for UI/debugging.

## Key symbols
- `_get_agent_configs()`
- `_build_chat_model(profile)`
- `get_agent_llm(agent_type)`
- `get_model_info(agent_type)`

## Supported agent types
- `planner`, `researcher`, `reasoner`, `synthesiser`, `tool_agent`
