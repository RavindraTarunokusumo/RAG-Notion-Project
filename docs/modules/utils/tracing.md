# src/utils/tracing.py

## Purpose
LangSmith tracing bootstrap and lightweight decorator for agent function tracing.

## Main responsibilities
- Set `LANGCHAIN_*` environment variables from project settings.
- Provide `agent_trace` decorator backed by `langsmith.traceable`.

## Key functions
- `initialize_tracing()`
- `agent_trace(agent_name, model=None, tags=None)`
