# src/tools/registry.py

## Purpose
Singleton registry for registering, discovering, and health-checking tool agents.

## Main responsibilities
- Register/unregister tool agents.
- Discover agent cards by capability.
- Provide agent lookup for invocation.
- Register built-in agents based on settings flags.

## Key symbols
- `ToolRegistry`
- `get_tool_registry()`
- `register_default_agents(registry)`
