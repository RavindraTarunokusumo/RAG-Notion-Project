# src/tools/client.py

## Purpose
Client used by core agents to discover and invoke A2A tool agents.

## Main responsibilities
- Discover available agent cards from registry.
- Invoke selected tool with timeout handling.
- Choose best agent via `can_handle` scoring.

## Key symbol
- `A2AToolClient`

## Core methods
- `discover_agents(capability=None)`
- `invoke_tool(agent_name, task, timeout=30.0)`
- `select_best_agent(task_description, available_agents)`
