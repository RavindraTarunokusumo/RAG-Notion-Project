# src/tools/base.py

## Purpose
Base protocol and shared models for A2A tool agents.

## Main responsibilities
- Define discoverable `AgentCard` metadata schema.
- Define standardized execution result model `ToolResult`.
- Define abstract `ToolAgent` interface for all tool implementations.

## Key symbols
- `AgentCard`
- `ToolResult`
- `ToolAgent`

## Contract
- Each tool agent must implement `get_agent_card()` and async `execute(task)`.
