# Execution Plan: ARCHITECTURE.md System Map

## Objective
Create a root-level `ARCHITECTURE.md` that documents the implemented system design, core components (Orchestrator, Agents, RAG), and end-to-end data flow.

## Source of Truth
- `src/orchestrator/graph.py`
- `src/orchestrator/state.py`
- `src/agents/*.py`
- `src/rag/*.py`
- `src/loaders/*.py`
- `src/ingest.py`
- `main.py`
- `app.py`
- `src/tools/*.py`
- `config/settings.py`

## Plan
1. Confirm runtime topology from orchestrator and agent nodes.
2. Confirm ingestion and retrieval paths from loaders + RAG modules.
3. Confirm entry points and execution modes from CLI/UI files.
4. Document tool-agent subsystem and current integration boundary.
5. Write `ARCHITECTURE.md` with diagrams, component responsibilities, and data flow.

## Validation
- Ensure component names in `ARCHITECTURE.md` match module names in `src/`.
- Ensure graph order matches `planner -> researcher -> reasoner -> synthesiser`.
- Ensure ingestion path matches `Notion/Arxiv -> chunking -> Chroma`.

## Status
- Completed
