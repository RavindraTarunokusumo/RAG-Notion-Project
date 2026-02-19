# src/orchestrator/state.py

## Purpose
Defines typed shared state structures passed between orchestration nodes.

## Types
- `SubTask`: planned work item with `id`, `task`, `priority`, `keywords`.
- `Analysis`: reasoner output per sub-task (findings, evidence, gaps, confidence).
- `AgentState`: full cross-agent state object.

## AgentState fields
- Input: `query`
- Planner: `sub_tasks`, `planning_reasoning`
- Researcher: `retrieved_docs`, `retrieval_metadata`
- Reasoner: `analysis`, `overall_assessment`
- Synthesiser: `final_answer`, `sources`
- Tool outputs: `tool_results`
- Error/status: `error`, `current_agent`

## Dependencies
- `langchain_core.documents.Document`
- `typing_extensions.TypedDict`
