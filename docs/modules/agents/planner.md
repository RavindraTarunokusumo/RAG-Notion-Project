# src/agents/planner.py

## Purpose
Planner node that decomposes user query into structured retrieval sub-tasks.

## Main responsibilities
- Build planning prompt.
- Request structured JSON output from LLM.
- Return `sub_tasks` and planning rationale into shared state.

## Key symbols
- `PlanOutput`
- `get_planner_prompt(parser)`
- `planner_node(state)`

## Dependencies
- `src.agents.llm_factory.get_agent_llm`
- `src.orchestrator.state.SubTask`
- `src.utils.tracing.agent_trace`
