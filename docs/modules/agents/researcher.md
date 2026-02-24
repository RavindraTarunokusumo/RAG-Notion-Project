# src/agents/researcher.py

## Purpose
Implements the Researcher node that converts planner sub-tasks into optimized retrieval queries and collects documents.

## Main responsibilities
- Build a prompt for query optimization.
- Use LLM structured output (`SearchQueryOutput`) for optimized queries.
- Retrieve documents with reranking.
- Deduplicate results by content hash.
- Attach retrieval provenance metadata to each document.

## Key symbols
- `SearchQueryOutput`: pydantic schema for optimized query list + filters.
- `get_query_optimizer_prompt(parser)`: prompt template factory.
- `researcher_node(state)`: node entrypoint used by the graph.

## Inputs and outputs
- Input from state: `sub_tasks`.
- Output to state: `retrieved_docs`, `retrieval_metadata`, `current_agent`.

### Retrieval metadata fields
- `total_docs`: number of unique documents returned.
- `processed_tasks`: number of sub-tasks attempted.
- `failed_tasks`: number of sub-tasks that raised errors.
- `fatal`: whether execution stopped due to a fatal provider error.
- `error_type`: classified fatal/non-fatal error category (`quota_exhausted`, `auth_failed`, `transient`).
- `aborted`: whether researcher exited early before processing all tasks.

## Dependencies
- `src.agents.llm_factory.get_agent_llm`
- `src.rag.retriever.get_retriever`
- `src.utils.tracing.agent_trace`
- LangChain parsers/prompts/document types

## Operational note
- Includes `time.sleep(...)` delays to mitigate trial-tier API rate limits.
- Fatal provider failures (`429` trial/quota, `401`, `403`) now abort the loop early and set `state.error`.
