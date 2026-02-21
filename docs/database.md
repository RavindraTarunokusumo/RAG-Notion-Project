# Data Model and Invariants

## Primary Data Sources

- Notion database metadata
- arXiv paper metadata and full text (when available)

## Notion Schema Expectations

`src/loaders/notion_loader.py` expects fields equivalent to:

- `Title`
- `Topic`
- `Keywords`
- `URL`
- `Type`
- `Publication Date`
- `Notes`

If the schema changes, update loader parsing and downstream metadata merge logic in the same iteration.

## Core Document Invariants

- Every stored chunk must preserve source identity metadata.
- Metadata values must be serializable for vector store persistence.
- Source URLs should remain traceable to the original system (Notion/arXiv).
- Merged documents should keep both Notion catalog context and arXiv enrichment.

## Vector Store Invariants

- Chroma persistence path is configured through settings.
- Embedding model and retrieval settings are centrally configured.
- Retrieval should fail gracefully and return deterministic shapes for downstream agents.

## AgentState Data Contract Invariants

Shared state in `src/orchestrator/state.py` must preserve:

- `query` as immutable user intent input
- `sub_tasks` as planner output for downstream steps
- `retrieved_docs` as researcher evidence payload
- `analysis` and `overall_assessment` as reasoner outputs
- `final_answer` and `sources` as synthesiser outputs
- `tool_results` as optional tool invocation envelope

Changes to state fields require synchronized updates to:

- graph node logic
- tests
- architecture docs
