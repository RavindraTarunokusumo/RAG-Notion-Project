# src/agents/synthesiser.py

## Purpose
Implements the Synthesiser node that writes the final response and emits structured source metadata.

## Main responsibilities
- Build a synthesis prompt with citation instructions.
- Transform reasoner output into prompt context.
- Extract, enrich, and deduplicate sources from retrieved documents.
- Return final markdown answer plus source list.

## Key symbols
- `get_synthesiser_prompt()`: prompt template with citation requirements.
- `synthesiser_node(state)`: graph node implementation.

## Inputs and outputs
- Input from state: `query`, `analysis`, `planning_reasoning`, `overall_assessment`, `retrieved_docs`.
- Output to state: `final_answer`, `sources`, `current_agent`.

## Dependencies
- `src.agents.llm_factory.get_agent_llm`
- `src.utils.tracing.agent_trace`
- LangChain `StrOutputParser`

## Source behavior
- Supports source-specific metadata for arXiv and Notion.
- Deduplicates by `source:title` key before numbering citations.
