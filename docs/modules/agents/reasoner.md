# src/agents/reasoner.py

## Purpose
Implements the Reasoner node that evaluates retrieved evidence against planned sub-tasks.

## Main responsibilities
- Convert retrieved docs into prompt-ready text.
- Request structured analysis from LLM.
- Return per-task findings and an overall assessment.
- Produce explicit error payloads on failure.
- Provide deterministic gap analysis when no documents are available.
- Parse structured model output robustly (raw parse, thinking-strip retry, JSON-substring fallback).

## Key symbols
- `ReasonerOutput`: schema containing `analysis` and `overall_assessment`.
- `get_reasoner_prompt(parser)`: prompt template factory.
- `format_docs(docs)`: helper for source-tagged document text formatting.
- `reasoner_node(state)`: graph node implementation.

## Inputs and outputs
- Input from state: `sub_tasks`, `retrieved_docs`.
- Output to state: `analysis`, `overall_assessment`, `current_agent`.

## Failure handling
- If `state.error` is already populated by an upstream node, reasoner skips model invocation.
- If retrieval produced zero documents, reasoner returns deterministic low-confidence gap analysis for each sub-task.

## Dependencies
- `src.agents.llm_factory.get_agent_llm`
- `src.orchestrator.state.Analysis`
- `src.utils.tracing.agent_trace`
