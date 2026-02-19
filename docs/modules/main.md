# main.py

## Purpose
`main.py` is the CLI entrypoint for running the RAG pipeline, ingestion utilities, and connection checks.

## Main responsibilities
- Parse command-line arguments.
- Run query execution through the LangGraph pipeline.
- Trigger ingestion (`--ingest`) and optional rebuild mode.
- Validate external integrations with `--test-conn`.
- Configure logging verbosity.

## Key functions
- `run_agentic_rag(query)`: builds initial state, invokes graph, prints answer and sources.
- `test_connection()`: checks tracing and embeddings connectivity.
- `main()`: argument dispatch and command routing.

## Inputs and outputs
- Input: CLI args and optional query string.
- Output: terminal logs + formatted answer/sources text.

## Dependencies
- `src.orchestrator.graph.create_rag_graph`
- `src.utils.tracing.initialize_tracing`
- `src.ingest.run_ingestion` (when ingestion command is used)
- `src.rag.embeddings.get_embeddings` (for connection test)
