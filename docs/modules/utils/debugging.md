# `src/utils/debugging.py`

Central observability module for runtime tracing and logging.

## Responsibilities

- Configure app-level logging to console + `logs/<app>.log`.
- Start and manage run-scoped trace sessions.
- Persist structured JSONL trace events under `logs/`.
- Serialize LangChain `Document` objects and complex state payloads safely.
- Compute per-node state deltas for transparent change tracking.

## Core APIs

- `configure_logging(app_name: str = "rag")`
- `debug_run(query, initial_state, mode="invoke")`
- `log_trace_event(event_type, payload)`
- `merge_state(state, update)`

## Related integrations

- `src/orchestrator/graph.py` node wrappers
- `src/tools/client.py` tool discovery/invocation tracing
- `main.py` and `app.py` run-level tracing contexts
