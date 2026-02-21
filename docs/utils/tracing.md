# Utils: tracing

Source module: `src/utils/tracing.py`

## Purpose

Initialize optional LangSmith tracing and provide trace wrappers for agent operations.

## Runtime Behavior

- Tracing is controlled by environment configuration.
- Tracing setup failures should be non-fatal to core query execution.
- Agent spans should preserve enough context for debugging and auditability.

## Operational Notes

- Verify environment variables in `.env` when tracing is expected but absent.
- Keep tracing paths lightweight to avoid unnecessary runtime overhead.
