# Testing Strategy

## Validation Gates

Default checks:

```bash
uv run ruff check .
uv run pytest
```

Run targeted tests during development and full suite before task completion when feasible.

## Test Organization

- `tests/test_agents.py` for core agent behavior
- `tests/test_tool_agents.py` for tool-agent framework behavior
- `tests/test_session_manager.py` and related UI/session tests
- `tests/test_e2e.py` for end-to-end execution path

## Test Categories

- Unit tests for deterministic logic and parsing boundaries.
- Integration tests for loader, retrieval, and orchestration interactions.
- End-to-end tests for complete query and answer flow.

## Environment-Sensitive Tests

- Some tests may require API keys or pre-ingested vector data.
- When environment constraints block tests, record:
  - exact failing command
  - reason for failure
  - unverified coverage area

## Adding Tests

- Place all new tests under `tests/`.
- Add regression tests for bug fixes.
- Keep fixtures small and explicit.
